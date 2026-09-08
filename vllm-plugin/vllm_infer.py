#!/usr/bin/env python3
"""Run rumik-oss-1 (NVFP4, vLLM-exported) through vLLM with the TinyAya
plugin, decode the resulting audio tokens via the Mimi codec, save wavs, and
print timing -- the "does this actually work, and is it fast" check.

Run inside vllm/vllm-openai:v0.20.0 (needs vLLM + the compiled Blackwell
NVFP4 kernels; our .venv doesn't have vLLM installed).
"""

import argparse
import re
import time
import wave
from pathlib import Path

import torch

AUDIO_RE = re.compile(r"^<(\d+)_(\d+)>$")

PROMPTS = [
    ("Ira", "Hey! This is a quick test of the rumik open source model running locally."),
    ("Siya", "<description=\"happy, fast pace\"> Namaste, aaj ka din kaisa hai?"),
    ("Zoya", "This is a longer sentence meant to measure steady state decode throughput across many audio tokens, so we can get a stable tokens per second estimate."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="/models/rumik-oss-1-nvfp4-vllm")
    ap.add_argument("--codec-dir", default=None, help="defaults to <repo-dir>/codec")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out-dir", default="/models/outputs_vllm")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--batch", action="store_true", help="submit all prompts as one llm.generate() call")
    ap.add_argument("--concurrency-levels", type=str, default=None,
                     help="comma-separated list of repeat counts to test in one engine load, e.g. "
                          "'1,4,8,16' -- each repeat count submits repeat*len(PROMPTS) requests "
                          "together in one llm.generate() call and reports aggregate tok/s. "
                          "Runs sequentially within the same process to amortize the one-time "
                          "engine startup (CUDA graph capture etc.) across all levels.")
    ap.add_argument("--save-wavs", action="store_true",
                     help="decode + save wav files (skip for large concurrency runs, just want tok/s)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoFeatureExtractor, AutoTokenizer, MimiModel

    repo_dir = Path(args.repo_dir)
    codec_dir = Path(args.codec_dir) if args.codec_dir else repo_dir / "codec"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    print("[load] tokenizer + codec...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(repo_dir, trust_remote_code=True)
    mimi = MimiModel.from_pretrained(codec_dir, dtype=torch.bfloat16).eval().to("cuda")
    sample_rate = int(AutoFeatureExtractor.from_pretrained(codec_dir).sampling_rate)
    vocab = tokenizer.get_vocab()
    audio_end_id = int(vocab["</audio>"])
    mapping = {
        int(tid): (int(m.group(1)), int(m.group(2)))
        for tok, tid in vocab.items()
        if (m := AUDIO_RE.match(tok))
    }

    print("[load] vLLM engine (this triggers the NVFP4 checkpoint load)...", flush=True)
    t0 = time.perf_counter()
    llm = LLM(
        model=str(repo_dir),
        trust_remote_code=True,
        quantization="modelopt_fp4",
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
        max_model_len=4096,
        enforce_eager=args.enforce_eager,
    )
    print(f"[load] engine ready in {time.perf_counter() - t0:.2f}s", flush=True)

    def save_wav(audio: torch.Tensor, path: Path):
        pcm = (audio.numpy() * 32767).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())

    def process(i, speaker, token_ids, wall_s):
        n_tokens = len(token_ids)
        tok_s = n_tokens / wall_s if wall_s > 0 else float("nan")

        values, frame, expected = [], [], 0
        for tid in token_ids:
            item = mapping.get(int(tid))
            if item is None:
                frame, expected = [], 0
                continue
            code, q = item
            if q == expected:
                frame.append(code)
                expected += 1
                if expected == 8:
                    values.extend(frame)
                    frame, expected = [], 0
            elif q == 0:
                frame, expected = [code], 1
            else:
                frame, expected = [], 0

        print(f"[{i}] speaker={speaker} tokens={n_tokens} wall_s={wall_s:.2f} tok/s={tok_s:.2f}", flush=True)

        if not values:
            print("    WARNING: no complete Mimi-Q8 frames decoded, skipping wav")
            return
        with torch.inference_mode():
            codes = torch.tensor(values, device="cuda").reshape(1, -1, 8).transpose(1, 2)
            audio = mimi.decode(codes).audio_values[0, 0].float().cpu().clamp(-1, 1)
        audio_duration_s = audio.shape[-1] / sample_rate
        rtf = wall_s / audio_duration_s if audio_duration_s > 0 else float("nan")
        wav_path = out_dir / f"vllm_sample_{i}.wav"
        save_wav(audio, wav_path)
        print(f"    audio_s={audio_duration_s:.2f} rtf={rtf:.2f} saved={wav_path}", flush=True)

    params = SamplingParams(
        temperature=0.8,
        top_k=30,
        max_tokens=args.max_tokens,
        stop_token_ids=[audio_end_id],
    )

    if args.concurrency_levels:
        levels = [int(x) for x in args.concurrency_levels.split(",")]
        print(f"\n[concurrency sweep] testing repeat levels {levels} "
              f"({[lvl * len(PROMPTS) for lvl in levels]} concurrent requests)", flush=True)
        results = []
        for lvl in levels:
            expanded = PROMPTS * lvl
            prompts = [f"<text>{speaker}: {text}<audio>" for speaker, text in expanded]
            n = len(prompts)
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, params)
            wall_s = time.perf_counter() - t0

            total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
            agg_tok_s = total_tokens / wall_s if wall_s > 0 else float("nan")
            results.append((n, wall_s, total_tokens, agg_tok_s))
            print(f"[concurrency={n}] wall_s={wall_s:.2f} total_tokens={total_tokens} "
                  f"aggregate_tok/s={agg_tok_s:.2f}", flush=True)

            if args.save_wavs and lvl == levels[0]:
                for i, ((speaker, _), out) in enumerate(zip(expanded, outputs)):
                    process(i, speaker, out.outputs[0].token_ids, wall_s)

        print("\n[concurrency sweep] summary:", flush=True)
        print(f"{'concurrency':>12} {'wall_s':>10} {'total_tok':>10} {'agg_tok/s':>12}", flush=True)
        for n, wall_s, total_tokens, agg_tok_s in results:
            print(f"{n:>12} {wall_s:>10.2f} {total_tokens:>10} {agg_tok_s:>12.2f}", flush=True)
    elif args.batch:
        prompts = [f"<text>{speaker}: {text}<audio>" for speaker, text in PROMPTS]
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params)
        wall_s = time.perf_counter() - t0
        print(f"[batch] all {len(prompts)} prompts, total wall_s={wall_s:.2f}", flush=True)
        for i, ((speaker, _), out) in enumerate(zip(PROMPTS, outputs)):
            process(i, speaker, out.outputs[0].token_ids, wall_s)
    else:
        for i, (speaker, text) in enumerate(PROMPTS):
            prompt = f"<text>{speaker}: {text}<audio>"
            t0 = time.perf_counter()
            outputs = llm.generate([prompt], params)
            wall_s = time.perf_counter() - t0
            process(i, speaker, outputs[0].outputs[0].token_ids, wall_s)


if __name__ == "__main__":
    main()
