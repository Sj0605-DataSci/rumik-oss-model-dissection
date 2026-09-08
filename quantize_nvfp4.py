#!/usr/bin/env python3
"""Weight-only NVFP4 quantization of rumik-oss-1's backbone via nvidia-modelopt.

Weight-only (W4A16) rather than full W4A4: decode at batch=1 is memory-
bandwidth-bound (see BENCHMARK.md), not compute-bound, so the win we want is
fewer bytes moved per step, not faster GEMM FLOPs -- weight-only NVFP4 gets
that without needing activation calibration/complexity.

Calibrates on a small set of representative `<text>{speaker}: {text}<audio>`
prompts (text-conditioning prefill only -- calibration just needs realistic
activation ranges through the backbone's Linear layers, doesn't need to run
full audio-token decode).
"""

import argparse
import re
import time
import wave
from pathlib import Path

import torch
import modelopt.torch.quantization as mtq
from transformers import AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer, MimiModel

AUDIO_RE = re.compile(r"^<(\d+)_(\d+)>$")

CALIB_PROMPTS = [
    "<text>Ira: Hey! This is a quick test of the rumik open source model.<audio>",
    "<text>Siya: <description=\"happy, fast pace\"> Namaste, aaj ka din kaisa hai?<audio>",
    "<text>Zoya: This is a longer sentence to exercise a wider range of activations.<audio>",
    "<text>Aisha: Testing another speaker voice and a different sentence structure entirely.<audio>",
    "<text>Ira: <description=\"sad, slow pace\"> I don't think that's quite right, actually.<audio>",
    "<text>Siya: Ek chhota sa test message hai yeh, dekhte hain kaisa lagta hai.<audio>",
]


@torch.inference_mode()
def run_benchmark(model, tokenizer, repo_dir: Path, device: str, max_new_tokens: int = 2048, n_prompts: int = 3):
    """Same decode/measurement logic as infer.py's synthesize(), but driven
    against an already-quantized in-memory model (no save/reload)."""
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    mimi = MimiModel.from_pretrained(repo_dir / "codec", dtype=dtype).eval().to(device)
    sample_rate = int(AutoFeatureExtractor.from_pretrained(repo_dir / "codec").sampling_rate)

    vocab = tokenizer.get_vocab()
    audio_end_id = int(vocab["</audio>"])
    mapping = {
        int(tid): (int(m.group(1)), int(m.group(2)))
        for tok, tid in vocab.items()
        if (m := AUDIO_RE.match(tok))
    }
    allowed_ids = torch.tensor(sorted([*mapping, audio_end_id]), device=device)

    prompts = [
        ("Ira", "Hey! This is a quick test of the rumik open source model running locally."),
        ("Siya", "<description=\"happy, fast pace\"> Namaste, aaj ka din kaisa hai?"),
        ("Zoya", "This is a longer sentence meant to measure steady state decode throughput across many audio tokens, so we can get a stable tokens per second estimate."),
    ]

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    for i, (speaker, text) in enumerate(prompts[:n_prompts]):
        prompt = f"<text>{speaker}: {text}<audio>"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        sync()
        t0 = time.perf_counter()
        output_ids = model.generate_audio(
            **inputs,
            allowed_ids=allowed_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=8,
            temperature=0.8,
            top_k=30,
            do_sample=True,
        )[0].tolist()
        sync()
        gen_wall_s = time.perf_counter() - t0

        start = inputs.input_ids.shape[1]
        try:
            end = output_ids.index(audio_end_id, start)
        except ValueError:
            end = len(output_ids)
        n_audio_tokens = end - start
        tok_s = n_audio_tokens / gen_wall_s if gen_wall_s > 0 else float("nan")

        values, frame, expected = [], [], 0
        for tid in output_ids[start:end]:
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

        audio_duration_s = float("nan")
        if values:
            codes = torch.tensor(values, device=device).reshape(1, -1, 8).transpose(1, 2)
            audio = mimi.decode(codes).audio_values[0, 0].float().cpu().clamp(-1, 1)
            audio_duration_s = audio.shape[-1] / sample_rate

        rtf = gen_wall_s / audio_duration_s if audio_duration_s and audio_duration_s == audio_duration_s else float("nan")
        print(f"[nvfp4-bench {i}] speaker={speaker} audio_tokens={n_audio_tokens} "
              f"wall_s={gen_wall_s:.2f} tok/s={tok_s:.2f} audio_s={audio_duration_s:.2f} rtf={rtf:.2f}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="./rumik-oss-1")
    ap.add_argument("--out-dir", default="./rumik-oss-1-nvfp4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-save", action="store_true",
                     help="skip save_pretrained (known broken for real-quantized/compressed modelopt "
                          "checkpoints -- packed tensor shapes don't round-trip through plain "
                          "AutoModelForCausalLM.from_pretrained). Use with --benchmark to test "
                          "in-process instead.")
    ap.add_argument("--benchmark", action="store_true",
                     help="run a quick decode benchmark on the quantized model in-process "
                          "(before any save/reload) and compare tok/s against the BF16 numbers "
                          "in BENCHMARK.md")
    ap.add_argument("--bench-max-new-tokens", type=int, default=2048)
    ap.add_argument("--bench-n-prompts", type=int, default=3)
    args = ap.parse_args()

    repo_dir = Path(args.repo_dir)
    out_dir = Path(args.out_dir)

    print("[1/4] loading BF16 model...")
    tokenizer = AutoTokenizer.from_pretrained(repo_dir, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            repo_dir, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .eval()
        .to(args.device)
    )

    n_params_before = sum(p.numel() for p in model.parameters())
    print(f"    {n_params_before/1e9:.3f}B params loaded")

    @torch.inference_mode()
    def forward_loop(m):
        for i, text in enumerate(CALIB_PROMPTS):
            inputs = tokenizer(text, return_tensors="pt").to(args.device)
            m(**inputs)
            print(f"    calib step {i+1}/{len(CALIB_PROMPTS)}")

    print("[2/4] quantizing to NVFP4 (weight-only, W4A16)...")
    model = mtq.quantize(model, mtq.W4A16_NVFP4_CFG, forward_loop=forward_loop)
    mtq.print_quant_summary(model)

    print("[3/4] compressing to real NVFP4 storage (mtq.compress)...")
    try:
        mtq.compress(model)  # in-place, returns None
        print("    compress() succeeded -- weights now stored as real NVFP4")
    except Exception as e:
        print(f"    compress() failed or unavailable: {e!r}")
        print("    falling back to fake-quant model (accuracy-representative, not bandwidth-reduced)")

    if args.benchmark:
        print("[bench] running decode benchmark on the quantized model IN-PROCESS (no save/reload)...", flush=True)
        run_benchmark(model, tokenizer, repo_dir, args.device,
                      max_new_tokens=args.bench_max_new_tokens, n_prompts=args.bench_n_prompts)

    if not args.skip_save:
        print(f"[4/4] saving to {out_dir} ...")
        out_dir.mkdir(exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        # copy over the custom modeling code + codec dir so the quantized repo is self-contained
        import shutil
        for fname in ["modeling_tinyaya.py", "configuration_tinyaya.py", "server.py"]:
            src = repo_dir / fname
            if src.exists():
                shutil.copy(src, out_dir / fname)
        codec_src = repo_dir / "codec"
        codec_dst = out_dir / "codec"
        if codec_src.exists() and not codec_dst.exists():
            shutil.copytree(codec_src, codec_dst)
        print(f"done saving to {out_dir} "
              f"(note: this checkpoint may not reload via plain AutoModelForCausalLM.from_pretrained "
              f"-- see BENCHMARK.md for what happened when we tried)")
    else:
        print("[4/4] skipped save (--skip-save)")


if __name__ == "__main__":
    main()
