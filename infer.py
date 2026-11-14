#!/usr/bin/env python3
"""Standalone inference + timing benchmark for rumik-oss-1.

Loads the model directly (no server), synthesizes one or more prompts, saves
wavs, and prints latency/throughput numbers: load time, time-to-first-audio-
token (TTFT), decode tokens/sec, total wall time, and real-time factor (RTF =
wall_time / audio_duration_sec — RTF < 1 means faster than real time).
"""

import argparse
import re
import time
import wave
from pathlib import Path

import torch
from transformers import AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer, MimiModel
from transformers.cache_utils import StaticCache

AUDIO_RE = re.compile(r"^<(\d+)_(\d+)>$")
MIMI_FRAME_HZ = 12.5  # Mimi produces 1 codebook-frame per (24000/1920) samples = 12.5 Hz


class Timer:
    def __init__(self):
        self.marks = {}

    def mark(self, name):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.marks[name] = time.perf_counter()

    def delta(self, a, b):
        return self.marks[b] - self.marks[a]


class RumikTTS:
    def __init__(self, repo_dir: Path, device: str, dtype: torch.dtype):
        self.device = device
        t = Timer()
        t.mark("start")
        self.tokenizer = AutoTokenizer.from_pretrained(repo_dir, trust_remote_code=True)
        t.mark("tokenizer_loaded")
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                repo_dir, trust_remote_code=True, dtype=dtype, attn_implementation="sdpa"
            )
            .eval()
            .to(device)
        )
        t.mark("model_loaded")
        self.mimi = MimiModel.from_pretrained(repo_dir / "codec", dtype=dtype).eval().to(device)
        t.mark("codec_loaded")
        self.sample_rate = int(AutoFeatureExtractor.from_pretrained(repo_dir / "codec").sampling_rate)

        vocab = self.tokenizer.get_vocab()
        self.audio_end_id = int(vocab["</audio>"])
        self.mapping = {
            int(tid): (int(m.group(1)), int(m.group(2)))
            for tok, tid in vocab.items()
            if (m := AUDIO_RE.match(tok))
        }
        self.allowed_ids = torch.tensor(sorted([*self.mapping, self.audio_end_id]), device=device)
        self.load_times = {
            "tokenizer_s": t.delta("start", "tokenizer_loaded"),
            "model_s": t.delta("tokenizer_loaded", "model_loaded"),
            "codec_s": t.delta("model_loaded", "codec_loaded"),
            "total_s": t.delta("start", "codec_loaded"),
        }

    @torch.inference_mode()
    def synthesize(self, text: str, speaker: str = "Ira", temperature=0.8, top_k=30, max_new_tokens=2048):
        t = Timer()
        prompt = f"<text>{speaker}: {text}<audio>"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        t.mark("gen_start")
        output_ids = self.model.generate_audio(
            **inputs,
            allowed_ids=self.allowed_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=8,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
        )[0].tolist()
        t.mark("gen_end")

        start = prompt_len
        try:
            end = output_ids.index(self.audio_end_id, start)
        except ValueError:
            end = len(output_ids)
        self.last_ids = torch.tensor(output_ids[: end + 1])  # prompt + audio (+ </audio>), used by interp/capture_acts.py
        n_audio_tokens = end - start
        gen_wall_s = t.delta("gen_start", "gen_end")

        values, frame, expected = [], [], 0
        for tid in output_ids[start:end]:
            item = self.mapping.get(int(tid))
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
        if not values:
            raise RuntimeError("No complete Mimi-Q8 frames generated")

        n_frames = len(values) // 8
        codes = torch.tensor(values, device=self.device).reshape(1, -1, 8).transpose(1, 2)
        t.mark("decode_start")
        audio = self.mimi.decode(codes).audio_values[0, 0].float().cpu().clamp(-1, 1)
        t.mark("decode_end")

        audio_duration_s = audio.shape[-1] / self.sample_rate
        stats = {
            "prompt_tokens": prompt_len,
            "audio_tokens_generated": n_audio_tokens,
            "codec_frames": n_frames,
            "audio_duration_s": audio_duration_s,
            "lm_decode_wall_s": gen_wall_s,
            "lm_tokens_per_sec": n_audio_tokens / gen_wall_s if gen_wall_s > 0 else float("nan"),
            "codec_decode_wall_s": t.delta("decode_start", "decode_end"),
            "total_wall_s": gen_wall_s + t.delta("decode_start", "decode_end"),
            "rtf": (gen_wall_s + t.delta("decode_start", "decode_end")) / audio_duration_s if audio_duration_s > 0 else float("nan"),
        }
        return audio, stats

    @staticmethod
    def _sample(scores, do_sample, temperature, top_k):
        if not do_sample:
            return scores.argmax(dim=-1, keepdim=True)
        scores = scores / max(float(temperature), 1e-5)
        if int(top_k) > 0:
            k = min(int(top_k), scores.shape[-1])
            cutoff = torch.topk(scores, k, dim=-1).values[:, -1:]
            scores = scores.masked_fill(scores < cutoff, torch.finfo(scores.dtype).min)
        return torch.multinomial(torch.softmax(scores.float(), dim=-1), 1)

    def _decode_step(self, input_ids, past_key_values, cache_position):
        """One autoregressive step against a StaticCache. Fixed input/cache
        shapes every call (given a StaticCache) -> safe to torch.compile with
        mode='reduce-overhead' (CUDA graphs)."""
        return self.model(
            input_ids=input_ids,
            attention_mask=None,  # let HF build the mask from the StaticCache (fixed shape)
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            output_hidden_states=True,
            return_dict=True,
        )

    def compile_decode_step(self):
        self._decode_step = torch.compile(self._decode_step, mode="reduce-overhead", dynamic=False)

    @torch.inference_mode()
    def synthesize_fast(self, text: str, speaker: str = "Ira", temperature=0.8, top_k=30,
                         max_new_tokens=2048, warmup_steps: int = 0):
        """Same output as synthesize(), but uses a preallocated StaticCache and
        (optionally, if compile_decode_step() was called) a compiled decode
        step -- avoids DynamicCache reallocation and Python/kernel-launch
        overhead per step."""
        t = Timer()
        prompt = f"<text>{speaker}: {text}<audio>"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]
        max_cache_len = prompt_len + max_new_tokens

        cache = StaticCache(self.model.config, max_cache_len=max_cache_len)

        t.mark("gen_start")
        prefill_cache_position = torch.arange(prompt_len, device=self.device)
        out = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=prefill_cache_position,
            output_hidden_states=True,
            return_dict=True,
        )
        scores = out.logits[:, -1, :]
        hidden = out.hidden_states[-1][:, -1:, :]

        emitted_tokens = []
        audio_end_id = int(self.audio_end_id)
        allowed_ids = self.allowed_ids

        for step in range(max_new_tokens):
            if step >= 8:
                stop = torch.sigmoid(self.model.stop_predictor(hidden).squeeze(-1))
                if bool((stop > 0.5).all()):
                    emitted_tokens.append(audio_end_id)
                    break
            restricted = torch.full_like(scores, torch.finfo(scores.dtype).min)
            restricted.index_copy_(1, allowed_ids, scores.index_select(1, allowed_ids))
            if step < 8:
                restricted[:, audio_end_id] = torch.finfo(scores.dtype).min
            token = self._sample(restricted, True, temperature, top_k)
            tok_id = int(token.item())
            emitted_tokens.append(tok_id)
            if tok_id == audio_end_id:
                break

            pos = torch.tensor([prompt_len + step], device=self.device)
            out = self._decode_step(token, cache, pos)
            scores = out.logits[:, -1, :]
            hidden = out.hidden_states[-1][:, -1:, :]
        t.mark("gen_end")

        try:
            end = emitted_tokens.index(audio_end_id)
        except ValueError:
            end = len(emitted_tokens)
        n_audio_tokens = end
        gen_wall_s = t.delta("gen_start", "gen_end")

        values, frame, expected = [], [], 0
        for tid in emitted_tokens[:end]:
            item = self.mapping.get(int(tid))
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
        if not values:
            raise RuntimeError("No complete Mimi-Q8 frames generated")

        n_frames = len(values) // 8
        codes = torch.tensor(values, device=self.device).reshape(1, -1, 8).transpose(1, 2)
        t.mark("decode_start")
        audio = self.mimi.decode(codes).audio_values[0, 0].float().cpu().clamp(-1, 1)
        t.mark("decode_end")

        audio_duration_s = audio.shape[-1] / self.sample_rate
        stats = {
            "prompt_tokens": prompt_len,
            "audio_tokens_generated": n_audio_tokens,
            "codec_frames": n_frames,
            "audio_duration_s": audio_duration_s,
            "lm_decode_wall_s": gen_wall_s,
            "lm_tokens_per_sec": n_audio_tokens / gen_wall_s if gen_wall_s > 0 else float("nan"),
            "codec_decode_wall_s": t.delta("decode_start", "decode_end"),
            "total_wall_s": gen_wall_s + t.delta("decode_start", "decode_end"),
            "rtf": (gen_wall_s + t.delta("decode_start", "decode_end")) / audio_duration_s if audio_duration_s > 0 else float("nan"),
        }
        return audio, stats

    def save_wav(self, audio: torch.Tensor, path: Path):
        pcm = (audio.numpy() * 32767).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm.tobytes())


DEFAULT_PROMPTS = [
    ("Ira", "Hey! This is a quick test of the rumik open source model running locally."),
    ("Siya", "<description=\"happy, fast pace\"> Namaste, aaj ka din kaisa hai?"),
    ("Zoya", "This is a longer sentence meant to measure steady state decode throughput across many audio tokens, so we can get a stable tokens per second estimate."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="./rumik-oss-1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="./outputs")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--engine", choices=["loop", "static-cache", "compiled"], default="loop",
                     help="loop = original generate_audio() with DynamicCache; "
                          "static-cache = preallocated StaticCache, same eager kernels; "
                          "compiled = static-cache + torch.compile(mode='reduce-overhead') decode step")
    args = ap.parse_args()

    repo_dir = Path(args.repo_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32

    print(f"[load] device={args.device} dtype={dtype}")
    engine = RumikTTS(repo_dir, args.device, dtype)
    print(f"[load] tokenizer={engine.load_times['tokenizer_s']:.2f}s "
          f"model={engine.load_times['model_s']:.2f}s "
          f"codec={engine.load_times['codec_s']:.2f}s "
          f"TOTAL={engine.load_times['total_s']:.2f}s")

    if args.engine == "compiled":
        engine.compile_decode_step()

    if args.engine in ("static-cache", "compiled"):
        print(f"[warmup] running a short synth to trigger cache alloc"
              + (" + torch.compile tracing" if args.engine == "compiled" else "") + " (excluded from timing)")
        wt0 = time.perf_counter()
        engine.synthesize_fast("Warming up the model.", speaker="Ira", max_new_tokens=64)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[warmup] done in {time.perf_counter() - wt0:.2f}s")

    for i, (speaker, text) in enumerate(DEFAULT_PROMPTS):
        print(f"\n[{i}] speaker={speaker} text={text[:60]!r} engine={args.engine}")
        if args.engine == "loop":
            audio, stats = engine.synthesize(text, speaker=speaker, max_new_tokens=args.max_new_tokens)
        else:
            audio, stats = engine.synthesize_fast(text, speaker=speaker, max_new_tokens=args.max_new_tokens)
        wav_path = out_dir / f"sample_{i}.wav"
        engine.save_wav(audio, wav_path)
        for k, v in stats.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        print(f"    saved -> {wav_path}")


if __name__ == "__main__":
    main()
