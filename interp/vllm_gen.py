#!/usr/bin/env python3
"""Stage A: generate audio token ids for all prompts with vLLM (NVFP4) in one batched call.
Runs inside vllm/vllm-openai:v0.20.0 (see interp/run_vllm_gen.sh).
Writes <out>/gen_ids.jsonl : {clip_id, prompt_ids, audio_ids} per clip.
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import build_prompts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="/models/rumik-oss-1-nvfp4-vllm")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", default="v1")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--out", type=Path, default=Path("/work/interp/acts"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    args.out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.repo_dir, trust_remote_code=True)
    audio_end = int(tok.get_vocab()["</audio>"])
    rows = build_prompts(args.n, args.seed, args.grid)
    texts = [f"<text>{r['speaker']}: {r['input']}<audio>" for r in rows]

    t0 = time.perf_counter()
    llm = LLM(model=args.repo_dir, trust_remote_code=True, quantization="modelopt_fp4",
              dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=4096)
    print(f"[vllm] engine ready in {time.perf_counter()-t0:.1f}s", flush=True)

    params = SamplingParams(temperature=0.8, top_k=30, max_tokens=args.max_tokens,
                            stop_token_ids=[audio_end])
    t0 = time.perf_counter()
    outs = llm.generate(texts, params, use_tqdm=True)
    wall = time.perf_counter() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[vllm] {len(outs)} clips, {n_tok} tokens in {wall:.1f}s = {n_tok/wall:.0f} tok/s", flush=True)

    with (args.out / "gen_ids.jsonl").open("w") as f:
        for r, o in zip(rows, outs):
            ids = [t for t in o.outputs[0].token_ids if t != audio_end]
            f.write(json.dumps({"clip_id": r["clip_id"], "prompt_ids": list(o.prompt_token_ids),
                                "audio_ids": ids}) + "\n")
    print(f"[vllm] wrote {args.out/'gen_ids.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
