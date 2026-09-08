#!/usr/bin/env python3
"""Quantize rumik-oss-1 to NVFP4 in vLLM's expected HF-checkpoint format.

Unlike quantize_nvfp4.py (which used mtq.compress() -- real 4-bit storage but
NO fused kernel to run it, since plain PyTorch/modelopt ships no compiled
CUDA extensions for this), this uses modelopt.torch.export.unified_export_hf
to write a checkpoint vLLM's native `modelopt_fp4` quant method can load
directly, backed by vLLM's own compiled CUTLASS NVFP4 kernels (confirmed
working with a real 2.3-4.3x batch=1 speedup on this GB10 -- see
benchmark_nvfp4_gemm.py results in NVFP4_PLAN.md).

Uses NVFP4_DEFAULT_CFG (full W4A4: weights AND activations to NVFP4) rather
than the weight-only config from quantize_nvfp4.py, since that's what the
fused CUTLASS GEMM kernel is built for.
"""

import argparse
from pathlib import Path

import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import unified_export_hf
from transformers import AutoModelForCausalLM, AutoTokenizer

CALIB_PROMPTS = [
    "<text>Ira: Hey! This is a quick test of the rumik open source model.<audio>",
    "<text>Siya: <description=\"happy, fast pace\"> Namaste, aaj ka din kaisa hai?<audio>",
    "<text>Zoya: This is a longer sentence to exercise a wider range of activations.<audio>",
    "<text>Aisha: Testing another speaker voice and a different sentence structure entirely.<audio>",
    "<text>Ira: <description=\"sad, slow pace\"> I don't think that's quite right, actually.<audio>",
    "<text>Siya: Ek chhota sa test message hai yeh, dekhte hain kaisa lagta hai.<audio>",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="./rumik-oss-1")
    ap.add_argument("--out-dir", default="./rumik-oss-1-nvfp4-vllm")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    repo_dir = Path(args.repo_dir)
    out_dir = Path(args.out_dir)

    print("[1/3] loading BF16 model...")
    tokenizer = AutoTokenizer.from_pretrained(repo_dir, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            repo_dir, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .eval()
        .to(args.device)
    )

    @torch.inference_mode()
    def forward_loop(m):
        for i, text in enumerate(CALIB_PROMPTS):
            inputs = tokenizer(text, return_tensors="pt").to(args.device)
            m(**inputs)
            print(f"    calib step {i+1}/{len(CALIB_PROMPTS)}", flush=True)

    print("[2/3] quantizing to NVFP4 (W4A4, matches vLLM's fused CUTLASS kernel)...", flush=True)
    # keep stop_predictor at full precision -- it's ~1M params (noise on the
    # bandwidth budget) and quantizing it buys nothing while risking the
    # stop-probability calibration; matches lm_head/embed_tokens already
    # being excluded by the base config.
    # rules apply sequentially and later ones override earlier matches for the
    # same quantizer, so the exclusion must be appended LAST to actually win
    # over the earlier "*weight_quantizer" enable-all rule in the base config.
    cfg = dict(mtq.NVFP4_DEFAULT_CFG)
    cfg["quant_cfg"] = list(cfg["quant_cfg"]) + [
        {"quantizer_name": "*stop_predictor*", "enable": False}
    ]
    model = mtq.quantize(model, cfg, forward_loop=forward_loop)

    print(f"[3/3] exporting HF checkpoint for vLLM to {out_dir} ...", flush=True)
    out_dir.mkdir(exist_ok=True, parents=True)
    unified_export_hf.export_hf_checkpoint(model, dtype=torch.bfloat16, export_dir=str(out_dir))

    # export_hf_checkpoint writes weights + hf_quant_config.json, but not the
    # tokenizer or our custom modeling code / codec -- copy those over so the
    # exported dir is a complete, self-contained repo.
    tokenizer.save_pretrained(out_dir)
    import shutil
    for fname in ["modeling_tinyaya.py", "configuration_tinyaya.py"]:
        src = repo_dir / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)
    codec_src = repo_dir / "codec"
    codec_dst = out_dir / "codec"
    if codec_src.exists() and not codec_dst.exists():
        shutil.copytree(codec_src, codec_dst)

    print("done.", flush=True)
    print("contents:", sorted(p.name for p in out_dir.iterdir()))


if __name__ == "__main__":
    main()
