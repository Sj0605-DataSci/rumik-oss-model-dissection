# rumik-oss-1 on DGX Spark (GB10) — measured results

Real numbers from `infer.py`, run on this box after fixing a GPU device-node
issue that required a reboot (see below). BF16, `attn_implementation="sdpa"`,
`transformers==4.57.6` (pinned to match the repo's `transformers_version`),
`torch==2.15.0.dev20260907+cu130` (nightly — needed for GB10/Blackwell sm_120
kernels; stable PyPI `torch` builds as of this writing ship zero compiled CUDA
architectures for this chip, see "environment notes" below).

## Setup notes (useful if you rerun this)
- `nvidia-smi` can see the GPU (`NVIDIA GB10`) even when CUDA is fully broken —
  it talks to the driver directly, not through `/dev/nvidia0`. The tell was
  `torch.cuda.is_available()` returning `False` with an empty
  `torch.cuda.get_arch_list()` across **every venv on the machine**, including
  ones with a known-good CUDA setup (`env_isaacsim`). Root cause: `/dev/nvidia0`,
  `/dev/nvidiactl` etc. were missing — `sudo nvidia-modprobe -u -c 0` recreated
  them but CUDA still reported `cuInit -> 100` (`CUDA_ERROR_NO_DEVICE`); a full
  system update + reboot fixed it.
- Stable `torch` (both the default PyPI wheel and `download.pytorch.org/whl/cu128`)
  had `torch.cuda.get_arch_list() == []` on this box — i.e. no compiled kernels
  for any architecture, effectively unusable for a GPU that's compute
  capability 12.1 (Blackwell). The nightly `cu130` index
  (`download.pytorch.org/whl/nightly/cu130`) has `sm_120`/`compute_120` in its
  arch list and actually initializes. If you're setting this up again, use the
  nightly index, not stable.
- HF's `~/.cache/huggingface/modules` dir was pre-existing and root-owned on
  this box (unrelated to this session) — worked around with
  `HF_MODULES_CACHE=./.hf_modules_cache` rather than touching it.

## Results (3 prompts, default `infer.py` run)

| | sample 0 (Ira, EN, short) | sample 1 (Siya, Hinglish, happy/fast) | sample 2 (Zoya, EN, long) |
|---|---|---|---|
| prompt tokens | 23 | 25 | 34 |
| audio tokens generated | 520 | 288 | 992 |
| audio duration | 5.20s | 2.88s | 9.92s |
| LM decode wall time | 20.16s | 10.71s | 37.25s |
| **LM decode throughput** | 25.8 tok/s | 26.9 tok/s | 26.6 tok/s |
| Mimi codec decode time | 0.14s | 0.02s | 0.02s |
| **total wall time** | 20.30s | 10.72s | 37.27s |
| **RTF (wall / audio duration)** | 3.90x | 3.72x | 3.76x |

- **Model load time: 37.7s** (36.99s of that is just `from_pretrained` loading
  the 6.7GB BF16 checkpoint — tokenizer 0.6s, codec 0.1s).
- **Decode throughput is flat around ~26 audio-tokens/sec regardless of
  utterance length** — as expected for autoregressive decode, it's a
  per-step cost, not amortized.
- **Mimi codec decode is essentially free** (14-140ms) — all the cost is in
  the LM's autoregressive loop, none of it is the audio reconstruction.
- **RTF is consistently ~3.7-3.9x — this is NOT real-time.** Generating 1
  second of audio takes ~3.7-3.9 seconds of wall clock. This is the raw,
  un-optimized HF `transformers` path (manual Python decode loop, dynamic
  KV cache, no CUDA graphs, no `torch.compile`, no batching) — Rumik's own
  ~300ms-latency claim for their hosted product almost certainly uses a
  purpose-built serving stack (continuous batching, static/CUDA-graph-captured
  decode, possibly speculative decoding), not this reference `server.py`.

## Why decode is ~26 tok/s: the math roughly checks out as bandwidth-bound
- Backbone weights: 3.38B params × 2 bytes (BF16) ≈ **6.76 GB** read once per
  decode step (every layer touched every step, standard autoregressive cost).
- **Correction to an earlier assumption in `NVFP4_PLAN.md`**: the output
  logits projection (`hidden @ embed_tokens.T`, tied embeddings) is a real
  matmul over the full 277,404-row vocab every step — not just a cheap
  gather. That's another `2048 × 277404 × 2 bytes ≈ 1.14 GB` moved per step.
  So actual bytes-per-token is closer to **~7.9 GB**, not 6.76 GB — the
  earlier doc's claim that this projection is "gather-only, weaker case for
  quantizing" was wrong for the output side; only the *input* embedding
  lookup is a cheap gather. The output projection is a real bandwidth cost
  and belongs in scope for quantization.
- Measured: ~38.5ms/decode-step → implied effective bandwidth ≈
  7.9 GB / 0.0385s ≈ **~205 GB/s**, against GB10's LPDDR5X unified-memory
  spec (~273 GB/s peak). That's roughly 75% of peak, which is a believable
  number for a naive `sdpa` + Python-loop decode path (kernel launch
  overhead, no fusion, no CUDA graphs) — i.e. this is genuinely
  memory-bandwidth-bound, confirming the core assumption in `NVFP4_PLAN.md`:
  **cutting bytes-per-token is the direct lever on latency here**, and NVFP4
  (~3.5x fewer bytes for the quantized portion) is a well-targeted
  optimization, now also correctly including the output projection.

## What would actually move the needle, roughly in order of effort
1. **NVFP4 the backbone + tied output projection** (see `NVFP4_PLAN.md`,
   now corrected above) — direct bytes-per-token reduction, the biggest
   lever given this is bandwidth-bound.
2. **CUDA graphs / static KV cache** — removes Python + kernel-launch
   overhead per step, closes the gap between "theoretical bandwidth-bound
   time" and the ~25% overhead observed above, independent of quantization.
3. **`torch.compile`** — likely helps for the same reason as #2; wasn't
   attempted here (kept `infer.py` simple/portable for a first measurement).
4. **Batching** — irrelevant for a single-user latency benchmark like this
   one, but the actual lever for serving throughput once this becomes a
   multi-request question.

## Attempt 3: vLLM + real NVFP4 kernel — this one actually worked

Unlike the naive `mtq.compress()` PyTorch path (Attempt 2, above), **vLLM
ships real compiled CUTLASS/FlashInfer NVFP4 kernels for GB10**. Confirmed by:
running vLLM's own `benchmark_nvfp4_gemm.py` inside `vllm/vllm-openai:v0.20.0`
on this box: at batch=1 (our decode regime), NVFP4 beat BF16 by **2.3-4.3x**
in raw GEMM TFLOP/s across several shapes — a real, hardware-level win.

To use it, rumik-oss-1 needed a small out-of-tree vLLM model plugin
(`vllm-plugin/`) since it's architecturally `Cohere2ForCausalLM` (which vLLM
supports natively) plus a custom `stop_predictor` head and constrained
audio-vocab decode — neither is a standard vLLM model. Both are folded into
a single `compute_logits()` override (see `vllm-plugin/tinyaya_vllm.py`):
1. Restrict logits to the audio-codec sub-vocabulary (always correct here,
   since every prompt ends with the literal `<audio>` marker).
2. Force `</audio>` when the `stop_predictor` head's probability crosses
   0.5, replacing the HF reference's separate stopping-criteria check.

Quantization: `quantize_for_vllm.py` uses `mtq.NVFP4_DEFAULT_CFG` (full
W4A4, matching what the fused kernel is built for — not the weight-only
config from Attempt 2) and `modelopt.torch.export.unified_export_hf`, which
writes vLLM's native `modelopt_fp4` checkpoint format directly (confirmed:
vLLM logs `Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM` and
`Resolved architecture: TinyAyaForCausalLM` on load). `stop_predictor` is
excluded from quantization (kept BF16) — small head, not worth the risk.

### Results (same 3 prompts, sequential single-request calls, CUDA graphs enabled)

| | sample 0 (Ira) | sample 1 (Siya) | sample 2 (Zoya) |
|---|---|---|---|
| audio tokens | 513 | 289 | 1041 |
| wall time | 8.67s | 4.62s | 16.77s |
| **tok/s** | **59.15** | **62.56** | **62.06** |
| audio duration | 5.12s | 2.88s | 10.40s |
| **RTF** | **1.69** | **1.60** | **1.61** |

Compare to the BF16 baseline above: **~26.4 tok/s → ~60 tok/s (≈2.3x), RTF
3.7-3.9x → 1.6-1.7x**. Real, measured, on this exact GB10 — not a
projection. Still not below RTF 1.0 (real-time), but this alone is roughly
the size of win `NVFP4_PLAN.md` originally projected from the bandwidth
math, now actually delivered via the correct execution path. Generated
audio durations closely track the BF16 reference run (5.12s vs 5.20s, 2.88s
vs 2.88s, 10.40s vs 9.92s), indicating the plugin's `compute_logits`
masking + stop-head logic reproduces the reference behavior correctly, not
just "generates *something* fast."

**Engine startup cost**: 97.89s for CUDA-graph capture + `torch.compile`
tracing (one-time, amortized across all requests in a real serving
scenario — irrelevant for a long-running server, significant for a
one-shot script like this).

**Not yet tried**: batched/concurrent requests (where vLLM's actual design
target — continuous batching throughput — would show a bigger win than
this single-stream comparison); `--batch` flag exists in `vllm_infer.py`
for this but wasn't benchmarked this session.

See `vllm-plugin/PR_NOTES.md` for what contributing this back to vLLM
in-tree would involve, and why we're not opening that PR yet (checkpoint
format still experimental, `min_new_tokens` floor dropped as a known
simplification, and Rumik hasn't been asked whether they want this).

## Attempt 4: concurrent request throughput (vLLM's actual design target)

All numbers above are single-request, sequential measurements. vLLM's
purpose is continuous batching across many concurrent requests, so we ran
`vllm_infer.py --concurrency-levels 1,4,8,16` (each level submits
`level * len(PROMPTS)` requests together in one `llm.generate()` call,
inside the same engine load to amortize the one-time CUDA-graph-capture
startup cost across all levels):

```
[concurrency=3]  wall_s=15.77  total_tokens=1795   aggregate_tok/s=113.80
[concurrency=12] wall_s=17.56  total_tokens=7060   aggregate_tok/s=402.06
[concurrency=24] wall_s=20.24  total_tokens=14040  aggregate_tok/s=693.65
[concurrency=48] wall_s=23.31  total_tokens=28696  aggregate_tok/s=1231.13
```

| Concurrent requests | Wall time | Total tokens | Aggregate tok/s | vs. single-request NVFP4 (~60 tok/s) | vs. BF16 baseline (~26 tok/s) |
|---|---|---|---|---|---|
| 3 | 15.77s | 1,795 | 113.8 | 1.9x | 4.4x |
| 12 | 17.56s | 7,060 | 402.1 | 6.7x | 15.5x |
| 24 | 20.24s | 14,040 | 693.7 | 11.6x | 26.7x |
| 48 | 23.31s | 28,696 | 1,231.1 | 20.5x | 47.4x |

Going from 3 to 48 concurrent requests (16x more load) increases aggregate
throughput ~10.8x while wall time only grows ~1.5x (15.77s → 23.31s) — this
is continuous batching doing its job: since decode is bandwidth-bound (see
Step 0), reading the model's weights once serves many sequences' next-token
predictions at once, so total system throughput scales much faster than
wall time does. The tradeoff is per-request latency: any individual
request among the 48 concurrent ones takes longer to finish than it would
running alone (23.3s of shared GPU time vs. the same request finishing in
a few seconds solo) — expected, and the right tradeoff for a server handling
multiple simultaneous users rather than optimizing for one user's
worst-case wait.

Not yet tested: concurrency levels beyond 48 (to find where this scaling
curve actually saturates on this GPU), and mixed-length/realistic traffic
patterns rather than the same 3 prompts repeated.

## Reproduce
```
cd rumik-oss-model-dissection

# BF16 baseline
HF_MODULES_CACHE=./.hf_modules_cache .venv/bin/python infer.py --repo-dir ./rumik-oss-1 --device cuda

# NVFP4 export for vLLM
HF_MODULES_CACHE=./.hf_modules_cache .venv/bin/python quantize_for_vllm.py

# run through vLLM (needs docker + the vllm/vllm-openai:v0.20.0 image)
docker run --rm --gpus all \
  -v $PWD/rumik-oss-1-nvfp4-vllm:/models/rumik-oss-1-nvfp4-vllm:ro \
  -v $PWD/vllm-plugin:/workspace/vllm-plugin \
  -v $PWD/outputs_vllm:/models/outputs_vllm \
  --entrypoint bash vllm/vllm-openai:v0.20.0 \
  -c "pip install -q -e /workspace/vllm-plugin && VLLM_PLUGINS=register_tinyaya python3 /workspace/vllm-plugin/vllm_infer.py"

# concurrency sweep (same mounts as above)
docker run --rm --gpus all \
  -v $PWD/rumik-oss-1-nvfp4-vllm:/models/rumik-oss-1-nvfp4-vllm:ro \
  -v $PWD/vllm-plugin:/workspace/vllm-plugin \
  --entrypoint bash vllm/vllm-openai:v0.20.0 \
  -c "pip install -q -e /workspace/vllm-plugin && VLLM_PLUGINS=register_tinyaya python3 -u /workspace/vllm-plugin/vllm_infer.py --concurrency-levels 1,4,8,16"
```
Outputs land in `./outputs/sample_{0,1,2}.wav` (BF16) and
`./outputs_vllm/vllm_sample_{0,1,2}.wav` (NVFP4/vLLM).

## Attempt 2: StaticCache + torch.compile — a real negative result, diagnosed

`infer.py` now supports `--engine {loop,static-cache,compiled}` (the original
`generate_audio()` path is `loop`; the other two are a hand-rolled decode loop
using `transformers.cache_utils.StaticCache` in place of the default
`DynamicCache`, with `compiled` additionally wrapping the per-step forward in
`torch.compile(mode="reduce-overhead")` — the standard CUDA-graph recipe for
autoregressive decode).

| engine | tok/s | RTF |
|---|---|---|
| `loop` (baseline, `DynamicCache`) | ~26.4 | ~3.7-3.9x |
| `static-cache` (no compile) | ~22.2 | ~4.5x |
| `compiled` (`torch.compile`, mode='reduce-overhead') | ~21.8 | ~4.6x |

**Neither optimization helped — both regressed slightly.** Two separate
findings, not one:

1. **`StaticCache` alone is slower here**, because Cohere2's mask-building
   computes attention over the *actual* used cache length for sliding-window
   layers (efficient), but `StaticCache`'s fixed-size buffers mean the
   full-attention layers (1/4 of the 36 layers, see `layer_types` in
   `ARCHITECTURE_NOTES.md`) always mask over the complete `max_cache_len`
   (2048+prompt tokens) rather than the actual sequence length so far — wasted
   compute that grows with `max_new_tokens`, outweighing any allocation
   savings versus `DynamicCache`'s incremental growth.

2. **`torch.compile` never actually got a CUDA graph.** The log shows
   `torch._dynamo hit config.recompile_limit (8)` — the sliding-window cache
   layers compute `kv_length` from the *current* `cumulative_length` (grows
   by 1 every decode step, up to the 4096-token `sliding_window`), which is a
   new tensor shape on every single step. Since our utterances (500-900 audio
   tokens) never reach the 4096-token sliding window where `kv_length` would
   stabilize, Dynamo retraces up to its recompile cap, then falls back to
   eager for the rest of generation — so `compiled` measured here is
   effectively still eager, plus the fixed cost of 8 wasted compilation
   attempts during warmup (43.5s warmup vs 3.5s for plain `static-cache`).

**Why this is fixable but wasn't attempted here**: HF's own `model.generate()`
with `cache_implementation="static"` is the officially-supported, actually
compile-tested path for exactly this scenario — it's built to keep shapes
stable across the sliding-window transition in ways a hand-rolled loop
(what `synthesize_fast()` is) doesn't automatically get right. Getting a real
compiled speedup here would mean reimplementing the model's custom sampling
(`allowed_ids` vocab restriction + the `stop_predictor` early-stop check) as
a `LogitsProcessor` + `StoppingCriteria` pair and driving generation through
`model.generate()` itself, rather than a manual step loop — a bigger refactor
than the version currently in `infer.py`. Left as a follow-up; the honest
conclusion right now is **`torch.compile` did not help this model given the
current implementation**, and NVFP4 (a more mechanical, lower-risk win — see
`NVFP4_PLAN.md`) is a better next lever to pull first.
