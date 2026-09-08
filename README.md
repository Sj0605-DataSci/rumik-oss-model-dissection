# Dissecting and Optimizing Rumik's Open Source TTS Model on DGX Spark

This is a writeup of pulling apart [rumik-ai/rumik-oss-1](https://huggingface.co/rumik-ai/rumik-oss-1) — Rumik's open source Indic text-to-speech model — and optimizing its inference speed on an NVIDIA DGX Spark (GB10, Blackwell).

## What the model is

rumik-oss-1 is a 3.38B parameter speech language model. It is not a purpose-built TTS architecture — it's an existing small multilingual text LLM taught to also emit audio tokens.

- **Backbone**: `CohereLabs/tiny-aya-fire`, a Cohere2/Command-R-family model. 36 layers, 2048 hidden size, 16 attention heads with 4 key/value heads (grouped-query attention). Every 4th layer uses full attention; the other three use a 4096-token sliding window (same pattern as Gemma2). Attention and MLP run in parallel per layer (both read the same normalized input and their outputs are summed), and the input/output embeddings are tied.
- **Audio representation**: Kyutai's Mimi neural codec (24kHz, 8 codebooks). The 8 codebooks are flattened into a single token stream and modeled autoregressively by the same transformer — this is the same trick used in the "Llama-Mimi" paper (arXiv:2509.14882).
- **The only genuinely new code** on top of the stock backbone is about 100 lines: a small `stop_predictor` head (LayerNorm → Linear → GELU → Linear, ~1M params) that predicts when to end generation, and a decode loop that restricts sampling to the audio-token vocabulary and checks the stop head at each step.
- Vocabulary size is 277,404 — almost entirely because of the audio-token space (8 codebooks × 2048 codes) plus the base model's multilingual text vocabulary.

Verified directly against the released weights: the `stop_predictor` is the only extra set of parameters, tied embeddings are confirmed (no separate output-head weight in the checkpoint), and there is exactly one LayerNorm per layer (confirming the parallel attention+MLP structure).

## Hardware

**NVIDIA DGX Spark**, GB10 Grace Blackwell superchip:
- Compute capability 12.1 (Blackwell — same generation that introduced native FP4 tensor core support)
- 128GB unified LPDDR5X memory, shared between CPU and GPU, ~273GB/s peak bandwidth
- Driver 580.159.03, CUDA 13.0

## Optimization process

The question we were answering: **can this model run faster on GB10, and how?** The model is only 6.7GB in BF16, so it comfortably fits in memory — the goal was lowering per-request latency, not fitting a bigger model.

### Step 0 — Baseline

Ran the model as released (BF16, standard HuggingFace `transformers` generation loop).

| | value |
|---|---|
| Model load time | 37.7s |
| Decode throughput | ~26 audio-tokens/sec |
| Real-time factor (RTF)* | 3.7 – 3.9x |

*RTF = wall-clock time to generate ÷ length of the audio produced. RTF above 1.0 means slower than real time.

We confirmed this is a memory-bandwidth problem, not a compute problem: at batch size 1, generating one token means reading the entire model's weights once (~6.76GB backbone + ~1.14GB for the output vocabulary projection, since embeddings are tied and that projection is a real matmul, not just a lookup). Measured throughput implied ~205GB/s of effective bandwidth use against a ~273GB/s peak — consistent with a bandwidth-bound workload with some overhead from Python and kernel-launch cost. This told us where the win had to come from: moving fewer bytes per token, i.e., quantization.

### Step 1 — `torch.compile` / CUDA graphs (did not help)

Tried replacing the model's dynamic KV-cache decode loop with a fixed-size cache (`StaticCache`) and compiling the per-step forward pass with `torch.compile(mode="reduce-overhead")`, which normally lets PyTorch capture CUDA graphs and cut per-step launch overhead.

Result: **21.8 tok/s, RTF 4.6x — slower than baseline.** The model's sliding-window attention layers compute their key/value length from how many tokens have been generated so far, which changes every single step. Our utterances (500–1000 tokens) never reach the 4096-token window where that length would stabilize, so `torch.compile` kept re-tracing the graph instead of reusing one, hit its retry limit, and fell back to plain eager execution — while still paying the cost of the failed compilation attempts. Dropped this approach.

### Step 2 — NVFP4 quantization via plain PyTorch (did not help)

NVFP4 is NVIDIA's 4-bit floating point format for Blackwell tensor cores. Quantizing the model's weights to NVFP4 should cut bytes-moved-per-token by roughly 3–4x, which — given the bandwidth-bound finding above — should translate fairly directly into lower latency.

Quantized the backbone with `nvidia-modelopt` (weight-only NVFP4, calibrated on a handful of representative prompts) and packed the weights into real 4-bit storage. This part worked correctly: the checkpoint shrank from 6.7GB to 3.0GB, a genuine 2.23x reduction.

Result: **1.8 tok/s — 14.7x slower than baseline.** The `modelopt` PyPI package ships no compiled CUDA kernels for matrix multiplication against packed 4-bit weights. The quantized checkpoint was correct, but running it meant unpacking every weight back to BF16 before each matmul — pure overhead with no fused kernel to make up for it. Quantizing the data format alone doesn't produce a speedup; something has to know how to compute against that format on the actual tensor cores.

### Step 3 — NVFP4 via vLLM (this is the one that worked)

vLLM ships real compiled NVFP4 kernels (CUTLASS/FlashInfer) for Blackwell. Confirmed this first with vLLM's own GEMM microbenchmark, run directly on this GB10: at batch size 1, NVFP4 beat BF16 by **2.3–4.3x** in raw compute throughput across several matrix shapes.

Getting rumik-oss-1 running through vLLM required two things vLLM doesn't support out of the box, since the model isn't a standard chat/completion model:

1. **A custom model plugin.** rumik-oss-1's backbone architecture (Cohere2) is natively supported by vLLM, so we subclassed vLLM's existing implementation and added the `stop_predictor` head. Both of the model's custom behaviors — restricting output to the audio-token vocabulary, and stopping generation when the `stop_predictor` fires — were implemented as a single override of vLLM's `compute_logits` step, which runs once per generated token.
2. **Re-exporting the quantized checkpoint** in vLLM's expected format, using full weight+activation NVFP4 (matching what the fused kernel is built for) rather than the weight-only version from Step 2, via `nvidia-modelopt`'s HuggingFace checkpoint exporter.

Result:

| | sample 1 | sample 2 | sample 3 |
|---|---|---|---|
| Audio tokens generated | 513 | 289 | 1041 |
| Throughput | 59.2 tok/s | 62.6 tok/s | 62.1 tok/s |
| RTF | 1.69 | 1.60 | 1.61 |

**~26 tok/s → ~60 tok/s (2.3x), RTF 3.7–3.9x → 1.6–1.7x.** Generated audio lengths closely matched the original BF16 run (durations within a few percent), confirming the plugin reproduces the reference model's behavior correctly rather than just running fast. Engine startup (CUDA graph capture + compilation) took about 98 seconds — a one-time cost, negligible for a long-running server, but worth knowing about for a single request like this.

### Step 4 — concurrent requests (vLLM's actual design target)

Everything above is a single-request, single-stream comparison. vLLM's real purpose is serving many requests at once via continuous batching, so we swept concurrency by submitting multiple requests together in one call and measuring aggregate throughput (total tokens generated across all of them, divided by wall time):

| Concurrent requests | Wall time | Total tokens | Aggregate throughput |
|---|---|---|---|
| 3 | 15.77s | 1,795 | 113.8 tok/s |
| 12 | 17.56s | 7,060 | 402.1 tok/s |
| 24 | 20.24s | 14,040 | 693.7 tok/s |
| 48 | 23.31s | 28,696 | 1,231.1 tok/s |

Aggregate throughput scales up to **~1,231 tok/s at 48 concurrent requests** — about 20x the single-request NVFP4 throughput (~60 tok/s) and ~47x the BF16 baseline (~26 tok/s), while wall time only grows from 15.8s to 23.3s (a 1.5x increase for 16x more requests). This is the practical payoff of the fused NVFP4 kernel: because decode is bandwidth-bound, batching many requests together lets the same weight-read serve many sequences at once, so total serving capacity scales far faster than single-request latency does. Individual requests do take somewhat longer to finish under heavier concurrent load (that's the cost of sharing the GPU), but for a real deployment serving multiple users, aggregate throughput — not single-stream latency — is the number that determines how many people you can serve at once.

## Summary

| Stage | Throughput | RTF | Outcome |
|---|---|---|---|
| BF16 baseline | ~26 tok/s | 3.7–3.9x | reference |
| + torch.compile / CUDA graphs | ~22 tok/s | 4.6x | regression — sliding-window cache doesn't compile stably at this sequence length |
| + NVFP4 (plain PyTorch/modelopt) | ~1.8 tok/s | ~55x | regression — no compiled kernel for the quantized format |
| + NVFP4 via vLLM (fused kernel) | ~60 tok/s | 1.6–1.7x | **2.3x faster than baseline** |

The lesson that generalizes: **quantizing a model's weight format and getting a speedup from that format are two separate problems.** The first is a data transformation; the second needs a kernel that runs on the actual hardware unit the format was designed for. On this generation of NVIDIA hardware, that kernel currently exists in two places — TensorRT-LLM and vLLM — and nowhere else.

## Other ecosystems checked: Ollama and Unsloth

- **Ollama**: no path to add this model. Ollama has no text-to-speech / audio-token-output support at all yet — this is tracked as an open feature request ([ollama/ollama#11021](https://github.com/ollama/ollama/issues/11021)), not something implemented today. There is no runtime in Ollama that can decode audio codec tokens, regardless of the model. Nothing to add until Ollama itself supports this category of model.
- **Unsloth**: already works, no changes needed. Unsloth advertises generic support for any `transformers`-compatible model via `trust_remote_code=True`. Verified directly (`test_unsloth.py`): `FastModel.from_pretrained("./rumik-oss-1", trust_remote_code=True)` loads the model correctly, and `FastModel.get_peft_model(...)` wraps it with LoRA (15.1M trainable / 3.4B total params) with zero code changes on either side. There's no PR to write here — it already works exactly as documented.

## What's in this repository

- `ARCHITECTURE_NOTES.md` — full architecture dissection, verified against the released weights
- `BENCHMARK.md` — detailed benchmark methodology, logs, and diagnosis for every step above
- `NVFP4_PLAN.md` — the quantization reasoning and what was tried
- `infer.py`, `quantize_nvfp4.py`, `quantize_for_vllm.py` — the scripts behind each step
- `vllm-plugin/` — the vLLM model plugin (`tinyaya_vllm.py`), its packaging, and `PR_NOTES.md` on what contributing this back to vLLM would involve
- `FULL_DUPLEX_DESIGN.md` — a separate design sketch for extending this model to full-duplex (simultaneous listening and speaking), not implemented
- `test_unsloth.py` — the Unsloth compatibility check described above

All benchmarks were run on a single DGX Spark; numbers will differ on other hardware.
