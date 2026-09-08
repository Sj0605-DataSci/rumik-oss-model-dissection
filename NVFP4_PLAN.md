# Should we convert rumik-oss-1 to NVFP4 for DGX Spark?

## tl;dr — UPDATED after actually trying it: not yet, on this box
The bandwidth-bound reasoning below is correct and confirmed by measurement
(see `BENCHMARK.md`) — NVFP4 *should* be a decode-latency win in principle.
But an actual attempt (see "Attempt made" section below) quantized and
compressed the model correctly (real 2.23x on-disk size reduction, 6.7GB →
3.0GB) and then measured **decode getting ~14.7x slower** (1.8 tok/s vs. 26.4
tok/s BF16), because `nvidia-modelopt==0.46.0` ships no compiled CUDA kernels
for GB10's compute capability yet — the compressed weights have nowhere fast
to run. The theory is sound; the tooling for this specific new chip isn't
there yet. Read the "Attempt made" section before acting on anything above it.

## Why GB10 is a good NVFP4 target
- GB10 = Grace Blackwell superchip, compute capability 12.1 — same Blackwell
  generation that introduced native FP4 (NVFP4) tensor core support, first
  exposed broadly through TensorRT-LLM / TensorRT Model Optimizer.
- NVFP4 (NVIDIA's FP4 format, E2M1 mantissa with a finer-grained per-block
  FP8 scale than OCP MXFP4) is specifically designed to retain more accuracy
  than plain FP4/INT4 at the same 4-bit storage cost — NVIDIA's own numbers
  on Blackwell show close-to-BF16 accuracy on LLMs at ~4x less memory
  bandwidth per weight.
- For a **decode-bound, single-batch, real-time TTS workload** (exactly this
  use case — one user, one utterance, low batch size), the bottleneck per
  step is "load all the weights past through the model once" — this is
  almost pure memory-bandwidth-bound, which is exactly where FP4 helps most
  (compute is not the bottleneck at batch=1, bandwidth is).

## What to actually quantize (and what NOT to)
- **Quantize**: the 36-layer Cohere2 backbone's `q/k/v/o_proj` and
  `mlp.{gate,up,down}_proj` weights — this is >99% of the 3.38B params and
  the thing being replayed once per audio-token decode step.
- **Leave alone**:
  - `stop_predictor` head — trivially small (~1M params), quantizing it buys
    nothing and risks destabilizing the stop-probability calibration that
    controls generation length/cutoff.
  - `embed_tokens` (tied with output) — **correction after measuring**: the
    *input* side is a cheap gather (no bandwidth argument for quantizing it),
    but because embeddings are tied, the same 277,404×2048 table is also used
    as the *output* logits projection (`hidden @ embed_tokens.T`) every
    decode step — a real matmul over the full vocab, not a gather. Measured
    on this box (see `BENCHMARK.md`), that's ~1.14GB moved per step on top of
    the ~6.76GB backbone, i.e. ~15% of total per-token bandwidth. This
    **should** be in scope for NVFP4, not excluded.
  - Mimi codec — frozen, small, runs once per synthesis (batched frame
    decode), not the hot loop. Not worth the accuracy risk; the audio
    reconstruction is far more make-or-break for perceived quality than the
    LM weights are.

## Recipe (TensorRT Model Optimizer, NVIDIA's own toolchain)
1. `pip install nvidia-modelopt[torch]` (matches the CUDA 12.8/Blackwell stack
   we're installing anyway).
2. Calibrate: NVFP4 weight-only or weight+activation quantization needs a
   calibration set — for this model, that's a few hundred `<text>{speaker}:
   {text}<audio>` prompts spanning the 22 languages + emotion tags, run
   through the model to collect activation statistics (`modelopt.torch.quantization.quantize`
   with an NVFP4 config, `forward_loop` over calibration prompts).
3. Export via `modelopt` to a TensorRT-LLM checkpoint, or keep it in
   PyTorch with modelopt's fake-quant/real-quant paths if TensorRT-LLM
   engine-building is more infra than you want for a first pass.
4. **Custom-code risk**: `modeling_tinyaya.py`/`configuration_tinyaya.py` are
   `trust_remote_code=True` HF classes subclassing `Cohere2ForCausalLM` —
   modelopt's automatic layer detection generally works on standard
   `nn.Linear` modules regardless of the wrapping class, but the custom
   `generate_audio()` loop (manual KV-cache stepping, not `model.generate()`)
   means you cannot rely on TensorRT-LLM's built-in generation loop out of
   the box — either port the constrained-decode + stop-head logic into
   TensorRT-LLM's runtime hooks, or run NVFP4 through PyTorch NVFP4 GEMM
   kernels directly (`modelopt`'s PyTorch quantized linear replacement) and
   keep the existing Python decode loop — much less integration work, still
   captures most of the bandwidth win since decode is one token at a time
   through the same set of Linear layers either way.
5. Validate: rerun `infer.py`'s benchmark before/after, and — more
   importantly — spot check audio quality (WER via the IndicConformer ASR
   the blog already benchmarks against, plus manual listening) since 4-bit
   weight quantization is exactly the kind of thing that can quietly
   degrade prosody/expressiveness even when text intelligibility holds up.

## Expected payoff — now grounded in a measured BF16 baseline (see `BENCHMARK.md`)
- Measured on this GB10: **~26 audio-tokens/sec decode, RTF ≈ 3.7-3.9x** (i.e.
  synthesizing 1s of audio takes ~3.7-3.9s wall clock — not real-time), and
  the workload is confirmed memory-bandwidth-bound (~205GB/s effective vs
  ~273GB/s peak LPDDR5X spec), matching this doc's core assumption.
- Weight memory: 6.7GB (BF16 backbone) + ~1.14GB (tied output projection,
  now correctly in scope, see correction above) → roughly ~2.3GB combined at
  NVFP4 (allowing for per-block scale overhead), from ~7.9GB today. That's
  the ~3.4x reduction in bytes-moved-per-token that should translate fairly
  directly into decode latency given the confirmed bandwidth-bound regime —
  naively, ~26 tok/s → ~85-90 tok/s if the reduction transfers cleanly
  (it won't be perfectly linear — kernel-launch overhead and non-weight
  traffic like the KV cache don't shrink with weight quantization — but the
  bulk of the ~38.5ms/step is weight bytes, so a large chunk of that speedup
  should materialize).
- Getting RTF below 1.0 (real-time) looks achievable from NVFP4 alone given
  this margin, without yet touching CUDA graphs/`torch.compile` (see
  `BENCHMARK.md`'s "what would move the needle" list) — those are
  independent, stackable wins on top.
- Bigger payoff appears at higher concurrency (serving many simultaneous
  TTS requests) where FP4's compute throughput advantage compounds with the
  bandwidth advantage — worth benchmarking at batch sizes >1 too if this
  becomes a serving question, not just a single-stream latency question.

## Attempt made — real result: quantization worked, inference got much slower

Actually ran this (`quantize_nvfp4.py`, `nvidia-modelopt==0.46.0`, weight-only
`mtq.W4A16_NVFP4_CFG`, calibrated on 6 representative prompts, then
`mtq.compress()` for real 4-bit storage). Two sub-results:

**Quantization + compression itself worked correctly:**
- 914 quantizers inserted, calibrated cleanly across all 36 layers' `q/k/v/o_proj`
  and `mlp.{gate,up,down}_proj` (`lm_head` excluded by modelopt's own default
  config — see correction below).
- Real on-disk compression confirmed: **6.7GB (BF16) → 3.0GB (NVFP4), a real
  2.23x reduction** (less than the theoretical ~3.5-4x because `lm_head`,
  biases, and per-block FP8 scale factors stay at higher precision).

**But decode got ~14.7x *slower*, not faster: measured 1.80 tok/s vs the
26.4 tok/s BF16 baseline** (single prompt, 48 tokens, in-process — i.e. the
already-quantized model object benchmarked directly, no save/reload involved,
so this isn't a checkpoint-loading artifact).

**Root cause, confirmed**: `find .venv -iname "*.so"` under the `modelopt`
package returns nothing — this pip install ships **zero compiled CUDA
extensions**. `mtq.compress()` genuinely packs weights into real 4-bit
storage, but the compressed `Linear` forward pass has no fused NVFP4 GEMM
kernel to call — it's doing unpack-then-matmul in plain PyTorch ops, which is
slower than a native BF16 matmul, not faster. Triton (3.8.0) is present in
the environment but modelopt isn't routing through a Triton NVFP4 kernel
here either (GB10/Blackwell software support this fresh means the kernel
paths that exist for older Blackwell/Hopper SKUs in this modelopt version
likely don't yet target sm_121 specifically).

**Correction to the "recipe" section above**: step 3 said "keep it in PyTorch
with modelopt's fake-quant/real-quant paths if TensorRT-LLM engine-building
is more infra than you want" — **this is now shown to be wrong for actual
speed** on this hardware/software combination. The PyTorch real-quant path
(`mtq.compress()`) is good for **validating accuracy and measuring real
compression ratio** (both of which it did correctly), but getting an actual
latency win on GB10 requires the compiled fused-kernel path — i.e. **an
actual TensorRT-LLM engine build**, not a bare `AutoModelForCausalLM` +
`modelopt` PyTorch path. That's a meaningfully bigger integration project
(needs the custom `generate_audio()` decode logic — `allowed_ids` masking +
`stop_predictor` — reimplemented against TensorRT-LLM's runtime, similar to
the `torch.compile` finding in `BENCHMARK.md`), and wasn't attempted in this
session.

## Practical next step
Both the BF16 baseline (`BENCHMARK.md`) and a first NVFP4 attempt are now
measured. The honest state: **NVFP4 is not yet a win on this box** with the
tooling available today (`nvidia-modelopt==0.46.0`, no compiled kernels for
GB10). Real progress requires either (a) a TensorRT-LLM engine build with a
genuine Blackwell FP4 kernel path, once TensorRT-LLM ships GB10-targeted
kernels, or (b) checking for a newer `nvidia-modelopt` release with compiled
extensions for this exact compute capability. Worth re-checking periodically
rather than pursuing further right now — the software stack for this chip is
weeks old.
