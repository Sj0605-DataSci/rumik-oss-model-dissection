### 🚀 The feature, motivation and pitch

[rumik-ai/rumik-oss-1](https://huggingface.co/rumik-ai/rumik-oss-1) is an open source Indic text-to-speech model released by Rumik. Architecturally it's `Cohere2ForCausalLM` (already supported in vLLM via `commandr.py`) plus two small additions:

1. A `stop_predictor` head (LayerNorm → Linear → GELU → Linear, ~1M params) that predicts when generation should end, used instead of/alongside sampling a literal end token.
2. Generation constrained to a fixed audio-codec sub-vocabulary (Mimi codec tokens) once past a `<audio>` prompt marker — the model should never emit ordinary text tokens in that region.

Neither of these needs new vLLM infrastructure — both fold cleanly into a single `compute_logits()` override on top of the existing `CohereForCausalLM`.

**We have a working out-of-tree plugin** implementing this, verified end-to-end on a DGX Spark (GB10, Blackwell): correct architecture resolution, real NVFP4 quantization via `nvidia-modelopt` running through vLLM's fused `FlashInferCutlassNvFp4LinearKernel`, and generated audio matching the reference HuggingFace implementation's output closely (durations within a few percent on the same prompts).

Benchmarks (single DGX Spark, GB10):

| | BF16 (HF `transformers`) | NVFP4 (via vLLM, this plugin) |
|---|---|---|
| Decode throughput | ~26 tok/s | ~60 tok/s (single request), up to ~1,231 tok/s aggregate at 48 concurrent requests |
| RTF | 3.7–3.9x | 1.6–1.7x |

Full writeup, code, and benchmark methodology: https://github.com/Sj0605-DataSci/rumik-oss-model-dissection (plugin itself is in `vllm-plugin/`).

### Alternatives

Could stay purely out-of-tree via vLLM's plugin system (`vllm.general_plugins` entry point) — which is exactly how we're running it today, and works fine. Raising this issue rather than opening a PR directly because:

- The NVFP4 checkpoint format is still marked experimental by vLLM itself (`Detected ModelOpt NVFP4 checkpoint... could change in future`) — an in-tree model file built against it inherits that instability.
- Our `compute_logits()` override intentionally drops the reference implementation's `min_new_tokens=8` floor, since vLLM's per-step hook has no per-sequence step counter under continuous batching. Worked fine in testing, but it's a real behavioral difference from the original model worth discussing before it's presented as a first-class supported architecture.
- We're independent contributors doing an architecture dissection, not affiliated with Rumik — wanted to check whether there's interest before proposing an in-tree PR, and to get guidance on what test coverage a codec-token-output model (rather than a text/chat model) should have in vLLM's model test suite, since this may be one of the first of its kind.

Happy to open the PR (a small, mechanical diff — new `tinyaya.py` model file + one registry line) if there's interest and guidance on the above.
