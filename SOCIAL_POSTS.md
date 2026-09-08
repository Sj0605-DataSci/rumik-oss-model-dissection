# Draft social posts

## LinkedIn

Rumik open-sourced their Indic TTS model (rumik-oss-1) a few days ago, so I spent the week pulling it apart on a DGX Spark and trying to make it faster.

What it actually is: a small multilingual LLM (Cohere2/Aya backbone) taught to emit audio codec tokens instead of words, plus a ~1M-parameter head that learns when to stop talking. No exotic new architecture — just a well-executed reuse of an existing model.

The interesting part was the optimization path, because most of it didn't work:

→ torch.compile / CUDA graphs: made it slower. The sliding-window attention layers never hit a stable shape at these sequence lengths, so it kept recompiling instead of caching.

→ Quantizing to NVFP4 (4-bit) with plain PyTorch: made it 15x slower, not faster. The checkpoint shrank correctly (6.7GB → 3.0GB), but there was no compiled kernel to actually run 4-bit matmuls on Blackwell's tensor cores — so every multiply had to unpack back to bf16 first. Turns out quantizing the data and speeding up the compute are two completely different problems.

→ NVFP4 through vLLM's fused kernels: this one worked. 26 → 60 tokens/sec on a single request (2.3x), and scaling to ~1,231 tok/s aggregate at 48 concurrent requests — a 47x improvement in serving throughput over the baseline. Getting there meant writing a small out-of-tree vLLM model plugin, since the model's custom stop-head and constrained decoding aren't things vLLM supports out of the box.

Full writeup with every number, every failed attempt, and the code: https://github.com/Sj0605-DataSci/rumik-oss-model-dissection
Quantized model on Hugging Face: https://huggingface.co/Sanyam0605/rumik-oss-1-nvfp4-vllm
Opened an issue with vLLM proposing this as a first-class supported model: https://github.com/vllm-project/vllm/issues/55933

## X / Twitter

Rumik open-sourced an Indic TTS model this week. Spent a few days seeing how fast it can actually go on a DGX Spark.

BF16 baseline: 26 tok/s, 3.8x real-time (slow)
+ torch.compile: slower (sliding-window attn never stabilizes)
+ NVFP4 (plain PyTorch): 15x SLOWER — no kernel for 4-bit matmul
+ NVFP4 via vLLM's fused kernel: 60 tok/s single-stream, 1,231 tok/s @ 48 concurrent

Quantizing a model and speeding it up are two different problems.

Writeup: https://github.com/Sj0605-DataSci/rumik-oss-model-dissection
HF model: https://huggingface.co/Sanyam0605/rumik-oss-1-nvfp4-vllm
