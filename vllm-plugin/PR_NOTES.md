# Contributing TinyAya (rumik-oss-1) support to vLLM

## Status
- Checked `vllm-project/vllm` for existing issues/PRs mentioning "rumik" or
  "tinyaya" via `gh search issues`/`gh search prs`: **none found**. This
  would be the first.
- A **working out-of-tree plugin** exists in this directory
  (`tinyaya_vllm.py` + `register.py` + `pyproject.toml`) and is verified end
  to end on this GB10: real NVFP4 kernel (`FlashInferCutlassNvFp4LinearKernel`),
  correct architecture resolution (`Resolved architecture: TinyAyaForCausalLM`),
  and generated audio that matches the HF reference implementation's output
  length/duration closely (see `BENCHMARK.md` for exact numbers: ~60 tok/s,
  RTF 1.6-1.7x vs. the HF path's ~26 tok/s, RTF 3.7-3.9x).
- This plugin is the reference implementation for what an in-tree PR would
  contain — the diff is small and mechanical to port.

## What an in-tree PR would add
1. **`vllm/model_executor/models/tinyaya.py`** — essentially
   `tinyaya_vllm.py` verbatim (the `TinyAyaForCausalLM` class subclassing
   `CohereForCausalLM`), with vLLM's standard file header/license comment.
2. **Registry entry** in `vllm/model_executor/models/registry.py`:
   ```python
   "TinyAyaForCausalLM": ("tinyaya", "TinyAyaForCausalLM"),
   ```
   next to the existing `"Cohere2ForCausalLM": ("commandr", "CohereForCausalLM")`
   line — same file pattern, adjacent entry.
3. **Docs**: an entry in vLLM's supported-models table/page for
   `rumik-ai/rumik-oss-1` (and future release TTS checkpoints), noting it's
   an audio-token-output model (not a chat/completion model) — this is a
   detail vLLM's docs don't currently have a clean category for (its
   "audio language models" docs are about audio *input* — Whisper-style
   ASR/understanding — not audio-token *output* like this).
4. **Tests**: vLLM's model test suite expects a small set of standard
   correctness checks (e.g. `tests/models/registry.py` entry,
   logits/generation smoke test). This model's output isn't text, so the
   standard "generate text, check it's plausible" test pattern doesn't
   directly apply — would need a custom test asserting the model emits only
   audio-vocabulary tokens post-`<audio>`, or a full round-trip through Mimi
   with a sanity check on the resulting waveform. Worth discussing with vLLM
   maintainers what test shape they'd want for a codec-output model, since
   this may be one of the first of its kind in the registry.

## Known gaps / open questions before proposing upstream (be upfront about these)
- **The `min_new_tokens=8` floor is dropped** in `compute_logits()` (see the
  inline comment in `tinyaya_vllm.py`) because vLLM's `compute_logits` hook
  has no per-sequence step counter in continuous batching. In practice this
  wasn't an issue in testing (generated lengths closely matched the HF
  reference), but it's a real behavioral difference worth flagging to
  reviewers rather than glossing over.
- **NVFP4 checkpoint format is experimental** — vLLM itself logs
  `WARNING [modelopt.py:1014] Detected ModelOpt NVFP4 checkpoint. Please
  note that the format is experimental and could change in future.` on
  load. A PR built on this format inherits that instability; worth noting
  in the PR description rather than presenting the checkpoint format as
  settled.
- **Only tested single-request, non-batched generation** so far. Haven't
  yet validated concurrent/batched requests, prefix caching behavior with
  the audio-vocab restriction, or multi-GPU/tensor-parallel correctness.
- **Whether Rumik wants this contributed at all** hasn't been asked. This
  is *our* dissection/reverse-engineering of their open release, not
  something coordinated with the Rumik team — worth reaching out to them
  before opening a public PR that references their model by name, both as
  a courtesy and because they may already have TensorRT-LLM/vLLM support
  planned or in progress that this would duplicate or conflict with.

## Recommendation
Given the open questions above (checkpoint format instability, dropped
min-tokens floor, unasked-about coordination with Rumik, no batching
validation yet), I'd suggest **not** opening the PR to `vllm-project/vllm`
today. Reasonable next steps, roughly in order of increasing commitment:
1. Reach out to Rumik (e.g. `research@rumik.ai`, per their model card) to
   ask if they're already working on this or would want it — avoids
   duplicate effort and respects that it's their model.
2. Post the out-of-tree plugin itself somewhere public (a gist, a small
   repo) as a "here's how to run rumik-oss-1 fast on Blackwell via vLLM"
   writeup — lower-commitment, immediately useful, doesn't require vLLM
   maintainer review cycles.
3. If there's real interest (from Rumik or the vLLM community), *then*
   open a vLLM issue first (not a PR) proposing the addition and asking
   about the test-shape question above, before writing the in-tree PR.
