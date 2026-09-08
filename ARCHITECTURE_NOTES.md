# Rumik-OSS-1 — architecture dissection

Source: [rumik-ai/rumik-oss-1](https://huggingface.co/rumik-ai/rumik-oss-1) (and `-base` variant), pulled 2026-09-08.
Blog: https://rumik.ai/research/rumik-oss

## What it is
A 3.38B param (BF16) autoregressive **speech LM**: text-conditioned TTS built by
extending a small multilingual **Cohere2 (Aya/Command-R family)** decoder-only LM
with an **audio token vocabulary** derived from Kyutai's **Mimi** neural codec, and
generating flattened RVQ codes exactly like the "Llama-Mimi" recipe
(arXiv:2509.14882 — flatten multi-codebook Mimi tokens into one sequence, model
autoregressively with a plain Transformer decoder, decode back with Mimi).

Base LM: `CohereLabs/tiny-aya-fire` (a small Aya/Cohere2 checkpoint) — they didn't
invent a new backbone, they repurposed an existing small multilingual LLM as the
speech backbone and fine-tuned it to also emit audio tokens. This is the same
trick as most 2025-26 "speech-LM" work: text LLM + codec vocab >> bespoke TTS arch.

## Backbone config (`config.json`)
- `architectures`: `TinyAyaForCausalLM` (subclasses `Cohere2ForCausalLM`)
- hidden_size 2048, 36 layers, 16 attn heads, **4 KV heads** (GQA, 4x compression)
- head_dim 128, intermediate_size 11008, SiLU gated activation
- **Local/global attention interleave** (Gemma2/Cohere2-style):
  `sliding_attention × 3, full_attention × 1` repeating every 4 layers
  (`sliding_window=4096`, `_sliding_window_pattern=4`) — cuts KV-cache cost on the
  3/4 of layers that don't need full context, only 1/4 layers pay for global attention.
- RoPE: `rope_theta=50000`, `position_embedding_type=rope_gptj`, full rotary (100%)
- `use_parallel_block=true` — parallel attention+MLP blocks (GPT-J/PaLM style: both
  branches read the same normed input and sum outputs, instead of sequential
  residual blocks). Saves one LayerNorm/sync per layer, helps throughput.
- `use_embedding_sharing=true` — tied input/output embeddings (important since
  vocab_size=277,404 is huge — sharing avoids a second giant weight matrix)
- `use_qk_norm=false`, `logit_scale=1.0` (Cohere2 clamps/scales logits; here left neutral)
- `max_position_embeddings=8192`
- vocab_size **277,404** — this is huge for a 2048-hidden model, almost entirely
  because of the audio token vocab: `num_quantizers=8 * codebook_size=2048` =
  16,384 possible `<code_quantizer>` tokens, times how they're enumerated per
  position, plus the base Aya multilingual text vocab (Indic scripts + English).

## The actual "invention" here is small and precise
`modeling_tinyaya.py` is only ~86 lines. It does two things on top of stock Cohere2:

1. **`stop_predictor` head** — a tiny LayerNorm→Linear→GELU→Linear MLP bolted onto
   the last hidden state, trained as a binary classifier ("should generation stop
   now"), sigmoid-thresholded at 0.5. This is their "frame-boundary termination
   loss" mentioned in the blog — instead of relying purely on sampling an
   `</audio>` token (which can be unreliable / miscalibrated), a dedicated small
   head predicts stop probability at every decoding step, checked only after
   `min_new_tokens`. Cheap, effective way to control clipped/rambling endings in
   audio LMs.

2. **`generate_audio()`** — custom greedy/sampled decode loop that:
   - restricts logits to only the audio-token id range on every step
     (`allowed_ids` = all `<code_q>` tokens + `</audio>`), so the LM can never
     wander into text vocab mid-generation — a hard vocabulary mask rather than
     hoping the model learned to stay in-distribution.
   - runs its own KV-cache loop manually (not `model.generate`) so the stop-head
     check can be interleaved every step.

Everything else — attention, MLP, rotary embeddings, GQA, sliding window — is
untouched HF `Cohere2ForCausalLM`. No new attention math, no new positional
scheme. All the "TTS-specific" engineering is: (a) codec-token vocab extension,
(b) a stop head, (c) a constrained decode loop, (d) training data/curriculum.

## Codec: Mimi (frozen)
- 24kHz, 8 codebooks (RVQ), 100 tokens/sec (per the blog)
- `server.py` shows the actual frame reassembly logic: tokens come out flattened
  as `<code_quantizer>`-style ids in *round-robin quantizer order*
  (`quantizer==expected → append; quantizer==0 resets frame`), reassembled into
  `(8, T)` code tensors, then decoded via `MimiModel.decode()`. This flatten/
  reassemble scheme is exactly the Llama-Mimi paper's approach — one flat token
  stream instead of hierarchical/delayed multi-codebook prediction.
- Mimi itself is frozen (not fine-tuned) — only the LM learns to emit its codes.

## Training recipe (from blog, `rumik-oss` post)
- Phase 1 (75K updates): English-only speech pretrain (Emilia, ~40k hrs)
- Phase 2 (30K updates): + multilingual/Indic voice adapt (IndicVoices 7,348 hrs,
  IndicVoices-R 1,704 hrs, RASA ~1,145 hrs) — 22 languages total
- Phase 3: SFT for style/accent/pace control (the `<description="...">` prompt
  format visible in `server.py`'s demo textarea)
- Post-training: GRPO against unspecified reward model(s)
- lr 1e-4 peak, global batch 96, CE token loss + frame-boundary/stop loss (see above)

## Verified against actual weights (safetensors header inspection, no torch needed)
- `model.embed_tokens.weight`: `[277404, 2048]` — no separate `lm_head.weight` anywhere
  → confirms tied embeddings (`use_embedding_sharing=true`)
- Per layer: exactly **one** `input_layernorm` (no second post-attn norm) →
  confirms `use_parallel_block=true` (attn and MLP both read the same normed
  input, PaLM/GPT-J style, not sequential pre-norm blocks)
- `self_attn.k_proj` / `v_proj`: `[512, 2048]` = 4 kv_heads × 128 head_dim →
  confirms GQA with 4 KV heads vs 16 query heads (4x KV-cache compression)
- `stop_predictor.{0,1,3}`: LayerNorm(2048) → Linear(2048→512) → GELU →
  Linear(512→1) — matches `modeling_tinyaya.py` exactly, ~1.05M extra params
  bolted onto the 3.38B backbone. The entire "novel" TTS-specific parameter
  budget is that ~1M-param head; everything else is stock Cohere2 weights.

## `rumik-oss-1` vs `rumik-oss-1-base` diff (config-level, verified)
Base = 3.95B params / 8.3GB (bigger than the released 3.38B!), pre-SFT checkpoint.
After stripping generic HF `GenerationConfig` boilerplate that the base config
still carries (defaults like `top_p`, `repetition_penalty`, etc. — the released
config was cleaned up), the real differences are:

- **`max_position_embeddings`**: base `500000` (inherited straight from
  `tiny-aya-fire`'s long-context default) → released `8192`. They deliberately
  clamp context for the TTS use case — no need for 500k tokens when an
  utterance is a few thousand audio tokens at most; this also shrinks the
  RoPE/KV-cache assumptions bundled with the released checkpoint.
- **`speakers`**: base has 4 voices (`Ira, Aisha, Siya, Zoya`) → released adds
  a 5th, `Silver` — a voice introduced specifically during the SFT/style phase,
  not present in the pretrain stage.
- **`vocab_size`**: base `277395` → released `277404` (+9 tokens) — new special
  tokens added at fine-tune time, consistent with adding the `Silver` speaker
  tag and/or additional emotion/style control tokens (`<laugh>`, `<excited>`,
  `<description="...">` style markers referenced in the blog and demo UI).
- Param count actually *shrinks* released vs base (3.95B → 3.38B) despite vocab
  growing — likely the base checkpoint carries some extra untrimmed
  generation/aux structure from the raw pretrain run that gets pruned before
  release (worth diffing tensor names directly if this matters later).

This confirms the 3-phase recipe cleanly: `-base` = phases 1+2 (English +
multilingual pretrain), `rumik-oss-1` = + phase 3 SFT (style/speaker/pace
control) + GRPO.

## Files in the repo
- `modeling_tinyaya.py`, `configuration_tinyaya.py` — the actual novel code (~110 lines total)
- `model-0000{1,2}-of-00002.safetensors` — 3.38B params, BF16, ~7.2GB total
- `server.py` — reference FastAPI server + browser demo UI, shows real inference contract
- `tokenizer.json` / `tokenizer_config.json` — extended Aya tokenizer w/ audio tokens
- `codec/` — Mimi codec weights (not yet inspected — check after download completes)
- `rumik-oss-1-base` variant — presumably pre-SFT/pre-GRPO checkpoint (English+multilingual
  pretrain only, no style-control SFT) — worth diffing configs/behavior against the main release

## Optimization / borrowable ideas for MiniK3 / our own work
- **Parallel transformer blocks** (`use_parallel_block`) — free-ish throughput win
  at small scale, worth trying if not already in MiniK3 ablations.
- **Local:global attention ratio 3:1 with 4k window** — good template for cheap
  long-context KV cache at small model sizes; directly reusable pattern.
- **Small bolt-on stop/boundary head instead of relying on EOS sampling** —
  generalizable trick anywhere we do constrained/structured autoregressive
  generation (not just audio) and want reliable, low-variance stopping.
- **Reuse an existing small multilingual LLM as the backbone for a new modality**
  rather than training a bespoke architecture from scratch — validates the
  "distill/adapt Qwen3, don't out-pretrain" philosophy already set for the
  Featherweight distillation goal.
- **Flattened single-stream codec tokens (Llama-Mimi-style) over hierarchical
  RVQ prediction** — simpler loss, simpler decode loop, and per this lineage
  performs competitively; relevant if we ever touch audio/multimodal token
  streams.
