# Building a full-duplex model on top of rumik-oss-1

## Why this is unusually tractable here
rumik-oss-1 already uses **Kyutai's Mimi codec** — the exact same codec that
Kyutai's own **Moshi** (the reference full-duplex speech-to-speech model) is
built on. Moshi's whole trick for full-duplex is architectural, not a new
codec: it models **two audio streams in lockstep, frame-by-frame** (the
user's incoming audio and the model's own outgoing audio), plus an inner
text "monologue" stream, all multiplexed into one sequence that a single
Transformer predicts autoregressively at a fixed frame rate. Since rumik-oss-1
is currently a **half-duplex, single-stream, turn-based** model (text prompt
in → one audio stream out, no listening while speaking), converting it is
"add the second stream and change the training objective," not "invent a new
model family."

## What "full duplex" actually requires (vs. what we have)

| Capability | rumik-oss-1 today | needed for full duplex |
|---|---|---|
| Streams modeled | 1 (LM's own audio) | 2 (user audio + LM audio), frame-aligned |
| Timing model | none — LM decides length via `stop_predictor`, turn ends | fixed frame clock (Mimi's 12.5Hz), both streams always advance |
| Text conditioning | one text prompt up front, no interleaving | inner monologue: text tokens interleaved with the LM's own audio stream, predicted *before* the audio codes at each step (Moshi's "Inner Monologue") |
| Barge-in / interruption | N/A (not real-time) | model must react to the user stream starting/stopping while still speaking |
| Silence / turn-taking | N/A | needs a way to emit silence codes on its own stream while listening, and pick up user energy mid-generation |

## Proposed architecture: "Rumik-Duplex-1"

### 1. Two parallel audio channels, one Transformer
Keep the exact `TinyAyaForCausalLM` (Cohere2) backbone. Change the sequence
layout from Moshi's RQ-Transformer-lite idea, adapted to this codebook count:

At each Mimi frame step `t` (12.5 Hz → 80ms/frame), the token sequence carries,
concatenated per-step:
```
[ inner_text_token_t ]  [ lm_audio_codes_t (8 quantizers) ]  [ user_audio_codes_t (8 quantizers) ]
```
- `inner_text_token_t`: the LM's own token-level "thinking out loud" text
  stream (what it *would* say), predicted first each step, same idea as
  Moshi's Inner Monologue — gives the model a text-level plan a few frames
  ahead of the audio it commits to, which is what keeps duplex TTS
  intelligible instead of babbling.
- `lm_audio_codes_t`: this model's own output audio for this frame (what
  rumik-oss-1 already predicts, just clocked to a fixed 80ms grid instead of
  running until a stop token).
- `user_audio_codes_t`: the incoming user audio for this same frame, encoded
  by the *same frozen Mimi encoder* — treated as another position in the
  sequence, not cross-attention or a separate encoder. The model just also
  learns to predict/consume this stream (during training it's teacher-forced
  from a real conversation recording; at inference it's live mic input,
  encoded on the fly).

Because rumik-oss-1 already flattens 8-codebook Mimi frames into one stream
(the Llama-Mimi trick), the natural extension is exactly "flatten two
codebook sets instead of one, plus a text slot" — same representation, wider
per-step token count (1 text + 8 LM-audio + 8 user-audio = 17 tokens/frame vs
today's 8/frame).

### 2. Streaming decode loop, not `generate_audio()`'s turn-based loop
Today's `generate_audio()` in `modeling_tinyaya.py` runs until `stop_predictor`
fires or `</audio>` is sampled — that's fundamentally a turn-based assumption.
Full duplex needs a **fixed-rate streaming loop**:
```
every 80ms:
    encode new 80ms of user mic audio -> 8 Mimi codes         (frozen Mimi encoder)
    feed [prev inner-text, prev lm-audio, new user-audio] into model
    sample next inner-text token, then next 8 lm-audio codes  (still causal, same forward pass)
    decode lm-audio codes -> 80ms of speaker output           (frozen Mimi decoder, streaming)
    play audio while accepting next mic chunk (this is where duplex actually happens)
```
This removes `min_new_tokens`/`max_new_tokens`/`allowed_ids` masking as
currently implemented (those assume "one bounded utterance") and replaces
them with an always-running loop with a KV-cache that just keeps sliding
(the existing `sliding_window=4096` local-attention layers are actually
perfect for this — bounded per-layer context is exactly what a
never-ending streaming conversation needs; the 1/4 full-attention layers
handle longer-range coherence).

### 3. What changes in the checkpoint/config
- `layer_types` / `sliding_window`: keep as-is, this pattern was clearly
  chosen with streaming in mind already (even though the released model
  isn't streaming) — a nice sign the backbone is duplex-ready.
- `vocab_size`: grows again — need a second bank of `<usercode_q_v>` tokens
  distinct from the existing `<code_q_v>` (LM) tokens, since the model must
  tell its own output codes apart from the user's input codes in the shared
  vocabulary/embedding table.
- `stop_predictor` head: **drop it**. There's no "stop" in a duplex model —
  the model should instead learn to emit Mimi's silence codes on its own
  channel when it has nothing to say (this is literally how Moshi handles
  turn-taking — silence is just another predicted code sequence, not a
  special control token). Replace with (optional) a lightweight
  **voice-activity / "am I about to speak" head** if you want an explicit
  signal for downstream UX (e.g. show a "listening vs speaking" indicator),
  but it's not required for the core model to function.
- New loss terms: cross-entropy on inner-text stream + cross-entropy on
  lm-audio stream (as today) + **no loss on the user-audio stream** (it's an
  input, not a prediction target) — the model consumes it but is never
  scored on reproducing it.

### 4. Training data — the real bottleneck
This is where it gets hard, and where rumik's own blog already flags the
core problem for their current model ("where do we get high-quality
expressive multi-indic TTS data?") — for duplex you need something strictly
harder: **two-channel conversational recordings with accurate turn overlap**
(real interruptions, backchannels like "mm-hmm", barge-in), not
single-speaker narration. Options, roughly in order of how much new data
collection they imply:
1. **Synthetic duplex data from existing turn-based data**: take existing
   IndicVoices/RASA-style single-speaker clips, synthesize plausible
   "listener" tracks (silence + occasional backchannel tokens) to bootstrap
   the two-channel format without recording anything new — gets the
   architecture/data-pipeline working before investing in real dialogue data.
2. **Two-channel conversational corpora**: real dialogue recordings with
   separate mic tracks per speaker (call-center-style data, or scripted
   conversational recording sessions) — this is what Moshi/dGSLM actually
   trained on. Given rumik's existing IndicVoices-R and RASA data
   relationships, this is plausibly extendable but needs re-collection, not
   reuse of existing single-speaker TTS data.
3. **Self-play**: once a first duplex checkpoint exists (even a rough one
   trained on synthetic data), have two instances of it talk to each other
   and mine the transcripts/audio for realistic turn-taking patterns to
   bootstrap phase-2 fine-tuning — same self-distillation trick used
   elsewhere in speech-LM literature.

### 5. Minimal path to a first prototype (fastest way to see if this works at all)
You do **not** need real duplex training data to validate the architecture
change first:
1. Take `rumik-oss-1-base` (pre-SFT, more headroom for further fine-tuning),
   extend the vocab with the user-audio-code token bank, extend
   `TinyAyaConfig`/`TinyAyaForCausalLM` to accept the wider per-frame
   sequence, drop `stop_predictor`.
2. Fine-tune on synthetic two-channel data built from existing IndicVoices
   clips: channel A = the existing TTS-style audio, channel B = silence
   (all-zero/Mimi-silence codes) for the first pass — this teaches the model
   "ignore an empty second channel, behave exactly like today's model," which
   is a correctness check on the plumbing (tokenizer, KV-cache streaming
   loop, codec round-trip) before touching real conversational dynamics.
3. Then start introducing real second-channel content (recorded
   backchannels, then real overlapping dialogue) incrementally, checking at
   each stage that the model doesn't regress on the plumbing-correctness
   check from step 2.

## Summary
- **Reuse almost everything**: backbone, tokenizer scheme, Mimi codec, the
  flattened-codebook-stream trick, even the local:global attention pattern
  (already streaming-friendly).
- **Replace**: the turn-based bounded decode loop → fixed-rate streaming loop;
  the `stop_predictor` head → silence-as-a-predicted-code + optional VAD head.
- **Add**: a second (user) audio-code token bank, an inner-text stream
  interleaved with the LM's own audio (Moshi's Inner Monologue trick).
- **Hardest part is data**, not architecture — two-channel conversational
  recordings with real turn-taking, which nothing in rumik's current stack
  (built for single-speaker/single-channel TTS) directly provides.
