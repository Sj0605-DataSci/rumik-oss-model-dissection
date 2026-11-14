# Inside rumik-oss-1: interpretability and weight editing

Where does a voice live inside a text-to-speech language model, and can we edit it?
This folder takes [rumik-oss-1](https://huggingface.co/rumik-ai/rumik-oss-1) (3B, TinyAya backbone, Mimi audio tokens,
four female voices) apart with sparse autoencoders, steering, circuit ablations and direct weight edits, all without gradients.

Listen first: [`demos/`](demos/) has before/after clips, and [`../social/out/`](../social/out/) has a carousel and two short videos.

## Results

| question | answer | evidence |
|---|---|---|
| Does an SAE find voice features? | Yes for speaker identity (effect size d up to 8.4). Tone is weak. The "accent" features were language features. | `sae_v2_L19_x16_k64/FEATURES.md` |
| Can one feature steer the voice? | Speaker: yes, Ira becomes Aisha, Zoya or Siya (6/6 by α = 0.4). Emotion and pace: only when features are chosen by measured effect. | `steer_v2_L19/` |
| SAE features vs a mean-difference vector | SAE wins on speaker identity. Mean difference wins on happy, laugh and fast pace. | `compare_v2_L19/COMPARE.md` |
| Is a speaker direction necessary? | Partly. Removing it from all 36 layers cuts Zoya 0.91 → 0.61. No single layer matters. | `q123/results.json` |
| What routes the speaker name into the audio? | About 16 of 576 heads, centred on layer 15, carry 82% of speaker binding (8 random heads: 0%). Removing 8 collapses every voice (Zoya 0.91 → 0.19). | `q123/results.json` |
| Is there an emotion plane? | Yes. Two axes hold 93% of tone variation; arousal is mostly tempo. | `q12/q12.json` |
| Do steering vectors add up? | No. Opposite vectors don't cancel; the larger one wins. | `q11/q11.json` |
| Can weight editing create a male voice? | Yes. A 4,096-number patch gives every voice a male twin (~108 Hz, P(male) 1.00, WER ≈ 0) while keeping identity. | `q13/` |

## Use the male voice
On Hugging Face: [Sanyam0605/rumik-oss-1-male](https://huggingface.co/Sanyam0605/rumik-oss-1-male).
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Sanyam0605/rumik-oss-1-male", trust_remote_code=True, dtype="bfloat16").cuda().eval()
model.set_voice("male")      # or None for the original voices; strength=1.5 for deeper
```
Or apply the 18 KB patch to stock rumik-oss-1 with `q13/rumik_male.py` (`load_rumik_male("rumik-oss-1")`).

## Pipeline
| step | script |
|---|---|
| generate clips (vLLM, NVFP4, in docker) | `vllm_gen.py`, `run_pipeline.sh` |
| capture bf16 residual activations | `capture_acts.py` |
| train BatchTopK SAE | `train_sae.py` |
| splice eval + feature labels | `analyze_sae.py` |
| steering sweeps / SAE vs DiffMean | `steer.py`, `steer_compare.py` |
| speaker probe (WavLM) | `speaker_probe.py`, `speaker_eval.py` |
| Q1–Q3 weight edits, head ablations | `q123_speaker.py` |
| Q11 composition, Q12 emotion plane | `q11_compose.py`, `q12_emotion_plane.py` |
| Q13 voice lab (Claude as weight editor) | `voice_lab.py`, `q13/edits/`, `q13/synth_male_dir.py` |

Activation shards (~20 GB) and SAE checkpoints (513 MB) are not committed. `run_pipeline.sh 2500 all interp/acts_v2 v2 1`
regenerates the data, then `train_sae.py --acts interp/acts_v2`.

Independent research, not affiliated with rumik.ai. rumik-oss-1 is CC-BY-NC 4.0 with Cohere's acceptable-use addendum.
