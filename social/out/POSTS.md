# Posts: rumik-oss-1 interpretability

Assets in this folder:
- `linkedin_carousel.pdf`: 10 slides (upload as a LinkedIn "document")
- `linkedin/slide_01..10.png`: the same slides as images
- `x/img_01..07.png`: 1600×900 thread images
- `x/voice_male.mp4` (18 s): Ira and Zoya, before and after the patch, with audio
- `x/circuit.mp4` (10 s): Zoya with and without the 8 binding heads, with audio

---

## LinkedIn (attach `linkedin_carousel.pdf`, and optionally `x/voice_male.mp4` as a comment)

I gave an all-female text-to-speech model a male voice by editing 4,096 numbers. No fine-tuning, no gradients. And you can try it now.

rumik.ai open-sourced rumik-oss-1, a 3B Indic TTS model with four built-in voices (Ira, Aisha, Siya, Zoya), all female. I wanted to know where, inside the network, a voice actually gets decided. So I took it apart on a DGX Spark.

What I found:

→ One feature per speaker. A sparse autoencoder on layer 19 has a clean feature for each voice. Turn up the "Aisha" feature while the prompt still says "Ira", and the voice becomes Aisha (an external speaker classifier agrees 6/6 times).

→ About 16 attention heads decide whose voice you hear. Out of 576 heads, roughly 16, mostly in layer 15, carry 82% of speaker identity. Delete just 8 and the model still speaks perfectly but forgets who it is: P(Zoya) falls from 0.91 to 0.19, below chance. Deleting 8 random heads changes nothing.

→ Emotion lives on a 2-D plane. Two axes explain 93% of tone variation: arousal and an "angry" axis. Arousal turns out to be mostly tempo; "excited" and "fast" point almost the same way.

→ The male voice. Directions learned from the four female voices could only push pitch down to about 165 Hz. So I made male versions of real clips with a vocoder, ran them through the model's own audio codec, and measured how its activations shift. Baking that shift into two layers as a constant gives every voice a male twin: about 108 Hz, a gender classifier at 1.00, words intact. Each voice still sounds like itself, because gender and speaker identity live in separate directions.

Try it yourself. The patched model is on Hugging Face; one line switches voices:
model.set_voice("male")   # or None for the original voices

🤗 Model: https://huggingface.co/Sanyam0605/rumik-oss-1-male
💻 Code, results and audio demos: https://github.com/Sj0605-DataSci/rumik-oss-model-dissection/tree/main/interp

Still open: tying the new voice to a new name ("Arjun:") didn't work in the 8 tries I gave it. Ideas welcome.

Independent research on an open model, not affiliated with rumik.ai. Built on rumik-ai/rumik-oss-1 (CC-BY-NC 4.0, non-commercial use).

#interpretability #TTS #speech #AI #opensource #mechanisticinterpretability

---

## X / Twitter thread

**1/** (attach `x/voice_male.mp4`)
I gave an all-female TTS model a male voice by editing 4,096 numbers.

No fine-tuning. No gradients. Just finding where a voice lives inside rumik-oss-1 and nudging it. 🧵

**2/** (attach `x/img_02.png`)
rumik-oss-1 is a 3B Indic TTS: a language model that emits audio-codec tokens. 4 voices, all female.

I trained a sparse autoencoder on layer 19. One feature per speaker. Turn up "Aisha" on an "Ira:" prompt and the voice becomes Aisha.

**3/** (attach `x/img_03.png`)
Where does the model decide whose voice to use?

I zeroed each of its 576 attention heads in turn. About 16 heads, mostly in layer 15, carry 82% of speaker identity. 8 random heads: 0%.

**4/** (attach `x/circuit.mp4`)
Delete just the top 8 and the model still talks fine, but forgets who it is.

P(Zoya) on a "Zoya:" prompt: 0.91 → 0.19. Below chance.

**5/** (attach `x/img_05.png`)
Making it male, attempt 1: push along a "pitch" direction learned from the female voices.

It hits a wall around 165 Hz. A gender classifier still says 0.01 male. You can't extrapolate out of what the model has seen.

**6/** (attach `x/img_06.png`)
The trick: show the model what "male" looks like in its own head.

Vocoder-shift real clips (pitch ×0.55, formants ×0.85) → encode with rumik's own Mimi codec → feed through the model → the activation shift is a male direction.

**7/** (attach `x/img_07.png`)
Bake it into two MLP layers as a constant: every voice gets a male twin.

~108 Hz, P(male) 1.00, Whisper WER ≈ 0, and each voice keeps its identity. Gender and speaker are separate directions.

**8/**
Still open: binding the new voice to a new name ("Arjun:") through those layer-15 heads didn't work yet.

Independent work on an open model (not affiliated with rumik.ai). Code: https://github.com/Sj0605-DataSci/rumik-oss-model-dissection
Model: https://huggingface.co/Sanyam0605/rumik-oss-1-male
