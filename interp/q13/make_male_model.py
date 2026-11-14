#!/usr/bin/env python3
"""Assemble `rumik-oss-1-male`: rumik-oss-1 + switchable male voice patch, in standard HF layout.

What changes vs rumik-oss-1:
  * voice-patches.safetensors : 2 x 2048 buffers (`voice_patches.male_L18`, `voice_patches.male_L26`)
  * model.safetensors.index.json : maps those keys to the new file
  * config.json : "voice_patches": {"male": [18, 26]}
  * modeling_tinyaya.py : adds model.set_voice("male" | None, strength=1.0), which adds the patch
                          to the MLP output of those layers (forward hooks; off by default)
  * server.py : speakers "Ira (male)", "Aisha (male)", "Zoya (male)", "Siya (male)"
  * README.md : attribution, license, modifications (CC-BY-NC adaptation requirements)
Base weights are linked (local) or copied (--copy, for upload).
"""
import argparse, json, os, shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "rumik-oss-1"
HERE = Path(__file__).resolve().parent

MODEL_ADDON = '''

# ---------------------------------------------------------------------------
# Voice patches (added in rumik-oss-1-male; not part of the original release).
# A voice patch is a constant vector added to the MLP output of a few layers. The "male" patch was
# found by weight-space interpretability (no gradients): see README "How the male voice was made".
# ---------------------------------------------------------------------------
_orig_init = TinyAyaForCausalLM.__init__


def _init_with_patches(self, config):
    _orig_init(self, config)
    self.voice_patches = nn.Module()
    self._voice_hooks = []
    for name, layers in (getattr(config, "voice_patches", None) or {}).items():
        for l in layers:
            self.voice_patches.register_buffer(f"{name}_L{l}", torch.zeros(int(config.hidden_size)), persistent=True)


def set_voice(self, voice=None, strength=1.0):
    """voice=None restores the original model; voice="male" applies the male patch at `strength`."""
    for h in self._voice_hooks:
        h.remove()
    self._voice_hooks = []
    if voice is None:
        return self
    layers = (getattr(self.config, "voice_patches", None) or {}).get(voice)
    if layers is None:
        raise ValueError(f"unknown voice patch {voice!r}; available: {list((self.config.voice_patches or {}).keys())}")
    for l in layers:
        vec = getattr(self.voice_patches, f"{voice}_L{l}")
        def hook(_m, _i, out, vec=vec, s=float(strength)):
            return out + s * vec.to(out.dtype)
        self._voice_hooks.append(self.model.layers[l].mlp.down_proj.register_forward_hook(hook))
    return self


TinyAyaForCausalLM.__init__ = _init_with_patches
TinyAyaForCausalLM.set_voice = set_voice
'''

README = """---
pipeline_tag: text-to-speech
base_model: rumik-ai/rumik-oss-1
license: other
license_name: cohere-cc-by-nc-4.0-with-acceptable-use-addendum
license_link: https://cohere.com/cohere-labs-cc-by-nc-license
tags: [tts, indic, multilingual, interpretability, model-editing]
---
# rumik-oss-1-male

**rumik-oss-1 with a switchable male voice.** The original model ships four female voices
(Ira, Aisha, Siya, Zoya). This variant adds a male version of each, created by editing the
model's weights directly: no fine-tuning, no gradients, 4,096 added numbers.

This is an independent research adaptation of [rumik-ai/rumik-oss-1](https://huggingface.co/rumik-ai/rumik-oss-1)
and is not affiliated with or endorsed by rumik.ai.

## Usage
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("<this repo>", trust_remote_code=True, dtype="bfloat16").cuda().eval()
model.set_voice("male")          # male Ira / Aisha / Zoya / Siya
# ... generate exactly as with rumik-oss-1 (prompt "Ira: <text><audio>", model.generate_audio(...))
model.set_voice(None)            # original female voices
model.set_voice("male", strength=1.5)   # deeper
```
The bundled `server.py` also exposes speakers `"Ira (male)"`, `"Aisha (male)"`, `"Zoya (male)"`, `"Siya (male)"`.

## Measured (6 clips per voice, 2 sentences)
| voice | F0 before | F0 with patch | P(male) | Whisper WER | speaker probe still hears |
|---|---|---|---|---|---|
| Ira | 232 Hz | 108 Hz | 1.00 | 0.00 | Ira 0.92 |
| Aisha | 222 Hz | 107 Hz | 1.00 | 0.02 | Aisha 0.97 |
| Zoya | 260 Hz | 115 Hz | 1.00 | 0.00 | Zoya 0.80 |
| Siya | 238 Hz | 103 Hz | 1.00 | 0.00 | Siya 0.94, but only 1 of 6 clips intact: try strength 0.7 |

P(male): prithivMLmods/Common-Voice-Gender-Detection. Speaker probe: ridge probe on WavLM features (99.4% CV).
Higher formants (F2 ≈ 1800 Hz) mean the voices may sound like a deep voice with a smaller vocal tract.

## How the male voice was made
1. Directions learned from the four female voices could only lower pitch to ~165 Hz.
2. Natural Ira clips were vocoder-shifted (WORLD: F0 x0.55, spectral envelope x0.85), encoded with the
   model's own Mimi codec (round trip: 127 Hz, P(male) 1.00) and teacher-forced through the model.
3. The mean activation difference (male - control) at layers 19 and 27 is the patch, added to the MLP
   output of layers 18 and 26 at its natural size.

Full study (SAE features, steering, the layer-15 speaker-binding circuit):
https://github.com/Sj0605-DataSci/rumik-oss-model-dissection

## Modifications relative to rumik-oss-1
* added `voice-patches.safetensors` (two 2048-d vectors) and its entries in the weight index
* `config.json`: added `voice_patches`
* `modeling_tinyaya.py`: added `set_voice()` (forward hooks; the original behaviour is unchanged until it is called)
* `server.py`: added "(male)" speakers
All original weights are unchanged.

## License
Same as the base model: CC-BY-NC 4.0 with Cohere Labs' acceptable-use addendum, research and non-commercial
use only. The Mimi codec is CC-BY-4.0. Please also cite rumik-oss 1 (Govindu et al., 2026).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "rumik-oss-1-male")
    ap.add_argument("--copy", action="store_true", help="copy base weights instead of symlinking (needed for upload)")
    a = ap.parse_args()
    out = a.out; out.mkdir(exist_ok=True)
    for p in SRC.iterdir():
        dst = out / p.name
        if dst.exists() or dst.is_symlink():
            dst.unlink() if not dst.is_dir() or dst.is_symlink() else shutil.rmtree(dst)
        real = p.resolve()
        if p.name.endswith(".safetensors") and not a.copy:
            os.symlink(real, dst)
        elif real.is_dir():
            shutil.copytree(real, dst, symlinks=False)
        else:
            shutil.copy2(real, dst)
    # patch tensors
    Z = np.load(HERE / "male_dirs.npz")
    vecs = {"voice_patches.male_L18": torch.tensor(Z["male_dir_L19"]) * float(Z["male_norm_L19"]),
            "voice_patches.male_L26": torch.tensor(Z["male_dir_L27"]) * float(Z["male_norm_L27"])}
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in vecs.items()}, str(out / "voice-patches.safetensors"),
              metadata={"format": "pt", "source": "rumik-oss-1-male voice patch (q13 synthetic-male direction)"})
    idx = json.loads((out / "model.safetensors.index.json").read_text())
    for k in vecs:
        idx["weight_map"][k] = "voice-patches.safetensors"
    (out / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2))
    cfg = json.loads((out / "config.json").read_text()); cfg["voice_patches"] = {"male": [18, 26]}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    mt = (out / "modeling_tinyaya.py"); mt.write_text(mt.read_text() + MODEL_ADDON)
    sv = (out / "server.py"); s = sv.read_text()
    s = s.replace('SPEAKERS = ("Ira", "Aisha", "Siya", "Zoya")',
                  'SPEAKERS = ("Ira", "Aisha", "Siya", "Zoya", "Ira (male)", "Aisha (male)", "Siya (male)", "Zoya (male)")')
    s = s.replace("        prompt = f'<text>{req.speaker}: {req.input}<audio>'",
                  "        name, male = req.speaker.replace(' (male)', ''), req.speaker.endswith('(male)')\n"
                  "        self.model.set_voice('male' if male else None)\n"
                  "        prompt = f'<text>{name}: {req.input}<audio>'")
    s = s.replace("<option>Zoya</option></select>", "<option>Zoya</option><option>Ira (male)</option><option>Aisha (male)</option><option>Zoya (male)</option><option>Siya (male)</option></select>")
    sv.write_text(s)
    (out / "README.md").write_text(README)
    print(f"built {out}  (weights {'copied' if a.copy else 'symlinked'})")


if __name__ == "__main__":
    main()
