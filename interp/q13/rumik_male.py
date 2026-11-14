#!/usr/bin/env python3
"""rumik-oss-1 + 'male voice' weight patch (Q13 result, pass p14).

The patch is two constant vectors baked into the MLP output of layers 18 and 26
(the synthetic-male direction from q13/male_dirs.npz at its natural size). It is a
pure weight edit: no gradients, no fine-tuning, 4,096 numbers in total.

    from rumik_male import load_rumik_male
    model = load_rumik_male("rumik-oss-1")        # every speaker now speaks in a male voice
    # ...generate exactly as with rumik-oss-1 (e.g. infer.py / server.py)

CLI:  python rumik_male.py --make-patch      -> writes male_patch.pt next to this file
"""
import argparse
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PATCH = HERE / "male_patch.pt"


def make_patch():
    Z = np.load(HERE / "male_dirs.npz")
    patch = {18: torch.tensor(Z["male_dir_L19"]) * float(Z["male_norm_L19"]),
             26: torch.tensor(Z["male_dir_L27"]) * float(Z["male_norm_L27"])}
    torch.save({"layers": patch, "source": "q13 synthetic-male direction (WORLD F0x0.55, formants x0.85), pass p14"}, PATCH)
    print(f"wrote {PATCH} ({sum(v.numel() for v in patch.values())} numbers)")


def apply_patch(model, strength=1.0, patch_path=PATCH):
    P = torch.load(patch_path)["layers"]
    for l, v in P.items():
        lin = model.model.layers[l].mlp.down_proj
        vec = (strength * v).to(lin.weight.dtype).to(lin.weight.device)
        lin.bias = torch.nn.Parameter(vec, requires_grad=False)
    return model


def load_rumik_male(repo, strength=1.0, **kw):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=kw.pop("dtype", torch.bfloat16), **kw)
    return apply_patch(m.eval(), strength)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--make-patch", action="store_true"); a = ap.parse_args()
    if a.make_patch:
        make_patch()
