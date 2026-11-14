#!/usr/bin/env python3
"""Editor tooling (CPU): acoustic directions at L19 from natural v2 clips.
pitch_dir : ridge regression of clip-mean audio activations on log-F0, after regressing out
            speaker + tone + pace + lang one-hots (so it's not just 'sad' or 'Siya').
f1_dir/f2_dir: same for formants F1, F2 (vocal-tract length cues).
Saves q13/dirs_L19.npz with unit vectors + per-unit effect sizes (Hz per unit of projection).
"""
import json, sys, wave
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "interp"))
from train_sae import load_acts  # noqa: E402
from steer import f0_stats  # noqa: E402
from voice_lab import formants  # noqa: E402

ACTS = ROOT / "interp/acts_v2"
x, meta = load_acts(ACTS, 19, audio_only=True)
clips = {json.loads(l)["clip_id"]: json.loads(l) for l in (ACTS / "clips.jsonl").open()}
ids = sorted(clips); ix = {c: i for i, c in enumerate(ids)}
co = torch.tensor([ix[int(c)] for c in meta["clip_id"]])
S = torch.zeros(len(ids), x.shape[1]); n = torch.zeros(len(ids))
for b in tqdm(range(0, x.shape[0], 65536), desc="clip means"):
    S.index_add_(0, co[b:b + 65536], x[b:b + 65536].float()); n.index_add_(0, co[b:b + 65536], torch.ones(min(65536, x.shape[0] - b)))
M = (S / n[:, None]).numpy(); del x
ac = []
for c in tqdm(ids, desc="acoustics"):
    w = wave.open(str(ACTS / "wavs" / f"{c:05d}.wav")); a = (np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768).astype(np.float32)
    f0, _, _ = f0_stats(a, 24000); F1, F2, _ = formants(a)
    ac.append((f0, F1, F2))
ac = np.array(ac)
rows = [clips[c] for c in ids]
Z = np.concatenate([np.stack([[r[k] == v for v in vals] for r in rows]).astype(float)
                    for k, vals in [("speaker", ["Ira", "Aisha", "Siya", "Zoya"]), ("tone", ["angry", "excited", "happy", "professional", "sad"]),
                                    ("pace", ["fast", "slow", "steady"]), ("lang", ["hi", "en"])]], 1)


def resid(A, Z):
    B = np.linalg.lstsq(Z, A, rcond=None)[0]; return A - Z @ B


Mr = resid(M - M.mean(0), Z)
out = {}
for j, nm in enumerate(["pitch", "f1", "f2"]):
    y = np.log(ac[:, j]) if nm == "pitch" else ac[:, j]
    ok = ~np.isnan(y); yr = resid(y[ok, None] - y[ok].mean(), Z[ok])[:, 0]
    X = Mr[ok]; lam = 10.0 * np.trace(X.T @ X) / X.shape[1]
    w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ yr)
    u = w / np.linalg.norm(w)
    proj = X @ u; slope = np.polyfit(proj, yr, 1)[0]; r = np.corrcoef(proj, yr)[0, 1]
    out[f"{nm}_dir"] = u.astype(np.float32); out[f"{nm}_slope"] = slope; out[f"{nm}_r"] = r
    print(f"[dirs] {nm}: in-sample r={r:+.2f}; slope {slope:.4f} {'log-Hz' if nm=='pitch' else 'Hz'} per unit; "
          f"projection sd {proj.std():.2f}", flush=True)
cos = lambda a, b: float(a @ b)
names = json.load(open(ROOT / "interp/q12/direction_names.json")); V = np.load(ROOT / "interp/q12/directions_L19.npy")
for nm in ["pitch", "f1", "f2"]:
    print(f"[dirs] cos({nm}, ...): " + ", ".join(f"{k.split('=')[1]} {cos(out[nm+'_dir'], V[i]):+.2f}" for i, k in enumerate(names)), flush=True)
np.savez(ROOT / "interp/q13/dirs_L19.npz", **out)
print("[dirs] saved q13/dirs_L19.npz")
