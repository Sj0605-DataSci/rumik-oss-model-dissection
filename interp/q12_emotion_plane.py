#!/usr/bin/env python3
"""Q12: is there an emotion plane? Geometry of tone directions at layer L (CPU only).

Uses clip-mean AUDIO activations from acts_v2. For each tone: DiffMean vs rest.
- cosine matrix between tone directions (and vs speaker / pace / lang directions)
- PCA of the 5 tone centroids (centred): variance explained by PC1/PC2
- correlate PC1/PC2 with acoustic arousal proxies measured on the natural clips
  (F0, F0 sd, rms, duration) -> is PC1 an arousal axis?
- within-speaker consistency: tone directions computed separately per speaker, cosine
  across speakers (is "sad" the same direction for every voice?)
"""
import json, sys, wave
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sae import load_acts  # noqa: E402
from steer import f0_stats  # noqa: E402

ACTS = ROOT / "interp/acts_v2"; LAYER = 19
OUT = ROOT / "interp/q12"; OUT.mkdir(exist_ok=True)
TONES = ["angry", "excited", "happy", "professional", "sad"]


def main():
    x, meta = load_acts(ACTS, LAYER, audio_only=True)
    clips = {json.loads(l)["clip_id"]: json.loads(l) for l in (ACTS / "clips.jsonl").open()}
    ids = sorted(clips); ix = {c: i for i, c in enumerate(ids)}
    co = torch.tensor([ix[int(c)] for c in meta["clip_id"]])
    S = torch.zeros(len(ids), x.shape[1]); n = torch.zeros(len(ids))
    for b in tqdm(range(0, x.shape[0], 65536), desc="clip means"):
        S.index_add_(0, co[b:b + 65536], x[b:b + 65536].float()); n.index_add_(0, co[b:b + 65536], torch.ones(min(65536, x.shape[0] - b)))
    M = (S / n[:, None]).numpy(); del x
    rows = [clips[c] for c in ids]
    col = lambda k: np.array([r.get(k) for r in rows])

    def dm(mask):
        d = M[mask].mean(0) - M[~mask].mean(0); return d / np.linalg.norm(d)

    tone = col("tone")
    D = {f"tone={t}": dm(tone == t) for t in TONES}
    for k, vals in [("speaker", ["Ira", "Aisha", "Siya", "Zoya"]), ("pace", ["fast", "slow"]), ("lang", ["hi", "en"])]:
        for v in vals:
            D[f"{k}={v}"] = dm(col(k) == v)
    names = list(D); V = np.stack([D[k] for k in names])
    C = V @ V.T
    print("[q12] cosine between directions:")
    print("      " + " ".join(f"{k.split('=')[1][:6]:>6}" for k in names))
    for i, k in enumerate(names):
        print(f"{k.split('=')[1][:6]:>6} " + " ".join(f"{C[i, j]:6.2f}" for j in range(len(names))))

    # PCA of tone centroids
    cent = np.stack([M[tone == t].mean(0) for t in TONES]); cc = cent - cent.mean(0)
    U, s, Vt = np.linalg.svd(cc, full_matrices=False)
    ve = s ** 2 / (s ** 2).sum()
    coords = cc @ Vt[:2].T
    print(f"[q12] tone-centroid PCA variance explained: " + ", ".join(f"PC{i+1} {v:.1%}" for i, v in enumerate(ve)))
    for t, (a, b) in zip(TONES, coords):
        print(f"   {t:12s} PC1 {a:+7.3f}  PC2 {b:+7.3f}")

    # acoustic proxies per clip -> correlate with clip projection on PC1/PC2
    ac = []
    for c in tqdm(ids, desc="acoustics"):
        w = wave.open(str(ACTS / "wavs" / f"{c:05d}.wav")); a = np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768
        f0, f0sd, _ = f0_stats(a.astype(np.float32), w.getframerate())
        ac.append((f0, f0sd, float(np.sqrt((a ** 2).mean())), len(a) / w.getframerate()))
    ac = np.array(ac)
    P = (M - M.mean(0)) @ Vt[:2].T
    print("[q12] correlation of clip projection with acoustics (Pearson r):")
    res = {}
    for pi in range(2):
        for j, nm in enumerate(["F0", "F0sd", "rms", "dur"]):
            ok = ~np.isnan(ac[:, j])
            r = float(np.corrcoef(P[ok, pi], ac[ok, j])[0, 1]); res[f"PC{pi+1}~{nm}"] = r
        print(f"   PC{pi+1}: " + "  ".join(f"{nm} {res[f'PC{pi+1}~{nm}']:+.2f}" for nm in ["F0", "F0sd", "rms", "dur"]))

    # within-speaker consistency of tone directions
    spk = col("speaker"); cons = {}
    for t in TONES:
        per = [dm((tone == t) & (spk == s)) if False else None for s in []]
        ds = []
        for s in ["Ira", "Aisha", "Siya", "Zoya"]:
            m = spk == s
            d = M[m & (tone == t)].mean(0) - M[m & (tone != t)].mean(0); ds.append(d / np.linalg.norm(d))
        ds = np.stack(ds); cm = ds @ ds.T
        cons[t] = float(cm[np.triu_indices(4, 1)].mean())
    print("[q12] tone direction consistency across speakers (mean pairwise cos): " +
          ", ".join(f"{t} {v:.2f}" for t, v in cons.items()))
    json.dump({"names": names, "cos": C.tolist(), "pca_var": ve.tolist(),
               "tone_coords": {t: coords[i].tolist() for i, t in enumerate(TONES)},
               "acoustic_corr": res, "cross_speaker_consistency": cons}, open(OUT / "q12.json", "w"), indent=2)
    np.save(OUT / "directions_L19.npy", V); json.dump(names, open(OUT / "direction_names.json", "w"))
    print(f"[q12] saved -> {OUT}")


if __name__ == "__main__":
    main()
