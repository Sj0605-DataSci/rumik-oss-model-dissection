#!/usr/bin/env python3
"""Trained speaker classifier on an EXTERNAL model (WavLM hidden states, mean-pooled),
fit on natural rumik clips with known speaker labels. Not circular w.r.t. rumik's
own activations. Picks the WavLM layer with best 5-fold CV accuracy, then scores
steered clips: P(speaker) per alpha.
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WavLMModel

from speaker_eval import read16k, REPO

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def feats(m, fe, paths):
    out = []
    for p in tqdm(paths, desc="wavlm", unit="wav", leave=False):
        a = read16k(p)
        if len(a) < 8000:
            out.append(None); continue
        x = fe(a, sampling_rate=16000, return_tensors="pt")
        hs = m(**x, output_hidden_states=True).hidden_states  # 13 x [1,T,768]
        out.append(np.stack([h[0].mean(0).numpy() for h in hs]))  # [13,768]
    return out


def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(X.shape[1]), Xs.T @ Y)
    return W, mu, sd


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z * 5); return e / e.sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, default=ROOT / "interp/acts")
    ap.add_argument("--steer", type=Path, nargs="*", default=[])
    ap.add_argument("--lam", type=float, default=100.0)
    args = ap.parse_args()
    torch.set_num_threads(8)
    fe = AutoFeatureExtractor.from_pretrained(REPO)
    m = WavLMModel.from_pretrained(REPO).eval()

    clips = [json.loads(l) for l in (args.ref / "clips.jsonl").open()]
    F = feats(m, fe, [args.ref / "wavs" / f"{c['clip_id']:05d}.wav" for c in clips])
    keep = [i for i, f in enumerate(F) if f is not None]
    spk = sorted({c["speaker"] for c in clips})
    y = np.array([spk.index(clips[i]["speaker"]) for i in keep])
    Xall = np.stack([F[i] for i in keep])  # [N,13,768]
    Y = np.eye(len(spk))[y]
    rng = np.random.default_rng(0); folds = rng.integers(0, 5, len(y))
    best = (0, -1)
    for L in range(Xall.shape[1]):
        X = Xall[:, L]; acc = []
        for k in range(5):
            tr, te = folds != k, folds == k
            W, mu, sd = ridge_fit(X[tr], Y[tr], args.lam)
            acc.append((((X[te] - mu) / sd @ W).argmax(1) == y[te]).mean())
        a = float(np.mean(acc))
        print(f"[probe] WavLM layer {L:2d}: 5-fold speaker acc {a:.1%}", flush=True)
        best = max(best, (a, L))
    acc, L = best
    W, mu, sd = ridge_fit(Xall[:, L], Y, args.lam)
    print(f"[probe] best layer {L}: {acc:.1%} (chance {1/len(spk):.0%}); speakers={spk}", flush=True)

    report = {"layer": L, "cv_acc": acc, "speakers": spk, "experiments": {}}
    for d in args.steer:
        wavs = sorted(d.glob("*.wav"))
        Fs = feats(m, fe, wavs)
        rows = defaultdict(list)
        for p, f in zip(wavs, Fs):
            if f is None:
                continue
            al = float(re.search(r"_a([+-][\d.]+)_", p.name).group(1))
            rows[al].append(softmax(((f[L] - mu) / sd) @ W))
        print(f"[probe] {d.name}:", flush=True)
        exp = {}
        for al in sorted(rows):
            P = np.mean(rows[al], 0); arg = np.bincount(np.argmax(rows[al], 1), minlength=len(spk))
            exp[al] = {"p": dict(zip(spk, P.round(3).tolist())), "argmax": dict(zip(spk, arg.tolist()))}
            print(f"   α={al:+.2f} n={len(rows[al])}  " + "  ".join(f"{s}:{p:.2f}" for s, p in zip(spk, P))
                  + "   argmax=" + str(dict(zip(spk, arg.tolist()))), flush=True)
        report["experiments"][d.name] = exp
    out = (args.steer[0].parent if args.steer else args.ref) / "speaker_probe.json"
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
