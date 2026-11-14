#!/usr/bin/env python3
"""Speaker-identity metric via WavLM x-vectors (microsoft/wavlm-base-plus-sv).

1. Reference: embed natural clips from an acts dir -> per-speaker centroids; report
   nearest-centroid accuracy (split-half) so we know the metric can tell speakers apart.
2. Steered: embed every wav in a steer experiment dir; report cosine to each centroid
   and the nearest speaker, per alpha.
CPU by default (don't share the GPU with a running job).
"""
import argparse, json, re, wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WavLMForXVector

ROOT = Path(__file__).resolve().parent.parent
REPO = "microsoft/wavlm-base-plus-sv"


def read16k(p):
    w = wave.open(str(p)); sr = w.getframerate()
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    a = resample_poly(a, 2, 3) if sr == 24000 else a
    # loudness-normalise (RMS over voiced-ish samples) so quiet clips can't fake a speaker
    act = np.abs(a) > 0.1 * np.abs(a).max() if a.size else a
    rms = np.sqrt(np.mean(a[act] ** 2)) if a.size and act.any() else 0.0
    return (a * (0.1 / rms)).clip(-1, 1).astype(np.float32) if rms > 1e-5 else a


class Embedder:
    def __init__(self, device):
        self.fe = AutoFeatureExtractor.from_pretrained(REPO)
        self.m = WavLMForXVector.from_pretrained(REPO).eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, paths):
        out = []
        for p in tqdm(paths, desc="embed", unit="wav", leave=False):
            a = read16k(p)
            if len(a) < 8000:
                out.append(None); continue
            x = self.fe(a, sampling_rate=16000, return_tensors="pt").to(self.device)
            e = self.m(**x).embeddings[0]
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, default=ROOT / "interp/acts")
    ap.add_argument("--steer", type=Path, nargs="*", default=[ROOT / "interp/steer_L19/to_zoya"])
    ap.add_argument("--max_ref", type=int, default=60, help="clips per speaker for centroids")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(8)
    emb = Embedder(args.device)

    clips = [json.loads(l) for l in (args.ref / "clips.jsonl").open()]
    by_spk = defaultdict(list)
    for c in clips:
        if len(by_spk[c["speaker"]]) < args.max_ref:
            by_spk[c["speaker"]].append(args.ref / "wavs" / f"{c['clip_id']:05d}.wav")
    E = {s: [e for e in emb(ps) if e is not None] for s, ps in by_spk.items()}
    spk = sorted(E)
    # split-half sanity: centroids from even, classify odd
    cent_a = {s: np.mean(E[s][0::2], 0) for s in spk}
    correct = total = 0
    for s in spk:
        for e in E[s][1::2]:
            pred = max(spk, key=lambda t: e @ cent_a[t] / np.linalg.norm(cent_a[t]))
            correct += pred == s; total += 1
    acc = correct / total
    cent = {s: np.mean(E[s], 0) / np.linalg.norm(np.mean(E[s], 0)) for s in spk}
    print(f"[spk] reference nearest-centroid accuracy (split-half): {acc:.1%} on {total} clips "
          f"(chance {1/len(spk):.0%})", flush=True)
    print("[spk] centroid cosine matrix:", flush=True)
    for s in spk:
        print("   " + s.ljust(6) + " ".join(f"{cent[s] @ cent[t]:.3f}" for t in spk), flush=True)

    report = {"ref_acc": acc, "speakers": spk, "experiments": {}}
    for d in args.steer:
        wavs = sorted(d.glob("*.wav"))
        es = emb(wavs)
        rows = defaultdict(list)
        for p, e in zip(wavs, es):
            if e is None:
                continue
            al = float(re.search(r"_a([+-][\d.]+)_", p.name).group(1))
            rows[al].append({s: float(e @ cent[s]) for s in spk})
        print(f"[spk] {d.name}:", flush=True)
        exp = {}
        for al in sorted(rows):
            r = rows[al]
            m = {s: float(np.mean([x[s] for x in r])) for s in spk}
            near = defaultdict(int)
            for x in r:
                near[max(x, key=x.get)] += 1
            exp[al] = {"mean_cos": m, "nearest": dict(near), "n": len(r)}
            print(f"   α={al:+.2f} n={len(r)}  " + "  ".join(f"{s}:{m[s]:.3f}" for s in spk)
                  + f"   nearest={dict(near)}", flush=True)
        report["experiments"][d.name] = exp
    (args.steer[0].parent / "speaker_eval.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
