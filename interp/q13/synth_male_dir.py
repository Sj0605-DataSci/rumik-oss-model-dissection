#!/usr/bin/env python3
"""Editor tool: a 'male' direction inside rumik's own representation, without gradients.

1. Take natural Ira clips (v2 data). WORLD-analyse; male version = F0 x PF, spectral envelope
   frequency-warped by FF (formants down = longer vocal tract); control = WORLD resynth x1.0.
2. Mimi-encode both (8 codebooks) -> rumik audio token ids (q0..q7 interleaved per frame).
3. Round-trip check: decode male tokens, gender classifier P(male) -> is 'male' reachable in codec space?
4. Teacher-force "<text>Ira: {input}<audio>" + tokens through bf16 rumik; clip-mean audio-position
   residuals at L19 and L27. male_dir = mean(male) - mean(control) (unit) + its raw norm.
"""
import json, sys, wave
from pathlib import Path

import numpy as np
import pyworld as pw
import torch
from scipy.signal import resample_poly
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, AutoModelForCausalLM, AutoTokenizer, MimiModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "interp"))
from capture_acts import AUDIO_RE  # noqa: E402
from prompts import build_prompts  # noqa: E402
from steer import f0_stats  # noqa: E402

N, PF, FF = 160, 0.55, 0.85
OUT = ROOT / "interp/q13"; (OUT / "synth").mkdir(exist_ok=True)
torch.set_grad_enabled(False); torch.cuda.set_per_process_memory_fraction(0.25)


def shift(x, fs, pf, ff):
    x = x.astype(np.float64)
    f0, sp, ap = pw.wav2world(x, fs)
    if ff != 1.0:
        nb = sp.shape[1]; src = np.clip(np.arange(nb) / ff, 0, nb - 1)
        sp = np.stack([np.interp(src, np.arange(nb), r) for r in sp]); ap = np.stack([np.interp(src, np.arange(nb), r) for r in ap])
    y = pw.synthesize(f0 * pf, np.ascontiguousarray(sp), np.ascontiguousarray(ap), fs)
    return np.clip(y / (np.abs(y).max() + 1e-9) * 0.8 * np.abs(x).max() / (np.abs(x).max() + 1e-9), -1, 1).astype(np.float32)


def main():
    repo = ROOT / "rumik-oss-1"; ACTS = ROOT / "interp/acts_v2"
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
    mimi = MimiModel.from_pretrained(repo / "codec", dtype=torch.float32).eval().cuda()
    vocab = tok.get_vocab(); inv = {}
    for s, t in vocab.items():
        if (m := AUDIO_RE.match(s)):
            inv[(int(m.group(1)), int(m.group(2)))] = int(t)
    G = "prithivMLmods/Common-Voice-Gender-Detection"
    gfe = AutoFeatureExtractor.from_pretrained(G); gclf = AutoModelForAudioClassification.from_pretrained(G).eval().cuda()
    male_i = [i for i, l in gclf.config.id2label.items() if l.lower().startswith("male")][0]
    pmale = lambda a24: float(gclf(**gfe(resample_poly(a24, 2, 3).astype(np.float32), sampling_rate=16000, return_tensors="pt").to("cuda")).logits.softmax(-1)[0, male_i])

    rows = {r["clip_id"]: r for r in build_prompts(2500, 1, "v2")}
    ira = [c for c in sorted(rows) if rows[c]["speaker"] == "Ira" and (ACTS / "wavs" / f"{c:05d}.wav").exists()][:N]
    grab = {}
    hooks = [model.model.layers[L].register_forward_hook(lambda m_, i_, o, L=L: grab.__setitem__(L, o.detach())) for L in (19, 27)]
    acc = {k: {19: [], 27: []} for k in ("male", "ctrl")}; checks = []

    def to_ids(a24):
        codes = mimi.encode(torch.tensor(a24, device="cuda")[None, None], num_quantizers=8).audio_codes[0]  # [8, frames]
        return [inv[(int(codes[q, f]), q)] for f in range(codes.shape[1]) for q in range(8)], codes

    for k, c in enumerate(tqdm(ira, desc="synth male", unit="clip", dynamic_ncols=True)):
        w = wave.open(str(ACTS / "wavs" / f"{c:05d}.wav")); x = (np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768).astype(np.float32)
        variants = {"male": shift(x, 24000, PF, FF), "ctrl": shift(x, 24000, 1.0, 1.0)}
        prompt = tok(f"<text>Ira: {rows[c]['input']}<audio>").input_ids
        for name, a in variants.items():
            ids, codes = to_ids(a)
            if k < 12:  # round-trip check on a subset
                dec = mimi.decode(codes[None]).audio_values[0, 0].float().cpu().clamp(-1, 1).numpy()
                f0, _, _ = f0_stats(dec, 24000)
                checks.append(dict(variant=name, f0=f0, p_male=pmale(dec)))
                if k < 3:
                    pcm = (dec * 32767).astype(np.int16)
                    with wave.open(str(OUT / "synth" / f"{name}_{c:05d}.wav"), "wb") as o:
                        o.setnchannels(1); o.setsampwidth(2); o.setframerate(24000); o.writeframes(pcm.tobytes())
            seq = torch.tensor(prompt + ids, device="cuda")[None]
            model.model(input_ids=seq, use_cache=False)
            for L in (19, 27):
                acc[name][L].append(grab[L][0, len(prompt):].float().mean(0).cpu())
    for h in hooks:
        h.remove()
    for name in ("male", "ctrl"):
        g = [r for r in checks if r["variant"] == name]
        print(f"[male] round-trip {name}: F0 {np.nanmean([r['f0'] for r in g]):.0f} Hz, P(male) {np.mean([r['p_male'] for r in g]):.2f} (n={len(g)})", flush=True)
    out = {}
    for L in (19, 27):
        M, C = torch.stack(acc["male"][L]), torch.stack(acc["ctrl"][L])
        d = M.mean(0) - C.mean(0); norm = float(d.norm())
        typ = float(C.norm(dim=1).mean())
        out[f"male_dir_L{L}"] = (d / norm).numpy(); out[f"male_norm_L{L}"] = norm; out[f"resid_norm_L{L}"] = typ
        proj = ((M - C) @ (d / norm)).numpy()
        print(f"[male] L{L}: |mean diff| = {norm:.2f} ({norm/typ:.2f} x typical residual); per-clip projection {proj.mean():.2f} ± {proj.std():.2f}", flush=True)
        p = np.load(OUT / f"dirs_L{L}.npz")["pitch_dir"] if (OUT / f"dirs_L{L}.npz").exists() else None
        if p is not None:
            print(f"[male] L{L}: cos(male_dir, pitch_dir) = {float((d / norm).numpy() @ p):+.2f}", flush=True)
    np.savez(OUT / "male_dirs.npz", **out)
    print("[male] saved q13/male_dirs.npz", flush=True)


if __name__ == "__main__":
    main()
