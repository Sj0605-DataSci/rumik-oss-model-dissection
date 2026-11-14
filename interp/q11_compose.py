#!/usr/bin/env python3
"""Q11: do steering directions compose linearly?

For pairs (A, B) of layer-19 DiffMean directions we generate: base, A, B, A+B (same seeds).
Metrics from external WavLM probes / acoustics. Additivity test per metric m:
    interaction = m(A+B) - m(A) - m(B) + m(base)   (0 = perfectly additive)
and retention: does adding B keep A's own effect? ret_A = [m_A(A+B)-m_A(base)] / [m_A(A)-m_A(base)]
Pairs mix near-orthogonal (speaker x emotion) and near-opposite (excited x sad, fast x sad).
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer, MimiModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_acts import to_codes, save_wav, AUDIO_RE  # noqa: E402
from steer import f0_stats, TEXTS  # noqa: E402
from steer_compare import Probes, generate_rows  # noqa: E402

ACTS = ROOT / "interp/acts_v2"; OUT = ROOT / "interp/q11"; LAYER = 19
ALPHA = {"speaker=Aisha": 0.3, "speaker=Zoya": 0.3, "tone=excited": 0.4, "tone=sad": 0.2,
         "tone=happy": 0.4, "pace=fast": 0.4, "pace=slow": 0.2}
PAIRS = [("speaker=Aisha", "tone=excited"), ("speaker=Aisha", "tone=sad"), ("speaker=Zoya", "pace=fast"),
         ("speaker=Zoya", "tone=happy"), ("tone=excited", "tone=sad"), ("pace=fast", "tone=sad")]


def metric(lab, r, base_dur):
    k, v = lab.split("=")
    if k == "speaker":
        return r["p_spk"][v]
    if k == "tone":
        return r["p_tone"][v]
    return (base_dur / r["dur"]) if v == "fast" else (r["dur"] / base_dur)


def main():
    torch.set_grad_enabled(False); torch.cuda.set_per_process_memory_fraction(0.35); OUT.mkdir(exist_ok=True)
    names = json.load(open(ROOT / "interp/q12/direction_names.json"))
    V = torch.tensor(np.load(ROOT / "interp/q12/directions_L19.npy"), dtype=torch.float32)
    D = {n: V[i] for i, n in enumerate(names)}
    R = 26.2
    repo = ROOT / "rumik-oss-1"
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
    mimi = MimiModel.from_pretrained(repo / "codec", dtype=torch.bfloat16).eval().cuda()
    sr = int(AutoFeatureExtractor.from_pretrained(repo / "codec").sampling_rate)
    vocab = tok.get_vocab(); end_id = int(vocab["</audio>"])
    mapping = {int(t): (int(m.group(1)), int(m.group(2))) for s, t in vocab.items() if (m := AUDIO_RE.match(s))}
    allowed = torch.tensor(sorted([*mapping, end_id]), device="cuda")
    probes = Probes(ACTS, "cuda")
    vec = lambda lab: ALPHA[lab] * R * D[lab]

    settings = [("base", torch.zeros(2048))]
    for A, B in PAIRS:
        for nm, v in ((A, vec(A)), (B, vec(B)), (f"{A}+{B}", vec(A) + vec(B))):
            if nm not in [s[0] for s in settings]:
                settings.append((nm, v))
    seeds = 4
    recs = []
    for ti in (0, 1):
        ids = tok(f'<text>Ira: <description="professional, Indian English, steady pace"> {TEXTS[ti]}<audio>', return_tensors="pt").input_ids.cuda()
        vecs = torch.stack([v for _, v in settings for _ in range(seeds)])
        toks, stopped = generate_rows(model, LAYER, ids, allowed, end_id, vecs, 1000, seed=21 + ti)
        for i, (tk, stp) in enumerate(tqdm(list(zip(toks, stopped)), desc=f"score t{ti}", dynamic_ncols=True)):
            nm = settings[i // seeds][0]; vals = to_codes(tk, mapping)
            if not vals:
                continue
            au = mimi.decode(torch.tensor(vals, device="cuda").reshape(1, -1, 8).transpose(1, 2)).audio_values[0, 0].float().cpu().clamp(-1, 1).numpy()
            (OUT / "wavs").mkdir(exist_ok=True)
            save_wav(OUT / "wavs" / f"{nm.replace('=', '_')}_t{ti}_s{i % seeds}.wav", torch.from_numpy(au), sr)
            fe = probes.embed(au); f0, _, vf = f0_stats(au, sr); rms = float(np.sqrt((au ** 2).mean()))
            if fe is None:
                continue
            recs.append(dict(setting=nm, text=ti, dur=len(au) / sr, f0=f0, rms=rms, intact=bool(stp and rms > 0.03 and vf > 0.2),
                             p_spk={c: probes.prob(fe, "speaker", c) for c in probes.tasks["speaker"]["classes"]},
                             p_tone={c: probes.prob(fe, "tone", c) for c in probes.tasks["tone"]["classes"]}))
        torch.cuda.empty_cache()

    out = {"pairs": []}
    for ti in (0, 1):
        rt = [r for r in recs if r["text"] == ti]
        bdur = np.mean([r["dur"] for r in rt if r["setting"] == "base"])
        m = lambda s, lab: float(np.mean([metric(lab, r, bdur) for r in rt if r["setting"] == s and r["intact"]] or [np.nan]))
        it = lambda s: float(np.mean([r["intact"] for r in rt if r["setting"] == s] or [0]))
        for A, B in PAIRS:
            AB = f"{A}+{B}"; row = {"text": ti, "A": A, "B": B, "intact": {s: it(s) for s in ("base", A, B, AB)}}
            for lab in (A, B):
                vals = {s: m(s, lab) for s in ("base", A, B, AB)}
                own = A if lab == A else B
                denom = vals[own] - vals["base"]
                row[lab] = dict(vals, interaction=vals[AB] - vals[A] - vals[B] + vals["base"],
                                retention=(vals[AB] - vals["base"]) / denom if abs(denom) > 1e-3 else np.nan)
            out["pairs"].append(row)
            print(f"[q11] t{ti} {A} + {B}  intact A/B/AB={row['intact'][A]:.0%}/{row['intact'][B]:.0%}/{row['intact'][AB]:.0%}", flush=True)
            for lab in (A, B):
                r = row[lab]
                print(f"      {lab:14s} base {r['base']:.2f}  A {r[A]:.2f}  B {r[B]:.2f}  A+B {r[AB]:.2f}  "
                      f"retention {r['retention']:.0%}  interaction {r['interaction']:+.2f}", flush=True)
    (OUT / "q11.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"[q11] done -> {OUT/'q11.json'}", flush=True)


if __name__ == "__main__":
    main()
