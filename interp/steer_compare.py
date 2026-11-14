#!/usr/bin/env python3
"""Step 6: SAE features vs DiffMean for steering rumik-oss-1 at layer L.

Per label we compare three directions:
  topd    : SAE feature with best Cohen's d (what v2 steer.py used)
  screen  : best of the top-K SAE candidates, chosen by measured steering effect ("output score"),
            strength scaled per feature: add c * p90_f * W_dec[f] (normalised-space units)
  diffmean: mean(clip-mean audio acts | label) - mean(| not label), unit-normalised, alpha * R

Scoring (external, not rumik's own activations):
  speaker=*  -> P(speaker) from WavLM-layer ridge probe trained on natural clips
  tone=*     -> P(tone) from a WavLM tone probe (5-way)            [+ F0/rms reported]
  tag=<laugh>-> P(laugh) from a WavLM laugh-vs-no-tag probe
  pace=*     -> duration ratio vs unsteered
A clip is INTACT if it stopped before max_new, rms > 0.03 and voiced frac > 0.2; a setting
only scores if >= 2/3 of its clips are intact.  Phase 1 screens on text 0; phase 2 re-runs
each label's best screen/diffmean/topd setting on text 1 (held-out confirmation).
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer, MimiModel, WavLMModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_sae import load_sae  # noqa: E402
from capture_acts import to_codes, save_wav, AUDIO_RE  # noqa: E402
from steer import f0_stats, TEXTS  # noqa: E402
from train_sae import load_acts  # noqa: E402

LABELS = ["speaker=Zoya", "speaker=Aisha", "speaker=Siya", "tone=excited", "tone=sad", "tone=happy",
          "tag=<laugh>", "pace=fast", "pace=slow"]
DESC = "professional, Indian English, steady pace"
WAVLM = "microsoft/wavlm-base-plus-sv"


# ---------------- audio / probes ----------------
def to16k(a24):
    a = resample_poly(a24, 2, 3).astype(np.float32)
    act = np.abs(a) > 0.1 * (np.abs(a).max() + 1e-9)
    rms = np.sqrt(np.mean(a[act] ** 2)) if act.any() else 0.0
    return (a * (0.1 / rms)).clip(-1, 1) if rms > 1e-5 else a


class Probes:
    """WavLM mean-pooled hidden state (best layer per task by 5-fold CV) + ridge one-vs-rest."""

    def __init__(self, acts_dir, device, lam=100.0):
        self.fe = AutoFeatureExtractor.from_pretrained(WAVLM)
        self.m = WavLMModel.from_pretrained(WAVLM).eval().to(device)
        self.dev = device
        clips = [json.loads(l) for l in (acts_dir / "clips.jsonl").open()]
        feats, keep = [], []
        import wave
        for c in tqdm(clips, desc="probe-feats", unit="wav", dynamic_ncols=True):
            p = acts_dir / "wavs" / f"{c['clip_id']:05d}.wav"
            if not p.exists():
                continue
            w = wave.open(str(p)); a = np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768
            f = self.embed(a.astype(np.float32))
            if f is not None:
                feats.append(f); keep.append(c)
        X = np.stack(feats)  # [N, 13, 768]
        self.tasks = {}
        specs = {"speaker": [c["speaker"] for c in keep],
                 "tone": [c["tone"] for c in keep],
                 "laugh": ["yes" if c["tag"] == "<laugh>" else "no" for c in keep if c["tag"] in (None, "<laugh>")]}
        laugh_idx = [i for i, c in enumerate(keep) if c["tag"] in (None, "<laugh>")]
        for task, ys in specs.items():
            Xt = X[laugh_idx] if task == "laugh" else X
            classes = sorted(set(ys)); y = np.array([classes.index(v) for v in ys])
            Y = np.eye(len(classes))[y]
            folds = np.random.default_rng(0).integers(0, 5, len(y))
            best = (-1, 0)
            for L in range(Xt.shape[1]):
                acc = []
                for k in range(5):
                    tr, te = folds != k, folds == k
                    W, mu, sd = self._fit(Xt[tr, L], Y[tr], lam)
                    acc.append((((Xt[te, L] - mu) / sd @ W).argmax(1) == y[te]).mean())
                best = max(best, (float(np.mean(acc)), L))
            acc, L = best
            W, mu, sd = self._fit(Xt[:, L], Y, lam)
            self.tasks[task] = dict(L=L, W=W, mu=mu, sd=sd, classes=classes, cv=acc)
            print(f"[probe] {task}: WavLM layer {L}, 5-fold acc {acc:.1%} (chance {1/len(classes):.0%}) {classes}", flush=True)

    @staticmethod
    def _fit(X, Y, lam):
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xs = (X - mu) / sd
        return np.linalg.solve(Xs.T @ Xs + lam * np.eye(X.shape[1]), Xs.T @ Y), mu, sd

    @torch.no_grad()
    def embed(self, a24):
        a = to16k(a24)
        if len(a) < 8000:
            return None
        x = self.fe(a, sampling_rate=16000, return_tensors="pt").to(self.dev)
        hs = self.m(**x, output_hidden_states=True).hidden_states
        return np.stack([h[0].mean(0).float().cpu().numpy() for h in hs])

    def prob(self, feat, task, cls):
        t = self.tasks[task]
        z = ((feat[t["L"]] - t["mu"]) / t["sd"]) @ t["W"]
        e = np.exp(5 * (z - z.max())); p = e / e.sum()
        return float(p[t["classes"].index(cls)])


# ---------------- generation with per-row steering vectors ----------------
@torch.no_grad()
def generate_rows(model, layer, ids, allowed, end_id, vecs, max_new, seed, temp=0.8, top_k=30):
    """vecs: [B, d] raw-space additive vectors (zeros row = unsteered)."""
    B = vecs.shape[0]
    V = vecs[:, None, :].to(torch.bfloat16).cuda()
    st = {"on": False}

    def hook(_m, _i, out):
        if st["on"]:
            return out + V
    h = model.model.layers[layer].register_forward_hook(hook)
    g = torch.Generator(device="cuda").manual_seed(seed)
    try:
        x = ids.expand(B, -1).contiguous(); mask = torch.ones_like(x)
        out = model(input_ids=x, attention_mask=mask, use_cache=True, output_hidden_states=True, return_dict=True)
        st["on"] = True
        cache, scores, hid = out.past_key_values, out.logits[:, -1, :], out.hidden_states[-1][:, -1:, :]
        done = torch.zeros(B, dtype=torch.bool, device="cuda"); stopped = torch.zeros_like(done)
        toks = [[] for _ in range(B)]; neg = torch.finfo(scores.dtype).min
        for step in tqdm(range(max_new), desc=f"decode B={B}", unit="tok", leave=False, dynamic_ncols=True):
            if step >= 8:
                s = torch.sigmoid(model.stop_predictor(hid).squeeze(-1)).view(-1) > 0.5
                stopped |= s & ~done; done |= s
            if bool(done.all()):
                break
            r = torch.full_like(scores, neg); r.index_copy_(1, allowed, scores.index_select(1, allowed))
            if step < 8:
                r[:, end_id] = neg
            r = r / temp
            r = r.masked_fill(r < torch.topk(r, top_k, -1).values[:, -1:], neg)
            t = torch.multinomial(torch.softmax(r.float(), -1), 1, generator=g)
            e = t.view(-1) == end_id
            stopped |= e & ~done; done |= e
            tl = t.view(-1).tolist(); dl = done.tolist()
            for b in range(B):
                if not dl[b]:
                    toks[b].append(tl[b])
            mask = torch.cat((mask, torch.ones_like(t)), 1)
            out = model(input_ids=t, attention_mask=mask, past_key_values=cache, use_cache=True,
                        output_hidden_states=True, return_dict=True)
            cache, scores, hid = out.past_key_values, out.logits[:, -1, :], out.hidden_states[-1][:, -1:, :]
    finally:
        h.remove()
    return toks, stopped.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", type=Path, default=ROOT / "interp/sae_v2_L19_x16_k64")
    ap.add_argument("--acts", type=Path, default=ROOT / "interp/acts_v2")
    ap.add_argument("--out", type=Path, default=ROOT / "interp/compare_v2_L19")
    ap.add_argument("--k_cand", type=int, default=10)
    ap.add_argument("--cs", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--confirm_seeds", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=1000)
    ap.add_argument("--labels", nargs="*", default=LABELS)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.cuda.set_per_process_memory_fraction(0.35)
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- directions ----
    sae, mean, scale, a = load_sae(args.sae); layer = a["layer"]
    feats = json.loads((args.sae / "features.json").read_text())
    x, meta = load_acts(args.acts, layer, audio_only=True)
    R = float(x[:50000].float().norm(dim=-1).median())
    clips = {json.loads(l)["clip_id"]: json.loads(l) for l in (args.acts / "clips.jsonl").open()}
    ids_sorted = sorted(clips); cidx = {c: i for i, c in enumerate(ids_sorted)}
    co = torch.tensor([cidx[int(c)] for c in meta["clip_id"]])
    C = len(ids_sorted)
    csum = torch.zeros(C, x.shape[1]); cn = torch.zeros(C)
    p90 = {}
    cand_all = sorted({f["feature"] for lab in args.labels for f in feats["labels"][lab][:args.k_cand]})
    cand_t = torch.tensor(cand_all, device="cuda"); act_store = []
    for b in tqdm(range(0, x.shape[0], 65536), desc="clip-means+p90", unit="chunk", dynamic_ncols=True):
        xb = x[b:b + 65536].float()
        csum.index_add_(0, co[b:b + 65536], xb); cn.index_add_(0, co[b:b + 65536], torch.ones(xb.shape[0]))
        if b < 400_000:
            z, _ = sae.encode((xb.cuda() - mean) * scale, train=False)
            act_store.append(z[:, cand_t].cpu())
    cmean = csum / cn[:, None].clamp_min(1)
    A = torch.cat(act_store)
    for j, f in enumerate(cand_all):
        v = A[:, j][A[:, j] > 0]
        p90[f] = float(v.quantile(0.9)) if v.numel() > 10 else 1.0
    del x, A
    rows_meta = [clips[c] for c in ids_sorted]

    def field_mask(lab):
        k, v = lab.split("=", 1)
        return torch.tensor([r.get(k) == v for r in rows_meta])

    dirs = {}  # label -> list of (method, id, strength, raw_vec)
    for lab in args.labels:
        m = field_mask(lab)
        dm = cmean[m].mean(0) - cmean[~m].mean(0); dm = dm / dm.norm()
        lst = [("diffmean", "dm", al, (al * R * dm)) for al in args.alphas]
        cands = feats["labels"][lab][:args.k_cand]
        for rank, fd in enumerate(cands):
            f = fd["feature"]; d = sae.W_dec[f].float().cpu()
            for c in args.cs:  # normalised-space add of c*p90 along decoder row -> raw space /scale
                lst.append(("screen" if rank else "screen", f"f{f}", c, (c * p90[f] * d / scale)))
        f0 = cands[0]["feature"]; d0 = sae.W_dec[f0].float().cpu()
        lst += [("topd", f"f{f0}", al, (al * R * d0 / d0.norm())) for al in (0.2, 0.3, 0.4)]
        dirs[lab] = lst
    print(f"[cmp] layer {layer}, R={R:.1f}; {sum(map(len, dirs.values()))} settings over {len(dirs)} labels", flush=True)

    # ---- models ----
    repo = ROOT / "rumik-oss-1"
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").eval().cuda()
    mimi = MimiModel.from_pretrained(repo / "codec", dtype=torch.bfloat16).eval().cuda()
    sr = int(AutoFeatureExtractor.from_pretrained(repo / "codec").sampling_rate)
    vocab = tok.get_vocab(); end_id = int(vocab["</audio>"])
    mapping = {int(t): (int(mm.group(1)), int(mm.group(2))) for s, t in vocab.items() if (mm := AUDIO_RE.match(s))}
    allowed = torch.tensor(sorted([*mapping, end_id]), device="cuda")
    probes = Probes(args.acts, "cuda")

    def run(lab, settings, text_idx, n_seeds, tag):
        """settings: list of (method,id,strength,vec); adds n_seeds unsteered rows. Returns records."""
        prompt = f'<text>Ira: <description="{DESC}"> {TEXTS[text_idx]}<audio>'
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        rows = [("base", "none", 0.0, torch.zeros(2048))] + list(settings)
        vecs = torch.stack([r[3] for r in rows for _ in range(n_seeds)])
        toks, stopped = generate_rows(model, layer, ids, allowed, end_id, vecs, args.max_new, seed=7 + text_idx)
        recs, od = [], args.out / tag / lab.replace("=", "_").replace("<", "").replace(">", "")
        od.mkdir(parents=True, exist_ok=True)
        for i, (tk, stp) in enumerate(zip(toks, stopped)):
            meth, fid, strength, _ = rows[i // n_seeds]; s = i % n_seeds
            rec = dict(label=lab, method=meth, id=fid, strength=strength, seed=s, text=text_idx,
                       n_tokens=len(tk), stopped=bool(stp))
            vals = to_codes(tk, mapping)
            if vals:
                au = mimi.decode(torch.tensor(vals, device="cuda").reshape(1, -1, 8).transpose(1, 2)
                                 ).audio_values[0, 0].float().cpu().clamp(-1, 1).numpy()
                save_wav(od / f"{meth}_{fid}_{strength:g}_s{s}.wav", torch.from_numpy(au), sr)
                f0, f0sd, vf = f0_stats(au, sr); rms = float(np.sqrt((au ** 2).mean()))
                rec.update(dur=len(au) / sr, f0=f0, f0sd=f0sd, voiced=vf, rms=rms)
                fe = probes.embed(au)
                if fe is not None:
                    rec["p_speaker"] = {c: probes.prob(fe, "speaker", c) for c in probes.tasks["speaker"]["classes"]}
                    rec["p_tone"] = {c: probes.prob(fe, "tone", c) for c in probes.tasks["tone"]["classes"]}
                    rec["p_laugh"] = probes.prob(fe, "laugh", "yes")
            rec["intact"] = bool(rec.get("stopped") and rec.get("rms", 0) > 0.03 and rec.get("voiced", 0) > 0.2)
            recs.append(rec)
        torch.cuda.empty_cache()
        return recs

    def target(lab, r, base_dur):
        k, v = lab.split("=", 1)
        if k == "speaker":
            return r.get("p_speaker", {}).get(v, np.nan)
        if k == "tone":
            return r.get("p_tone", {}).get(v, np.nan)
        if k == "tag":
            return r.get("p_laugh", np.nan)
        if k == "pace":
            d = r.get("dur", np.nan)
            return base_dur / d if v == "fast" else d / base_dur
        return np.nan

    def summarise(lab, recs):
        base = [r for r in recs if r["method"] == "base"]
        bdur = float(np.nanmean([r.get("dur", np.nan) for r in base]))
        btgt = float(np.nanmean([target(lab, r, bdur) for r in base]))
        groups = defaultdict(list)
        for r in recs:
            if r["method"] != "base":
                groups[(r["method"], r["id"], r["strength"])].append(r)
        out = []
        for (meth, fid, st), g in groups.items():
            intact = np.mean([r["intact"] for r in g])
            tg = float(np.nanmean([target(lab, r, bdur) for r in g if r["intact"]])) if intact > 0 else np.nan
            out.append(dict(method=meth, id=fid, strength=st, intact=float(intact),
                            target=tg if intact >= 2 / 3 else np.nan, base=btgt,
                            dur=float(np.nanmean([r.get("dur", np.nan) for r in g])),
                            f0=float(np.nanmean([r.get("f0", np.nan) for r in g])),
                            rms=float(np.nanmean([r.get("rms", np.nan) for r in g]))))
        return out, btgt

    report = {}
    print("[cmp] phase 1: screening on text 0", flush=True)
    for lab in tqdm(args.labels, desc="phase1", unit="label", dynamic_ncols=True):
        recs = run(lab, dirs[lab], 0, args.seeds, "screen")
        summ, btgt = summarise(lab, recs)
        best = {}
        for meth in ("screen", "diffmean", "topd"):
            ok = [s for s in summ if s["method"] == meth and not np.isnan(s["target"])]
            best[meth] = max(ok, key=lambda s: s["target"]) if ok else None
        report[lab] = {"phase1": summ, "base1": btgt, "best1": best}
        tqdm.write(f"[cmp] {lab:15s} base={btgt:.3f}  " + "  ".join(
            f"{m}:{(b['id']+'@'+format(b['strength'],'g')+'='+format(b['target'],'.3f')) if b else 'none-intact'}"
            for m, b in best.items()))
        (args.out / "report.json").write_text(json.dumps(report, indent=2, default=float))

    print("[cmp] phase 2: confirm winners on held-out text 1", flush=True)
    for lab in tqdm(args.labels, desc="phase2", unit="label", dynamic_ncols=True):
        b = report[lab]["best1"]
        settings = [s for s in dirs[lab] if any(bb and s[0] == bb["method"] and s[1] == bb["id"] and s[2] == bb["strength"]
                                                for bb in b.values())]
        if not settings:
            continue
        recs = run(lab, settings, 1, args.confirm_seeds, "confirm")
        summ, btgt = summarise(lab, recs)
        report[lab].update(phase2=summ, base2=btgt)
        tqdm.write(f"[cmp] CONFIRM {lab:15s} base={btgt:.3f}  " + "  ".join(
            f"{s['method']}:{s['id']}@{s['strength']:g}={s['target']:.3f}(intact {s['intact']:.0%})" for s in summ))
        (args.out / "report.json").write_text(json.dumps(report, indent=2, default=float))

    # ---- markdown ----
    md = [f"# Steering comparison, layer {layer} (R={R:.1f})\n",
          "Target = P(label) from external WavLM probe (speaker/tone/laugh) or duration ratio (pace). "
          "Held-out = text 1, %d seeds. 'n/a' = <2/3 clips intact.\n" % args.confirm_seeds,
          "| label | base | top-d SAE | screened SAE | DiffMean |", "|---|---|---|---|---|"]
    for lab in args.labels:
        r = report[lab]; p2 = {s["method"]: s for s in r.get("phase2", [])}
        cell = lambda m: (f"{p2[m]['target']:.2f} ({p2[m]['id']}@{p2[m]['strength']:g})"
                          if m in p2 and not np.isnan(p2[m]["target"]) else "n/a")
        md.append(f"| {lab} | {r.get('base2', float('nan')):.2f} | {cell('topd')} | {cell('screen')} | {cell('diffmean')} |")
    md.append("\nProbe CV accuracy: " + ", ".join(f"{t} {v['cv']:.1%}" for t, v in probes.tasks.items()))
    (args.out / "COMPARE.md").write_text("\n".join(md) + "\n")
    print(f"[cmp] done -> {args.out/'COMPARE.md'}", flush=True)


if __name__ == "__main__":
    main()
