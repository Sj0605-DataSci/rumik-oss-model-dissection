#!/usr/bin/env python3
"""Q1 (necessity + locus), Q2 (speaker-binding heads), Q3 (Silver) via WEIGHT edits.

Teacher-forced speaker-preference score (no generation):
  for natural clip c of speaker s, with audio tokens a:
    gap(c) = mean_t [ log p(a_t | "<s>: ...") - log p(a_t | "<s'>: ...") ]   (s' = other speakers)
  log p restricted to the audio sub-vocabulary (what the sampler can emit).
  "binding" = mean gap; an edit that destroys speaker routing drives it to 0.

Weight edits (in place, restored after each test):
  dir-ablate(u, layers): W_o <- (I-uu^T) W_o ; W_down <- (I-uu^T) W_down   (o_proj, down_proj outputs)
  head-ablate(l,h)     : o_proj.weight[:, h*128:(h+1)*128] = 0

Stages:
  tf_dir   : Q1 locus - project each speaker direction out of ONE layer at a time (36 layers) + all layers
  tf_heads : Q2 - ablate each of 576 heads, rank by binding drop
  gen      : generation + WavLM speaker probe confusion for: base, Zoya-dir ablated (all layers),
             random-dir control, top-k heads ablated, and Q3 "Silver:" prompts
"""
import argparse, json, sys
from collections import defaultdict
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
from prompts import build_prompts  # noqa: E402

SPK = ["Ira", "Aisha", "Siya", "Zoya"]
ACTS = ROOT / "interp/acts_v2"
OUT = ROOT / "interp/q123"


class Edits:
    def __init__(self, model):
        self.m = model; self.saved = {}

    def _save(self, name, p):
        if name not in self.saved:
            self.saved[name] = p.data.clone()

    def dir_ablate(self, u, layers):
        u = (u / u.norm()).to(torch.bfloat16).cuda()
        P = torch.eye(u.numel(), device="cuda", dtype=torch.bfloat16) - torch.outer(u, u)
        for l in layers:
            L = self.m.model.layers[l]
            for nm, lin in (("o", L.self_attn.o_proj), ("d", L.mlp.down_proj)):
                self._save(f"{l}.{nm}", lin.weight); lin.weight.data = P @ lin.weight.data

    def head_ablate(self, pairs, hd=128):
        for l, h in pairs:
            lin = self.m.model.layers[l].self_attn.o_proj
            self._save(f"{l}.o", lin.weight); lin.weight.data[:, h * hd:(h + 1) * hd] = 0

    def restore(self):
        for k, w in self.saved.items():
            l, nm = k.split(".")
            L = self.m.model.layers[int(l)]
            (L.self_attn.o_proj if nm == "o" else L.mlp.down_proj).weight.data = w
        self.saved = {}


def build_tf_set(tok, n_per_spk, seed=0):
    """natural v2 clips: (speaker, desc+text input, audio ids)."""
    rows = {r["clip_id"]: r for r in build_prompts(2500, 1, "v2")}
    gens = [json.loads(l) for l in (ACTS / "gen_ids.jsonl").open()]
    rng = np.random.default_rng(seed); out = []
    for s in SPK:
        g = [x for x in gens if rows[x["clip_id"]]["speaker"] == s and len(x["audio_ids"]) < 700]
        for x in rng.choice(len(g), n_per_spk, replace=False):
            out.append((s, rows[g[x]["clip_id"]]["input"], g[x]["audio_ids"]))
    return out


@torch.no_grad()
def binding(model, tok, tf, audio_W, audio_ids_t, batch=8):
    """mean over clips of [logp(a|true spk) - mean_{s'} logp(a|s')], per audio token."""
    seqs, owner = [], []
    for ci, (s, inp, a) in enumerate(tf):
        for s2 in SPK:
            p = tok(f"<text>{s2}: {inp}<audio>").input_ids
            seqs.append((p, a, s2 == s)); owner.append(ci)
    lp = []
    for b0 in range(0, len(seqs), batch):
        chunk = seqs[b0:b0 + batch]
        T = max(len(p) + len(a) for p, a, _ in chunk)
        ids = torch.zeros(len(chunk), T, dtype=torch.long); att = torch.zeros_like(ids)
        for i, (p, a, _) in enumerate(chunk):
            s = torch.tensor(p + a); ids[i, :len(s)] = s; att[i, :len(s)] = 1
        hs = model.model(input_ids=ids.cuda(), attention_mask=att.cuda(), use_cache=False).last_hidden_state
        for i, (p, a, _) in enumerate(chunk):
            h = hs[i, len(p) - 1:len(p) + len(a) - 1]                       # positions predicting a
            lg = (h @ audio_W.T).float() * model.config.logit_scale           # restricted vocab
            tgt = torch.searchsorted(audio_ids_t, torch.tensor(a, device="cuda"))
            lp.append((lg.log_softmax(-1).gather(1, tgt[:, None]).mean()).item())
        del hs
    lp = np.array(lp).reshape(len(tf), len(SPK))
    true = np.array([SPK.index(s) for s, _, _ in tf])
    gaps = [lp[i, true[i]] - np.mean(np.delete(lp[i], true[i])) for i in range(len(tf))]
    per_spk = {s: float(np.mean([g for g, (ss, _, _) in zip(gaps, tf) if ss == s])) for s in SPK}
    return float(np.mean(gaps)), per_spk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["tf_dir", "tf_heads", "gen"])
    ap.add_argument("--n_tf", type=int, default=4, help="clips per speaker for teacher-forced score")
    ap.add_argument("--top_heads", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    torch.set_grad_enabled(False); torch.cuda.set_per_process_memory_fraction(0.35)
    OUT.mkdir(exist_ok=True)
    res = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else {}
    save = lambda: (OUT / "results.json").write_text(json.dumps(res, indent=2))

    repo = ROOT / "rumik-oss-1"
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").eval().cuda()
    vocab = tok.get_vocab(); end_id = int(vocab["</audio>"])
    mapping = {int(t): (int(m.group(1)), int(m.group(2))) for s, t in vocab.items() if (m := AUDIO_RE.match(s))}
    audio_ids_t = torch.tensor(sorted(mapping), device="cuda")
    audio_W = model.lm_head.weight[audio_ids_t]
    E = Edits(model)
    names = json.load(open(ROOT / "interp/q12/direction_names.json"))
    V = torch.tensor(np.load(ROOT / "interp/q12/directions_L19.npy"), dtype=torch.float32)
    D = {n: V[i] for i, n in enumerate(names)}
    g = torch.Generator().manual_seed(0); rnd = torch.randn(2048, generator=g); rnd /= rnd.norm()
    D["random"] = rnd
    tf = build_tf_set(tok, args.n_tf)
    nL, nH = model.config.num_hidden_layers, model.config.num_attention_heads

    base, base_spk = binding(model, tok, tf, audio_W, audio_ids_t)
    res["base_binding"] = base; res["base_binding_per_spk"] = base_spk
    print(f"[q123] base binding gap = {base:.4f} nats/token  per speaker {json.dumps({k: round(v, 3) for k, v in base_spk.items()})}", flush=True)

    if "tf_dir" in args.stages:  # ---- Q1 locus ----
        res["tf_dir"] = {}
        for dn in ["speaker=Zoya", "speaker=Aisha", "speaker=Siya", "speaker=Ira", "random"]:
            E.dir_ablate(D[dn], range(nL)); allg, alls = binding(model, tok, tf, audio_W, audio_ids_t); E.restore()
            per = []
            for l in tqdm(range(nL), desc=f"locus {dn}", unit="layer", dynamic_ncols=True):
                E.dir_ablate(D[dn], [l]); gl, sl = binding(model, tok, tf, audio_W, audio_ids_t); E.restore()
                per.append({"layer": l, "gap": gl, "per_spk": sl})
            res["tf_dir"][dn] = {"all_layers": allg, "all_layers_per_spk": alls, "per_layer": per}
            top = sorted(per, key=lambda r: r["gap"])[:3]
            print(f"[q123] ablate {dn:14s} all layers -> gap {allg:.4f} (per-spk {json.dumps({k: round(v, 3) for k, v in alls.items()})}); "
                  f"most damaging single layers: " + ", ".join(f"L{r['layer']}:{r['gap']:.3f}" for r in top), flush=True)
            save()

    if "tf_heads" in args.stages:  # ---- Q2 heads ----
        hs = []
        for l in tqdm(range(nL), desc="heads", unit="layer", dynamic_ncols=True):
            for h in range(nH):
                E.head_ablate([(l, h)]); gh, sh = binding(model, tok, tf, audio_W, audio_ids_t); E.restore()
                hs.append({"layer": l, "head": h, "gap": gh, "drop": base - gh, "per_spk": sh})
        hs.sort(key=lambda r: -r["drop"])
        res["tf_heads"] = hs
        print("[q123] top heads by binding drop: " + ", ".join(f"L{r['layer']}h{r['head']}:{r['drop']:+.3f}" for r in hs[:12]), flush=True)
        drops = np.array([r["drop"] for r in hs])
        print(f"[q123] head drop distribution: median {np.median(drops):+.4f}, 95th pct {np.percentile(drops,95):+.4f}, max {drops.max():+.4f}", flush=True)
        # joint ablation of top-k
        for k in (1, 2, 4, 8, 16):
            E.head_ablate([(r["layer"], r["head"]) for r in hs[:k]]); gk, sk = binding(model, tok, tf, audio_W, audio_ids_t); E.restore()
            res[f"top{k}_heads_gap"] = gk
            print(f"[q123] ablate top-{k:2d} heads jointly -> gap {gk:.4f} ({gk/base:.0%} of base)", flush=True)
        # random-head control (same count)
        rng = np.random.default_rng(1); rp = [(int(l), int(h)) for l, h in zip(rng.integers(0, nL, 8), rng.integers(0, nH, 8))]
        E.head_ablate(rp); gr, _ = binding(model, tok, tf, audio_W, audio_ids_t); E.restore()
        res["random8_heads_gap"] = gr
        print(f"[q123] ablate 8 RANDOM heads -> gap {gr:.4f} ({gr/base:.0%} of base)", flush=True)
        save()

    if "gen" in args.stages:  # ---- generation confirmation + Q3 Silver ----
        mimi = MimiModel.from_pretrained(repo / "codec", dtype=torch.bfloat16).eval().cuda()
        sr = int(AutoFeatureExtractor.from_pretrained(repo / "codec").sampling_rate)
        allowed = torch.tensor(sorted([*mapping, end_id]), device="cuda")
        probes = Probes(ACTS, "cuda")
        desc = "professional, Indian English, steady pace"
        heads = res.get("tf_heads", [])[:args.top_heads]
        conds = {"base": lambda: None,
                 "ablate_Zoya_dir": lambda: E.dir_ablate(D["speaker=Zoya"], range(nL)),
                 "ablate_Aisha_dir": lambda: E.dir_ablate(D["speaker=Aisha"], range(nL)),
                 "ablate_random_dir": lambda: E.dir_ablate(D["random"], range(nL)),
                 f"ablate_top{len(heads)}_heads": lambda: E.head_ablate([(r["layer"], r["head"]) for r in heads])}
        prompt_spk = SPK + ["Silver"]
        res["gen"] = {}
        zeros = torch.zeros(args.seeds, 2048)
        for cname, apply in conds.items():
            apply(); res["gen"][cname] = {}
            for ps in tqdm(prompt_spk, desc=f"gen {cname}", unit="spk", dynamic_ncols=True):
                recs = []
                for ti in (0, 1):
                    ids = tok(f'<text>{ps}: <description="{desc}"> {TEXTS[ti]}<audio>', return_tensors="pt").input_ids.cuda()
                    toks, stopped = generate_rows(model, 19, ids, allowed, end_id, zeros, 1000, seed=11 + ti)
                    for si, (tk, stp) in enumerate(zip(toks, stopped)):
                        vals = to_codes(tk, mapping)
                        if not vals:
                            continue
                        au = mimi.decode(torch.tensor(vals, device="cuda").reshape(1, -1, 8).transpose(1, 2)).audio_values[0, 0].float().cpu().clamp(-1, 1).numpy()
                        od = OUT / "wavs" / cname; od.mkdir(parents=True, exist_ok=True)
                        save_wav(od / f"{ps}_t{ti}_s{si}.wav", torch.from_numpy(au), sr)
                        fe = probes.embed(au)
                        rms = float(np.sqrt((au ** 2).mean())); _, _, vf = f0_stats(au, sr)
                        if fe is not None:
                            recs.append({"p": {c: probes.prob(fe, "speaker", c) for c in SPK}, "intact": bool(stp and rms > 0.03 and vf > 0.2), "dur": len(au) / sr})
                P = {c: float(np.mean([r["p"][c] for r in recs])) for c in SPK} if recs else {}
                arg = defaultdict(int)
                for r in recs:
                    arg[max(r["p"], key=r["p"].get)] += 1
                res["gen"][cname][ps] = {"mean_p": P, "argmax": dict(arg), "intact": float(np.mean([r["intact"] for r in recs])) if recs else 0, "n": len(recs)}
                tqdm.write(f"[q123] {cname:22s} prompt={ps:6s} -> " + " ".join(f"{c}:{P.get(c, 0):.2f}" for c in SPK)
                           + f"  argmax={dict(arg)} intact={res['gen'][cname][ps]['intact']:.0%}")
            E.restore(); save()
    print(f"[q123] done -> {OUT/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
