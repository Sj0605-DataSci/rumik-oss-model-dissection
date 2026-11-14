#!/usr/bin/env python3
"""Steps 3+4: (a) splice eval - replace layer-L residual with SAE reconstruction and
measure next-audio-token CE; (b) feature labelling against clip metadata.

(a) loss recovered = (CE_mean_ablate - CE_sae) / (CE_mean_ablate - CE_clean)
(b) per feature: audio/text firing split, and per-label effect size (Cohen's d on
    clip-mean activation, one-vs-rest) for speaker / tone / accent / pace / tag.
Outputs into the SAE dir: splice.json, features.json, FEATURES.md
"""
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sae import BatchTopKSAE, load_acts  # noqa: E402


def load_sae(sae_dir):
    ck = torch.load(sae_dir / "sae.pt")
    a = ck["args"]
    sae = BatchTopKSAE(2048, 2048 * a["expansion"], a["k"])
    sae.load_state_dict(ck["state_dict"])
    return sae.cuda().eval(), ck["mean"].cuda(), ck["scale"], a


@torch.no_grad()
def splice_eval(sae, mean, scale, layer, acts_dir, n_clips, batch):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(ROOT / "rumik-oss-1", trust_remote_code=True,
                                                 dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
    gens = [json.loads(l) for l in (acts_dir / "gen_ids.jsonl").open()][:n_clips]
    mode = {"m": "clean"}

    def hook(_m, _i, out):
        if mode["m"] == "clean":
            return None
        x = out.float()
        if mode["m"] == "mean":
            xh = mean.expand_as(x)
        else:
            z, _ = sae.encode((x - mean) * scale, train=False)
            xh = sae.decode(z) / scale + mean
        return xh.to(out.dtype)

    h = model.model.layers[layer].register_forward_hook(hook)
    sums = defaultdict(float); cnt = 0
    for b0 in tqdm(range(0, len(gens), batch), desc="splice", unit="batch"):
        gb = gens[b0:b0 + batch]
        seqs = [torch.tensor(g["prompt_ids"] + g["audio_ids"]) for g in gb]
        T = max(map(len, seqs))
        ids = torch.zeros(len(seqs), T, dtype=torch.long); att = torch.zeros_like(ids)
        tgt_mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, (g, s) in enumerate(zip(gb, seqs)):
            ids[i, :len(s)] = s; att[i, :len(s)] = 1
            tgt_mask[i, len(g["prompt_ids"]) - 1:len(s) - 1] = True  # positions predicting audio tokens
        ids, att, tgt_mask = ids.cuda(), att.cuda(), tgt_mask.cuda()
        targets = ids[:, 1:][tgt_mask[:, :-1]]
        for m in ("clean", "sae", "mean"):
            mode["m"] = m
            hs = model.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
            sel = hs[tgt_mask]
            ce = 0.0
            for c in range(0, sel.shape[0], 2048):  # chunk lm_head: never materialise full logits
                lg = model.lm_head(sel[c:c + 2048]).float() * model.config.logit_scale
                ce += torch.nn.functional.cross_entropy(lg, targets[c:c + 2048], reduction="sum").item()
            sums[m] += ce
            del hs, sel
        cnt += targets.numel()
        torch.cuda.empty_cache()
    h.remove()
    del model; torch.cuda.empty_cache()
    ce = {m: sums[m] / cnt for m in sums}
    ce["loss_recovered"] = (ce["mean"] - ce["sae"]) / (ce["mean"] - ce["clean"])
    ce["delta_ce"] = ce["sae"] - ce["clean"]
    ce["n_targets"] = cnt
    return ce


@torch.no_grad()
def label_features(sae, mean, scale, layer, acts_dir, top):
    x, meta = load_acts(acts_dir, layer, audio_only=False)
    clips = {json.loads(l)["clip_id"]: json.loads(l) for l in (acts_dir / "clips.jsonl").open()}
    ids = sorted(clips); cidx = {c: i for i, c in enumerate(ids)}
    C, F = len(ids), sae.n
    fire_audio = torch.zeros(F, device="cuda"); fire_text = torch.zeros(F, device="cuda")
    clip_sum = torch.zeros(C, F, device="cuda"); clip_n = torch.zeros(C, device="cuda")
    is_audio = meta["is_audio"].cuda(); clip_of = torch.tensor([cidx[int(c)] for c in meta["clip_id"]], device="cuda")
    for b in tqdm(range(0, x.shape[0], 8192), desc="encode", unit="chunk"):
        xb = (x[b:b + 8192].cuda().float() - mean) * scale
        z, _ = sae.encode(xb, train=False)
        a = is_audio[b:b + 8192]; co = clip_of[b:b + 8192]
        f = (z > 0).float()
        fire_audio += f[a].sum(0); fire_text += f[~a].sum(0)
        za = z[a]  # clip-mean over AUDIO positions (what the voice is doing)
        clip_sum.index_add_(0, co[a], za); clip_n.index_add_(0, co[a], torch.ones(int(a.sum()), device="cuda"))
    clip_mean = clip_sum / clip_n[:, None].clamp_min(1)
    n_a, n_t = int(is_audio.sum()), int((~is_audio).sum())
    rate_a, rate_t = fire_audio / n_a, fire_text / n_t
    modality = torch.where(rate_a > 3 * rate_t, 1, torch.where(rate_t > 3 * rate_a, -1, 0))  # 1 audio, -1 text, 0 both

    labels = {}
    rows = [clips[c] for c in ids]
    for field in ("speaker", "tone", "accent", "lang", "pace"):
        for v in sorted({r.get(field) for r in rows} - {None}):
            labels[f"{field}={v}"] = torch.tensor([r.get(field) == v for r in rows], device="cuda")
    for t in ("<laugh>", "<sigh>", "<chuckle>"):
        labels[f"tag={t}"] = torch.tensor([r["tag"] == t for r in rows], device="cuda")

    alive = (fire_audio + fire_text) > 0
    out = {"layer": layer, "n_tokens": x.shape[0], "modality_counts": {
        "audio": int(((modality == 1) & alive).sum()), "text": int(((modality == -1) & alive).sum()),
        "both": int(((modality == 0) & alive).sum()), "dead": int((~alive).sum())}, "labels": {}}
    for name, m in labels.items():
        if m.sum() < 5 or (~m).sum() < 5:
            continue
        a, b = clip_mean[m], clip_mean[~m]
        pooled = torch.sqrt(0.5 * (a.var(0) + b.var(0))).clamp_min(1e-6)
        d = (a.mean(0) - b.mean(0)) / pooled
        d[~alive] = 0
        # require the feature to actually fire on a decent share of positive clips
        d[(a > 0).float().mean(0) < 0.3] = 0
        v, i = d.topk(top)
        out["labels"][name] = [{"feature": int(f), "cohens_d": round(float(dd), 3),
                                "mean_pos": round(float(a[:, f].mean()), 4), "mean_neg": round(float(b[:, f].mean()), 4),
                                "modality": ["text", "both", "audio"][int(modality[f]) + 1],
                                "top_clips": [ids[j] for j in clip_mean[:, f].topk(3).indices.tolist()]}
                               for f, dd in zip(i.tolist(), v.tolist())]
    return out


def write_md(sae_dir, splice, feats):
    L = [f"# SAE layer {feats['layer']} - features\n"]
    if splice:
        L.append("## Splice eval (next audio-token CE)\n")
        L.append(f"| clean | SAE spliced | mean-ablated | ΔCE | loss recovered |\n|---|---|---|---|---|\n"
                 f"| {splice['clean']:.4f} | {splice['sae']:.4f} | {splice['mean']:.4f} | {splice['delta_ce']:+.4f} | **{splice['loss_recovered']:.1%}** |\n")
    mc = feats["modality_counts"]
    L.append(f"## Modality (fires ≥3× more on one type)\naudio-only {mc['audio']} · text-only {mc['text']} · both {mc['both']} · dead {mc['dead']}\n")
    L.append("## Top features per label (Cohen's d, clip-mean over audio positions)\n")
    for name, fs in feats["labels"].items():
        L.append(f"**{name}**: " + ", ".join(f"f{f['feature']} (d={f['cohens_d']}, {f['modality']})" for f in fs[:5]) + "\n")
    (sae_dir / "FEATURES.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", type=Path, default=ROOT / "interp/sae_L19_x16_k64")
    ap.add_argument("--acts", type=Path, default=ROOT / "interp/acts")
    ap.add_argument("--splice_clips", type=int, default=80)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.35)
    sae, mean, scale, a = load_sae(args.sae)
    layer = a["layer"]

    print(f"[analyze] splice eval on {args.splice_clips} clips, layer {layer}", flush=True)
    splice = splice_eval(sae, mean, scale, layer, args.acts, args.splice_clips, args.batch)
    print("[analyze] splice:", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in splice.items()}), flush=True)
    (args.sae / "splice.json").write_text(json.dumps(splice, indent=2))

    print("[analyze] labelling features", flush=True)
    feats = label_features(sae, mean, scale, layer, args.acts, args.top)
    (args.sae / "features.json").write_text(json.dumps(feats, indent=2, ensure_ascii=False))
    write_md(args.sae, splice, feats)
    print(f"[analyze] modality: {feats['modality_counts']}", flush=True)
    for name, fs in feats["labels"].items():
        print(f"[analyze] {name:22s} " + "  ".join(f"f{f['feature']}(d={f['cohens_d']:.2f},{f['modality'][0]})" for f in fs[:4]), flush=True)
    print(f"[analyze] done -> {args.sae/'FEATURES.md'}", flush=True)


if __name__ == "__main__":
    main()
