#!/usr/bin/env python3
"""Step 5: steer rumik-oss-1 with SAE decoder directions at layer L.

For each experiment we pick the top feature for a label (from features.json), then
generate the SAME prompt at several steering strengths: x_L += alpha * R * d_f on
every decode step (audio positions), R = median residual norm at L, d_f = unit decoder
direction. Rows of one batch = alphas x seeds (identical prompt -> no padding).

Metrics per clip (scipy/numpy only): duration, mean F0 + F0 std (autocorrelation pitch),
RMS energy. Writes interp/steer_L{L}/{exp}/*.wav + results.json + STEER.md
"""
import argparse, json, re, sys, wave
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer, MimiModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_sae import load_sae  # noqa: E402
from capture_acts import to_codes, save_wav, AUDIO_RE  # noqa: E402

# (experiment name, label to pull feature from, base prompt speaker, description)
EXPERIMENTS = [
    ("pace_fast", "pace=fast", "Ira", "professional, Indian English, steady pace"),
    ("pace_slow", "pace=slow", "Ira", "professional, Indian English, steady pace"),
    ("sad", "tone=sad", "Ira", "professional, Indian English, steady pace"),
    ("happy", "tone=happy", "Ira", "professional, Indian English, steady pace"),
    ("laugh", "tag=<laugh>", "Ira", "professional, Indian English, steady pace"),
    ("to_zoya", "speaker=Zoya", "Ira", "professional, Indian English, steady pace"),
    ("excited", "tone=excited", "Ira", "professional, Indian English, steady pace"),
    ("to_aisha", "speaker=Aisha", "Ira", "professional, Indian English, steady pace"),
    ("to_siya", "speaker=Siya", "Ira", "professional, Indian English, steady pace"),
]
TEXTS = ["Please make sure the report is ready by tomorrow morning.",
         "Let me check that for you, it will only take a minute."]


def f0_stats(audio, sr):
    """Frame autocorrelation pitch on voiced frames; returns (mean_f0, std_f0, voiced_frac)."""
    fl, hop = int(0.04 * sr), int(0.01 * sr)
    lo, hi = int(sr / 400), int(sr / 70)
    f0s, n = [], 0
    rms_all = np.sqrt(np.mean(audio ** 2)) + 1e-9
    for s in range(0, len(audio) - fl, hop):
        fr = audio[s:s + fl]; n += 1
        if np.sqrt(np.mean(fr ** 2)) < 0.5 * rms_all:
            continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, "full")[fl - 1:]
        if ac[0] <= 0:
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] / ac[0] > 0.45:
            f0s.append(sr / lag)
    if not f0s:
        return float("nan"), float("nan"), 0.0
    f0s = np.array(f0s)
    return float(np.median(f0s)), float(np.std(f0s)), len(f0s) / max(n, 1)


@torch.no_grad()
def generate_batch(model, ids, allowed, end_id, steer_vec, alphas, max_new, temp=0.8, top_k=30):
    """ids [1,T] prompt; returns list of token lists (audio only). steer_vec [d] raw-space direction."""
    B = len(alphas)
    alpha_t = torch.tensor(alphas, device="cuda", dtype=torch.bfloat16)[:, None, None]
    state = {"on": False}

    def hook(_m, _i, out):
        if state["on"]:
            return out + alpha_t * steer_vec.to(out.dtype)
    h = model.model.layers[LAYER].register_forward_hook(hook)
    try:
        ids = ids.expand(B, -1).contiguous()
        mask = torch.ones_like(ids)
        state["on"] = STEER_PROMPT
        out = model(input_ids=ids, attention_mask=mask, use_cache=True, output_hidden_states=True, return_dict=True)
        state["on"] = True
        cache, scores, hidden = out.past_key_values, out.logits[:, -1, :], out.hidden_states[-1][:, -1:, :]
        done = torch.zeros(B, dtype=torch.bool, device="cuda")
        toks = [[] for _ in range(B)]
        neg = torch.finfo(scores.dtype).min
        for step in range(max_new):
            if step >= 8:
                stop = torch.sigmoid(model.stop_predictor(hidden).squeeze(-1)).view(-1) > 0.5
                done |= stop
            if bool(done.all()):
                break
            r = torch.full_like(scores, neg)
            r.index_copy_(1, allowed, scores.index_select(1, allowed))
            if step < 8:
                r[:, end_id] = neg
            r = r / temp
            cut = torch.topk(r, top_k, dim=-1).values[:, -1:]
            r = r.masked_fill(r < cut, neg)
            tok = torch.multinomial(torch.softmax(r.float(), -1), 1)
            done |= tok.view(-1) == end_id
            for b in range(B):
                if not done[b]:
                    toks[b].append(int(tok[b]))
            mask = torch.cat((mask, torch.ones_like(tok)), 1)
            out = model(input_ids=tok, attention_mask=mask, past_key_values=cache, use_cache=True,
                        output_hidden_states=True, return_dict=True)
            cache, scores, hidden = out.past_key_values, out.logits[:, -1, :], out.hidden_states[-1][:, -1:, :]
    finally:
        h.remove()
    return toks


def main():
    global LAYER, STEER_PROMPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", type=Path, default=ROOT / "interp/sae_L19_x16_k64")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max_new", type=int, default=1200)
    ap.add_argument("--steer_prompt", action="store_true", help="also steer prompt positions")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--acts", type=Path, default=ROOT / "interp/acts")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    STEER_PROMPT = args.steer_prompt
    torch.set_grad_enabled(False)
    torch.cuda.set_per_process_memory_fraction(0.35)

    sae, mean, scale, a = load_sae(args.sae)
    LAYER = a["layer"]
    feats = json.loads((args.sae / "features.json").read_text())
    acts0 = torch.load(args.acts / f"acts_L{LAYER}_shard1.pt")[:20000].float()
    R = float(acts0.norm(dim=-1).median()); del acts0
    out_root = args.out or ROOT / f"interp/steer_L{LAYER}"
    out_root.mkdir(exist_ok=True)

    repo = ROOT / "rumik-oss-1"
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").eval().cuda()
    mimi = MimiModel.from_pretrained(repo / "codec", dtype=torch.bfloat16).eval().cuda()
    sr = int(AutoFeatureExtractor.from_pretrained(repo / "codec").sampling_rate)
    vocab = tok.get_vocab(); end_id = int(vocab["</audio>"])
    mapping = {int(t): (int(m.group(1)), int(m.group(2))) for s, t in vocab.items() if (m := AUDIO_RE.match(s))}
    allowed = torch.tensor(sorted([*mapping, end_id]), device="cuda")
    print(f"[steer] layer {LAYER}, R={R:.1f}, alphas={args.alphas} x {args.seeds} seeds", flush=True)

    exps = [e for e in EXPERIMENTS if not args.only or e[0] in args.only]
    results = []
    rows_alpha = [al for al in args.alphas for _ in range(args.seeds)]
    pbar = tqdm(total=len(exps) * len(TEXTS), desc="steer", unit="batch", dynamic_ncols=True)
    for name, label, spk, desc in exps:
        cand = feats["labels"].get(label)
        if not cand:
            tqdm.write(f"[steer] no feature for {label}, skip"); pbar.update(len(TEXTS)); continue
        f = cand[0]["feature"]
        d = sae.W_dec[f].detach().float()
        vec = (d / d.norm() * R).cuda()
        (out_root / name).mkdir(exist_ok=True)
        for ti, text in enumerate(TEXTS):
            prompt = f'<text>{spk}: <description="{desc}"> {text}<audio>'
            ids = tok(prompt, return_tensors="pt").input_ids.cuda()
            torch.manual_seed(1234 + ti)
            toks = generate_batch(model, ids, allowed, end_id, vec, rows_alpha, args.max_new)
            for r, (al, tk) in enumerate(zip(rows_alpha, toks)):
                vals = to_codes(tk, mapping)
                rec = dict(exp=name, label=label, feature=f, alpha=al, seed=r % args.seeds, text_idx=ti, n_tokens=len(tk))
                if vals:
                    codes = torch.tensor(vals, device="cuda").reshape(1, -1, 8).transpose(1, 2)
                    audio = mimi.decode(codes).audio_values[0, 0].float().cpu().clamp(-1, 1)
                    save_wav(out_root / name / f"t{ti}_a{al:+.2f}_s{r % args.seeds}.wav", audio, sr)
                    au = audio.numpy()
                    f0, f0sd, vf = f0_stats(au, sr)
                    rec.update(dur_s=len(au) / sr, f0=f0, f0_std=f0sd, voiced=vf, rms=float(np.sqrt((au ** 2).mean())))
                results.append(rec)
            pbar.update(1)
            torch.cuda.empty_cache()
        # running summary
        rs = [x for x in results if x["exp"] == name and "dur_s" in x]
        for al in args.alphas:
            g = [x for x in rs if x["alpha"] == al]
            if g:
                tqdm.write(f"[steer] {name:10s} f{f} α={al:+.2f}  dur {np.mean([x['dur_s'] for x in g]):5.2f}s  "
                           f"F0 {np.nanmean([x['f0'] for x in g]):6.1f}Hz  F0sd {np.nanmean([x['f0_std'] for x in g]):5.1f}  "
                           f"rms {np.mean([x['rms'] for x in g]):.3f}")
    pbar.close()
    (out_root / "results.json").write_text(json.dumps(results, indent=2))

    md = [f"# Steering at layer {LAYER} (R={R:.1f}; vector = α·R·unit decoder dir)\n",
          "| exp | feature | α | dur s | F0 Hz | F0 sd | rms | n |", "|---|---|---|---|---|---|---|---|"]
    for name, *_ in exps:
        for al in args.alphas:
            g = [x for x in results if x["exp"] == name and x["alpha"] == al and "dur_s" in x]
            if g:
                md.append(f"| {name} | f{g[0]['feature']} | {al:+.2f} | {np.mean([x['dur_s'] for x in g]):.2f} | "
                          f"{np.nanmean([x['f0'] for x in g]):.1f} | {np.nanmean([x['f0_std'] for x in g]):.1f} | "
                          f"{np.mean([x['rms'] for x in g]):.3f} | {len(g)} |")
    (out_root / "STEER.md").write_text("\n".join(md) + "\n")
    print(f"[steer] done -> {out_root/'STEER.md'}", flush=True)


LAYER, STEER_PROMPT = 19, False
if __name__ == "__main__":
    main()
