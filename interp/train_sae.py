#!/usr/bin/env python3
"""Step 2: train a BatchTopK SAE (Bussmann et al. 2024, as used in arXiv:2606.10029)
on residual activations captured by capture_acts.py.

BatchTopK: keep the top (k * batch) pre-activations across the whole batch (variable
per-token sparsity, avg k). At inference use a learned threshold (EMA of the min kept
activation) -> JumpReLU-style per-token encoding.
Aux loss (AuxK) revives dead latents by reconstructing the residual with top dead ones.
"""
import argparse, json, math, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent


class BatchTopKSAE(nn.Module):
    def __init__(self, d, n, k):
        super().__init__()
        self.d, self.n, self.k = d, n, k
        W = torch.randn(n, d) / math.sqrt(d)
        W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W.clone())          # [n, d], unit-norm rows
        self.W_enc = nn.Parameter(W.t().clone())      # [d, n]
        self.b_enc = nn.Parameter(torch.zeros(n))
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.register_buffer("threshold", torch.tensor(-1.0))

    def pre(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def encode(self, x, train=True):
        a = self.pre(x)
        if train:
            flat = a.flatten()
            kk = min(self.k * x.shape[0], flat.numel())
            vals, idx = flat.topk(kk, sorted=False)
            z = torch.zeros_like(flat).scatter_(0, idx, vals).view_as(a)
            return z, a
        return a * (a > self.threshold), a

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    @torch.no_grad()
    def renorm(self):
        # keep decoder rows unit norm; remove gradient parallel to rows
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def project_grad(self):
        if self.W_dec.grad is not None:
            W = self.W_dec.data
            self.W_dec.grad -= (self.W_dec.grad * W).sum(1, keepdim=True) * W


def load_acts(acts_dir, layer, audio_only):
    xs, metas = [], []
    for f in sorted(acts_dir.glob(f"acts_L{layer}_shard*.pt"), key=lambda p: int(p.stem.split("shard")[1])):
        k = f.stem.split("shard")[1]
        x = torch.load(f)
        m = torch.load(acts_dir / f"meta_shard{k}.pt")
        xs.append(x); metas.append(m)
    x = torch.cat(xs)
    meta = {k: torch.cat([m[k] for m in metas]) for k in metas[0]}
    # drop BOS / first 2 positions (huge-norm attention-sink outliers)
    keep = meta["pos"] >= 2
    if audio_only:
        keep &= meta["is_audio"]
    return x[keep], {k: v[keep] for k, v in meta.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", type=Path, default=ROOT / "interp/acts")
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--expansion", type=int, default=16)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--aux_k", type=int, default=512)
    ap.add_argument("--aux_coef", type=float, default=1 / 32)
    ap.add_argument("--dead_steps", type=int, default=500)
    ap.add_argument("--audio_only", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / f"interp/sae_L{args.layer}_x{args.expansion}_k{args.k}"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    x, meta = load_acts(args.acts, args.layer, args.audio_only)
    N, d = x.shape
    print(f"[sae] layer {args.layer}: {N} tokens x {d} (audio frac {meta['is_audio'].float().mean():.2f}) "
          f"loaded in {time.time()-t0:.1f}s", flush=True)

    # normalise so E||x||^2 = d  (scale factor stored for steering later)
    x = x.cuda()
    # chunked stats: never materialise an fp32 copy of all activations (unified memory)
    mean = sum(x[i:i + 65536].float().sum(0) for i in range(0, N, 65536)) / N
    scale = math.sqrt(d) / (sum((x[i:i + 65536].float() - mean).norm(dim=1).sum().item()
                                for i in range(0, N, 65536)) / N)
    perm = torch.randperm(N, device="cuda")
    n_val = min(20_000, N // 20)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def get(idx):
        return (x[idx].float() - mean) * scale

    n_lat = d * args.expansion
    sae = BatchTopKSAE(d, n_lat, args.k).cuda()
    with torch.no_grad():  # init b_dec at geometric-median-ish (mean of normalised data = 0)
        sae.b_dec.zero_()
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr, betas=(0.9, 0.999))
    steps_per_epoch = len(tr_idx) // args.batch
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 500) * (1.0 if s < 0.8 * total else max(0.0, (total - s) / (0.2 * total))))
    since_fired = torch.zeros(n_lat, device="cuda", dtype=torch.long)
    log = (out / "train_log.jsonl").open("w")

    print(f"[sae] {n_lat} latents, k={args.k}, {total} steps ({args.epochs} epochs x {steps_per_epoch})", flush=True)
    pbar = tqdm(total=total, desc=f"SAE L{args.layer}", unit="step", dynamic_ncols=True)
    step = 0
    for ep in range(args.epochs):
        order = tr_idx[torch.randperm(len(tr_idx), device="cuda")]
        for b in range(steps_per_epoch):
            xb = get(order[b * args.batch:(b + 1) * args.batch])
            z, a = sae.encode(xb, train=True)
            xh = sae.decode(z)
            err = xb - xh
            mse = err.pow(2).sum(-1).mean()

            fired = (z > 0).any(0)
            since_fired = torch.where(fired, 0, since_fired + 1)
            dead = since_fired > args.dead_steps
            aux = torch.tensor(0.0, device="cuda")
            if dead.any():
                ad = a * dead
                kk = min(args.aux_k, int(dead.sum()))
                v, i = ad.topk(kk, dim=-1)
                za = torch.zeros_like(ad).scatter_(-1, i, v)
                aux = (err.detach() - za @ sae.W_dec).pow(2).sum(-1).mean()
            loss = mse + args.aux_coef * aux

            opt.zero_grad(set_to_none=True)
            loss.backward()
            sae.project_grad()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            opt.step(); sched.step(); sae.renorm()

            with torch.no_grad():  # threshold for inference = EMA of min kept activation
                pos = z[z > 0]
                if pos.numel():
                    m = pos.min()
                    sae.threshold.copy_(m if sae.threshold < 0 else 0.99 * sae.threshold + 0.01 * m)

            step += 1
            if step % 50 == 0 or step == total:
                fvu = (err.pow(2).sum() / (xb - xb.mean(0)).pow(2).sum()).item()
                rec = dict(step=step, mse=mse.item(), fvu=fvu, aux=aux.item(), l0=(z > 0).sum(-1).float().mean().item(),
                           dead=int(dead.sum()), lr=sched.get_last_lr()[0])
                log.write(json.dumps(rec) + "\n"); log.flush()
                pbar.set_postfix(fvu=f"{fvu:.3f}", l0=f"{rec['l0']:.0f}", dead=rec["dead"], ep=ep)
            pbar.update(1)
    pbar.close()

    # validation with inference-time (threshold) encoding
    with torch.no_grad():
        num = den = l0 = 0.0
        alive = torch.zeros(n_lat, dtype=torch.bool, device="cuda")
        for b in range(0, n_val, args.batch):
            xb = get(val_idx[b:b + args.batch])
            z, _ = sae.encode(xb, train=False)
            e = xb - sae.decode(z)
            num += e.pow(2).sum().item(); den += (xb - xb.mean(0)).pow(2).sum().item()
            l0 += (z > 0).sum().item(); alive |= (z > 0).any(0)
        res = dict(layer=args.layer, n_latents=n_lat, k=args.k, val_fvu=num / den, val_l0=l0 / n_val,
                   alive_frac=alive.float().mean().item(), threshold=sae.threshold.item(),
                   n_tokens=N, minutes=(time.time() - t0) / 60)
    print("[sae] val:", json.dumps(res), flush=True)
    torch.save({"state_dict": sae.state_dict(), "mean": mean.cpu(), "scale": scale, "args": vars(args) | {"out": str(out), "acts": str(args.acts)}},
               out / "sae.pt")
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print(f"[sae] saved {out/'sae.pt'}", flush=True)


if __name__ == "__main__":
    main()
