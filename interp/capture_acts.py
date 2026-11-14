#!/usr/bin/env python3
"""Stage B: teacher-forced bf16 forward over vLLM-generated sequences; dump residual
activations from selected layers + Mimi-decode wavs.

Input : <out>/gen_ids.jsonl from vllm_gen.py (stage A)
Output: acts_L{layer}_shard{k}.pt  fp16 [N_tokens, 2048]
        meta_shard{k}.pt           token_ids, clip_id, pos, is_audio
        clips.jsonl, wavs/{clip_id}.wav
"""
import argparse, json, re, sys, time, wave
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer, MimiModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import build_prompts  # noqa: E402

class _Stop(Exception):
    pass


AUDIO_RE = re.compile(r"^<(\d+)_(\d+)>$")


def to_codes(ids, mapping):
    values, frame, expected = [], [], 0
    for tid in ids:
        item = mapping.get(int(tid))
        if item is None:
            frame, expected = [], 0
            continue
        code, q = item
        if q == expected:
            frame.append(code); expected += 1
            if expected == 8:
                values.extend(frame); frame, expected = [], 0
        elif q == 0:
            frame, expected = [code], 1
        else:
            frame, expected = [], 0
    return values


def save_wav(path, audio, sr):
    pcm = (audio.numpy() * 32767).astype("int16")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[11, 19, 27])
    ap.add_argument("--out", type=Path, default=ROOT / "interp/acts")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", default="v1")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--shard_tokens", type=int, default=200_000)
    args = ap.parse_args()

    rows = {r["clip_id"]: r for r in build_prompts(args.n, args.seed, args.grid)}
    gens = [json.loads(l) for l in (args.out / "gen_ids.jsonl").open()]
    gens.sort(key=lambda g: len(g["prompt_ids"]) + len(g["audio_ids"]))  # length-bucket batches
    (args.out / "wavs").mkdir(exist_ok=True)

    repo = ROOT / "rumik-oss-1"
    torch.cuda.set_per_process_memory_fraction(0.35)  # unified memory: never starve the OS
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").eval().cuda()
    mimi = MimiModel.from_pretrained(repo / "codec", dtype=torch.bfloat16).eval().cuda()
    sr = int(AutoFeatureExtractor.from_pretrained(repo / "codec").sampling_rate)
    vocab = tok.get_vocab()
    audio_end = int(vocab["</audio>"])
    mapping = {int(t): (int(m.group(1)), int(m.group(2))) for s, t in vocab.items() if (m := AUDIO_RE.match(s))}
    audio_ids_t = torch.tensor(sorted(mapping))
    # forward hooks: capture residual output of chosen layers, abort after the last one
    last = max(args.layers)
    def mk(L):
        def hook(_m, _inp, out):
            grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
            if L == last:
                raise _Stop
        return hook
    for L in args.layers:
        model.model.layers[L].register_forward_hook(mk(L))
    print(f"[capture] bf16 model loaded in {time.time()-t0:.1f}s; {len(gens)} clips, layers={args.layers}", flush=True)

    buf = {L: [] for L in args.layers}
    meta = {"token_ids": [], "clip_id": [], "pos": [], "is_audio": []}
    state = {"shard": 0, "n": 0, "total": 0}

    def flush():
        if state["n"] == 0:
            return
        k = state["shard"]
        for L in args.layers:
            torch.save(torch.cat(buf[L]), args.out / f"acts_L{L}_shard{k}.pt"); buf[L].clear()
        torch.save({kk: torch.cat(v) for kk, v in meta.items()}, args.out / f"meta_shard{k}.pt")
        for v in meta.values():
            v.clear()
        tqdm.write(f"[capture] wrote shard {k} ({state['n']} tokens)")
        state["shard"] += 1; state["n"] = 0

    clips_f = (args.out / "clips.jsonl").open("w")
    pbar = tqdm(range(0, len(gens), args.batch), desc="forward", unit="batch", dynamic_ncols=True)
    for b0 in pbar:
        batch = gens[b0:b0 + args.batch]
        seqs = [torch.tensor(g["prompt_ids"] + g["audio_ids"] + [audio_end]) for g in batch]
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), tok.pad_token_id or 0)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):  # right-pad; causal => valid positions unaffected
            ids[i, :len(s)] = s; att[i, :len(s)] = 1
        grabbed = {}
        with torch.inference_mode():
            try:  # backbone only (no 277k-vocab logits), stop right after the deepest hooked layer
                model.model(input_ids=ids.cuda(), attention_mask=att.cuda(), use_cache=False)
            except _Stop:
                pass
        for i, (g, s) in enumerate(zip(batch, seqs)):
            n = len(s)
            for L in args.layers:
                buf[L].append(grabbed[L][i, :n].to(torch.float16).cpu())
            meta["token_ids"].append(s)
            meta["clip_id"].append(torch.full((n,), g["clip_id"], dtype=torch.int32))
            meta["pos"].append(torch.arange(n, dtype=torch.int32))
            meta["is_audio"].append(torch.isin(s, audio_ids_t))
            state["n"] += n; state["total"] += n

            vals = to_codes(g["audio_ids"], mapping)
            audio_s = 0.0
            if vals:
                with torch.inference_mode():
                    codes = torch.tensor(vals, device="cuda").reshape(1, -1, 8).transpose(1, 2)
                    audio = mimi.decode(codes).audio_values[0, 0].float().cpu().clamp(-1, 1)
                save_wav(args.out / "wavs" / f"{g['clip_id']:05d}.wav", audio, sr)
                audio_s = audio.shape[-1] / sr
            rec = dict(rows[g["clip_id"]]); rec.update(n_tokens=n, prompt_len=len(g["prompt_ids"]), audio_s=round(audio_s, 2))
            clips_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        del grabbed
        torch.cuda.empty_cache()
        pbar.set_postfix(tokens=state["total"], seq_len=T)
        if state["n"] >= args.shard_tokens:
            flush()
    flush()
    clips_f.close()
    print(f"[capture] done: {state['total']} tokens in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
