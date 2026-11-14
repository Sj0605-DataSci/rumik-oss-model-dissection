#!/usr/bin/env python3
"""Q13 voice lab: persistent server for Claude-as-editor weight-space programming on rumik-oss-1.

Rules (MIT WSP harness style): no gradients; one PASS = one batched generation + scoring;
pass budget enforced from budget.json. Editor submits jobs into q13/queue/<id>.json:
  {"id": "p03", "edit": "q13/edits/p03.py" | null, "speaker": "Ira",
   "prompt_fmt": "<text>{spk}: {text}<audio>",            # optional override
   "texts": [0, 1], "seeds": 3, "max_new": 1000, "note": "why"}
An edit file defines  apply(model, H)  and may change any weights in place (H = helpers).
Weights are snapshotted before and restored after every job.

Per clip metrics: F0 median, formants F1/F2/F3 (LPC, voiced frames), P(male) (gender clf),
WER vs the input text (whisper-large-v3-turbo), speaker-probe max P over the 4 voices,
min cosine distance to the 4 speaker centroids (WavLM layer feats), intact.
"""
import importlib.util, json, re, shutil, sys, time, traceback, wave
from pathlib import Path

import numpy as np
import torch
from scipy.signal import lfilter, resample_poly
from tqdm import tqdm
from transformers import (AutoFeatureExtractor, AutoModelForAudioClassification, AutoModelForCausalLM,
                          AutoTokenizer, MimiModel, pipeline)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_acts import to_codes, save_wav, AUDIO_RE  # noqa: E402
from steer import f0_stats  # noqa: E402
from steer_compare import Probes, generate_rows  # noqa: E402

LAB = ROOT / "interp/q13"
TEXTS = ["Please make sure the report is ready by tomorrow morning.",
         "Let me check that for you, it will only take a minute.",
         "The weather has been so strange this whole month."]
GENDER = "prithivMLmods/Common-Voice-Gender-Detection"


def formants(a24, sr=24000):
    """median F1..F3 over voiced frames via LPC root-finding at 10 kHz."""
    a = resample_poly(a24, 5, 12).astype(np.float64)  # 10 kHz
    fs = 10000; fl = int(0.025 * fs); hop = int(0.01 * fs); order = 12
    a = lfilter([1, -0.63], 1, a)
    rms_all = np.sqrt(np.mean(a ** 2)) + 1e-9
    F = []
    win = np.hamming(fl)
    for s in range(0, len(a) - fl, hop):
        fr = a[s:s + fl] * win
        if np.sqrt(np.mean(fr ** 2)) < 0.6 * rms_all:
            continue
        r = np.correlate(fr, fr, "full")[fl - 1:fl + order]
        if r[0] <= 0:
            continue
        R = np.array([[r[abs(i - j)] for j in range(order)] for i in range(order)]) + 1e-9 * np.eye(order)
        try:
            lpc = np.linalg.solve(R, r[1:order + 1])
        except np.linalg.LinAlgError:
            continue
        roots = np.roots(np.concatenate([[1], -lpc]))
        roots = roots[np.imag(roots) > 0]
        fr_hz = np.sort(np.angle(roots) * fs / (2 * np.pi))
        bw = -0.5 * fs / np.pi * np.log(np.abs(roots[np.argsort(np.angle(roots))]))
        f = [x for x, b in zip(fr_hz, bw) if 200 < x < 4500 and b < 400]
        if len(f) >= 3:
            F.append(f[:3])
    if not F:
        return [np.nan] * 3
    return np.median(np.array(F), 0).tolist()


def wer(ref, hyp):
    norm = lambda s: re.sub(r"[^\w\s']", " ", s.lower()).split()
    r, h = norm(ref), norm(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)


class Helpers:
    """Exposed to edit files as H."""
    def __init__(self, model, tok, D):
        self.model, self.tok, self.D = model, tok, D

    def direction(self, name):           # q12 L19 DiffMean directions (unit, raw space)
        return self.D[name].clone()

    def token_ids(self, s):
        return self.tok(s, add_special_tokens=False).input_ids

    def layer(self, l):
        return self.model.model.layers[l]

    def embed(self):
        return self.model.model.embed_tokens.weight

    def audio_token_ids(self):
        return self.audio_ids

    def add_bias(self, l, vec, where="mlp"):
        """bake a constant residual-stream vector into layer l as an output bias (weight edit)."""
        lin = self.layer(l).mlp.down_proj if where == "mlp" else self.layer(l).self_attn.o_proj
        v = torch.as_tensor(vec, dtype=lin.weight.dtype, device=lin.weight.device)
        if lin.bias is None:
            lin.bias = torch.nn.Parameter(v.clone(), requires_grad=False)
        else:
            lin.bias.data += v

    def acoustic_dir(self, name):            # q13/dirs_L19.npz : pitch / f1 / f2 (unit, raw space)
        import numpy as _np
        return torch.tensor(_np.load(ROOT / "interp/q13/dirs_L19.npz")[name + "_dir"])


class Lab:
    def __init__(self):
        torch.set_grad_enabled(False); torch.cuda.set_per_process_memory_fraction(0.45)
        repo = ROOT / "rumik-oss-1"
        self.tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16,
                                                          attn_implementation="sdpa").eval().cuda()
        self.mimi = MimiModel.from_pretrained(repo / "codec", dtype=torch.bfloat16).eval().cuda()
        self.sr = int(AutoFeatureExtractor.from_pretrained(repo / "codec").sampling_rate)
        vocab = self.tok.get_vocab(); self.end_id = int(vocab["</audio>"])
        self.mapping = {int(t): (int(m.group(1)), int(m.group(2))) for s, t in vocab.items() if (m := AUDIO_RE.match(s))}
        self.allowed = torch.tensor(sorted([*self.mapping, self.end_id]), device="cuda")
        self.probes = Probes(ROOT / "interp/acts_v2", "cuda")
        names = json.load(open(ROOT / "interp/q12/direction_names.json"))
        V = torch.tensor(np.load(ROOT / "interp/q12/directions_L19.npy"), dtype=torch.float32)
        self.H = Helpers(self.model, self.tok, {n: V[i] for i, n in enumerate(names)})
        self.H.audio_ids = torch.tensor(sorted(self.mapping))
        self.asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
                            torch_dtype=torch.float16, device="cuda")
        self.gfe = AutoFeatureExtractor.from_pretrained(GENDER)
        self.gclf = AutoModelForAudioClassification.from_pretrained(GENDER).eval().cuda()
        self.male_idx = [i for i, l in self.gclf.config.id2label.items() if l.lower().startswith("male")][0]
        print(f"[lab] gender labels {self.gclf.config.id2label}", flush=True)
        self.snap = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}  # restore point
        print("[lab] ready", flush=True)

    def restore(self):
        for L in self.model.model.layers:      # drop any bias added by H.add_bias
            for lin in (L.mlp.down_proj, L.self_attn.o_proj):
                if lin.bias is not None:
                    lin.bias = None
        sd = self.model.state_dict()
        for k, v in self.snap.items():
            if not torch.equal(sd[k].cpu(), v):
                sd[k].copy_(v.to(sd[k].device))

    def p_male(self, a16):
        x = self.gfe(a16, sampling_rate=16000, return_tensors="pt").to("cuda")
        return float(self.gclf(**x).logits.softmax(-1)[0, self.male_idx])

    def score(self, au, text):
        a16 = resample_poly(au, 2, 3).astype(np.float32)
        f0, f0sd, vf = f0_stats(au, self.sr)
        F1, F2, F3 = formants(au, self.sr)
        hyp = self.asr({"raw": a16, "sampling_rate": 16000}, generate_kwargs={"language": "en"})["text"]
        fe = self.probes.embed(au)
        spk = {c: self.probes.prob(fe, "speaker", c) for c in self.probes.tasks["speaker"]["classes"]} if fe is not None else {}
        return dict(f0=f0, f0sd=f0sd, voiced=vf, F1=F1, F2=F2, F3=F3, p_male=self.p_male(a16),
                    wer=wer(text, hyp), asr=hyp.strip(), spk=spk, spk_max=max(spk.values()) if spk else np.nan,
                    rms=float(np.sqrt((au ** 2).mean())), dur=len(au) / self.sr)

    def run(self, job):
        jid = job["id"]; od = LAB / "results" / jid; od.mkdir(parents=True, exist_ok=True)
        if job.get("edit"):
            spec = importlib.util.spec_from_file_location("edit_" + jid, ROOT / "interp" / job["edit"])
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            mod.apply(self.model, self.H)
        fmt = job.get("prompt_fmt", "<text>{spk}: {text}<audio>")
        recs = []
        try:
            for ti in job.get("texts", [0, 1]):
                text = TEXTS[ti]
                ids = self.tok(fmt.format(spk=job.get("speaker", "Ira"), text=text), return_tensors="pt").input_ids.cuda()
                n = job.get("seeds", 3)
                toks, stopped = generate_rows(self.model, 19, ids, self.allowed, self.end_id, torch.zeros(n, 2048),
                                              job.get("max_new", 1000), seed=100 + ti)
                for s, (tk, stp) in enumerate(zip(toks, stopped)):
                    vals = to_codes(tk, self.mapping)
                    if not vals:
                        recs.append(dict(text=ti, seed=s, intact=False)); continue
                    au = self.mimi.decode(torch.tensor(vals, device="cuda").reshape(1, -1, 8).transpose(1, 2)
                                          ).audio_values[0, 0].float().cpu().clamp(-1, 1).numpy()
                    save_wav(od / f"t{ti}_s{s}.wav", torch.from_numpy(au), self.sr)
                    r = self.score(au, text); r.update(text=ti, seed=s)
                    r["intact"] = bool(stp and r["rms"] > 0.02 and r["voiced"] > 0.2 and r["wer"] < 0.5)
                    recs.append(r)
        finally:
            self.restore(); torch.cuda.empty_cache()
        ok = [r for r in recs if r.get("intact")]
        agg = lambda k: float(np.nanmean([r[k] for r in ok])) if ok else float("nan")
        summ = dict(id=jid, note=job.get("note", ""), n=len(recs), intact=len(ok) / max(len(recs), 1),
                    f0=agg("f0"), F1=agg("F1"), F2=agg("F2"), F3=agg("F3"), p_male=agg("p_male"),
                    wer=float(np.nanmean([r.get("wer", 1.0) for r in recs])), spk_max=agg("spk_max"))
        (od / "result.json").write_text(json.dumps({"job": job, "summary": summ, "clips": recs}, indent=2, default=float))
        return summ


def main():
    (LAB / "queue").mkdir(parents=True, exist_ok=True); (LAB / "done").mkdir(exist_ok=True); (LAB / "edits").mkdir(exist_ok=True)
    bf = LAB / "budget.json"
    if not bf.exists():
        bf.write_text(json.dumps({"budget": 25, "used": 0}))
    lab = Lab()
    while True:
        jobs = sorted((LAB / "queue").glob("*.json"))
        if not jobs:
            time.sleep(2); continue
        jf = jobs[0]; job = json.loads(jf.read_text()); B = json.loads(bf.read_text())
        free = job.get("free", False)  # calibration passes (baseline reference) don't count
        if not free and B["used"] >= B["budget"]:
            print(f"[lab] budget exhausted, refusing {job['id']}", flush=True); shutil.move(jf, LAB / "done" / jf.name); continue
        tag = "free" if free else "pass %d/%d" % (B["used"] + 1, B["budget"])
        print(f"[lab] >>> {job['id']} ({tag}): {job.get('note', '')}", flush=True)
        t0 = time.time()
        try:
            s = lab.run(job)
            print(f"[lab] <<< {job['id']} intact {s['intact']:.0%}  F0 {s['f0']:.0f}Hz  F1/F2/F3 {s['F1']:.0f}/{s['F2']:.0f}/{s['F3']:.0f}  "
                  f"P(male) {s['p_male']:.2f}  WER {s['wer']:.2f}  spk_max {s['spk_max']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
            if not free:
                B["used"] += 1; bf.write_text(json.dumps(B))
        except Exception:
            print(f"[lab] !!! {job['id']} failed:\n{traceback.format_exc()}", flush=True)
            lab.restore()
        if jf.exists():  # editor may have withdrawn the job file mid-run
            shutil.move(jf, LAB / "done" / jf.name)


if __name__ == "__main__":
    main()
