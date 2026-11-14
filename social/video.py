#!/usr/bin/env python3
"""Short square videos (1080x1080) with real audio + live waveform, for X / LinkedIn.
Each segment = rendered title card (HTML -> PNG) + ffmpeg showwaves overlay of the clip's audio.
"""
import subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import CSS, render, H, OUT  # noqa: E402

AUD = Path("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/7d6d4930-93c9-42bb-be05-0bc30845249e/scratchpad/audio")
TMP = OUT / "x" / "_seg"; TMP.mkdir(parents=True, exist_ok=True)

CARD = """<!doctype html><html style="--W:1080px;--H:1080px"><head><meta charset="utf-8"><style>{css}
.slide{{justify-content:space-between}} .tag{{font-family:var(--f-m);font-size:24px;color:var(--ink3);letter-spacing:.08em;text-transform:uppercase}}
.who{{font-family:var(--f-d);font-weight:800;font-size:96px;line-height:1;letter-spacing:-.02em}} .sub{{font-size:32px;color:var(--ink2);max-width:900px;line-height:1.35}}
.stats{{display:flex;gap:18px;flex-wrap:wrap}} .stat{{font-family:var(--f-m);font-size:26px;border:2px solid var(--ink);border-radius:999px;padding:8px 20px}}
.stat.on{{background:var(--accent);border-color:var(--accent);color:#fff}} .wavezone{{height:300px}}</style></head>
<body><div class="slide"><div class="top"><span><i class="star"></i>Inside rumik-oss-1</span><span>{step}</span></div>
<div style="display:flex;flex-direction:column;gap:22px"><span class="tag">{tag}</span><div class="who">{who}</div><div class="sub">{sub}</div>
<div class="stats">{stats}</div></div><div class="wavezone"></div>
<div class="foot"><span>Sanyam · @Sj0605-DataSci</span><span>same sentence · same seed</span></div></div></body></html>"""


def seg(name, wav, step, tag, who, sub, stats, color):
    hp = H / f"card_{name}.html"
    hp.write_text(CARD.format(css=CSS, step=step, tag=tag, who=who, sub=sub,
                              stats="".join(f'<span class="stat {"on" if on else ""}">{s}</span>' for s, on in stats)))
    png = TMP / f"{name}.png"; render(hp, png, 1080, 1080)
    mp4 = TMP / f"{name}.mp4"
    fc = (f"[1:a]apad=pad_dur=0.6,aformat=channel_layouts=mono,showwaves=s=920x260:mode=cline:rate=30:colors={color}:scale=sqrt:draw=full[w];"
          f"[0:v][w]overlay=80:600:format=auto,format=yuv420p[v]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(png), "-i", str(wav), "-filter_complex", fc,
                    "-map", "[v]", "-map", "1:a", "-af", "apad=pad_dur=0.6", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-shortest", "-r", "30", str(mp4)], check=True)
    return mp4


def title(name, step, head, sub, secs=2.6):
    hp = H / f"card_{name}.html"
    hp.write_text(CARD.format(css=CSS, step=step, tag="", who=head, sub=sub, stats="").replace('<div class="wavezone"></div>', ""))
    png = TMP / f"{name}.png"; render(hp, png, 1080, 1080)
    mp4 = TMP / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-t", str(secs), "-i", str(png), "-f", "lavfi", "-t", str(secs),
                    "-i", "anullsrc=r=48000:cl=mono", "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
                    "-r", "30", "-shortest", str(mp4)], check=True)
    return mp4


def concat(parts, out):
    lst = TMP / (out.stem + ".txt"); lst.write_text("".join(f"file '{p}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)], check=True)
    print("wrote", out, flush=True)


BLACK, ORANGE = "0x1d1d1f", "0xea6a1f"
male = [title("m0", "male voice", 'An all-female TTS model.<br><span class="hl">One 4,096-number edit.</span>', "Same prompt, same seed. Listen to what changes."),
        seg("m1", AUD / "ira_base.wav", "1 / 4", "before", "Ira", "rumik-oss-1, unedited", [("F0 ≈ 232 Hz", False), ("P(male) 0.00", False)], BLACK),
        seg("m2", AUD / "ira_male.wav", "2 / 4", "after the weight patch", 'Ira, <span class="hl">male</span>', "Same model + a constant on two MLP layers. No training.",
            [("F0 ≈ 108 Hz", True), ("P(male) 1.00", True), ("still sounds like Ira", False)], ORANGE),
        seg("m3", AUD / "zoya_base.wav", "3 / 4", "before", "Zoya", "a different built-in voice", [("F0 ≈ 260 Hz", False), ("P(male) 0.00", False)], BLACK),
        seg("m4", AUD / "zoya_male.wav", "4 / 4", "after the same patch", 'Zoya, <span class="hl">male</span>', "Gender flips. Identity stays: the probe still hears Zoya.",
            [("F0 ≈ 115 Hz", True), ("P(male) 1.00", True), ("still Zoya 0.80", False)], ORANGE)]
concat(male, OUT / "x" / "voice_male.mp4")

circ = [title("c0", "circuit", 'Delete <span class="hl">8 of 576</span> attention heads.', "Same \"Zoya:\" prompt. Watch the identity disappear."),
        seg("c1", AUD / "circuit_zoya_base.wav", "1 / 2", "unedited", "Zoya", "speaker-binding heads intact", [("P(Zoya) 0.91", False)], BLACK),
        seg("c2", AUD / "circuit_zoya_heads_off.wav", "2 / 2", "top 8 binding heads removed", 'Zoya<span class="hl">?</span>',
            "Speech is intact; the model no longer knows whose voice to use.", [("P(Zoya) 0.19", True), ("below chance", True)], ORANGE)]
concat(circ, OUT / "x" / "circuit.mp4")
