#!/usr/bin/env python3
"""Build social assets for the rumik-oss-1 interpretability study.

Outputs (social/out/):
  linkedin/slide_XX.png  1080x1350  +  linkedin_carousel.pdf
  x/img_XX.png           1600x900   (thread images)
  x/voice_male.mp4, x/circuit.mp4  (waveform videos with real audio)
Style: rumik-inspired (warm paper, near-black ink, orange accent, dotted waveform), no rumik logo.
Rendering: HTML -> headless Chromium screenshot. Charts are HTML/CSS/SVG drawn from real result data.
"""
import json, math, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "social/out"; H = ROOT / "social/html"
for d in (OUT / "linkedin", OUT / "x", H):
    d.mkdir(parents=True, exist_ok=True)
DATA = json.load(open("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/7d6d4930-93c9-42bb-be05-0bc30845249e/scratchpad/report_data.json"))
AUDIO = Path("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/7d6d4930-93c9-42bb-be05-0bc30845249e/scratchpad/audio")
HANDLE = "Sanyam · @Sj0605-DataSci"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
:root{--paper:#f6f5f2;--paper2:#ecebe6;--ink:#1d1d1f;--ink2:#55555b;--ink3:#8d8d93;--rule:#d9d7d0;--accent:#ea6a1f;--accent2:#c2410c;--soft:#fbe3d2;--blue:#2a78d6;
 --f-d:'Outfit',system-ui,sans-serif;--f-b:'Inter',system-ui,sans-serif;--f-m:'JetBrains Mono',monospace}
*{box-sizing:border-box;margin:0}
html,body{width:var(--W);height:var(--H);background:var(--paper);color:var(--ink);font-family:var(--f-b);overflow:hidden}
.slide{position:relative;width:100%;height:100%;padding:84px 80px 72px;display:flex;flex-direction:column;gap:30px}
.land .slide{padding:64px 84px 56px;gap:22px}
.top{display:flex;justify-content:space-between;align-items:center;font-family:var(--f-m);font-size:20px;color:var(--ink3);letter-spacing:.06em;text-transform:uppercase}
.star{display:inline-block;width:12px;height:12px;background:var(--accent);border-radius:50%;margin-right:10px;vertical-align:1px}
h1{font-family:var(--f-d);font-weight:800;font-size:78px;line-height:1.02;letter-spacing:-.02em}
.land h1{font-size:64px}
h2{font-family:var(--f-d);font-weight:700;font-size:60px;line-height:1.05;letter-spacing:-.015em}
.land h2{font-size:52px}
.hl{color:var(--accent)}
p.lead{font-size:30px;line-height:1.4;color:var(--ink2);max-width:880px}
.land p.lead{font-size:26px;max-width:1300px}
.body{flex:1;display:flex;flex-direction:column;gap:26px;justify-content:center;min-height:0}
.land .body.row{flex-direction:row;align-items:center;gap:56px}
.foot{display:flex;justify-content:space-between;font-family:var(--f-m);font-size:19px;color:var(--ink3)}
.dots{position:absolute;left:0;right:0;bottom:0;height:120px;pointer-events:none;opacity:.5}
.hasdots .slide{padding-bottom:150px}
.foot{position:relative;z-index:2}
.big{font-family:var(--f-d);font-weight:800;font-size:150px;line-height:.9;letter-spacing:-.03em}
.land .big{font-size:120px}
.card{background:#fff;border:2px solid var(--ink);border-radius:22px;padding:30px 34px}
.mono{font-family:var(--f-m)}
.barrow{display:grid;grid-template-columns:190px 1fr 120px;align-items:center;gap:18px;font-size:26px}
.barrow .lab{font-family:var(--f-d);font-weight:600}
.track{height:34px;background:var(--paper2);border-radius:8px;overflow:hidden;position:relative}
.fill{height:100%;border-radius:8px;background:var(--ink)}
.fill.a{background:var(--accent)}
.val{font-family:var(--f-m);font-size:24px;text-align:right}
.chips{display:flex;flex-wrap:wrap;gap:12px}
.chip{font-family:var(--f-m);font-size:20px;border:2px solid var(--ink);border-radius:999px;padding:6px 16px}
.chip.on{background:var(--ink);color:var(--paper)}
.legend{display:flex;gap:26px;font-size:22px;color:var(--ink2)}
.legend i{display:inline-block;width:18px;height:18px;border-radius:5px;margin-right:8px;vertical-align:-2px}
table.t{border-collapse:collapse;width:100%;font-size:26px}
table.t th{font-family:var(--f-m);font-weight:400;font-size:19px;color:var(--ink3);text-align:left;padding:10px 12px;border-bottom:2px solid var(--ink)}
table.t td{padding:16px 12px;border-bottom:1px solid var(--rule)}
table.t td.n{font-family:var(--f-m)}
.step{display:flex;gap:22px;align-items:flex-start;font-size:27px;line-height:1.35}
.step b.k{flex:none;width:52px;height:52px;border-radius:50%;background:var(--ink);color:var(--paper);display:grid;place-items:center;font-family:var(--f-d);font-size:26px}
.arrow{font-family:var(--f-d);font-size:44px;color:var(--accent);text-align:center}
"""

DOTS = """<svg class="dots" viewBox="0 0 1600 150" preserveAspectRatio="none">{}</svg>"""


def dots_svg(seed=1):
    out = []; cols = 160
    for i in range(cols):
        x = i * 10 + 5
        amp = 0.25 + 0.75 * math.exp(-((i - cols * 0.55) / (cols * 0.22)) ** 2) * (0.6 + 0.4 * math.sin(i * 0.9 + seed))
        n = max(1, int(amp * 9))
        for k in range(n):
            out.append(f'<rect x="{x}" y="{142 - k * 15}" width="2.4" height="7" fill="#1d1d1f"/>')
    return DOTS.format("".join(out))


def page(body, W, Hh, land=False, n=None, total=None, dots=False):
    footer = f'<div class="foot"><span>{HANDLE}</span><span>{f"{n} / {total}" if n else ""}</span></div>'
    return f"""<!doctype html><html class="{'land' if land else ''} {'hasdots' if dots else ''}" style="--W:{W}px;--H:{Hh}px"><head><meta charset="utf-8"><style>{CSS}</style></head>
<body><div class="slide">
<div class="top"><span><i class="star"></i>Inside rumik-oss-1 · interpretability</span><span>3B Indic TTS</span></div>
{body}{footer}</div>{dots_svg(n or 1) if dots else ''}</body></html>"""


# ------------------------------------------------------------------ chart helpers
def bars(rows, maxv=1.0, fmt="{:.2f}", accent_idx=()):
    h = ""
    for i, (lab, v) in enumerate(rows):
        w = max(1.5, 100 * v / maxv)
        h += f'<div class="barrow"><span class="lab">{lab}</span><div class="track"><div class="fill {"a" if i in accent_idx else ""}" style="width:{w}%"></div></div><span class="val">{fmt.format(v)}</span></div>'
    return f'<div style="display:flex;flex-direction:column;gap:18px">{h}</div>'


def heatmap(cell=22, gap=3):
    hd = DATA["heads"]; mx = max(max(r) for r in hd)
    flat = sorted(((v, l, h) for l, r in enumerate(hd) for h, v in enumerate(r)), reverse=True)[:8]
    top = {(l, h) for _, l, h in flat}
    W = 36 * (cell + gap) + 60; Hh = 16 * (cell + gap) + 50
    s = f'<svg viewBox="0 0 {W} {Hh}" style="width:100%;height:auto">'
    ramp = ["#ecebe6", "#f6cfb3", "#f0a26f", "#ea6a1f", "#b8470f", "#6e2a08"]
    for l in range(36):
        for h in range(16):
            v = hd[l][h]; t = max(0, v) / mx
            idx = 0 if v <= 0.001 else min(5, 1 + int(math.sqrt(t) * 5))
            x = 50 + l * (cell + gap); y = 6 + h * (cell + gap)
            s += f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="4" fill="{ramp[idx]}"/>'
            if (l, h) in top:
                s += f'<rect x="{x-1.5}" y="{y-1.5}" width="{cell+3}" height="{cell+3}" rx="5" fill="none" stroke="#1d1d1f" stroke-width="3"/>'
        if l % 5 == 0 or l == 15:
            s += f'<text x="{50 + l*(cell+gap) + cell/2}" y="{Hh-12}" text-anchor="middle" font-family="JetBrains Mono" font-size="17" fill="{"#1d1d1f" if l==15 else "#8d8d93"}" font-weight="{600 if l==15 else 400}">L{l}</text>'
    s += f'<text x="0" y="{6+cell-4}" font-family="JetBrains Mono" font-size="16" fill="#8d8d93">h0</text><text x="0" y="{6+15*(cell+gap)+cell-4}" font-family="JetBrains Mono" font-size="16" fill="#8d8d93">h15</text></svg>'
    return s


def line_chart(series, W=900, Hh=420, xs=None, xlab="", ylab=""):
    m = dict(l=70, r=150, t=20, b=60); xs = xs or list(range(len(series[0][1])))
    X = lambda i: m["l"] + (xs[i] - xs[0]) / (xs[-1] - xs[0]) * (W - m["l"] - m["r"])
    Y = lambda v: m["t"] + (1 - v) * (Hh - m["t"] - m["b"])
    s = f'<svg viewBox="0 0 {W} {Hh}" style="width:100%;height:auto">'
    for v in (0, .5, 1):
        s += f'<line x1="{m["l"]}" x2="{W-m["r"]}" y1="{Y(v)}" y2="{Y(v)}" stroke="#d9d7d0" stroke-width="2"/><text x="{m["l"]-14}" y="{Y(v)+7}" text-anchor="end" font-family="JetBrains Mono" font-size="20" fill="#8d8d93">{v:.1f}</text>'
    for i, xv in enumerate(xs):
        s += f'<text x="{X(i)}" y="{Hh-m["b"]+34}" text-anchor="middle" font-family="JetBrains Mono" font-size="20" fill="#8d8d93">{xv}</text>'
    s += f'<text x="{(m["l"]+W-m["r"])/2}" y="{Hh-4}" text-anchor="middle" font-family="Inter" font-size="20" fill="#55555b">{xlab}</text>'
    for name, ys, col in series:
        d = " ".join(("M" if i == 0 else "L") + f"{X(i)} {Y(v)}" for i, v in enumerate(ys))
        s += f'<path d="{d}" fill="none" stroke="{col}" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>'
        for i, v in enumerate(ys):
            s += f'<circle cx="{X(i)}" cy="{Y(v)}" r="8" fill="{col}" stroke="#f6f5f2" stroke-width="3"/>'
        s += f'<text x="{X(len(ys)-1)+18}" y="{Y(ys[-1])+8}" font-family="Outfit" font-weight="700" font-size="26" fill="{col}">{name}</text>'
    return s + "</svg>"


# ------------------------------------------------------------------ slide content (shared by both formats)
dose = DATA["dose"]; A = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]
dose_series = [("→ Aisha", [dose["to_aisha"][a]["Aisha"] for a in A], "#ea6a1f"),
               ("→ Zoya", [dose["to_zoya"][a]["Zoya"] for a in A], "#1d1d1f"),
               ("→ Siya", [dose["to_siya"][a]["Siya"] for a in A], "#8d8d93")]
topk = DATA["topk"]; base = DATA["base"]
gen = DATA["gen"]


def S(key, land):
    if key == "cover":
        return f"""<div class="body"><h1>I gave an all-female TTS model a <span class="hl">male voice</span> by editing 4,096 numbers.</h1>
<p class="lead">No fine-tuning. No gradients. Just reading rumik-oss-1's internals, finding where a voice lives, and nudging it.</p>
<div class="chips"><span class="chip on">sparse autoencoders</span><span class="chip">steering</span><span class="chip">circuit discovery</span><span class="chip">weight editing</span></div></div>"""
    if key == "setup":
        return f"""<div class="body {'row' if land else ''}"><div style="flex:1;display:flex;flex-direction:column;gap:24px"><h2>The model: a language model that speaks</h2>
<p class="lead">rumik-oss-1 is a 3B text-to-speech model. It reads text and emits audio-codec tokens, 8 per 80 ms frame. It ships four voices, all female: Ira, Aisha, Siya, Zoya.</p></div>
<div class="card" style="flex:1;display:flex;flex-direction:column;gap:16px;font-size:26px">
<div><b class="mono">prompt</b> → <span class="mono">Ira: Please make sure the report is ready…</span></div>
<div class="arrow">↓ 36 transformer layers</div>
<div><b class="mono">output</b> → Mimi codec tokens → 24 kHz audio</div>
<div style="color:var(--ink2);font-size:22px">The question: where inside, exactly, is "Ira" decided?</div></div></div>"""
    if key == "steer":
        return f"""<div class="body {'row' if land else ''}"><div style="flex:1;display:flex;flex-direction:column;gap:20px"><h2>One feature switches the speaker</h2>
<p class="lead">I trained a sparse autoencoder on layer 19 (32k features, 1.39M tokens). Turning up a single "Aisha" feature while the prompt still says <b>Ira</b> turns the voice into Aisha.</p></div>
<div style="flex:1.1">{line_chart(dose_series, xs=[0, .1, .2, .3, .4, .5], xlab="steering strength α")}<p style="font-size:20px;color:var(--ink3);margin-top:8px">P(target voice) from an external WavLM speaker probe (99.4% accurate). 6 clips per point.</p></div></div>"""
    if key == "circuit":
        return f"""<div class="body"><h2>16 of 576 attention heads decide <span class="hl">whose voice</span> you hear</h2>
{heatmap()}
<p class="lead">I zeroed each head in turn and measured how much the model still prefers the right speaker. Darker = bigger drop; outlined = top 8. Layer 15 lights up.</p></div>"""
    if key == "collapse":
        rows = [("1 head", topk["1"] / base), ("2 heads", topk["2"] / base), ("4 heads", topk["4"] / base), ("8 heads", topk["8"] / base), ("16 heads", topk["16"] / base), ("8 random", DATA["rand8"] / base)]
        z = gen["base"]["Zoya"]["p"]["Zoya"]; z8 = gen["ablate_top8_heads"]["Zoya"]["p"]["Zoya"]
        return f"""<div class="body {'row' if land else ''}"><div style="flex:1;display:flex;flex-direction:column;gap:22px"><h2>Remove them and the model forgets who it is</h2>
<p class="lead">Speaker binding left after removing the top-k heads. Eight random heads change nothing.</p>
{bars(rows, maxv=1.1, fmt="{:.0%}", accent_idx=(5,))}</div>
<div class="card" style="flex:.8;display:flex;flex-direction:column;gap:10px"><span class="mono" style="color:var(--ink3);font-size:20px">"Zoya:" prompt, P(sounds like Zoya)</span>
<div style="display:flex;align-items:baseline;gap:22px"><span class="big">{z:.2f}</span><span class="arrow">→</span><span class="big hl">{z8:.2f}</span></div>
<span style="font-size:24px;color:var(--ink2)">Below chance (0.25). Speech stays intact; identity is gone.</span></div></div>"""
    if key == "emotion":
        tc = DATA["q12"]["tone_coords"]
        W, Hh = 820, 520
        sx = lambda v: 60 + (v + 1.6) / 3.2 * (W - 120); sy = lambda v: Hh - 50 - (v + 1.3) / 2.6 * (Hh - 100)
        svg = f'<svg viewBox="0 0 {W} {Hh}" style="width:100%;height:auto"><line x1="{sx(-1.6)}" x2="{sx(1.6)}" y1="{sy(0)}" y2="{sy(0)}" stroke="#d9d7d0" stroke-width="2"/><line x1="{sx(0)}" x2="{sx(0)}" y1="{sy(-1.3)}" y2="{sy(1.3)}" stroke="#d9d7d0" stroke-width="2"/>'
        svg += f'<text x="{sx(1.55)}" y="{sy(0)-14}" text-anchor="end" font-family="JetBrains Mono" font-size="20" fill="#8d8d93">calm / slow →</text><text x="{sx(-1.55)}" y="{sy(0)-14}" font-family="JetBrains Mono" font-size="20" fill="#8d8d93">← aroused / fast</text><text x="{sx(0)+12}" y="{sy(1.25)}" font-family="JetBrains Mono" font-size="20" fill="#8d8d93">↑ angry</text>'
        for t, (a, b) in tc.items():
            col = "#ea6a1f" if t in ("excited", "sad") else "#1d1d1f"
            svg += f'<circle cx="{sx(a)}" cy="{sy(b)}" r="16" fill="{col}"/><text x="{sx(a)+24}" y="{sy(b)+9}" font-family="Outfit" font-weight="700" font-size="30" fill="#1d1d1f">{t}</text>'
        svg += "</svg>"
        return f"""<div class="body {'row' if land else ''}"><div style="flex:1;display:flex;flex-direction:column;gap:20px"><h2>Emotion lives on a 2-D plane</h2>
<p class="lead">93% of tone variation sits on two axes: arousal (65%) and an "angry" axis (28%). The catch: arousal is mostly tempo. "Excited" ≈ "fast" (cos 0.79).</p></div>
<div style="flex:1.1">{svg}</div></div>"""
    if key == "wall":
        rows = [("unedited", 232), ("pitch −4", 179), ("pitch −6", 165), ("pitch −8", 164), ("stacked", 162)]
        return f"""<div class="body {'row' if land else ''}"><div style="flex:1;display:flex;flex-direction:column;gap:20px"><h2>Making it male: first, I hit a wall</h2>
<p class="lead">Directions learned from four female voices can only move pitch within the female range. Pushing harder stalls at ~165 Hz, and a gender classifier still says 0.01 male.</p></div>
<div style="flex:1">{bars(rows, maxv=260, fmt="{:.0f} Hz", accent_idx=(3,))}<p style="font-size:20px;color:var(--ink3);margin-top:14px">Typical male F0: 85–155 Hz</p></div></div>"""
    if key == "trick":
        return f"""<div class="body"><h2>The trick: show the model what "male" looks like <span class="hl">in its own head</span></h2>
<div style="display:flex;flex-direction:column;gap:22px">
<div class="step"><b class="k">1</b><span>Take real Ira clips. A vocoder (WORLD) lowers pitch ×0.55 and shrinks formants ×0.85.</span></div>
<div class="step"><b class="k">2</b><span>Encode with rumik's own Mimi codec. Round-trip check: <b>127 Hz, P(male) 1.00</b>.</span></div>
<div class="step"><b class="k">3</b><span>Feed those tokens through rumik. The activation difference, male − original, is a "male" direction inside the model.</span></div>
<div class="step"><b class="k">4</b><span>Bake it in as a constant on two layers' MLP output. 4,096 numbers. No gradients.</span></div></div></div>"""
    if key == "result":
        rows = [("Ira", 232, 108, "Ira 0.92"), ("Aisha", 222, 107, "Aisha 0.97"), ("Zoya", 260, 115, "Zoya 0.80")]
        tr = "".join(f'<tr><td style="font-family:Outfit;font-weight:700">{a}</td><td class="n">{b} Hz</td><td class="n" style="color:var(--accent);font-weight:600">{c} Hz</td><td class="n">1.00</td><td class="n">{d}</td></tr>' for a, b, c, d in rows)
        return f"""<div class="body"><h2>Result: every voice gets a male twin, and keeps its identity</h2>
<table class="t"><tr><th>voice</th><th>before</th><th>after patch</th><th>P(male)</th><th>probe still hears</th></tr>{tr}</table>
<p class="lead">Words survive (Whisper WER ≈ 0). Gender and speaker identity are separate directions in rumik, so you can flip one without touching the other.</p></div>"""
    if key == "takeaways":
        return f"""<div class="body"><h2>What I took away</h2>
<div style="display:flex;flex-direction:column;gap:20px;font-size:28px;line-height:1.4">
<div>● <b>Identity is a circuit.</b> About 16 heads around layer 15 copy the speaker's name into the audio.</div>
<div>● <b>SAE features win on discrete traits</b> (speaker). Plain mean-difference vectors win on diffuse ones (happy, laugh, pace).</div>
<div>● <b>Gender and identity are separable.</b> A 4,096-number edit creates voices the model never trained on.</div>
<div>● <b>Still open:</b> binding the new voice to a new name ("Arjun:") didn't work in 8 tries.</div></div>
<p style="font-size:22px;color:var(--ink3)">Model: rumik-ai/rumik-oss-1 (CC-BY-NC). Independent research, not affiliated with rumik.ai. Code: github.com/Sj0605-DataSci/rumik-oss-model-dissection</p></div>"""
    raise KeyError(key)


LINKEDIN = ["cover", "setup", "steer", "circuit", "collapse", "emotion", "wall", "trick", "result", "takeaways"]
X = ["cover", "steer", "circuit", "collapse", "wall", "trick", "result"]


def render(html_path, png_path, W, Hh):
    subprocess.run(["/snap/bin/chromium", "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                    f"--window-size={W},{Hh}", "--virtual-time-budget=4000", f"--screenshot={png_path}", f"file://{html_path}"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    from PIL import Image
    pngs = []
    for i, k in enumerate(LINKEDIN, 1):
        hp = H / f"li_{i:02d}.html"; hp.write_text(page(S(k, False), 1080, 1350, False, i, len(LINKEDIN), dots=(k in ("cover", "takeaways"))))
        pp = OUT / "linkedin" / f"slide_{i:02d}.png"; render(hp, pp, 1080, 1350); pngs.append(pp); print("li", i, k, flush=True)
    # PDF via Chromium print (this Pillow build lacks JPEG): one 1080x1350 page per slide
    book = H / "carousel.html"
    book.write_text("<!doctype html><html><head><style>@page{size:1080px 1350px;margin:0}html,body{margin:0}img{display:block;width:1080px;height:1350px;page-break-after:always}</style></head><body>"
                    + "".join(f'<img src="file://{p}">' for p in pngs) + "</body></html>")
    subprocess.run(["/snap/bin/chromium", "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT / 'linkedin_carousel.pdf'}",
                    "--virtual-time-budget=4000", f"file://{book}"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for i, k in enumerate(X, 1):
        hp = H / f"x_{i:02d}.html"; hp.write_text(page(S(k, True), 1600, 900, True, i, len(X), dots=(k == "cover")))
        render(hp, OUT / "x" / f"img_{i:02d}.png", 1600, 900); print("x", i, k, flush=True)
    print("done", OUT)


if __name__ == "__main__":
    main()
