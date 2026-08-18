"""The cache trick that makes it slower - prereg #106's refuted P-3.

Every number is read from weights/data/prereg106_reproduce.json, the raw arm output, so this
chart regenerates from committed data or it does not render at all.

The hero is the priming point. Common advice for a slow model file is to read it once so the
OS caches it (`cat model.gguf > /dev/null`). We staked that at +1.0 tok/s and measured -1.95.
The chart shows why: the six runs before it were climbing, and priming threw that away.
"""
from __future__ import annotations
import json
import os

import brand as B

# Chips are 172 tall and the footer owns the bottom 140. Laid out from the top:
# header 270 + chart 430 + gap 34 + why-panel 178 + gap 30 + chip 172 + gap 40 + footer 140.
W, H = 1600, 1300
PUBLISHED = 14.86            # what the model card used to claim, as a point estimate
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                    "prereg106_reproduce.json")

# chart box
GX, GY, GW, GH = 80, 300, 950, 430
PLOT_L, PLOT_R = GX + 96, GX + GW - 150      # room for the y labels and the primed column
PLOT_T, PLOT_B = GY + 54, GY + GH - 62
YMIN, YMAX = 11.0, 15.3


def py(v):
    return PLOT_B - (v - YMIN) / (YMAX - YMIN) * (PLOT_B - PLOT_T)


def main():
    d = json.load(open(DATA, encoding="utf-8"))
    runs = [r["tok_s"] for r in d["headline_runs"] if r.get("tok_s")]
    primed = d["primed"]["tok_s"]
    ctl = [r["tok_s"] for r in d["control_small"] if r.get("tok_s")]
    mean = sum(runs) / len(runs)
    best = max(runs)
    spread_big = (max(runs) - min(runs)) / mean * 100
    spread_ctl = (max(ctl) - min(ctl)) / (sum(ctl) / len(ctl)) * 100
    gib = d["model_bytes"] / 2**30
    free = d["headline_runs"][0]["free_gb"]

    s = [B.svg_open(W, H),
         B.header(W, f"MEASURED &#183; Qwen3.6-35B-A3B &#183; {gib:.2f} GiB FILE, "
                     f"{free} GB FREE RAM &#183; ONE SESSION",
                  "The cache trick that makes it slower",
                  "Reading a model file once to \"warm the cache\" is standard advice. When the "
                  "file is bigger than your RAM, it costs you.")]

    # ---- chart
    s.append(B.panel(GX, GY, GW, GH))
    s.append(f'<text x="{GX+28}" y="{GY+36}" fill="{B.MUT}" font-size="19" letter-spacing="3">'
             f'DECODE tok/s &#183; SIX CONSECUTIVE RUNS OF ONE UNCHANGED COMMAND</text>')

    for v in (11, 12, 13, 14, 15):
        s.append(f'<line x1="{PLOT_L}" y1="{py(v)}" x2="{PLOT_R+112}" y2="{py(v)}" '
                 f'stroke="{B.GRID}" stroke-width="1.5"/>')
        s.append(f'<text x="{PLOT_L-18}" y="{py(v)+7}" text-anchor="end" fill="{B.MUT}" '
                 f'font-size="19">{v}</text>')

    # what the card used to promise, and never reached
    s.append(f'<line x1="{PLOT_L}" y1="{py(PUBLISHED)}" x2="{PLOT_R+112}" y2="{py(PUBLISHED)}" '
             f'stroke="{B.MUT}" stroke-width="3" stroke-dasharray="10 8"/>')
    s.append(f'<text x="{PLOT_L+10}" y="{py(PUBLISHED)-14}" fill="{B.MUT}" font-size="20">'
             f'{PUBLISHED} &#8212; what we used to publish. Never reached in 6 tries.</text>')

    step = (PLOT_R - PLOT_L) / (len(runs) - 1)
    pts = [(PLOT_L + i * step, py(v)) for i, v in enumerate(runs)]
    s.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) +
             f'" fill="none" stroke="{B.TEAL}" stroke-width="6" stroke-linejoin="round"/>')
    for i, ((x, y), v) in enumerate(zip(pts, runs)):
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="{B.TEAL}"/>')
        s.append(f'<text x="{x:.1f}" y="{y-26:.1f}" text-anchor="middle" fill="{B.INK}" '
                 f'font-size="21" font-weight="bold">{v:.2f}</text>')
        s.append(f'<text x="{x:.1f}" y="{PLOT_B+34:.1f}" text-anchor="middle" fill="{B.MUT}" '
                 f'font-size="19">run {i+1}</text>')

    # the primed run, set apart so it reads as a treatment and not a seventh sample
    px = PLOT_R + 92          # the 11.89 label is centred here; keep it inside the panel
    s.append(f'<line x1="{PLOT_R+40}" y1="{PLOT_T-10}" x2="{PLOT_R+40}" y2="{PLOT_B+10}" '
             f'stroke="{B.EDGE}" stroke-width="2" stroke-dasharray="6 6"/>')
    s.append(f'<line x1="{pts[-1][0]:.1f}" y1="{pts[-1][1]:.1f}" x2="{px}" y2="{py(primed):.1f}" '
             f'stroke="{B.DISK}" stroke-width="6" stroke-dasharray="12 7"/>')
    s.append(f'<circle cx="{px}" cy="{py(primed):.1f}" r="15" fill="{B.DISK}"/>')
    s.append(f'<text x="{px}" y="{py(primed)+46:.1f}" text-anchor="middle" fill="{B.DISK}" '
             f'font-size="30" font-weight="bold">{primed:.2f}</text>')
    s.append(f'<text x="{px}" y="{PLOT_B+34:.1f}" text-anchor="middle" fill="{B.DISK}" '
             f'font-size="19" letter-spacing="1">PRIMED</text>')

    # ---- the verdict column
    VX, VW = GX + GW + 30, W - (GX + GW + 30) - 80
    s.append(B.panel(VX, GY, VW, GH, stroke=B.DISK, sw=2))
    s.append(f'<text x="{VX+28}" y="{GY+44}" fill="{B.DISK}" font-size="19" letter-spacing="3">'
             f'THE PREDICTION THAT FAILED</text>')
    s.append(f'<text x="{VX+28}" y="{GY+104}" fill="{B.SUB}" font-size="21">We staked priming at'
             f'</text>')
    s.append(f'<text x="{VX+28}" y="{GY+152}" fill="{B.SUB}" font-size="40" font-weight="bold">'
             f'+1.0 tok/s</text>')
    s.append(f'<text x="{VX+28}" y="{GY+196}" fill="{B.MUT}" font-size="21">It measured</text>')
    s.append(f'<text x="{VX+28}" y="{GY+262}" fill="{B.DISK}" font-size="72" font-weight="bold">'
             f'{primed-mean:+.2f}</text>')
    s.append(B.paragraph(VX + 28, GY + 312,
                         "The sign came out wrong. Staked before the arm ran, so it ships as a "
                         "miss at the same size as the hits.", 21, B.SUB, VW - 56))

    # ---- why, and the control that rules out "our box is just noisy"
    CY_ = GY + GH + 34
    s.append(B.panel(GX, CY_, W - 160, 178, stroke=B.TEAL, sw=2))
    s.append(f'<text x="{GX+28}" y="{CY_+40}" fill="{B.TEAL}" font-size="19" letter-spacing="3">'
             f'WHY &#8212; AND WHY IT IS NOT JUST A NOISY MACHINE</text>')
    s.append(B.paragraph(
        GX + 28, CY_ + 78,
        f"A {gib:.2f} GiB file cannot fit in {free} GB of free RAM, so a sequential read ends "
        f"with the cache holding the file's LAST ~12 GB. Six real runs instead leave it holding "
        f"the pages the model actually re-reads &#8212; for a sparse MoE, the hot experts. "
        f"Priming swaps a frequency-adapted cache for a position-adapted one, and position is "
        f"the wrong key. Control: a 4.36 GiB model that DOES fit held a {spread_ctl:.1f}% spread "
        f"with no ramp at all, against this file's {spread_big:.1f}%.",
        22, B.SUB, W - 216))

    # ---- chips
    KY = CY_ + 178 + 30
    CW_ = (W - 160 - 2 * 24) / 3
    s.append(B.chip(GX, KY, CW_, "WARM-UP, RUN 1 TO BEST",
                    f"+{(best/runs[0]-1)*100:.1f}%", "same command, nothing changed", B.TEAL))
    s.append(B.chip(GX + CW_ + 24, KY, CW_, "COST OF PRIMING",
                    f"{(primed/mean-1)*100:.0f}%", "vs the six-run mean", B.DISK))
    s.append(B.chip(GX + 2 * (CW_ + 24), KY, CW_, "SPREAD, FITS vs DOES NOT",
                    f"{spread_ctl:.1f}% / {spread_big:.1f}%",
                    "4.36 GiB model vs this one", B.VRAM))

    s.append(B.footer(W, H, "prereg #106 &#183; scored 2/4 &#183; L-29, D-29"))
    s.append("</svg>")
    B.save("cache_trap.svg", "".join(s))


if __name__ == "__main__":
    main()
