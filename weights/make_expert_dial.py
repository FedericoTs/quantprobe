"""The MoE speed dial: what it buys, what it costs - prereg #107, scored 4/4.

Every number read from weights/data/prereg107_kcurve.json, so the chart regenerates from the
raw arms or it does not render.

Two panels because there are two stories and they need each other. Left: Law 4 predicted this
curve from the FILE, before any measurement, to within 2%. Right: every point it predicts costs
more quality than the speed is worth. Either alone is half an argument.
"""
from __future__ import annotations
import json
import math
import os

import brand as B

# Laid out from the top: header 270 + charts 430 + gap 34 + verdict 200 + gap 30 + chip 172
# + gap 44 + footer 140. The first pass used a 170 verdict panel and the fifth line of prose
# came out struck through by its own border.
W, H = 1600, 1340
# Law 4's predictions, staked in prereg #107 before the arms ran.
PREDICTED = {8: 1.000, 4: 1.125, 2: 1.200, 1: 1.242}
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                    "prereg107_kcurve.json")

GY, GH = 300, 430
LX, LW = 80, 740
RX, RW = 860, 660
KS = [8, 4, 2, 1]


def main():
    d = json.load(open(DATA, encoding="utf-8"))
    sp = {int(k): (sum(v) / len(v), (max(v) - min(v)) / 2) for k, v in d["speed"].items() if v}
    ppl = {int(k): v for k, v in d["ppl"].items()}
    base = sp[8][0]
    gain = {k: sp[k][0] / base for k in sp}

    s = [B.svg_open(W, H),
         B.header(W, "MEASURED &#183; Qwen3.6-35B-A3B &#183; 256 EXPERTS, k=8 &#183; "
                     "PREDICTED FROM THE FILE FIRST",
                  "The MoE speed dial nobody should turn",
                  "Using fewer experts is traded as free speed on small hardware. It is bounded "
                  "by arithmetic you can read off the file - and it is never free.")]

    # ---------- left: speed, predicted vs measured
    s.append(B.panel(LX, GY, LW, GH))
    s.append(f'<text x="{LX+28}" y="{GY+36}" fill="{B.MUT}" font-size="19" letter-spacing="3">'
             f'SPEEDUP vs k=8 &#183; LAW 4 PREDICTED THIS BEFORE MEASURING</text>')
    # Both legends live here rather than floating in the plot. The first pass put them inside and
    # the k=1 annotation crossed the very line it was labelling.
    s.append(f'<text x="{LX+28}" y="{GY+62}" fill="{B.MUT}" font-size="18">'
             f'dashed = predicted from the file&#8217;s byte split &#183; '
             f'<tspan fill="{B.VRAM}">k=1 beat that ceiling by 17%</tspan></text>')
    pl, pr = LX + 96, LX + LW - 60
    pt, pb = GY + 96, GY + GH - 62
    ymin, ymax = 0.95, 1.55

    def py(v):
        return pb - (v - ymin) / (ymax - ymin) * (pb - pt)

    for v in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
        s.append(f'<line x1="{pl}" y1="{py(v):.1f}" x2="{pr}" y2="{py(v):.1f}" '
                 f'stroke="{B.GRID}" stroke-width="1.5"/>')
        s.append(f'<text x="{pl-16}" y="{py(v)+7:.1f}" text-anchor="end" fill="{B.MUT}" '
                 f'font-size="18">{v:.1f}x</text>')
    step = (pr - pl) / (len(KS) - 1)
    xs = {k: pl + i * step for i, k in enumerate(KS)}

    pred = [(xs[k], py(PREDICTED[k])) for k in KS]
    s.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pred) +
             f'" fill="none" stroke="{B.MUT}" stroke-width="4" stroke-dasharray="10 7"/>')
    meas = [(xs[k], py(gain[k])) for k in KS]
    s.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in meas) +
             f'" fill="none" stroke="{B.TEAL}" stroke-width="6" stroke-linejoin="round"/>')
    for k in KS:
        x = xs[k]
        s.append(f'<circle cx="{x:.1f}" cy="{py(PREDICTED[k]):.1f}" r="8" fill="{B.MUT}"/>')
        col = B.VRAM if k == 1 else B.TEAL       # k=1 is the arm that beat the ceiling
        s.append(f'<circle cx="{x:.1f}" cy="{py(gain[k]):.1f}" r="13" fill="{col}"/>')
        # The k=8 value sits on the axis and a centred label crowds the tick numbers, so the
        # baseline label starts to the right of its point instead of straddling it.
        anch, lx = ("start", x + 16) if k == KS[0] else ("middle", x)
        s.append(f'<text x="{lx:.1f}" y="{py(gain[k])-26:.1f}" text-anchor="{anch}" '
                 f'fill="{B.INK}" font-size="22" font-weight="bold">{gain[k]:.2f}x</text>')
        s.append(f'<text x="{x:.1f}" y="{pb+34:.1f}" text-anchor="middle" fill="{B.MUT}" '
                 f'font-size="20">k={k}</text>')

    # ---------- right: what it costs
    s.append(B.panel(RX, GY, RW, GH, stroke=B.DISK, sw=2))
    s.append(f'<text x="{RX+28}" y="{GY+36}" fill="{B.DISK}" font-size="19" letter-spacing="3">'
             f'PERPLEXITY &#183; LOG SCALE &#183; LOWER IS BETTER</text>')
    ql, qr = RX + 110, RX + RW - 50
    lo, hi = math.log10(5.0), math.log10(4000.0)

    def qy(v):
        return pb - (math.log10(v) - lo) / (hi - lo) * (pb - pt)

    for v in (10, 100, 1000):
        s.append(f'<line x1="{ql}" y1="{qy(v):.1f}" x2="{qr}" y2="{qy(v):.1f}" '
                 f'stroke="{B.GRID}" stroke-width="1.5"/>')
        s.append(f'<text x="{ql-16}" y="{qy(v)+7:.1f}" text-anchor="end" fill="{B.MUT}" '
                 f'font-size="18">{v}</text>')
    qstep = (qr - ql) / (len(KS) - 1)
    qxs = {k: ql + i * qstep for i, k in enumerate(KS)}
    qpts = [(qxs[k], qy(ppl[k])) for k in KS]
    s.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in qpts) +
             f'" fill="none" stroke="{B.DISK}" stroke-width="6" stroke-linejoin="round"/>')
    for k in KS:
        x = qxs[k]
        s.append(f'<circle cx="{x:.1f}" cy="{qy(ppl[k]):.1f}" r="13" fill="{B.DISK}"/>')
        lbl = f"{ppl[k]:,.0f}" if ppl[k] >= 100 else f"{ppl[k]:.2f}"
        s.append(f'<text x="{x:.1f}" y="{qy(ppl[k])-26:.1f}" text-anchor="middle" '
                 f'fill="{B.INK}" font-size="22" font-weight="bold">{lbl}</text>')
        s.append(f'<text x="{x:.1f}" y="{pb+34:.1f}" text-anchor="middle" fill="{B.MUT}" '
                 f'font-size="20">k={k}</text>')

    # ---------- the verdict
    VY = GY + GH + 34
    s.append(B.panel(80, VY, W - 160, 200, stroke=B.TEAL, sw=2))
    s.append(f'<text x="108" y="{VY+42}" fill="{B.TEAL}" font-size="19" letter-spacing="3">'
             f'THERE IS NO GOOD POINT ON THIS CURVE</text>')
    s.append(B.paragraph(
        108, VY + 82,
        f"Halving the experts buys {(gain[4]-1)*100:.0f}% - a gain most people would not notice - "
        f"for +{ppl[4]-ppl[8]:.2f} perplexity, which they would. Going to one expert of 256 buys "
        f"{(gain[1]-1)*100:.0f}% and the model is gone. All of it was readable before anything ran: "
        f"the routed experts own 22% of the active bytes, so 78% of the work is untouched no matter "
        f"what k does. quantprobe v1.30 prints that ceiling for your own file.",
        22, B.SUB, W - 216))

    # ---------- chips
    KY = VY + 200 + 30
    CW = (W - 160 - 2 * 24) / 3
    s.append(B.chip(80, KY, CW, "LAW 4 ERROR AT k=4 AND k=2", "+1.9% / -2.1%",
                    "predicted from the file, before measuring", B.TEAL))
    s.append(B.chip(80 + CW + 24, KY, CW, "WHAT k=4 COSTS", f"+{ppl[4]-ppl[8]:.2f} PPL",
                    f"for {(gain[4]-1)*100:.0f}% more speed", B.DISK))
    s.append(B.chip(80 + 2 * (CW + 24), KY, CW, "CEILING, FROM THE FILE ALONE", "1.24x",
                    "even at k=1 - 22% of bytes are routed", B.VRAM))

    s.append(B.footer(W, H, "prereg #107 &#183; scored 4/4 &#183; L-30, V-22"))
    s.append("</svg>")
    B.save("expert_dial.svg", "".join(s))


if __name__ == "__main__":
    main()
