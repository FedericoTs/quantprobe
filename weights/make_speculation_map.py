"""Speculation is not a switch - twelve measured cells on ONE machine, 2.41x to 0.61x.

Every cell here was measured on the reference box (GTX 1060 6GB + DDR4-3000, llama-server,
temp 0), which is the point: the 4x swing is not hardware variance, it is the CELL. Placement,
drafter kind and workload each flip the sign, and llama.cpp's own default draft length sits
among the losses on the very pair where K=2 pays +33.5%.

Bars are drawn on a LOG scale about 1.00x because these are ratios: on a linear axis a 0.61x
looks like a small dent next to a 2.41x, when multiplicatively it is nearly as far from
break-even as 1.64x is on the other side. The printed number is the measurement; the geometry
just has to not lie about it.

Sources are register ids and prereg numbers, one per row - mixing boxes or protocols silently
is the failure this footer exists to prevent.
"""
from __future__ import annotations
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import brand as B                                            # noqa: E402

W = 1600
PAD = 80
# Columns are laid out so nothing can collide: text, then a bar-only lane, then a fixed gutter
# for the numbers. The first attempt let loss bars grow leftward out of the lane and straight
# through the detail text, and put each label beside its own bar - so the labels moved with the
# data and overlapped it. A bar that grows into a text column is not a layout to nudge.
BAR_L, BAR_R = 720, 1380     # the lane bars may occupy, and nothing else
NUM_X = 1500                 # every multiplier prints here, right-aligned, whatever the bar does
SPAN = 333                   # px per log2 unit; sized so 0.61x and 2.41x both land inside the lane
MID = 956                    # x of the 1.00x break-even line
ROWH = 46

# (multiplier, drafter, placement + workload, source). Reference box throughout.
CELLS = [
    (2.41, "n-gram draft",        "30B MoE split · edit/copy task · 89% acceptance", "V-04 · #28"),
    (1.335, "0.6B draft, K=2",    "dense target all-in-VRAM · novel text",           "V-20 · #67"),
    (1.114, "MTP head, K=2",      "35B MoE split · 93.2% acceptance",                "#71"),
    (1.03, "n-gram-mod",          "30B MoE split · novel · drafted nothing",         "D-10"),
    (1.00, "n-gram draft",        "30B MoE split · novel prose",                     "V-04 · #28"),
    (0.98, "n-gram draft",        "30B MoE split · novel code",                      "V-04 · #28"),
    (0.93, "n-gram-cache",        "30B MoE split · novel text",                      "D-10"),
    (0.79, "0.6B draft",          "30B MoE split · novel prose · 81-83% accepted",   "D-09 · #42"),
    (0.76, "MTP head",            "MoE split · previous generation",                 "Law 6 arm S-e"),
    (0.72, "0.6B draft",          "30B MoE split · novel code · 81-83% accepted",    "D-09 · #42"),
    (0.64, "0.6B draft, K=3",     "dense all-in-VRAM · novel · llama.cpp DEFAULT",   "V-20 · #67"),
    (0.61, "MTP head, K=4",       "35B MoE split · the union tax",                   "#71"),
]

DEFAULT_ROW = 0.64           # the one a reader can act on tonight


def x_of(mult):
    return MID + math.log2(mult) * SPAN


def main():
    H = B.HEADER_H + 176 + len(CELLS) * ROWH + 176 + B.FOOTER_H
    best, worst = max(c[0] for c in CELLS), min(c[0] for c in CELLS)

    s = [B.svg_open(W, H),
         B.header(W, "MEASURED · GTX 1060 6GB + DDR4 · llama-server · TEMP 0 · ONE BOX",
                  "Speculation is not a switch",
                  "Twelve cells, one machine. Placement, drafter and workload each flip the "
                  "sign - so “turn on speculative decoding” is not advice.")]

    y = B.HEADER_H + 16
    s.append(B.panel(PAD, y, W - 2 * PAD, 132, stroke=B.TEAL, sw=2))
    s.append(f'<text x="{PAD+38}" y="{y+44}" fill="{B.MUT}" font-size="19" letter-spacing="3">'
             f'SAME BOX, SAME SERVER, SAME TEMPERATURE</text>')
    s.append(f'<text x="{PAD+38}" y="{y+106}" fill="{B.TEAL}" font-size="62" '
             f'font-weight="bold">{best:.2f}x</text>')
    s.append(f'<text x="{PAD+212}" y="{y+106}" fill="{B.SUB}" font-size="26">down to</text>')
    s.append(f'<text x="{PAD+330}" y="{y+106}" fill="{B.DISK}" font-size="62" '
             f'font-weight="bold">{worst:.2f}x</text>')
    s.append(f'<text x="{PAD+510}" y="{y+106}" fill="{B.SUB}" font-size="24">'
             f'- a {best/worst:.1f}x swing that has nothing to do with the hardware</text>')
    y += 176

    top = y
    # break-even line first, so bars sit on top of it
    s.append(f'<line x1="{MID}" y1="{top-16}" x2="{MID}" y2="{top + len(CELLS)*ROWH + 4}" '
             f'stroke="{B.EDGE}" stroke-width="2"/>')
    s.append(f'<text x="{MID}" y="{top-26}" text-anchor="middle" fill="{B.MUT}" '
             f'font-size="19">1.00x  no speculation</text>')

    for mult, drafter, detail, src in CELLS:
        win = mult > 1.0
        col = B.TEAL if mult > 1.02 else (B.DISK if mult < 0.98 else B.MUT)
        x = x_of(mult)
        x0, x1 = (MID, x) if win else (x, MID)
        s.append(f'<rect x="{x0:.1f}" y="{y+9}" width="{max(x1-x0, 2):.1f}" height="26" rx="5" '
                 f'fill="{col}" opacity="{1 if mult == DEFAULT_ROW else 0.9}"/>')
        s.append(f'<text x="{NUM_X}" y="{y+29}" text-anchor="end" fill="{B.INK}" '
                 f'font-size="21" font-weight="bold">{mult:.2f}x</text>')
        s.append(f'<text x="{PAD}" y="{y+29}" fill="{B.INK}" font-size="20" '
                 f'font-family="Consolas, Menlo, monospace">{drafter}</text>')
        # Provenance rides with the row rather than in a column of its own - a separate
        # right-hand column is what the bars kept running into.
        s.append(f'<text x="{PAD+230}" y="{y+29}" fill="{B.MUT}" font-size="16">{detail}'
                 f'<tspan fill="{B.EDGE}">  · {src}</tspan></text>')
        if mult == DEFAULT_ROW:
            s.append(f'<rect x="{PAD-10}" y="{y+2}" width="{W-2*PAD+20}" height="40" rx="8" '
                     f'fill="none" stroke="{B.DISK}" stroke-width="2"/>')
        y += ROWH

    y += 44
    s.append(f'<text x="{PAD}" y="{y}" fill="{B.INK}" font-size="26" font-weight="bold">'
             f'The one to act on</text>')
    s.append(B.paragraph(PAD, y + 40,
                         "On the same model pair, a draft length of 2 pays +33.5% and llama.cpp's "
                         "default of 3 measures 0.64x - a loss. The knob that decides whether "
                         "speculation helps you is the one most people never touch.",
                         21, B.SUB, W - 2 * PAD))

    s.append(B.footer(W, H, "preregs #28 / #42 / #67 / #71 · register V-04, V-20, D-09, D-10"))
    s.append("</svg>")
    B.save("speculation_map.svg", "".join(s))


if __name__ == "__main__":
    main()
