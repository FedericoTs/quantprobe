"""The format ladder - same gigabytes, very different speed (atlas piece 15).

  python weights/make_format_ladder.py

Data: quantprobe/spec.py FORMAT_EBW - the MEASURED effective decode bandwidths per quant
format on the reference box (preregs #31/#52/#53/#58/#70/#77). Measured entries only; the
derived rows are excluded rather than mixed (cite-or-refuse applies to provenance class).
The story: format choice is a 2.6x speed decision at similar size, and IQ4_NL is the twist -
IQ by name, Q4-class by kernel.
"""
from __future__ import annotations
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import brand
from brand import INK, SUB, MUT, GRID, TEAL

# (format, GB/s, provenance prereg, codebook?) - measured rows of FORMAT_EBW only
ROWS = [
    ("Q4_0", 119.1, "#52/#53", False),
    ("IQ4_NL", 117.0, "#70", False),
    ("Q4_K", 106.4, "#52/#53", False),
    ("Q6_K", 100.0, "#58", False),
    ("IQ3_XXS", 68.3, "#70", True),
    ("Q2_K", 65.4, "#52/#53", False),
    ("IQ3_S", 61.1, "#70", True),
    ("Q3_K", 57.3, "#52/#53", False),
    ("IQ2_XS", 51.1, "#70", True),
    ("IQ2_XXS", 46.0, "#77", True),
]


def main():
    W, H = 1600, 1350
    s = [brand.svg_open(W, H),
         brand.header(W, "MEASURED ON ONE MACHINE - GTX 1060, PREREGISTERED",
                      f'your quant format is a <tspan fill="{TEAL}">2.6x speed decision</tspan>',
                      "effective decode bandwidth by format, e2e measured - the kernel class "
                      "matters more than the label")]
    s.append(f'<text x="80" y="306" fill="{SUB}" font-size="22">'
             f'<tspan fill="{brand.VRAM}" font-weight="bold">bandwidth-shaped kernels</tspan>'
             f'    <tspan fill="{brand.DISK}" font-weight="bold">codebook kernels (a lookup every decode)</tspan>'
             f'    <tspan fill="{TEAL}" font-weight="bold">the twist: IQ4_NL - IQ by name, Q4-class kernel</tspan></text>')
    x0, bar_h, gap, top = 320, 56, 22, 360
    xmax = 130.0
    def XW(v):
        return (W - x0 - 220) * v / xmax
    for gx in (25, 50, 75, 100, 125):
        gxp = x0 + XW(gx)
        s.append(f'<line x1="{gxp}" y1="{top-14}" x2="{gxp}" y2="{top + len(ROWS)*(bar_h+gap)}" stroke="{GRID}" stroke-width="1.5"/>'
                 f'<text x="{gxp}" y="{top + len(ROWS)*(bar_h+gap) + 40}" text-anchor="middle" fill="{MUT}" font-size="21">{gx}</text>')
    for i, (fmt, bw, src, cb) in enumerate(ROWS):
        y = top + i * (bar_h + gap)
        col = brand.DISK if cb else (TEAL if fmt == "IQ4_NL" else brand.VRAM)
        s.append(f'<text x="{x0-24}" y="{y+bar_h/2+9}" text-anchor="end" fill="{INK}" font-size="27" font-weight="bold">{fmt}</text>')
        s.append(f'<rect x="{x0}" y="{y}" width="{XW(bw)}" height="{bar_h}" rx="8" fill="{col}"/>')
        s.append(f'<text x="{x0+XW(bw)+18}" y="{y+bar_h/2+10}" fill="{INK}" font-size="26" font-weight="bold">{bw:g}</text>')
        s.append(f'<text x="{x0+XW(bw)+95}" y="{y+bar_h/2+9}" fill="{MUT}" font-size="18">GB/s - prereg {src}</text>')
    s.append(f'<text x="{x0 + XW(65)}" y="{top + len(ROWS)*(bar_h+gap) + 74}" text-anchor="middle" '
             f'fill="{SUB}" font-size="21">effective decode bandwidth, GB/s</text>')
    s += [brand.footer(W, H, "e2e measured, r>=3, one machine state per set - preregistrations #31/#52/#53/#58/#70/#77"),
          '</svg>']
    brand.save("format_ladder.svg", "".join(s))


if __name__ == "__main__":
    main()
