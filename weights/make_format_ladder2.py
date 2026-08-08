"""The format ladder - same gigabytes, 2.6x the decode speed, and the popular rule is wrong.

Renders FORMAT_EBW straight from quantprobe/spec.py (cite-or-refuse: the chart cannot state a
number the tool does not ship). The headline is the spread; the useful part is the correction.
"IQ is slow" is a rule people apply wholesale, and it is false - IQ4_NL measures 117.0, beside
Q4_0's 119.1. The real divide is CODEBOOK vs not (L-15 amendment, prereg #70), and the
mechanism under the K-quant deficit is metadata application DENSITY (L-16).
"""
from __future__ import annotations
import os, sys
import brand as B

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantprobe.spec import FORMAT_EBW                       # noqa: E402

W, H = 1600, 1400
X0, XW = 300, 830          # bar origin and full-scale width
TOP, ROWH = 300, 40

CODEBOOK = {"IQ2_XXS", "IQ2_XS", "IQ3_S", "IQ3_XXS"}         # grid lookup in the decode path
NOUNPACK = {"F16", "F32", "BF16"}


def klass(name, v):
    if name in NOUNPACK:
        return "no unpack at all", B.MUT
    if name in CODEBOOK:
        return "codebook lookup", B.DISK
    if name.startswith("Q") and "_K" in name:
        return "K-quant: fine metadata", B.VRAM
    return "simple unpack (dp4a)", B.TEAL


def main():
    rows = sorted(FORMAT_EBW.items(), key=lambda kv: -kv[1])
    top = max(v for _, v in rows)
    fastq = max(v for k, v in rows if k not in NOUNPACK)
    slowest = min(v for _, v in rows)
    spread = fastq / slowest

    s = [B.svg_open(W, H),
         B.header(W, "MEASURED · GTX 1060 6GB · EFFECTIVE DECODE BANDWIDTH PER FORMAT",
                  "Same gigabytes, 2.6x the speed",
                  "A quantization is not just a size. On an ALU-limited GPU the format sets "
                  "decode speed as much as the byte count does.")]

    for i, (name, v) in enumerate(rows):
        y = TOP + i * ROWH
        lab, col = klass(name, v)
        bw = v / top * XW
        s.append(f'<text x="{X0-22}" y="{y+24}" text-anchor="end" fill="{B.INK}" '
                 f'font-size="21" font-weight="bold" '
                 f'font-family="Consolas, Menlo, monospace">{name}</text>')
        s.append(f'<rect x="{X0}" y="{y+4}" width="{bw:.1f}" height="30" rx="6" fill="{col}" '
                 f'opacity="{0.55 if name in NOUNPACK else 1}"/>')
        s.append(f'<text x="{X0+bw+14}" y="{y+26}" fill="{B.INK}" font-size="20" '
                 f'font-weight="bold">{v:g}</text>')
        s.append(f'<text x="{X0+bw+82}" y="{y+26}" fill="{B.MUT}" font-size="17">GB/s</text>')
        if name.startswith("IQ"):
            s.append(f'<text x="{X0+XW+150}" y="{y+26}" fill="{col}" font-size="17" '
                     f'font-weight="bold">IQ</text>')

    yb = TOP + len(rows) * ROWH

    # legend: the four kernel classes, in speed order
    lx = X0
    for lab, col in (("no unpack", B.MUT), ("simple unpack (dp4a)", B.TEAL),
                     ("K-quant: fine metadata", B.VRAM), ("codebook lookup", B.DISK)):
        s.append(f'<rect x="{lx}" y="{yb+16}" width="20" height="20" rx="4" fill="{col}"/>')
        s.append(f'<text x="{lx+28}" y="{yb+33}" fill="{B.SUB}" font-size="18">{lab}</text>')
        lx += 40 + len(lab) * 9.6

    # the two claims, side by side
    PY = yb + 62
    s.append(B.panel(92, PY, 700, 150, stroke=B.TEAL, sw=2))
    s.append(f'<text x="126" y="{PY+38}" fill="{B.TEAL}" font-size="19" '
             f'letter-spacing="3">THE SPREAD</text>')
    s.append(f'<text x="126" y="{PY+92}" fill="{B.INK}" font-size="46" '
             f'font-weight="bold">{spread:.1f}x</text>')
    s.append(f'<text x="126" y="{PY+126}" fill="{B.SUB}" font-size="20">'
             f'Q4_0 {fastq:g} vs IQ2_XXS {slowest:g} GB/s at comparable size.</text>')

    s.append(B.panel(816, PY, W - 816 - 92, 150, stroke=B.DISK, sw=2))
    s.append(f'<text x="850" y="{PY+38}" fill="{B.DISK}" font-size="19" '
             f'letter-spacing="3">AND THE RULE EVERYONE USES IS WRONG</text>')
    s.append(f'<text x="850" y="{PY+76}" fill="{B.INK}" font-size="22">'
             f'"IQ is slow" is false. <tspan font-weight="bold">IQ4_NL measures 117.0</tspan>'
             f' - beside Q4_0.</text>')
    s.append(f'<text x="850" y="{PY+106}" fill="{B.SUB}" font-size="20">'
             f'The divide is CODEBOOK vs not, not IQ vs K. Codebook formats</text>')
    s.append(f'<text x="850" y="{PY+134}" fill="{B.SUB}" font-size="20">'
             f'pay a grid lookup inside the decode loop; IQ4_NL does not.</text>')

    s.append(f'<text x="92" y="{PY+186}" fill="{B.MUT}" font-size="17">'
             f'Mechanism (L-16): the K-quant deficit is metadata application DENSITY - scale+min '
             f'per 16 weights forces a metadata FMA every 4 bytes at 2 bits. Confirmation arm, '
             f'identical loads and dp4a count,</text>')
    s.append(f'<text x="92" y="{PY+210}" fill="{B.MUT}" font-size="17">'
             f'scale applied per-u32 instead of per-quad: 83.8 -> 103.2 GB/s (+23%). '
             f'Scope: one Pascal card; these are upper bounds carrying the measured model\'s FFN '
             f'shape and row width.</text>')

    s.append(B.footer(W, H, "quantprobe/spec.py FORMAT_EBW · L-15/L-16 · preregs #52/#70"))
    s.append("</svg>")
    B.save("format_ladder_v2.svg", "".join(s))


if __name__ == "__main__":
    main()
