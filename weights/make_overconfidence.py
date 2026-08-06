"""Test-writer overconfidence is size-independent (atlas 34, Phase B verdict asset).

  python weights/make_overconfidence.py
"""
from __future__ import annotations
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import brand
from brand import INK, SUB, MUT, TEAL


def main():
    W, H = 1600, 1090
    s = [brand.svg_open(W, H),
         brand.header(W, "PHASE B DATA ENGINE - MEASURED 2026-08-06",
                      f'scale does not fix <tspan fill="{TEAL}">hallucinated tests</tspan>',
                      "share of self-written test sets asserting WRONG expected values for "
                      "their own correct solution - same gates, same sandbox")]
    bars = [("Qwen3.5-4B", "writes tests for others' code", 54.7, 4866, brand.VRAM),
            ("Qwen3-Coder-30B", "writes tests for its OWN code - 7.5x larger", 55.7, 492, brand.RAM)]
    bx, bw, gap, base_y, maxh = 260, 420, 240, 760, 380
    for i, (name, sub2, pct, n, col) in enumerate(bars):
        x = bx + i * (bw + gap)
        h = maxh * pct / 60.0
        s.append(f'<rect x="{x}" y="{base_y-h}" width="{bw}" height="{h}" rx="14" fill="{col}"/>')
        s.append(f'<text x="{x+bw/2}" y="{base_y-h-28}" text-anchor="middle" fill="{INK}" font-size="64" font-weight="bold">{pct}%</text>')
        s.append(f'<text x="{x+bw/2}" y="{base_y+46}" text-anchor="middle" fill="{INK}" font-size="30" font-weight="bold">{name}</text>')
        s.append(f'<text x="{x+bw/2}" y="{base_y+82}" text-anchor="middle" fill="{SUB}" font-size="21">{sub2}</text>')
        s.append(f'<text x="{x+bw/2}" y="{base_y+114}" text-anchor="middle" fill="{MUT}" font-size="19">n = {n:,} test sets</text>')
    s.append(f'<text x="{W/2}" y="912" text-anchor="middle" fill="{TEAL}" font-size="30" font-weight="bold">the fix is not a bigger model - it is running the tests</text>')
    s.append(f'<text x="{W/2}" y="948" text-anchor="middle" fill="{SUB}" font-size="20">every failing set was caught by execution and never trained on - the null and mutation gates held</text>')
    s += [brand.footer(W, H, "weights/data/phaseb_gen.log + phaseb_verdict.json - prereg 2026-08-05-phase-b-data-engine"),
          '</svg>']
    brand.save("overconfidence.svg", "".join(s))


if __name__ == "__main__":
    main()
