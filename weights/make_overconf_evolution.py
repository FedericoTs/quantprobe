"""Overconfidence, cumulative - the estimate settling in front of you (atlas 34b).

  python weights/make_overconf_evolution.py

Parses phaseb_gen.log's periodic cumulative ledgers (every 100 attempts for the 4B, every 50
for the 30B) and plots the running wrong-expectation rate vs attempts. The story the endpoint
chart cannot tell: both curves flatten early and stay flat for thousands of attempts - the
54.7/55.7 split is a stable property, not endpoint noise. A third curve joins when B4b's
repair-loop v2 lands.
"""
from __future__ import annotations
import ast, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import brand
from brand import BG, INK, SUB, MUT, GRID, TEAL

LOG = os.path.join(HERE, "data", "phaseb_gen.log")


def curves():
    f1, f2 = [], []
    f1_final, f2_final = [False], [False]
    for line in open(LOG, encoding="utf-8"):
        m = re.search(r"\[feed1\] (\d+)/5000 kept \d+ dropped (\{.*\})", line)
        if m:
            n, d = int(m.group(1)), ast.literal_eval(m.group(2))
            den = n - d["no_fn"] - d["gen_fail"] - d["null_passed"]
            if den > 0:
                f1.append((n, 100 * d["ref_failed"] / den))
        m = re.search(r"\[feed2\] (\d+)/500 kept \d+ dropped (\{.*\})", line)
        if m:
            n, d = int(m.group(1)), ast.literal_eval(m.group(2))
            if "repaired" in d:
                continue                     # v2 curve joins after B4b - not this arm
            den = n - d["parse"] - d["no_fn"] - d.get("gen_fail", 0)
            if den > 0:
                f2.append((n, 100 * d["ref_failed"] / den))
        m = re.search(r"feed1: kept (\d+), dropped (\{.*\})", line)
        if m and int(m.group(1)) >= 1000 and not f1_final[0]:
            # kept-count gate: PROBE summaries (kept 2, zero drops) share this line shape
            # and the first version of this parser graphed them as a 0% endpoint -
            # caught on render review, the plunge to zero was the tell
            d = ast.literal_eval(m.group(2))
            den = 5000 - d["no_fn"] - d["gen_fail"] - d["null_passed"]
            f1.append((5000, 100 * d["ref_failed"] / den)); f1_final[0] = True
        m = re.search(r"feed2: kept (\d+), dropped (\{.*\})", line)
        if m and int(m.group(1)) >= 100 and "repaired" not in m.group(2) and not f2_final[0]:
            d = ast.literal_eval(m.group(2))
            den = 500 - d["parse"] - d["no_fn"] - d.get("gen_fail", 0)
            f2.append((500, 100 * d["ref_failed"] / den)); f2_final[0] = True
    f1.sort(); f2.sort()
    return f1, f2


def main():
    f1, f2 = curves()
    assert len(f1) >= 10 and len(f2) >= 3, "log missing checkpoints - refuse to render thin data"
    W, H = 1600, 1100
    s = [brand.svg_open(W, H),
         brand.header(W, "PHASE B DATA ENGINE - CUMULATIVE, FROM THE RUN LOG",
                      f'the overconfidence rate <tspan fill="{TEAL}">settles, and stays</tspan>',
                      "running share of self-written test sets with wrong expected values - "
                      "every checkpoint the run logged, nothing smoothed")]
    px0, px1, py0, py1 = 190, W - 90, H - 250, 340
    xmax = 5000.0
    ymin, ymax = 0.0, 80.0
    def X(n):
        return px0 + (px1 - px0) * n / xmax
    def Y(r):
        return py0 - (py0 - py1) * (r - ymin) / (ymax - ymin)
    for gy in range(0, 81, 20):
        s.append(f'<line x1="{px0}" y1="{Y(gy)}" x2="{px1}" y2="{Y(gy)}" stroke="{GRID}" stroke-width="1.5"/>'
                 f'<text x="{px0-18}" y="{Y(gy)+8}" text-anchor="end" fill="{MUT}" font-size="21">{gy}%</text>')
    for gx in (1000, 2000, 3000, 4000, 5000):
        s.append(f'<text x="{X(gx)}" y="{py0+40}" text-anchor="middle" fill="{MUT}" font-size="21">{gx:,}</text>')
    s.append(f'<text x="{(px0+px1)/2}" y="{py0+84}" text-anchor="middle" fill="{SUB}" font-size="22">test sets written (cumulative attempts)</text>')
    for pts, col, lw in ((f1, brand.VRAM, 5), (f2, brand.RAM, 5)):
        poly = " ".join(f"{X(n)},{Y(r)}" for n, r in pts)
        s.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="{lw}" stroke-linejoin="round"/>')
        n, r = pts[-1]
        s.append(f'<circle cx="{X(n)}" cy="{Y(r)}" r="12" fill="{col}" stroke="{BG}" stroke-width="4"/>')
    n1, r1 = f1[-1]
    n2, r2 = f2[-1]
    s.append(f'<text x="{X(n1)-16}" y="{Y(r1)-56}" text-anchor="end" fill="{brand.VRAM}" font-size="27" font-weight="bold">Qwen3.5-4B - {r1:.1f}%</text>')
    s.append(f'<text x="{X(n1)-16}" y="{Y(r1)-28}" text-anchor="end" fill="{SUB}" font-size="19">4,866 checked - 31 logged checkpoints</text>')
    s.append(f'<text x="{X(n2)+24}" y="{Y(r2)+50}" fill="{brand.RAM}" font-size="27" font-weight="bold">Qwen3-Coder-30B - {r2:.1f}%</text>')
    s.append(f'<text x="{X(n2)+24}" y="{Y(r2)+78}" fill="{SUB}" font-size="19">492 checked, its OWN solutions - 4 checkpoints, 7.5x larger</text>')
    s.append(f'<text x="{px0+14}" y="{py1-30}" fill="{INK}" font-size="29" font-weight="bold">two models, 7.5x apart in size, one flat line - overconfidence is not a scale problem</text>')
    s += [brand.footer(W, H, "parsed from weights/data/phaseb_gen.log checkpoints - v2 repair-loop curve joins when B4b lands"),
          '</svg>']
    brand.save("overconf_evolution.svg", "".join(s))


if __name__ == "__main__":
    main()
