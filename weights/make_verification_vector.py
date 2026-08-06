"""The verification vector v2 - accuracy vs wall-clock, single -> lanes16 arrows per model.

  python weights/make_verification_vector.py

Design: the geometry IS the hero - four fat arrows, the 4B's crossing the 30B's dashed
line. One takeaway sentence, three labels per model, nothing under 18px. Data: committed
Phase A grid JSONs only (cite-or-refuse).
"""
from __future__ import annotations
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import brand
from brand import BG, INK, SUB, MUT, GRID, TEAL

DATA = os.path.join(HERE, "data")
MODELS = [("0.6B", "0.6B", "#7d879c"), ("7B", "7B", brand.VRAM),
          ("4B", "4B", brand.TEAL), ("30B", "30B coder", brand.RAM)]


def cell(bench, model, mode):
    p = os.path.join(DATA, f"grid_{bench}_{model}_{mode}.json")
    if not os.path.isfile(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    return None if d.get("unrunnable") else (d["median_wall"], d["rate"] * 100)


def main(bench, bench_label, takeaway):
    W, H = 1600, 1200
    s = [brand.svg_open(W, H),
         brand.header(W, "ONE 2016 GPU - GTX 1060 6GB",
                      f'16 verified tries <tspan fill="{TEAL}">beat model size</tspan>',
                      f"{bench_label} - arrows: single shot -> best-of-16 (picked on visible tests, scored on hidden)")]
    px0, px1, py0, py1 = 190, W - 110, H - 260, 350
    xmin, xmax = 1, 60
    ymin, ymax = 15, 97
    def X(w):
        return px0 + (px1 - px0) * (math.log10(w / xmin)) / math.log10(xmax / xmin)
    def Y(r):
        return py0 - (py0 - py1) * (r - ymin) / (ymax - ymin)
    for gy in range(20, 100, 20):
        s.append(f'<line x1="{px0}" y1="{Y(gy)}" x2="{px1}" y2="{Y(gy)}" stroke="{GRID}" stroke-width="1.5"/>'
                 f'<text x="{px0-20}" y="{Y(gy)+8}" text-anchor="end" fill="{MUT}" font-size="22">{gy}%</text>')
    for gx in (1, 3, 10, 30):
        s.append(f'<text x="{X(gx)}" y="{py0+42}" text-anchor="middle" fill="{MUT}" font-size="22">{gx}s</text>')
    s.append(f'<text x="{(px0+px1)/2}" y="{py0+88}" text-anchor="middle" fill="{SUB}" font-size="23">median seconds per task (log) - all 16 candidates + selection charged</text>')
    s.append(f'<text x="115" y="{(py0+py1)/2}" fill="{SUB}" font-size="23" transform="rotate(-90 115 {(py0+py1)/2})" text-anchor="middle">tasks solved</text>')
    s.append('<defs><marker id="arr" viewBox="0 0 12 12" refX="9" refY="6" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 12 6 L 0 12 z" fill="context-stroke"/></marker></defs>')
    t30 = cell(bench, "30B", "single")
    if t30:
        s.append(f'<line x1="{px0}" y1="{Y(t30[1])}" x2="{px1}" y2="{Y(t30[1])}" stroke="{brand.RAM}" stroke-width="2.5" stroke-dasharray="10,8" opacity="0.8"/>')
        s.append(f'<text x="{px0+14}" y="{Y(t30[1])-16}" fill="{brand.RAM}" font-size="21" opacity="0.9">the 30B single-shot line</text>')
    # per-point label placement, hand-set to de-collide the 7-10s cluster (reviewed on render)
    NAME_OFF = {"0.6B": (0, 48, "middle"), "4B": (-34, 10, "end"),
                "7B": (30, 52, "start"), "30B": (26, 46, "start")}
    PCT_OFF = {"0.6B": (0, -24, "middle"), "4B": (-34, -22, "end"),
               "7B": (32, 22, "start"), "30B": (26, -22, "start")}
    for key, name, col in MODELS:
        a, b = cell(bench, key, "single"), cell(bench, key, "lanes")
        if a is None:
            continue
        ax, ay = X(a[0]), Y(a[1])
        if b:
            bx, by = X(b[0]), Y(b[1])
            s.append(f'<line x1="{ax}" y1="{ay}" x2="{bx}" y2="{by}" stroke="{col}" stroke-width="7" stroke-linecap="round" marker-end="url(#arr)"/>')
            s.append(f'<circle cx="{bx}" cy="{by}" r="15" fill="{col}" stroke="{BG}" stroke-width="4"/>')
            lane_dy = -30 if key != "7B" else 44
            s.append(f'<text x="{bx+4}" y="{by+lane_dy}" text-anchor="middle" fill="{INK}" font-size="34" font-weight="bold">{b[1]:.1f}%</text>')
        s.append(f'<circle cx="{ax}" cy="{ay}" r="11" fill="{BG}" stroke="{col}" stroke-width="4"/>')
        ndx, ndy, na = NAME_OFF[key]
        pdx, pdy, pa = PCT_OFF[key]
        s.append(f'<text x="{ax+ndx}" y="{ay+ndy}" text-anchor="{na}" fill="{col}" font-size="26" font-weight="bold">{name}</text>')
        s.append(f'<text x="{ax+pdx}" y="{ay+pdy}" text-anchor="{pa}" fill="{SUB}" font-size="23">{a[1]:.1f}%</text>')
    s += [f'<text x="{px0+14}" y="{py1-36}" fill="{INK}" font-size="30" font-weight="bold">{takeaway}</text>',
          brand.footer(W, H, "every point: weights/data/grid_*.json - misses published at full size"),
          '</svg>']
    brand.save(f"verification_vector_{bench}.svg", "".join(s))


if __name__ == "__main__":
    main("humaneval", "HumanEval+ (164 tasks, hidden plus tests)",
         "a 4B with verification lands above the 30B - on a card from 2016")
    main("mbpp", "MBPP+ (371 tasks, hidden plus tests)",
         "verification closes the size gap: every arrow crosses toward the 30B line")
