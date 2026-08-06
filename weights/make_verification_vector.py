"""The verification vector - accuracy vs wall-clock, single -> lanes16 arrows per model.

  python weights/make_verification_vector.py

The viral geometry: each model is two points (single shot, verified best-of-16) joined by an
arrow; the 30B sits alone (lanes excluded by U-39). The 4B's arrowhead lands ABOVE the 30B on
HumanEval+ - strategy beating size, visible without reading a number. Data: committed Phase A
grid JSONs only (cite-or-refuse).
"""
from __future__ import annotations
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import brand

DATA = os.path.join(HERE, "data")
MODELS = [("0.6B", "Qwen3-0.6B", brand.MUT), ("4B", "Qwen3.5-4B", brand.TEAL_HI),
          ("7B", "Qwen2.5-7B", brand.VRAM), ("30B", "Qwen3-Coder-30B", brand.RAM)]


def cell(bench, model, mode):
    p = os.path.join(DATA, f"grid_{bench}_{model}_{mode}.json")
    if not os.path.isfile(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    return None if d.get("unrunnable") else (d["median_wall"], d["rate"] * 100)


def main(bench="humaneval", label="HumanEval+ (164 tasks, hidden plus tests)"):
    W, H = 1500, 1120
    s = [brand.svg_open(W, H),
         brand.header(W, "THE VERIFICATION VECTOR - ONE 2016 MACHINE",
                      f'what <tspan fill="{brand.TEAL_HI}">16 verified tries</tspan> buy, per model',
                      f"{label} - GTX 1060 6GB - arrow: single shot -> verified best-of-16 "
                      f"(picked by base tests, scored on hidden tests)")]
    px0, px1, py0, py1 = 150, W - 90, H - 220, 270
    xmin, xmax = 1, 60
    ymin, ymax = 15, 95
    def X(w):
        return px0 + (px1 - px0) * (math.log10(w) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin))
    def Y(r):
        return py0 - (py0 - py1) * (r - ymin) / (ymax - ymin)
    for gy in range(20, 100, 10):
        s.append(f'<line x1="{px0}" y1="{Y(gy)}" x2="{px1}" y2="{Y(gy)}" stroke="{brand.EDGE}"/>'
                 f'<text x="{px0-14}" y="{Y(gy)+5}" text-anchor="end" fill="{brand.MUT}" font-size="15">{gy}%</text>')
    for gx in (1, 2, 5, 10, 20, 50):
        s.append(f'<line x1="{X(gx)}" y1="{py0}" x2="{X(gx)}" y2="{py1}" stroke="{brand.EDGE}" stroke-dasharray="2,6"/>'
                 f'<text x="{X(gx)}" y="{py0+30}" text-anchor="middle" fill="{brand.MUT}" font-size="15">{gx}s</text>')
    s.append(f'<text x="{(px0+px1)/2}" y="{py0+64}" text-anchor="middle" fill="{brand.SUB}" font-size="16">median wall-clock per task (log)</text>')
    s.append(f'<text x="{px0-90}" y="{(py0+py1)/2}" fill="{brand.SUB}" font-size="16" transform="rotate(-90 {px0-90} {(py0+py1)/2})" text-anchor="middle">solve rate</text>')
    s.append(f'<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>')
    for key, name, col in MODELS:
        a, b = cell(bench, key, "single"), cell(bench, key, "lanes")
        if a is None:
            continue
        ax, ay = X(a[0]), Y(a[1])
        if b:
            bx, by = X(b[0]), Y(b[1])
            s.append(f'<line x1="{ax}" y1="{ay}" x2="{bx}" y2="{by}" stroke="{col}" stroke-width="3.5" marker-end="url(#arr)" opacity="0.9"/>')
            s.append(f'<circle cx="{bx}" cy="{by}" r="11" fill="{col}" stroke="{brand.BG}" stroke-width="3"/>')
            s.append(f'<text x="{bx+2}" y="{by-20}" text-anchor="middle" fill="{brand.INK}" font-size="21" font-weight="bold">{b[1]:.1f}</text>')
        s.append(f'<circle cx="{ax}" cy="{ay}" r="8" fill="{brand.BG}" stroke="{col}" stroke-width="3"/>')
        s.append(f'<text x="{ax}" y="{ay+34}" text-anchor="middle" fill="{col}" font-size="16" font-weight="bold">{name}</text>')
        s.append(f'<text x="{ax}" y="{ay-16}" text-anchor="middle" fill="{brand.SUB}" font-size="16">{a[1]:.1f}</text>')
    tb = cell(bench, "30B", "single")
    if tb:
        s.append(f'<line x1="{px0}" y1="{Y(tb[1])}" x2="{px1}" y2="{Y(tb[1])}" stroke="{brand.RAM}" stroke-width="1.5" stroke-dasharray="8,6" opacity="0.7"/>')
        s.append(f'<text x="{px1}" y="{Y(tb[1])-10}" text-anchor="end" fill="{brand.RAM}" font-size="15">30B single-shot line - crossed by a 4B with verification</text>')
    s += [f'<text x="{px0+10}" y="{py1-24}" fill="{brand.SUB}" font-size="15">hollow dot = single shot - filled dot = verified best-of-16 - 30B lanes excluded by staked prior evidence (U-39)</text>',
          brand.footer(W, H, f"every point cites weights/data/grid_{bench}_*.json - wall-clock charged honestly: all 16 candidates + selection execution"),
          '</svg>']
    brand.save(f"verification_vector_{bench}.svg", "".join(s))


if __name__ == "__main__":
    main("humaneval", "HumanEval+ (164 tasks, hidden plus tests)")
    main("mbpp", "MBPP+ (371 tasks, hidden plus tests)")
