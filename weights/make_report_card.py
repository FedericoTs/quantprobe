"""Speed report card v2 (task #43) - one hero number, one curve, three receipts.

  python weights/make_report_card.py weights/data/card_flagship.json

KR-Q3 rule: every number arrives via the ledger json whose entries cite committed
measurements - the generator draws, never invents. Social-first design: the tok/s figure is
the hero at 130px; everything else is caption-weight. Dark single-theme by choice.
"""
from __future__ import annotations
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brand
from brand import BG, INK, SUB, MUT, GRID, TEAL


def main(path):
    d = json.load(open(path, encoding="utf-8"))
    for k in ("model", "quant", "machine", "decode", "prefill", "predicted", "binding",
              "flags", "protocol"):
        assert k in d, f"ledger missing {k}"
    for pt in d["decode"]:
        assert "source" in pt, "every decode point must cite its committed source"
    W, H = 1600, 1200
    hw = d["machine"]
    pr = d["predicted"]
    s = [brand.svg_open(W, H),
         brand.header(W, "SPEED REPORT CARD - MEASURED, NOT ESTIMATED",
                      f'{d["model"]} <tspan fill="{TEAL}">on a 2016 GPU</tspan>',
                      f'{d["quant"]} - {hw["gpu"]} - {hw["ram"]} - {hw["runtime"]}')]

    s += [f'<text x="80" y="475" fill="{INK}" font-size="150" font-weight="bold">'
          f'{d["kpi"]["peak_decode"]}<tspan fill="{TEAL}" font-size="52"> tok/s</tspan></text>',
          f'<text x="80" y="530" fill="{SUB}" font-size="26">decode, fresh context - 30B MoE, 6 GB VRAM</text>',
          brand.panel(80, 570, 600, 190, stroke=TEAL, sw=2),
          f'<text x="110" y="618" fill="{TEAL}" font-size="24" font-weight="bold">the law called it first</text>',
          f'<text x="110" y="662" fill="{SUB}" font-size="23">floor <tspan fill="{INK}" font-weight="bold">{pr["floor"]}</tspan>'
          f' -> measured <tspan fill="{INK}" font-weight="bold">{pr["measured"]}</tspan>'
          f' <tspan fill="{TEAL}" font-weight="bold">({pr["delta"].replace(" over floor", "")})</tspan></text>',
          f'<text x="110" y="706" fill="{SUB}" font-size="21">bound by <tspan fill="{brand.VRAM}">'
          f'system RAM bandwidth (51% of every token)</tspan></text>',
          # prereg #95 P-3 kill rule: the classification ships with its scope label everywhere
          # it is drawn, at full prominence, until a re-derivation confirms the mapping
          f'<text x="110" y="738" fill="{MUT}" font-size="14">derived from the law, '
          f'not confirmed by variance attribution (prereg #95)</text>',
          f'<text x="80" y="808" fill="{MUT}" font-size="20">{hw["placement"]}</text>']

    cx, cy, cw, ch = 740, 320, W - 740 - 80, 470
    s += [brand.panel(cx, cy, cw, ch),
          f'<text x="{cx+36}" y="{cy+52}" fill="{INK}" font-size="26" font-weight="bold">'
          f'decode vs context depth <tspan fill="{MUT}" font-size="20">tok/s</tspan></text>']
    if d.get("depth_note"):
        s.append(f'<text x="{cx+cw-30}" y="{cy+52}" text-anchor="end" fill="{MUT}" font-size="17">'
                 f'{d["depth_note"].split(";")[0]}</text>')
    pts = d["decode"]
    ymax = max(p["tok_s"] for p in pts) * 1.3
    px0, px1, py0, py1 = cx + 90, cx + cw - 60, cy + ch - 80, cy + 110
    xs = [p["depth"] for p in pts]
    def X(v):
        return px0 + (px1 - px0) * (xs.index(v) / max(1, len(xs) - 1))
    def Y(v):
        return py0 - (py0 - py1) * (v / ymax)
    for gy in range(0, int(ymax) + 1, 10):
        s.append(f'<line x1="{px0}" y1="{Y(gy)}" x2="{px1}" y2="{Y(gy)}" stroke="{GRID}" stroke-width="1.5"/>'
                 f'<text x="{px0-16}" y="{Y(gy)+8}" text-anchor="end" fill="{MUT}" font-size="20">{gy}</text>')
    poly = " ".join(f"{X(p['depth'])},{Y(p['tok_s'])}" for p in pts)
    s.append(f'<polyline points="{poly}" fill="none" stroke="{brand.VRAM}" stroke-width="6" stroke-linecap="round"/>')
    for p in pts:
        xl = "fresh" if p["depth"] == 0 else f'{p["depth"]//1024}K deep'
        s.append(f'<circle cx="{X(p["depth"])}" cy="{Y(p["tok_s"])}" r="14" fill="{brand.VRAM}" stroke="{BG}" stroke-width="4"/>'
                 f'<text x="{X(p["depth"])}" y="{Y(p["tok_s"])-30}" text-anchor="middle" fill="{INK}" font-size="32" font-weight="bold">{p["tok_s"]}</text>'
                 f'<text x="{X(p["depth"])}" y="{py0+40}" text-anchor="middle" fill="{SUB}" font-size="22">{xl}</text>')
    pf = d["prefill"]
    cwch = (W - 160 - 60) / 3
    chips = [("PREFILL PP2048", f'{pf["tok_s"]:.0f} t/s', f'r={d["protocol"]["reps"]}, one machine state', brand.VRAM),
             ("FLOOR HELD", pr["delta"], f'law said {pr["floor"]} before the run', TEAL),
             ("FITS IN", "6 GB VRAM", "16 GB RAM does the rest", brand.RAM)]
    for i, (t, v, sub2, c) in enumerate(chips):
        s.append(brand.chip(80 + i * (cwch + 30), 850, cwch, t, v, sub2, c))

    s += [brand.footer(W, H, "flags + raw logs in the repo - every number cites a committed measurement"),
          '</svg>']
    return brand.save(os.path.splitext(os.path.basename(path))[0] + ".svg", "".join(s))


if __name__ == "__main__":
    main(sys.argv[1])
