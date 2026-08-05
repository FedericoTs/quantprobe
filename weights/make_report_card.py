"""Speed report card generator (task #43) - the dark, screenshot-native per-run card.

  python weights/make_report_card.py weights/data/card_flagship.json

Renders an X-ready SVG from a LEDGER JSON only - the KR-Q3 rule from the quality-ladder
stake applies here too: every number on the card must arrive via the json, whose values must
cite committed measurements in their "source" fields. The generator draws; it never invents.
Deliberately dark single-theme (an X screenshot asset, not a doc page - the one sanctioned
exception to the both-themes rule, chosen, not omitted).
"""
from __future__ import annotations
import json, os, sys

BG, PANEL, EDGE = "#0c0d12", "#14161f", "#262a3a"
INK, SUB, MUT = "#f2f2f0", "#b9bcc9", "#7c8093"
PURPLE, ORANGE, TEAL, BLUE = "#a99df2", "#f5a623", "#5dcaa5", "#85b7eb"


def badge(x, y, w, title, value, sub, color):
    return (f'<g><rect x="{x}" y="{y}" width="{w}" height="150" rx="14" fill="{PANEL}" '
            f'stroke="{color}" stroke-width="1.5"/>'
            f'<text x="{x+w/2}" y="{y+38}" text-anchor="middle" fill="{color}" '
            f'font-size="15" letter-spacing="2">{title}</text>'
            f'<text x="{x+w/2}" y="{y+88}" text-anchor="middle" fill="{INK}" '
            f'font-size="40" font-weight="bold">{value}</text>'
            f'<text x="{x+w/2}" y="{y+122}" text-anchor="middle" fill="{SUB}" '
            f'font-size="14">{sub}</text></g>')


def main(path):
    d = json.load(open(path, encoding="utf-8"))
    for k in ("model", "quant", "machine", "decode", "prefill", "predicted", "binding",
              "flags", "protocol"):
        assert k in d, f"ledger missing {k} - the card renders committed data only"
    for pt in d["decode"]:
        assert "source" in pt, "every decode point must cite its committed source"
    W, H = 1500, 1120
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{BG}"/>',
         f'<text x="60" y="66" fill="{MUT}" font-size="17" letter-spacing="6">SPEED BENCHMARK REPORT CARD</text>',
         f'<text x="60" y="126" fill="{INK}" font-size="52" font-weight="bold">{d["model"]} '
         f'<tspan fill="{PURPLE}">{d["quant"]}</tspan></text>',
         f'<rect x="60" y="150" width="640" height="44" rx="22" fill="{PANEL}" stroke="{EDGE}"/>',
         f'<text x="380" y="178" text-anchor="middle" fill="{SUB}" font-size="17">{d["config_strip"]}</text>']
    hw = d["machine"]
    s += [f'<rect x="{W-430}" y="40" width="370" height="150" rx="14" fill="{PANEL}" stroke="{EDGE}"/>',
          f'<text x="{W-410}" y="78" fill="{TEAL}" font-size="20" font-weight="bold">{hw["gpu"]}</text>',
          f'<text x="{W-410}" y="108" fill="{BLUE}" font-size="17">{hw["ram"]}</text>',
          f'<text x="{W-410}" y="138" fill="{SUB}" font-size="15">{hw["runtime"]}</text>',
          f'<text x="{W-410}" y="166" fill="{MUT}" font-size="13">{hw["placement"]}</text>']

    cx, cy, cw, ch = 60, 250, 780, 470
    s += [f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="14" fill="{PANEL}" stroke="{EDGE}"/>',
          f'<text x="{cx+30}" y="{cy+42}" fill="{INK}" font-size="24" font-weight="bold">'
          f'<tspan fill="{ORANGE}">decode</tspan> vs context depth</text>']
    pts = d["decode"]
    xs = [p["depth"] for p in pts]
    ys = [p["tok_s"] for p in pts]
    ymax = max(ys) * 1.25
    px0, px1, py0, py1 = cx + 90, cx + cw - 50, cy + ch - 70, cy + 90
    def X(v):
        return px0 + (px1 - px0) * (xs.index(v) / max(1, len(xs) - 1))
    def Y(v):
        return py0 - (py0 - py1) * (v / ymax)
    for gy in range(0, int(ymax) + 1, 5):
        s.append(f'<line x1="{px0}" y1="{Y(gy)}" x2="{px1}" y2="{Y(gy)}" stroke="{EDGE}" stroke-width="1"/>'
                 f'<text x="{px0-14}" y="{Y(gy)+5}" text-anchor="end" fill="{MUT}" font-size="14">{gy}</text>')
    poly = " ".join(f"{X(p['depth'])},{Y(p['tok_s'])}" for p in pts)
    s.append(f'<polyline points="{poly}" fill="none" stroke="{ORANGE}" stroke-width="4" stroke-linecap="round"/>')
    for p in pts:
        xlabel = "fresh" if p["depth"] == 0 else f'd{p["depth"] // 1024}K'
        s.append(f'<circle cx="{X(p["depth"])}" cy="{Y(p["tok_s"])}" r="9" fill="{ORANGE}" stroke="{BG}" stroke-width="3"/>'
                 f'<text x="{X(p["depth"])}" y="{Y(p["tok_s"])-22}" text-anchor="middle" fill="{INK}" '
                 f'font-size="22" font-weight="bold">{p["tok_s"]}</text>'
                 f'<text x="{X(p["depth"])}" y="{py0+34}" text-anchor="middle" fill="{SUB}" font-size="16">{xlabel}</text>')
    if d.get("depth_note"):
        s.append(f'<text x="{cx+30}" y="{cy+ch-24}" fill="{MUT}" font-size="13">{d["depth_note"]}</text>')

    tx = 880
    s += [f'<rect x="{tx}" y="250" width="{W-tx-60}" height="220" rx="14" fill="{PANEL}" stroke="{EDGE}"/>',
          f'<text x="{tx+30}" y="292" fill="{ORANGE}" font-size="20" font-weight="bold">decode (measured)</text>']
    yy = 330
    for p in pts:
        lbl = "fresh (tg128)" if p["depth"] == 0 else f'd{p["depth"]//1024}K ({p["kind"]})'
        s.append(f'<text x="{tx+30}" y="{yy}" fill="{SUB}" font-size="17">{lbl}</text>'
                 f'<text x="{W-100}" y="{yy}" text-anchor="end" fill="{INK}" font-size="17" '
                 f'font-weight="bold">{p["tok_s"]} +/- {p["err"]}</text>')
        yy += 34
    pf = d["prefill"]
    s.append(f'<text x="{tx+30}" y="{yy+6}" fill="{PURPLE}" font-size="17">prefill {pf["kind"]}</text>'
             f'<text x="{W-100}" y="{yy+6}" text-anchor="end" fill="{INK}" font-size="17" '
             f'font-weight="bold">{pf["tok_s"]} +/- {pf["err"]} t/s</text>')

    pr = d["predicted"]
    s += [f'<rect x="{tx}" y="500" width="{W-tx-60}" height="220" rx="14" fill="{PANEL}" stroke="{TEAL}" stroke-width="1.5"/>',
          f'<text x="{tx+30}" y="542" fill="{TEAL}" font-size="20" font-weight="bold">the law, before the run</text>',
          f'<text x="{tx+30}" y="586" fill="{SUB}" font-size="17">predicted floor: <tspan fill="{INK}" font-weight="bold">{pr["floor"]} t/s</tspan></text>',
          f'<text x="{tx+30}" y="620" fill="{SUB}" font-size="17">measured: <tspan fill="{INK}" font-weight="bold">{pr["measured"]} t/s</tspan> '
          f'<tspan fill="{TEAL}" font-weight="bold">({pr["delta"]})</tspan></text>',
          f'<text x="{tx+30}" y="654" fill="{SUB}" font-size="16">binding constraint: <tspan fill="{ORANGE}">{d["binding"]}</tspan></text>',
          f'<text x="{tx+30}" y="694" fill="{MUT}" font-size="14">predict yours before downloading: pip install quantprobe</text>']

    bw = (W - 120 - 60) / 4
    labels = [("PEAK DECODE", f'{d["kpi"]["peak_decode"]} t/s', d["kpi"]["peak_decode_at"], ORANGE),
              ("PREFILL PP2048", f'{pf["tok_s"]} t/s', "measured, r=" + str(d["protocol"]["reps"]), PURPLE),
              ("FLOOR HELD", pr["delta"], f'law said {pr["floor"]}, C-02 semantics', TEAL),
              ("30B ON A 2016 GPU", d["kpi"]["headline"], d["kpi"]["headline_sub"], BLUE)]
    for i, (t, v, sub2, c) in enumerate(labels):
        s.append(badge(60 + i * (bw + 20), 760, bw, t, v, sub2, c))

    s += [f'<text x="60" y="965" fill="{MUT}" font-size="14">flags: {d["flags"]}</text>',
          f'<text x="60" y="992" fill="{MUT}" font-size="14">{d["protocol"]["line"]}</text>',
          f'<text x="60" y="1050" fill="{SUB}" font-size="19" font-weight="bold">quantprobe - '
          f'<tspan fill="{TEAL}">falsification-tested laws for local LLMs</tspan>  '
          f'<tspan fill="{MUT}" font-size="15">github.com/FedericoTs/quantprobe - every number on this card is in the repo</tspan></text>',
          '</svg>']
    out = os.path.splitext(path)[0] + ".svg"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("".join(s))
    print("card ->", out)
    return out


if __name__ == "__main__":
    main(sys.argv[1])
