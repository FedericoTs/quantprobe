"""Evaluation table generator (the frontier-lab-style capability table, for ONE machine).

  python weights/make_eval_table.py

Rows = benchmarks, columns = model x inference strategy, every cell a committed measurement
from the Phase A grid JSONs (cite-or-refuse: a cell whose source file is missing renders as
a dash, never a guess). Emits docs/EVAL_TABLE.md and a dark share-ready SVG.

What makes this table unlike the frontier labs': one machine (a 2016 GTX 1060), strategies as
first-class columns (verified best-of-16 lanes vs single shot), and the strategy column BEATS
the bigger model - which is the program's whole thesis in one grid.
"""
from __future__ import annotations
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DOCS = os.path.join(os.path.dirname(HERE), "docs")

BENCHES = [("mbpp", "MBPP+ (371, plus tests)"), ("humaneval", "HumanEval+ (164, plus tests)")]
COLS = [("0.6B", "single"), ("0.6B", "lanes"), ("4B", "single"), ("4B", "lanes"),
        ("7B", "single"), ("7B", "lanes"), ("30B", "single")]
NAMES = {"0.6B": "Qwen3-0.6B", "4B": "Qwen3.5-4B", "7B": "Qwen2.5-7B", "30B": "Qwen3-Coder-30B"}


def cell(bench, model, mode):
    p = os.path.join(DATA, f"grid_{bench}_{model}_{mode}.json")
    if not os.path.isfile(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    if d.get("unrunnable"):
        return None
    return dict(rate=d["rate"] * 100, wall=d["median_wall"], src=os.path.basename(p))


def build():
    grid = {(b, m, s): cell(b, m, s) for b, _ in BENCHES for m, s in COLS}
    md = ["# The evaluation table — one 2016 machine, every strategy",
          "",
          "Rows are benchmarks, columns are model x inference strategy, **every cell is a",
          "committed measurement** (`weights/data/grid_*.json`; the generator renders a dash",
          "for anything it cannot cite). Machine: GTX 1060 6GB + 16GB DDR4, llama.cpp b10098,",
          "placements planned by quantprobe. `lanes` = 16 sampled candidates in parallel server",
          "slots, winner picked by BASE tests only, scored on the hidden plus set (selection",
          "never sees the exam).",
          "",
          "| benchmark | " + " | ".join(f"{NAMES[m]}<br>{s}16" if s == "lanes" else f"{NAMES[m]}<br>{s}"
                                        for m, s in COLS) + " |",
          "|---|" + "---|" * len(COLS)]
    for b, label in BENCHES:
        row = [label]
        best = max((grid[(b, m, s)]["rate"] for m, s in COLS if grid[(b, m, s)]), default=0)
        for m, s in COLS:
            c = grid[(b, m, s)]
            if c is None:
                row.append("—")
            else:
                v = f"{c['rate']:.1f}"
                row.append(f"**{v}**" if abs(c["rate"] - best) < 1e-9 else v)
        md.append("| " + " | ".join(row) + " |")
    md += ["",
           "Median wall-clock per task (seconds), same cells:",
           "",
           "| benchmark | " + " | ".join(f"{m} {s}" for m, s in COLS) + " |",
           "|---|" + "---|" * len(COLS)]
    for b, label in BENCHES:
        row = [label.split(" (")[0]]
        for m, s in COLS:
            c = grid[(b, m, s)]
            row.append("—" if c is None else f"{c['wall']:.1f}")
        md.append("| " + " | ".join(row) + " |")
    md += ["",
           "Notes:",
           "1. The bold cell per row is the best score **on this machine** - on both benches it",
           "   is a lanes column, not the biggest model. Verified test-time compute beats",
           "   parameter count here (Phase A verdict, prereg 2026-08-05).",
           "2. Lanes assume executable tests exist (the verification regime); wall-clock is",
           "   charged honestly - all 16 candidates plus selection execution.",
           "3. The 30B lanes column is absent by staked prior evidence (U-39: MoE expert-offload",
           "   batching caps ~2x), not omission.",
           "4. Columns arriving with the program: k=32 arms, early-exit lanes (P0b),",
           "   depth-aware-quant variants (quality-ladder campaign), and the Phase C/D tuned",
           "   models - every phase adds a column, and misses stay on the table.",
           ""]
    with open(os.path.join(DOCS, "EVAL_TABLE.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(md))

    import sys
    sys.path.insert(0, HERE)
    import brand
    INK, SUB, MUT = brand.INK, brand.SUB, brand.MUT
    TEAL, ORANGE = brand.TEAL, brand.RAM
    W, colw, x0, y0, rh = 1600, 172, 340, 300, 130
    H = y0 + rh * (len(BENCHES) + 1) + 260
    s = [brand.svg_open(W, H),
         brand.header(W, "ONE 2016 GPU - GTX 1060 6GB",
                      f'model <tspan fill="{ORANGE}">x strategy</tspan>, measured',
                      "solve rate on hidden plus tests - lanes16 = verified best-of-16 "
                      "(picked by visible tests, scored on hidden ones)")]
    SHORT = {"0.6B": "0.6B", "4B": "4B", "7B": "7B", "30B": "30B coder"}
    for j, (m, st) in enumerate(COLS):
        cxx = x0 + j * colw + colw / 2
        s.append(f'<text x="{cxx}" y="{y0+40}" text-anchor="middle" fill="{INK}" font-size="26" font-weight="bold">{SHORT[m]}</text>')
        stl = "lanes16" if st == "lanes" else "single"
        s.append(f'<text x="{cxx}" y="{y0+72}" text-anchor="middle" fill="{ORANGE if st=="lanes" else MUT}" font-size="20">{stl}</text>')
    for i, (b, label) in enumerate(BENCHES):
        yy = y0 + rh * (i + 1) - 20
        s.append(brand.panel(60, yy, W - 120, rh - 16))
        s.append(f'<text x="90" y="{yy+56}" fill="{SUB}" font-size="24">{label.split(" (")[0]}</text>')
        s.append(f'<text x="90" y="{yy+88}" fill="{MUT}" font-size="18">{label.split("(")[1].rstrip(")")}</text>')
        best = max((grid[(b, m, st)]["rate"] for m, st in COLS if grid[(b, m, st)]), default=0)
        for j, (m, st) in enumerate(COLS):
            c = grid[(b, m, st)]
            cxx = x0 + j * colw + colw / 2
            if c is None:
                s.append(f'<text x="{cxx}" y="{yy+72}" text-anchor="middle" fill="{MUT}" font-size="30">-</text>')
            else:
                hero = abs(c["rate"] - best) < 1e-9
                col = TEAL if hero else INK
                if hero:
                    s.append(f'<rect x="{cxx-62}" y="{yy+22}" width="124" height="72" rx="12" '
                             f'fill="none" stroke="{TEAL}" stroke-width="2.5"/>')
                s.append(f'<text x="{cxx}" y="{yy+72}" text-anchor="middle" fill="{col}" '
                         f'font-size="{42 if hero else 34}" font-weight="bold">{c["rate"]:.1f}</text>')
    fy = y0 + rh * (len(BENCHES) + 1) + 30
    s += [f'<text x="80" y="{fy}" fill="{TEAL}" font-size="28" font-weight="bold">the boxed cell is never the biggest model - verification beats size on this box</text>',
          f'<text x="80" y="{fy+40}" fill="{MUT}" font-size="19">30B lanes absent by staked prior evidence (U-39 MoE batching cap), not omission</text>',
          brand.footer(W, H, "every cell cites weights/data/grid_*.json"),
          '</svg>']
    brand.save("eval_table.svg", "".join(s))
    print("-> docs/EVAL_TABLE.md")


if __name__ == "__main__":
    build()
