"""The capability ladder - where model size stops buying accuracy on one 12 GB card.

Four models, one machine, five standard benchmarks, full sets, temp 0 (EV-1, prereg
2026-08-06). The headline is a tie nobody expects: a 2.83 GB 4B matches an 11.33 GB 30B on
MATH-500, and beats a 4.68 GB 7B on every benchmark measured.

Cite-or-refuse. Every number renders from weights/data/ev1/**/results_*.json via ev1_report,
which REFUSES to hand over a metric that is exactly 0.0 across three or more models unless the
mechanism is on record (C-25) - so this chart cannot accidentally publish a scorer artifact as
a capability. Rows still running are drawn as explicit gaps, never omitted and never inferred:
a missing cell is data about what we have not measured yet.
"""
from __future__ import annotations
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import brand as B                                            # noqa: E402
import ev1_report as R                                       # noqa: E402
from gridbench import MODELS                                 # noqa: E402

W = 1600
X0, XW = 486, 830                    # bar origin, full-scale width (100%)
ROWH, GROUP_GAP = 38, 46

# Display order and the plain-English name. The suite is what people actually quote.
BENCHES = [
    ("math500_boxed",      "MATH-500",  "competition math, 500 items"),
    ("gsm8k_cot_zeroshot", "GSM8K",     "grade-school word problems, 1,319 items"),
    ("ifeval",             "IFEval",    "literal instruction compliance, 541 prompts"),
    ("aime24_boxed",       "AIME 2024", "olympiad qualifier, 30 items"),
    ("aime25_boxed",       "AIME 2025", "olympiad qualifier, 30 items - post-cutoff"),
    ("gpqa_main_zeroshot", "GPQA",      "graduate science, multiple choice"),
]
ORDER = ["0.6B", "4B", "7B", "30B"]
HERO = "4B"


def sizes():
    out = {}
    for k, p in MODELS.items():
        p = str(p).replace("\\", "/")
        if os.path.isfile(p):
            out[k] = os.path.getsize(p) / 1e9
    return out


def main():
    rows = R.load_rows()
    R.check_publishable(rows)                # refuses on an undiagnosed uniform zero
    gb = sizes()

    scored = {}
    for model, task, metric, value, _ in R.table(rows):
        scored[(model, task)] = value * 100

    live = [b for b in BENCHES if any((m, b[0]) in scored for m in ORDER)]
    H = B.HEADER_H + sum(ROWH * len(ORDER) + GROUP_GAP + 34 for _ in live) + B.FOOTER_H + 250

    m4 = scored.get((HERO, "math500_boxed"))
    m30 = scored.get(("30B", "math500_boxed"))

    s = [B.svg_open(W, H),
         B.header(W, "MEASURED · RTX 3060 12GB · lm-eval 0.4.12 · FULL SETS · TEMP 0",
                  "Where size stops buying accuracy",
                  "Four models on one card. Same harness, same prompts, same day - the only "
                  "thing that changes is the model.")]

    y = B.HEADER_H + 24

    # The hero comparison, stated once, in numbers big enough to read in a timeline.
    if m4 is not None and m30 is not None:
        s.append(B.panel(80, y, W - 160, 152, stroke=B.TEAL, sw=2))
        s.append(f'<text x="118" y="{y+50}" fill="{B.MUT}" font-size="19" '
                 f'letter-spacing="3">MATH-500, THE TIE</text>')
        s.append(f'<text x="118" y="{y+118}" fill="{B.TEAL}" font-size="72" '
                 f'font-weight="bold">{m4:.1f}%</text>')
        s.append(f'<text x="330" y="{y+118}" fill="{B.SUB}" font-size="26">'
                 f'4B · {gb.get("4B", 0):.2f} GB</text>')
        s.append(f'<text x="700" y="{y+118}" fill="{B.INK}" font-size="72" '
                 f'font-weight="bold">{m30:.1f}%</text>')
        s.append(f'<text x="912" y="{y+118}" fill="{B.SUB}" font-size="26">'
                 f'30B · {gb.get("30B", 0):.2f} GB</text>')
        s.append(f'<text x="{W-118}" y="{y+82}" text-anchor="end" fill="{B.INK}" '
                 f'font-size="30" font-weight="bold">'
                 f'{gb.get("30B", 1) / gb.get("4B", 1):.1f}x the bytes</text>')
        s.append(f'<text x="{W-118}" y="{y+120}" text-anchor="end" fill="{B.MUT}" '
                 f'font-size="24">for {m30 - m4:+.1f} points</text>')
        y += 200

    for task, label, blurb in live:
        s.append(f'<text x="80" y="{y+20}" fill="{B.INK}" font-size="26" '
                 f'font-weight="bold">{label}</text>')
        # Fixed column, not a width estimate off len(label) - proportional type made the gap
        # after "IFEval" visibly wider than the one after "MATH-500".
        s.append(f'<text x="266" y="{y+20}" fill="{B.MUT}" font-size="19">{blurb}</text>')
        y += 34
        for model in ORDER:
            v = scored.get((model, task))
            hero = model == HERO
            col = B.TEAL if hero else (B.VRAM if model == "30B" else B.SUB)
            s.append(f'<text x="{X0-150}" y="{y+26}" fill="{B.INK if hero else B.SUB}" '
                     f'font-size="21" font-weight="{"bold" if hero else "normal"}" '
                     f'font-family="Consolas, Menlo, monospace">{model}</text>')
            s.append(f'<text x="{X0-22}" y="{y+26}" text-anchor="end" fill="{B.MUT}" '
                     f'font-size="18">{gb.get(model, 0):.2f} GB</text>')
            s.append(f'<line x1="{X0}" y1="{y+19}" x2="{X0+XW}" y2="{y+19}" '
                     f'stroke="{B.GRID}" stroke-width="1"/>')
            if v is None:
                # A row still running is drawn, not dropped. Omitting it would quietly imply
                # the suite is complete, and the gap is exactly what a reader should see.
                s.append(f'<rect x="{X0}" y="{y+6}" width="150" height="26" rx="6" '
                         f'fill="none" stroke="{B.EDGE}" stroke-width="1.5" '
                         f'stroke-dasharray="6 5"/>')
                s.append(f'<text x="{X0+165}" y="{y+26}" fill="{B.MUT}" font-size="18">'
                         f'row still running</text>')
            else:
                bw = max(v / 100 * XW, 2)
                s.append(f'<rect x="{X0}" y="{y+6}" width="{bw:.1f}" height="26" rx="6" '
                         f'fill="{col}" opacity="{1 if hero else 0.85}"/>')
                s.append(f'<text x="{X0+bw+14}" y="{y+26}" fill="{B.INK}" font-size="21" '
                         f'font-weight="bold">{v:.1f}</text>')
            y += ROWH
        y += GROUP_GAP

    # Hand-wrapped: one long line ran past the 1600px canvas and clipped its own last words.
    notes = ["GSM8K is reported as flexible-extract. Its strict-match filter demands the literal "
             "sentence “The answer is N.”, which the zero-shot",
             "prompt never asks for - 0 of 3,957 responses matched it, on any model (C-25). "
             "GPQA is part of this suite and has not run yet: no cell is shown"
             " because none was measured."]
    for i, line in enumerate(notes):
        s.append(f'<text x="80" y="{y + 6 + i * 26}" fill="{B.MUT}" '
                 f'font-size="18">{line}</text>')

    s.append(B.footer(W, H, "prereg 2026-08-06-ev1-standard-benches · "
                            "weights/data/ev1/**/results_*.json"))
    s.append("</svg>")
    B.save("capability_ladder.svg", "".join(s))


if __name__ == "__main__":
    main()
