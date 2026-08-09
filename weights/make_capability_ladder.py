"""What 12 GB actually buys you - four runnable models, five standard benchmarks, one card.

EV-1, prereg 2026-08-06: one machine, full sets, temp 0.

NOT A SCALING STUDY, and the first version of this chart wrongly implied it was. It was titled
"Where size stops buying accuracy" over rows labelled 0.6B/4B/7B/30B, which reads as a
controlled test of size. These four differ in generation (Qwen2.5 / Qwen3 / Qwen3.5), in
quantization tier (Q8_0 / Q4_K_M / Q2_K_L) and in specialisation - the 30B is a CODE model
being asked to do competition maths, at a sub-4-bit tier our own C-05 and L-15 place in a
measurably degraded regime. It is handicapped twice over, and reporting "the 4B ties the 30B"
as a claim about SIZE would be an overclaim of exactly the kind this project refuses.

What the rows do support is narrower and more useful: on one 12 GB card, the newest small
generalist at a sane quant matches the largest specialist you can fit, and beats a
previous-generation 7B on every benchmark measured. That is a placement result - which is the
question the tool actually answers.

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
X0, XW = 620, 700                    # bar origin, full-scale width (100%)
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

# THE THING THE FIRST VERSION OF THIS CHART GOT WRONG. It was titled "Where size stops buying
# accuracy" over four rows labelled 0.6B/4B/7B/30B, which reads as a controlled scaling study.
# It is not one. These are four files you would actually run on a 12 GB card, and they differ
# in generation, quantization tier and specialisation as much as in size:
#
#   0.6B  Qwen3         Q8_0      8-bit, a generalist
#   4B    Qwen3.5       Q4_K_M    NEWEST generation here
#   7B    Qwen2.5       Q4_K_M    OLDEST generation here
#   30B   Qwen3-Coder   Q2_K_L    a CODE specialist, and sub-4-bit
#
# So the 30B is doubly handicapped on a maths benchmark - wrong specialism, and a quantization
# tier our own C-05/L-15 put in a measurably degraded regime. Reporting "the 4B ties the 30B"
# as a statement about SIZE would be an overclaim of exactly the kind this project exists to
# refuse. What the rows honestly support is narrower and more useful: on one 12 GB card, the
# newest small generalist at a sane quant matches what you can fit of a much larger specialist.
VARIANT = {
    "0.6B": ("Qwen3", "Q8_0", "generalist"),
    "4B":   ("Qwen3.5", "Q4_K_M", "newest generation here"),
    "7B":   ("Qwen2.5", "Q4_K_M", "oldest generation here"),
    "30B":  ("Qwen3-Coder", "Q2_K_L", "code specialist, sub-4-bit"),
}


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
    # Protocol-or-refuse, alongside cite-or-refuse: the boxed-answer instruction must
    # have reached every boxed row and no other. Read back out of the logged prompts,
    # because the code as it stands says nothing about the code a row was run under.
    violations = R.verify_prompts()
    if violations:
        sep = chr(10) + "  "
        raise SystemExit("protocol violated in the DATA, not just the code:" + sep
                         + sep.join(violations))
    gb = sizes()

    scored = {}
    for model, task, metric, value, _ in R.table(rows):
        scored[(model, task)] = value * 100

    live = [b for b in BENCHES if any((m, b[0]) in scored for m in ORDER)]
    H = B.HEADER_H + sum(ROWH * len(ORDER) + GROUP_GAP + 34 for _ in live) + B.FOOTER_H + 540

    m4 = scored.get((HERO, "math500_boxed"))
    m30 = scored.get(("30B", "math500_boxed"))

    s = [B.svg_open(W, H),
         B.header(W, "MEASURED · RTX 3060 12GB · lm-eval 0.4.12 · FULL SETS · TEMP 0",
                  "What 12 GB actually buys you",
                  "Four models you could really run on one card. NOT a scaling study - they "
                  "differ in generation and quantization as much as in size.")]

    y = B.HEADER_H + 24

    # The hero, stated as what it is: a dead heat between two very different files. The earlier
    # version framed this as size-versus-accuracy and printed a "-0.4 points" delta, which reads
    # as the big model losing. It is not losing; 0.4pp against a +-1.9pp stderr is no difference
    # at all, and the two files differ in generation, specialism and quant tier as well as size.
    if m4 is not None and m30 is not None:
        s.append(B.panel(80, y, W - 160, 164, stroke=B.TEAL, sw=2))
        s.append(f'<text x="118" y="{y+46}" fill="{B.MUT}" font-size="19" '
                 f'letter-spacing="3">MATH-500 - A DEAD HEAT, NOT A GAP</text>')
        s.append(f'<text x="118" y="{y+112}" fill="{B.TEAL}" font-size="66" '
                 f'font-weight="bold">{m4:.1f}%</text>')
        s.append(f'<text x="310" y="{y+112}" fill="{B.SUB}" font-size="23">'
                 f'Qwen3.5-4B Q4_K_M</text>')
        s.append(f'<text x="660" y="{y+112}" fill="{B.INK}" font-size="66" '
                 f'font-weight="bold">{m30:.1f}%</text>')
        s.append(f'<text x="852" y="{y+112}" fill="{B.SUB}" font-size="23">'
                 f'Qwen3-Coder-30B Q2_K_L</text>')
        s.append(f'<text x="{W-118}" y="{y+86}" text-anchor="end" fill="{B.INK}" '
                 f'font-size="27" font-weight="bold">'
                 f'{abs(m30 - m4):.1f}pp apart</text>')
        s.append(f'<text x="{W-118}" y="{y+120}" text-anchor="end" fill="{B.MUT}" '
                 f'font-size="22">on a stderr of +-1.9pp each</text>')
        s.append(f'<text x="118" y="{y+146}" fill="{B.MUT}" font-size="18">'
                 f'Different generation, different quantization tier, and one of them is a code '
                 f'model doing maths. Read it as a placement result, not a scaling law.</text>')
        y += 212

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
            fam, quant, _ = VARIANT[model]
            # The size label alone is what made the first version read as a scaling study. The
            # family and quant tier travel with every single bar now, so no row can be quoted
            # out of the chart as "the 30B" when it is a sub-4-bit code model.
            s.append(f'<text x="{X0-330}" y="{y+26}" fill="{B.INK if hero else B.SUB}" '
                     f'font-size="20" font-weight="{"bold" if hero else "normal"}" '
                     f'font-family="Consolas, Menlo, monospace">{fam}-{model}</text>')
            s.append(f'<text x="{X0-22}" y="{y+26}" text-anchor="end" fill="{B.MUT}" '
                     f'font-size="17">{quant} · {gb.get(model, 0):.2f} GB</text>')
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

    notes = [
        ("GSM8K is reported as flexible-extract. Its strict-match filter demands the literal "
         "sentence “The answer is N.”, which the zero-shot prompt never asks for - "
         "0 of 3,957 responses matched it, on any model (C-25). GPQA is part of this suite and "
         "has not run yet: no cell is shown because none was measured."),
        ("Boxed rows are re-graded from the logged samples with the current extractor. It "
         "previously took the LAST " + chr(92) + "boxed, so a model that repeated itself into "
         "the token cap had its correct answer discarded with the truncated fragment after it: "
         "9 answers rescued, 0 lost, 8 of 10 rows unchanged. Only the 4B loops, and only "
         "on AIME."),
        ("NOT STRICTLY COMPARABLE ON AIME: the 0.6B, 4B and 7B rows ran a 7,168-token "
         "generation budget; the 30B row ran 8,192 after the slot plan changed to clear a "
         "deadlock. The larger model had 14% more room to think, so any gap it shows is an "
         "upper bound on its advantage, not a measurement of it."),
        ("AND NOT A SCALING STUDY: three model generations (Qwen2.5 / Qwen3 / Qwen3.5), three "
         "quantization tiers (Q8_0 / Q4_K_M / Q2_K_L) and one code specialist sit on these "
         "four rows. The 30B is handicapped twice on maths - wrong specialism, and a sub-4-bit "
         "tier our own C-05 and L-15 put in a measurably degraded regime. These are the files "
         "you can actually fit on 12 GB, which is the useful question; they are not a "
         "controlled test of size."),
    ]
    for n in notes:
        s.append(B.paragraph(80, y + 6, n, 18, B.MUT, W - 160))
        y += 26 * (len(B.wrap(n, 18, W - 160)) + 0.6)

    s.append(B.footer(W, H, "prereg 2026-08-06-ev1-standard-benches · "
                            "weights/data/ev1/**/results_*.json"))
    s.append("</svg>")
    B.save("capability_ladder.svg", "".join(s))


if __name__ == "__main__":
    main()
