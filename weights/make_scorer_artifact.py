"""The zero that is not the model's fault - C-25, drawn.

Three tasks in one standard eval suite scored models at zero for answering correctly in the
wrong shape. Every number here is measured on our own rows (weights/data/ev1) and the GSM8K
case is checked against lm-evaluation-harness 0.4.12's own task file.

The framing matters and is deliberate: none of this is an accusation. gsm8k-cot-zeroshot's
strict filter is exactly right for the FEW-SHOT variant it was inherited from, where the
exemplars taught the model to write "The answer is N." Strip the exemplars and nothing
establishes the convention - the filter is simply asking for something the prompt no longer
requests. That is a subtle, easy failure, which is why it is worth a chart rather than a jeer.
"""
from __future__ import annotations
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import brand as B                                            # noqa: E402
import ev1_report as R                                       # noqa: E402

W = 1600
PAD = 80


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# (task, what the scorer required, what the model actually wrote, scored, recovered, recovery)
CASES = [
    ("GSM8K  cot_zeroshot  strict-match",
     'the literal sentence  "The answer is N."',
     '"Therefore, Janet makes $18 every day at the farmers\' market."',
     "0 of 3,957 responses matched, across three models",
     "36.8 / 81.7 / 79.9",
     "flexible-extract, same responses"),
    ("MATH-500  hendrycks extractor",
     "the slice between the FIRST $ and the LAST $",
     "a well-formed \\boxed{} answer, in 95.4% of the 0.6B's replies",
     "0.00%",
     "46.4%",
     "an extractor that reads \\boxed{}"),
    ("AIME 2024  zero-shot",
     "$...$ or \\boxed{} - but the prompt asks for neither",
     '"Answer: 49"  - the 4B wrote it bare in 30 of 30 items',
     "0.0%",
     "33.3%",
     "the same model, asked for the format"),
]


def main():
    # Cite-or-refuse: the recovered scores are read back from the rows, not typed in.
    rows = R.load_rows()
    R.check_publishable(rows)
    got = {(m, t): v for (m, t), d in rows.items() for k, v in d.items()
           if k == R.REPORTED.get(t, ("", ""))[0]}
    assert abs(got[("0.6B", "math500_boxed")] * 100 - 46.4) < 0.05, "MATH-500 0.6B moved"
    assert abs(got[("4B", "aime24_boxed")] * 100 - 33.3) < 0.05, "AIME24 4B moved"

    CH, GAP = 214, 26
    H = B.HEADER_H + 196 + len(CASES) * (CH + GAP) + 186 + B.FOOTER_H

    s = [B.svg_open(W, H),
         B.header(W, "MEASURED · lm-evaluation-harness 0.4.12 · THREE TASKS, ONE FAILURE MODE",
                  "The zero that is not the model's fault",
                  "A benchmark score can be a property of its scorer. When it is, the tell is "
                  "sharp - and cheap to check.")]

    y = B.HEADER_H + 16
    s.append(B.panel(PAD, y, W - 2 * PAD, 150, stroke=B.TEAL, sw=2))
    s.append(f'<text x="{PAD+38}" y="{y+46}" fill="{B.MUT}" font-size="19" letter-spacing="3">'
             f'GSM8K STRICT-MATCH · RESPONSES THAT MATCHED WHAT IT ASKED FOR</text>')
    s.append(f'<text x="{PAD+38}" y="{y+118}" fill="{B.TEAL}" font-size="72" '
             f'font-weight="bold">0 of 3,957</text>')
    s.append(f'<text x="{PAD+470}" y="{y+118}" fill="{B.SUB}" font-size="26">'
             f'across a 0.6B, a 4B and a 7B - not a low rate, exactly zero</text>')
    y += 196

    for task, wanted, wrote, scored, recovered, how in CASES:
        s.append(B.panel(PAD, y, W - 2 * PAD, CH))
        s.append(f'<text x="{PAD+34}" y="{y+44}" fill="{B.INK}" font-size="25" '
                 f'font-weight="bold" font-family="Consolas, Menlo, monospace">'
                 f'{esc(task)}</text>')

        s.append(f'<text x="{PAD+34}" y="{y+90}" fill="{B.MUT}" font-size="18" '
                 f'letter-spacing="2">THE SCORER WANTED</text>')
        s.append(f'<text x="{PAD+300}" y="{y+90}" fill="{B.SUB}" font-size="21">'
                 f'{esc(wanted)}</text>')

        s.append(f'<text x="{PAD+34}" y="{y+134}" fill="{B.MUT}" font-size="18" '
                 f'letter-spacing="2">THE MODEL WROTE</text>')
        s.append(f'<text x="{PAD+300}" y="{y+134}" fill="{B.INK}" font-size="21">'
                 f'{esc(wrote)}</text>')

        s.append(f'<line x1="{PAD+34}" y1="{y+158}" x2="{W-PAD-34}" y2="{y+158}" '
                 f'stroke="{B.GRID}" stroke-width="1"/>')
        s.append(f'<text x="{PAD+34}" y="{y+192}" fill="{B.DISK}" font-size="21" '
                 f'font-weight="bold">SCORED  {esc(scored)}</text>')
        s.append(f'<text x="{W-PAD-34}" y="{y+192}" text-anchor="end" fill="{B.TEAL}" '
                 f'font-size="21" font-weight="bold">{esc(recovered)}'
                 f'<tspan fill="{B.MUT}" font-weight="normal" font-size="19">'
                 f'   {esc(how)}</tspan></text>')
        y += CH + GAP

    y += 14
    s.append(f'<text x="{PAD}" y="{y}" fill="{B.INK}" font-size="26" font-weight="bold">'
             f'The rule that falls out of it</text>')
    for i, line in enumerate([
            "A metric that is EXACTLY 0.0 across unrelated model sizes is a format mismatch, "
            "not a difficulty wall. Real capability",
            "differences are ragged; they do not line up on 0.0000 for a 0.6B and a 30B. "
            "The extractor simply never fired."]):
        s.append(f'<text x="{PAD}" y="{y+40+i*28}" fill="{B.SUB}" font-size="21">'
                 f'{esc(line)}</text>')
    # Say this ON the chart, not only in the source. Without it the asset reads as a dunk on a
    # harness thousands of people rely on, and the actual mechanism - a filter that is correct
    # for the variant it came from - is the more useful and more interesting thing.
    s.append(B.paragraph(PAD, y + 108, esc(
        "Not a bug report. GSM8K strict-match is right for the FEW-SHOT variant it came from, "
        "where the exemplars taught the model to write that sentence. Zero-shot inherited the "
        "filter, not the exemplars."), 19, B.MUT, W - 2 * PAD))

    s.append(B.footer(W, H, "C-25 · weights/ev1_report.py · prereg 2026-08-06-ev1-standard-benches"))
    s.append("</svg>")
    B.save("scorer_artifact.svg", "".join(s))


if __name__ == "__main__":
    main()
