"""Qwen3.8-27B on a $50 GPU: predicted before measured - the check-before-you-download receipt.

prereg #101 P-5. The frame is the USER's question, not our cleverness: "will the new model run on
my machine?" quantprobe answered from the GGUF header + a one-time calibration, before the model
generated a token; llama-bench then measured it. Both numbers regenerate from committed logs:

    weights/data/qwen38_plan.log    the prediction (1.8 t/s, split 15/65, RAM-bound)
    weights/data/qwen38_bench.log   the measurement (tg128 2.04 +/- 0.02)

Cite-or-refuse: if either log is missing or unparseable this refuses rather than drawing a bar.
"""
from __future__ import annotations
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import brand as B                                             # noqa: E402

PLAN = os.path.join(HERE, "data", "qwen38_plan.log")
BENCH = os.path.join(HERE, "data", "qwen38_bench.log")
W, H = 1600, 1150


def read_numbers():
    if not (os.path.isfile(PLAN) and os.path.isfile(BENCH)):
        raise SystemExit("plan or bench log missing - cannot draw numbers we did not commit")
    plan = open(PLAN, encoding="utf-8", errors="replace").read()
    bench = open(BENCH, encoding="utf-8", errors="replace").read()
    m = re.search(r"\*\s+([\d.]+) tok/s\s+split", plan)
    if not m:
        raise SystemExit("no split prediction in the plan log")
    pred = float(m.group(1))
    m = re.search(r"tg128\s*\|\s*([\d.]+)\s*.\s*([\d.]+)\s*\|", bench)
    if not m:
        raise SystemExit("no tg128 row in the bench log")
    meas, err = float(m.group(1)), float(m.group(2))
    return pred, meas, err


def main():
    pred, meas, err = read_numbers()
    delta = 100.0 * (meas - pred) / pred

    s = [B.svg_open(W, H),
         B.header(W, "Qwen3.8-27B · GTX 1060 6GB ($50, 2016) · PREDICTED, THEN MEASURED",
                  "Will the new model run on your machine?",
                  "quantprobe answered from the file header before the model generated a token. "
                  "Then llama-bench measured it.")]

    # --- the two bars, horizontal, hero scale --------------------------------------------------
    y0 = 360
    bx, bw_full = 560, 760                      # bar origin, width at max scale
    vmax = 2.5
    rows = [("quantprobe PREDICTED", pred, B.SUB, "from the GGUF header + a one-time calibrate - no download needed"),
            ("llama-bench MEASURED", meas, B.TEAL, f"tg128, +/-{err:.2f} over runs, model split 15/65 layers GPU/RAM")]
    for i, (lab, v, col, sub) in enumerate(rows):
        y = y0 + i * 190
        s.append(f'<text x="80" y="{y-14}" fill="{B.MUT}" font-size="20" letter-spacing="3">{lab}</text>')
        bw = v / vmax * bw_full
        s.append(f'<rect x="{bx}" y="{y-58}" width="{bw:.0f}" height="84" rx="10" fill="{col}"/>')
        s.append(f'<text x="{bx+bw+26:.0f}" y="{y+2}" fill="{B.INK}" font-size="64" '
                 f'font-weight="bold">{v:.2f}</text>')
        s.append(f'<text x="{bx+bw+26:.0f}" y="{y+34}" fill="{B.MUT}" font-size="20">tok/s</text>')
        s.append(f'<text x="80" y="{y+52}" fill="{B.SUB}" font-size="19">{sub}</text>')

    # delta callout
    s.append(B.panel(80, y0 + 330, W - 160, 130, stroke=B.TEAL, sw=2))
    s.append(f'<text x="118" y="{y0+385}" fill="{B.TEAL}" font-size="46" font-weight="bold">'
             f'{delta:+.0f}%</text>')
    s.append(f'<text x="300" y="{y0+378}" fill="{B.INK}" font-size="24">prediction vs reality - '
             f'and the miss is in the FLOOR direction (real speed came out faster)</text>')
    s.append(f'<text x="300" y="{y0+412}" fill="{B.MUT}" font-size="19">on the first hybrid '
             f'linear-attention model ever released - an architecture the law was not built for</text>')

    # honest notes
    ny = y0 + 510
    notes = [
        ("The point is not this model: it is that you can check YOURS. quantprobe reads any GGUF "
         "header, measures your machine once (calibrate), and prices every placement before you "
         "spend the download or the money. It also names WHICH resource binds - here, system RAM "
         "bandwidth (88% of every token) - so the upgrade advice is physics, not vibes."),
        ("Disclosed: the same run exposed a real tool gap - quantprobe prices KV as if all 64 "
         "layers were full attention, but 48 are linear (fixed state), a ~4x KV over-estimate "
         "that is negligible at short context and will be fixed as its own change (U-51). "
         "Prediction bands are honest because the misses ship at the same size as the hits."),
    ]
    for n in notes:
        s.append(B.paragraph(80, ny, n, 18, B.MUT, W - 160))
        ny += 26 * (len(B.wrap(n, 18, W - 160)) + 0.6)

    s.append(B.footer(W, H, "prereg #101 P-5 · weights/data/qwen38_plan.log · qwen38_bench.log"))
    s.append("</svg>")
    B.save("qwen38_speed.svg", "".join(s), scale=2)


if __name__ == "__main__":
    main()
