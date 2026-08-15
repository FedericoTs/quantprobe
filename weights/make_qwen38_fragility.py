"""Qwen3.8-27B: the 48 linear-attention layers did not flatten the fragility curve.

prereg #101, staked 2026-08-14 (launch day) BEFORE the probe. The stake: does depth-localized
fragility survive a HYBRID linear-attention model? Qwen3.8-27B has 48 of 64 layers as linear
attention, spread evenly across depth (full attention at every 4th layer). If fragility tracked
attention TYPE, the 4-band depth profile would be FLAT - every band holds four full-attention
layers. It is not flat. It is back-heavy, exactly like every full-attention Qwen.

Cite-or-refuse: the four band deltas render from the committed probe log
(weights/data/prereg101_probe_qwen38_q4.log). The chart cannot show a number the probe did not
measure - if the log is missing or malformed, this refuses rather than drawing a made-up curve.
"""
from __future__ import annotations
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import brand as B                                             # noqa: E402

LOG = os.path.join(HERE, "data", "prereg101_probe_qwen38_q4.log")
W = 1600


def read_probe():
    """Bands + the probe's OWN median, both from the log. Refuses on a bad log.

    The median is PARSED, not recomputed: the probe reported '2.04x vs median 0.29' and the
    committed record (prereg, register) cites 2.04x. Re-deriving my own median here once gave
    2.79x off a different definition - so the chart reads the probe's number rather than
    inventing a rival one. Cite-or-refuse applies to the ratio, not just the bars.
    """
    if not os.path.isfile(LOG):
        raise SystemExit(f"probe log missing: {LOG} - cannot draw a curve we did not measure")
    txt = open(LOG).read()
    rows = []
    for m in re.finditer(r"layers (\d+)-(\d+): PPL [\d.]+\s+\(delta ([\d.]+)\)", txt):
        lo, hi, d = int(m.group(1)), int(m.group(2)), float(m.group(3))
        rows.append((f"{lo}-{hi}", lo, hi, d))
    if len(rows) != 4:
        raise SystemExit(f"expected 4 bands in the log, found {len(rows)} - refusing a partial curve")
    mm = re.search(r"fragile band: layers \d+-\d+ \(delta \+[\d.]+ vs median ([\d.]+)\)", txt)
    if not mm:
        raise SystemExit("could not read the probe's own median from the log - refusing to guess one")
    return rows, float(mm.group(1))


def main():
    bands, median = read_probe()
    deltas = [d for _, _, _, d in bands]
    worst_i = max(range(4), key=lambda i: deltas[i])
    worst = deltas[worst_i]
    ratio = worst / median                     # 0.593 / 0.29 = 2.04, the probe's own figure
    mean = sum(deltas) / 4                      # the counterfactual "flat" height

    H = 1600
    s = [B.svg_open(W, H),
         B.header(W, "MEASURED · Qwen3.8-27B · PROBED 2026-08-14, LAUNCH DAY · wikitext-2",
                  "The 48 linear layers didn't flatten it",
                  "Qwen3.8-27B is the first hybrid we've probed: 48 of 64 layers are linear "
                  "attention. Our depth-fragility law survived it.")]

    # --- hero: the 4-band fragility curve, back-heavy ------------------------------------------
    cx0, cw = 250, 1120                        # chart origin x, width
    base, ch = 760, 360                        # baseline y, chart height
    scale = ch / 0.65                          # 0.65 headroom above the tallest (0.593)
    slot = cw / 4
    barw = 150

    s.append(f'<text x="80" y="322" fill="{B.MUT}" font-size="20" letter-spacing="3">'
             f'FRAGILITY PER DEPTH BAND  ·  DELTA PERPLEXITY WHEN THAT BAND IS PUSHED TO 2-BIT</text>')

    # the counterfactual flat line: what attention-type-localized fragility would look like.
    # Label left-anchored over the SHORT front bars so it never runs behind the tall back column.
    fy = base - mean * scale
    s.append(f'<line x1="{cx0}" y1="{fy:.0f}" x2="{cx0+cw}" y2="{fy:.0f}" stroke="{B.VRAM}" '
             f'stroke-width="2.5" stroke-dasharray="10 7" opacity="0.85"/>')
    s.append(f'<text x="{cx0}" y="{fy-16:.0f}" fill="{B.VRAM}" font-size="20">'
             f'attention-type fragility would sit FLAT on this line</text>')

    # baseline
    s.append(f'<line x1="{cx0}" y1="{base}" x2="{cx0+cw}" y2="{base}" stroke="{B.EDGE}" '
             f'stroke-width="2"/>')

    for i, (lab, lo, hi, d) in enumerate(bands):
        bx = cx0 + i * slot + (slot - barw) / 2
        bh = d * scale
        fragile = i == worst_i
        col = B.TEAL if fragile else B.SUB
        s.append(f'<rect x="{bx:.0f}" y="{base-bh:.0f}" width="{barw}" height="{bh:.0f}" rx="8" '
                 f'fill="{col}" opacity="{1 if fragile else 0.9}"/>')
        # delta value on top
        s.append(f'<text x="{bx+barw/2:.0f}" y="{base-bh-18:.0f}" text-anchor="middle" '
                 f'fill="{B.INK}" font-size="30" font-weight="bold">{d:.3f}</text>')
        # band label under baseline
        s.append(f'<text x="{bx+barw/2:.0f}" y="{base+36}" text-anchor="middle" fill="{B.INK}" '
                 f'font-size="24" font-weight="{"bold" if fragile else "normal"}">layers {lab}</text>')
        if fragile:
            s.append(f'<text x="{bx+barw/2:.0f}" y="{base-bh-56:.0f}" text-anchor="middle" '
                     f'fill="{B.TEAL}" font-size="26" font-weight="bold">{ratio:.2f}x median</text>')
            s.append(f'<text x="{bx+barw/2:.0f}" y="{base+64}" text-anchor="middle" '
                     f'fill="{B.TEAL}" font-size="20" font-weight="bold">FRAGILE</text>')

    s.append(f'<text x="{cx0}" y="{base+36}" fill="{B.MUT}" font-size="20">FRONT</text>')
    s.append(f'<text x="{cx0+cw}" y="{base+36}" text-anchor="end" fill="{B.MUT}" '
             f'font-size="20">BACK</text>')

    # --- architecture strip: 64 layers, the 16 full-attention ones marked, evenly spread -------
    ay = base + 132
    s.append(f'<text x="80" y="{ay-14}" fill="{B.MUT}" font-size="19" letter-spacing="2">'
             f'THE 64 LAYERS  ·  full attention every 4th (evenly across all four bands), '
             f'linear attention between</text>')
    tick_w = cw / 64
    for L in range(64):
        full = (L % 4) == 3                    # full-attention at 3,7,11..63
        tx = cx0 + L * tick_w
        c = B.VRAM if full else B.GRID
        h = 26 if full else 14
        s.append(f'<rect x="{tx:.1f}" y="{ay+ (26-h)}" width="{max(tick_w-2,2):.1f}" height="{h}" '
                 f'fill="{c}"/>')
    s.append(f'<text x="{cx0+cw}" y="{ay+52}" text-anchor="end" fill="{B.VRAM}" font-size="18">'
             f'16 full-attention layers, 4 in every band - so attention-type fragility would read '
             f'FLAT</text>')

    # --- receipt chips -------------------------------------------------------------------------
    cy = ay + 96
    gap = 26
    cwid = (W - 160 - 2 * gap) / 3
    s.append(B.chip(80, cy, cwid, "FRAGILE BAND",
                    f"{bands[worst_i][0]}", "the back - like every full-attn Qwen", B.TEAL))
    s.append(B.chip(80 + cwid + gap, cy, cwid, "vs STAKED BAR",
                    f"{ratio:.2f}x", "staked >= 1.3x median, before the probe", B.INK))
    s.append(B.chip(80 + 2 * (cwid + gap), cy, cwid, "PROFILE SHAPE",
                    "monotone", "back-heavy, not flat -> depth, not type", B.SUB))

    # --- honest notes --------------------------------------------------------------------------
    ny = cy + 200
    notes = [
        ("STAKED BEFORE THE PROBE (prereg #101, launch day). The stake was the SHAPE of the depth "
         "curve: back-heavy (worst band >= 1.3x median) confirms depth-localized fragility; a flat "
         "profile within 15% would have meant fragility lived in the evenly-spread full-attention "
         "layers instead. It came back monotone at " + f"{ratio:.2f}x" + " - depth wins."),
        ("Measured from the Q4_K_M source, not BF16: the true-original probe is infeasible on a "
         "6GB/16GB box (a 27B Q6 reference exceeds 16GB RAM). Relative band deltas survive a "
         "quantized source - the source's own quant is common-mode and cancels. The declared risk "
         "that Q4 could blunt fragility into a false flat did NOT fire: a 2x signal survived it."),
        ("What this does NOT show yet: the benchmark ceiling (naive vs recipe on real tasks) and "
         "the Law-4 decode prediction for linear attention both need the BF16 27B, which is weeks "
         "per arm here - deferred to rented hardware, not claimed."),
    ]
    for n in notes:
        s.append(B.paragraph(80, ny, n, 18, B.MUT, W - 160))
        ny += 26 * (len(B.wrap(n, 18, W - 160)) + 0.5)

    s.append(B.footer(W, H, "prereg 2026-08-14-qwen38-27b-hybrid-fragility · "
                            "weights/data/prereg101_probe_qwen38_q4.log"))
    s.append("</svg>")
    # 2x for X: it JPEG-compresses uploads, and a 3200px source keeps the note text crisp.
    B.save("qwen38_fragility.svg", "".join(s), scale=2)


if __name__ == "__main__":
    main()
