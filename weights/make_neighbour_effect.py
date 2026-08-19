"""Your benchmark runs are contaminating each other - L-31, found under prereg #108.

Parsed from weights/data/prereg108_run2.log, so the chart regenerates from the raw arms.

The hero is the k=8 column: five runs of ONE unchanged command, 11.3 to 70.7 tok/s. Split them
by which config ran immediately before and the scatter disappears - the three whose predecessor
matched sit inside 1.8%, the two whose predecessor did not are the outliers. Same for k=4.
"""
from __future__ import annotations
import os
import re

import brand as B

W, H = 1600, 1280
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "prereg108_run2.log")
GY, GH = 300, 520
SHOWN = [8, 4]                      # the two arms that had a mismatched predecessor


def read():
    """-> {k: [(value, predecessor_k), ...]} in run order."""
    seq, out, prev = [], {}, None
    for line in open(LOG, encoding="utf-8"):
        m = re.match(r"\s+(warm \d+/\d+|p\d+ k=(\d+)):\s+([\d.]+) tok/s", line)
        if not m:
            continue
        k = 8 if m.group(1).startswith("warm") else int(m.group(2))
        if not m.group(1).startswith("warm") and prev is not None:
            out.setdefault(k, []).append((float(m.group(3)), prev))
        prev = k
    return out


def main():
    d = read()
    ymin, ymax = 0, 130
    pl, pr = 200, 1010
    pt, pb = GY + 92, GY + GH - 70

    def py(v):
        return pb - (v - ymin) / (ymax - ymin) * (pb - pt)

    s = [B.svg_open(W, H),
         B.header(W, "MEASURED &#183; Qwen3.6-35B-A3B &#183; 13.15 GiB FILE, 12.5 GB FREE RAM "
                     "&#183; ONE SESSION",
                  "Your benchmark runs contaminate each other",
                  "Five runs of one unchanged command, 11.3 to 70.7 tok/s. The scatter is not "
                  "noise - it is which config ran immediately before.")]

    s.append(B.panel(80, GY, 960, GH))
    s.append(f'<text x="108" y="{GY+38}" fill="{B.MUT}" font-size="19" letter-spacing="3">'
             f'PREFILL tok/s &#183; EVERY ARM, SPLIT BY ITS PREDECESSOR</text>')
    s.append(f'<text x="108" y="{GY+64}" fill="{B.TEAL}" font-size="18">'
             f'&#9679; predecessor used the SAME k'
             f'<tspan fill="{B.DISK}">     &#9679; predecessor used a LOWER k</tspan></text>')

    for v in (0, 25, 50, 75, 100, 125):
        s.append(f'<line x1="{pl}" y1="{py(v):.1f}" x2="{pr}" y2="{py(v):.1f}" '
                 f'stroke="{B.GRID}" stroke-width="1.5"/>')
        s.append(f'<text x="{pl-18}" y="{py(v)+7:.1f}" text-anchor="end" fill="{B.MUT}" '
                 f'font-size="18">{v}</text>')

    colw = (pr - pl) / len(SHOWN)
    for i, k in enumerate(SHOWN):
        cx = pl + colw * (i + 0.5)
        rows = d.get(k, [])
        s.append(f'<text x="{cx:.1f}" y="{pb+42:.1f}" text-anchor="middle" fill="{B.INK}" '
                 f'font-size="24" font-weight="bold">k={k}</text>')
        s.append(f'<text x="{cx:.1f}" y="{pb+68:.1f}" text-anchor="middle" fill="{B.MUT}" '
                 f'font-size="18">{len(rows)} runs, identical command</text>')
        # jitter across the column so overlapping values stay readable
        for j, (v, pred) in enumerate(rows):
            x = cx - 105 + j * 52
            col = B.TEAL if pred >= k else B.DISK
            s.append(f'<circle cx="{x:.1f}" cy="{py(v):.1f}" r="14" fill="{col}"/>')
            s.append(f'<text x="{x:.1f}" y="{py(v)-26:.1f}" text-anchor="middle" fill="{B.INK}" '
                     f'font-size="19" font-weight="bold">{v:g}</text>')
        matched = [v for v, p in rows if p >= k]
        if matched:
            lo, hi = min(matched), max(matched)
            s.append(f'<rect x="{cx-125:.1f}" y="{py(hi)-16:.1f}" width="250" '
                     f'height="{max(py(lo)-py(hi)+32, 34):.1f}" rx="8" fill="none" '
                     f'stroke="{B.TEAL}" stroke-width="2" stroke-dasharray="7 5"/>')
            spread = ((hi - lo) / 2) / (sum(matched) / len(matched)) * 100
            # The last column's label would run past the panel edge if it sat on the right,
            # so it flips to the left of its own box instead.
            right = cx + 140 + 120 < 80 + 960
            lx, anch = (cx + 140, "start") if right else (cx - 140, "end")
            s.append(f'<text x="{lx:.1f}" y="{py((lo+hi)/2)+7:.1f}" text-anchor="{anch}" '
                     f'fill="{B.TEAL}" font-size="19">{spread:.1f}% spread</text>')

    # ---- the verdict
    VX = 1070
    s.append(B.panel(VX, GY, W - VX - 80, GH, stroke=B.DISK, sw=2))
    s.append(f'<text x="{VX+28}" y="{GY+44}" fill="{B.DISK}" font-size="19" letter-spacing="3">'
             f'FROM RUN ORDER ALONE</text>')
    s.append(f'<text x="{VX+28}" y="{GY+140}" fill="{B.DISK}" font-size="96" '
             f'font-weight="bold">6.3x</text>')
    s.append(B.paragraph(
        VX + 28, GY + 186,
        "span between the fastest and slowest reading of the same command - 70.7 against 11.3 "
        "tok/s.", 21, B.SUB, W - VX - 136))
    s.append(f'<text x="{VX+28}" y="{GY+286}" fill="{B.TEAL}" font-size="19" letter-spacing="3">'
             f'WHY</text>')
    s.append(B.paragraph(
        VX + 28, GY + 322,
        "The file is larger than free RAM, so the page cache carries the previous run's working "
        "set into the next process, so a config that touched fewer experts leaves the wrong "
        "pages resident for the one that needs more.", 21, B.SUB, W - VX - 136))

    # ---- what to do
    KY = GY + GH + 34
    s.append(B.panel(80, KY, W - 160, 232, stroke=B.TEAL, sw=2))
    s.append(f'<text x="108" y="{KY+42}" fill="{B.TEAL}" font-size="19" letter-spacing="3">'
             f'WHAT TO DO INSTEAD</text>')
    s.append(B.paragraph(
        108, KY + 82,
        "Interleave your arms and repeat them - run A B C, then C B A, then A B C again, and "
        "compare only readings whose predecessor matched. Three passes of that put every arm "
        "inside 2%. If you A/B two configs back to back on a model bigger than your RAM, you are "
        "comparing cache states as much as configurations, and the difference you publish may be "
        "entirely run order. quantprobe v1.31 warns when your model is in this regime.",
        22, B.SUB, W - 216))

    s.append(B.footer(W, H, "prereg #108 &#183; L-31 &#183; found while measuring something else"))
    s.append("</svg>")
    B.save("neighbour_effect.svg", "".join(s))


if __name__ == "__main__":
    main()
