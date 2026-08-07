"""Prediction-vs-reality - the trust chart. Log-log scatter of predicted vs measured tok/s.

Sources (all committed):
- weights/data/unattended_20260801_002809_ladder_result.json - the 14-row ladder measured
  under the quiesced-machine protocol the README publishes (median |err| 8.4%). The earlier
  PRE_v124 reference reads 9.0% on the same rows; the README calls that move "unchanged
  inside our +/-1 point noise floor", and the charts cite the same dataset as the README so
  one number exists in public. The 20260731_1913 re-run is contaminated from row 9 and is NOT
  plotted at all (overlap law).
- E-08 (register): RTX 5070, pred 58.1 vs 54-57 measured; 9B floor 45.7 vs 71-76.
- E-13 (register): RX 5700 XT, pred 73.1 vs 73.18 +/- 0.16.
- U-06 anchor (prereg #66): 35B Q8_0 disk-streaming, pred 2.0 vs 0.66 (-67%) - THE miss,
  plotted at full size; the tier ships labeled unvalidated.
Floor semantics: all-in-VRAM rows are one-sided floors (C-02, 'typically 1.1-1.8x above') -
the wedge above the diagonal is drawn, not hidden.
"""
from __future__ import annotations
import json, math, os
import brand as B

W, H = 1600, 1200
PX0, PY0, PS = 150, 330, 670          # plot box (square, log-log)
VMIN, VMAX = 0.45, 230.0


def px(v):
    return PX0 + (math.log10(v) - math.log10(VMIN)) / (math.log10(VMAX) - math.log10(VMIN)) * PS


def py(v):
    return PY0 + PS - (math.log10(v) - math.log10(VMIN)) / (math.log10(VMAX) - math.log10(VMIN)) * PS


def main():
    rows = json.load(open(os.path.join(os.path.dirname(__file__), "data",
                                       "unattended_20260801_002809_ladder_result.json")))
    errs = sorted(abs(r["predicted"] / r["measured"] - 1) * 100 for r in rows)
    med = errs[len(errs) // 2] if len(errs) % 2 else (errs[len(errs)//2-1] + errs[len(errs)//2]) / 2
    s = [B.svg_open(W, H),
         B.header(W, "EVERY POINT A COMMITTED MEASUREMENT · MISSES AT FULL SIZE",
                  "Predicted vs measured",
                  "The 14-model ladder on our GTX 1060, plus the first out-of-sample externals "
                  "(Blackwell, AMD) - and the disk-tier miss on the same axes.")]

    # bands first (under everything): +/-25% printed band, then the disclosed floor wedge
    def band_path(k_lo, k_hi):
        x1, x2 = VMIN, VMAX
        return (f'M {px(x1)} {py(x1*k_lo)} L {px(x2)} {py(x2*k_lo)} '
                f'L {px(x2)} {py(x2*k_hi)} L {px(x1)} {py(x1*k_hi)} Z')
    s.append(f'<clipPath id="plot"><rect x="{PX0}" y="{PY0}" width="{PS}" height="{PS}"/></clipPath>')
    s.append(f'<g clip-path="url(#plot)">')
    s.append(f'<path d="{band_path(0.75, 1.25)}" fill="{B.TEAL_DEEP}" opacity="0.13"/>')
    s.append(f'<path d="{band_path(1.1, 1.8)}" fill="{B.VRAM}" opacity="0.10"/>')
    s.append(f'<line x1="{px(VMIN)}" y1="{py(VMIN)}" x2="{px(VMAX)}" y2="{py(VMAX)}" '
             f'stroke="{B.INK}" stroke-width="3" stroke-dasharray="2 6" opacity="0.85"/>')
    s.append('</g>')

    # grid + ticks
    for t in (0.5, 1, 2, 5, 10, 20, 50, 100, 200):
        s.append(f'<line x1="{px(t)}" y1="{PY0}" x2="{px(t)}" y2="{PY0+PS}" '
                 f'stroke="{B.GRID}" stroke-width="1"/>')
        s.append(f'<line x1="{PX0}" y1="{py(t)}" x2="{PX0+PS}" y2="{py(t)}" '
                 f'stroke="{B.GRID}" stroke-width="1"/>')
        lab = f"{t:g}"
        s.append(f'<text x="{px(t)}" y="{PY0+PS+34}" text-anchor="middle" fill="{B.MUT}" '
                 f'font-size="19">{lab}</text>')
        s.append(f'<text x="{PX0-16}" y="{py(t)+7}" text-anchor="end" fill="{B.MUT}" '
                 f'font-size="19">{lab}</text>')
    s.append(f'<text x="{PX0+PS/2}" y="{PY0+PS+72}" text-anchor="middle" fill="{B.SUB}" '
             f'font-size="22">predicted tok/s (log)</text>')
    s.append(f'<text x="{PX0-104}" y="{PY0+PS/2}" fill="{B.SUB}" font-size="22" '
             f'transform="rotate(-90 {PX0-104} {PY0+PS/2})" text-anchor="middle">measured tok/s (log)</text>')

    # ladder points: circle = all-in-VRAM (floor semantics), diamond = split rows
    for r in rows:
        x, y = px(r["predicted"]), py(r["measured"])
        vram = "all in VRAM" in r["placement"]
        if vram:
            s.append(f'<circle cx="{x}" cy="{y}" r="13" fill="{B.VRAM}" '
                     f'stroke="{B.BG}" stroke-width="2.5"/>')
        else:
            s.append(f'<path d="M {x} {y-15} L {x+15} {y} L {x} {y+15} L {x-15} {y} Z" '
                     f'fill="{B.TEAL}" stroke="{B.BG}" stroke-width="2.5"/>')

    # externals: squares with whiskers
    ext = [(58.1, 54.0, 57.0, "RTX 5070 - 35B MoE split"),
           (45.7, 71.0, 76.0, "RTX 5070 - 9B floor"),
           (73.1, 73.02, 73.34, "RX 5700 XT - 7B +0.1%")]
    for p, lo, hi, _ in ext:
        x, mid = px(p), (lo + hi) / 2
        s.append(f'<line x1="{x}" y1="{py(lo)}" x2="{x}" y2="{py(hi)}" '
                 f'stroke="{B.TEAL_DEEP}" stroke-width="5"/>')
        s.append(f'<rect x="{x-12}" y="{py(mid)-12}" width="24" height="24" '
                 f'fill="{B.TEAL_DEEP}" stroke="{B.INK}" stroke-width="2.5"/>')

    # the miss, full size - caption stacked in the empty space above the point
    xm, ym = px(2.0), py(0.66)
    s.append(f'<circle cx="{xm}" cy="{ym}" r="16" fill="{B.DISK}" stroke="{B.INK}" '
             f'stroke-width="3"/>')
    s.append(f'<text x="{xm-6}" y="{ym-88}" fill="{B.DISK}" font-size="26" '
             f'font-weight="bold">-67%</text>')
    s.append(f'<text x="{xm-6}" y="{ym-58}" fill="{B.SUB}" font-size="19">35B streaming from disk</text>')
    s.append(f'<text x="{xm-6}" y="{ym-32}" fill="{B.SUB}" font-size="19">ships labeled unvalidated</text>')

    # point labels (storied only)
    labels = [(124.0, 151.76, "0.5B: floor runs hot", -26, 4, "end"),
              (73.1, 73.18, "first AMD: +0.1%", -24, -12, "end"),
              (58.1, 55.5, "first Blackwell: +2..7.6%", 24, 44, "start"),
              (45.7, 73.5, "floor 1.1-1.8x: held", -24, 44, "end")]
    for vx, vy, t, dx, dy, anch in labels:
        s.append(f'<text x="{px(vx)+dx}" y="{py(vy)+dy}" fill="{B.SUB}" font-size="20" '
                 f'text-anchor="{anch}">{t}</text>')
    # cluster note for the split pack, below-left of the pack
    s.append(f'<text x="{px(15)}" y="{py(8.2)}" fill="{B.SUB}" font-size="20" '
             f'text-anchor="end">8 split placements,</text>')
    s.append(f'<text x="{px(15)}" y="{py(8.2)+28}" fill="{B.SUB}" font-size="20" '
             f'text-anchor="end">all inside ±17%</text>')

    # legend (bottom-right of plot, empty below-diagonal zone)
    LX, LY = PX0 + PS - 344, PY0 + PS - 192
    s.append(B.panel(LX, LY, 320, 158))
    s.append(f'<circle cx="{LX+34}" cy="{LY+36}" r="11" fill="{B.VRAM}"/>')
    s.append(f'<text x="{LX+58}" y="{LY+43}" fill="{B.SUB}" font-size="19">all-in-VRAM (floor)</text>')
    s.append(f'<path d="M {LX+34} {LY+62} L {LX+46} {LY+74} L {LX+34} {LY+86} L {LX+22} {LY+74} Z" '
             f'fill="{B.TEAL}"/>')
    s.append(f'<text x="{LX+58}" y="{LY+81}" fill="{B.SUB}" font-size="19">split placement</text>')
    s.append(f'<rect x="{LX+24}" y="{LY+100}" width="20" height="20" fill="{B.TEAL_DEEP}"/>')
    s.append(f'<text x="{LX+58}" y="{LY+117}" fill="{B.SUB}" font-size="19">other people\'s GPUs</text>')
    s.append(f'<text x="{LX+24}" y="{LY+147}" fill="{B.MUT}" font-size="18">band: printed ±25% · '
             f'wedge: floor 1.1-1.8x</text>')

    # right column: KPI chips
    CX, CW = 1010, 500
    s.append(B.chip(CX, 380, CW, "MEDIAN ERROR, 14-ROW LADDER", f"{med:.1f}%",
                    "one law + one calibration, 0.5B to 35B", B.TEAL))
    s.append(B.chip(CX, 590, CW, "FIRST FOREIGN SILICON (AMD)", "+0.1%",
                    "RX 5700 XT, Vulkan - preset GPU constants", B.TEAL_DEEP))
    s.append(B.chip(CX, 800, CW, "THE MISS, ON THE CHART", "-67%",
                    "disk tier over-promised - relabeled, not hidden", B.DISK))

    s.append(B.footer(W, H, "ladder 2026-08-01 quiesced run · E-08/E-13 register · prereg #66 disk anchor"))
    s.append("</svg>")
    B.save("prediction_vs_reality.svg", "".join(s))


if __name__ == "__main__":
    main()
