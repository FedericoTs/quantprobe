"""Charts for the 2026-07-30 findings. Pure-SVG (no matplotlib on this box), one file per
finding, dark/light neutral so they render on GitHub either way.

  python weights/make_charts.py     ->  weights/data/chart_*.svg
"""
import json
import os

W, H, PAD = 720, 380, 58
INK, SUB, LINE = "#16181d", "#5c6066", "#d8d7d2"
A, B, WARN, GOOD = "#0f766e", "#b45309", "#b91c1c", "#15803d"
HERE = os.path.dirname(os.path.abspath(__file__))


def frame(title, sub, ylab, xlab):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI,system-ui,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{PAD}" y="30" font-size="16" font-weight="650" fill="{INK}">{title}</text>',
        f'<text x="{PAD}" y="49" font-size="12" fill="{SUB}">{sub}</text>',
        f'<text x="14" y="{H/2}" font-size="11" fill="{SUB}" transform="rotate(-90 14 {H/2})" text-anchor="middle">{ylab}</text>',
        f'<text x="{W/2}" y="{H-8}" font-size="11" fill="{SUB}" text-anchor="middle">{xlab}</text>',
    ]


def axes(x0, y0, x1, y1):
    return [f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{LINE}"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{LINE}"/>']


def chart_shape():
    """L-20: bandwidth vs rows/tensor, two row widths (preregs #80/#81)."""
    # BOTH curves from the SAME session (prereg81_knee.log). #80's standalone K=2048 run gave
    # 31.0/64.5/88.8/100.5 - within session variance, but plotting two curves from two sessions
    # is the cross-state comparison C-14 exists to forbid, so the chart uses one run only.
    k2 = [(128, 30.2), (256, 45.4), (512, 61.1), (1024, 76.5), (2048, 88.4),
          (4096, 94.2), (8192, 96.9), (16384, 98.7)]
    k4 = [(128, 49.2), (256, 67.3), (512, 92.3), (1024, 111.6), (2048, 121.4),
          (4096, 128.7), (8192, 131.6), (16384, 138.5)]
    x0, y0, x1, y1 = PAD + 22, 70, W - 150, H - 46
    ymax = 200.0
    import math
    def px(r): return x0 + (math.log2(r) - 7) / 7 * (x1 - x0)
    def py(v): return y1 - v / ymax * (y1 - y0)
    s = frame("L-20: decode bandwidth is set by tensor GEOMETRY, not format alone",
              "same card, same 4.5-bit format, same kernel, same bytes swept - only the shape changes (preregs #80/#81)",
              "effective GB/s", "rows per tensor (log2)")
    s += axes(x0, y0, x1, y1)
    s.append(f'<line x1="{x0}" y1="{py(192.2)}" x2="{x1}" y2="{py(192.2)}" stroke="{SUB}" '
             f'stroke-dasharray="4 4"/><text x="{x1+6}" y="{py(192.2)+4}" font-size="10" fill="{SUB}">spec peak 192</text>')
    for name, pts, col in (("K=4096 (2304 B/row)", k4, B), ("K=2048 (1152 B/row)", k2, A)):
        d = " ".join(f"{'M' if i == 0 else 'L'}{px(r):.1f},{py(v):.1f}" for i, (r, v) in enumerate(pts))
        s.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        for r, v in pts:
            s.append(f'<circle cx="{px(r):.1f}" cy="{py(v):.1f}" r="3.5" fill="{col}"/>')
        lx, ly = px(pts[-1][0]), py(pts[-1][1])
        s.append(f'<text x="{lx+8}" y="{ly+4}" font-size="11" fill="{col}">{name}</text>')
    for r in (128, 512, 2048, 8192, 16384):
        s.append(f'<text x="{px(r):.1f}" y="{y1+15}" font-size="10" fill="{SUB}" text-anchor="middle">{r}</text>')
    for v in (0, 50, 100, 150):
        s.append(f'<text x="{x0-8}" y="{py(v)+4}" font-size="10" fill="{SUB}" text-anchor="end">{v}</text>')
    kx = px(4096)
    s.append(f'<line x1="{kx:.1f}" y1="{y0}" x2="{kx:.1f}" y2="{y1}" stroke="{GOOD}" stroke-dasharray="3 3"/>')
    s.append(f'<text x="{kx+6:.1f}" y="{y0+14}" font-size="10" fill="{GOOD}">knee ~4096 rows - SAME for both widths</text>')
    s.append(f'<text x="{kx+6:.1f}" y="{y0+28}" font-size="10" fill="{GOOD}">(occupancy floor, not launch cost)</text>')
    s.append(f'<text x="{px(180):.1f}" y="{py(31)-12:.1f}" font-size="10" fill="{WARN}">attention-shaped: -23% vs FFN</text>')
    s.append("</svg>")
    return "chart_L20_tensor_shape.svg", "\n".join(s)


def chart_ladder():
    """The state-locked ladder: predicted vs measured, one machine state."""
    p = os.path.join(HERE, "data", "ladder_state_locked.json")
    rows = [r for r in json.load(open(p, encoding="utf-8")) if r.get("measured")]
    x0, y0, x1, y1 = PAD + 30, 70, W - 40, H - 46
    n = len(rows)
    lo, hi = -30.0, 30.0
    def py(i): return y0 + (i + 0.5) / n * (y1 - y0)
    def px(e): return x0 + (max(lo, min(hi, e)) - lo) / (hi - lo) * (x1 - x0)
    s = frame("Predicted vs measured - all 14 models, ONE locked machine state",
              f"state {rows[0]['cal_id']}: predictions and measurements from the same calibration (C-14). "
              "Median |error| 8.8%, 14/14 inside the printed band.",
              "", "prediction error (%)   negative = we under-promised")
    s += axes(x0, y0, x1, y1)
    for band, col, op in ((5, GOOD, 0.10), (25, SUB, 0.05)):
        s.append(f'<rect x="{px(-band):.1f}" y="{y0}" width="{px(band)-px(-band):.1f}" '
                 f'height="{y1-y0}" fill="{col}" opacity="{op}"/>')
    s.append(f'<line x1="{px(0):.1f}" y1="{y0}" x2="{px(0):.1f}" y2="{y1}" stroke="{INK}" stroke-width="1"/>')
    for i, r in enumerate(rows):
        e = r["err_pct"]; y = py(i)
        col = GOOD if abs(e) <= 5 else (A if abs(e) <= 10 else (B if abs(e) <= 25 else WARN))
        s.append(f'<line x1="{px(0):.1f}" y1="{y:.1f}" x2="{px(e):.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="2"/>')
        s.append(f'<circle cx="{px(e):.1f}" cy="{y:.1f}" r="4" fill="{col}"/>')
        s.append(f'<text x="{x0-6}" y="{y+3.5:.1f}" font-size="9.5" fill="{INK}" text-anchor="end">{r["name"][:24]}</text>')
        s.append(f'<text x="{px(e)+ (7 if e>=0 else -7):.1f}" y="{y+3.5:.1f}" font-size="9" fill="{col}" '
                 f'text-anchor="{"start" if e>=0 else "end"}">{e:+.0f}%</text>')
    for v in (-25, -10, 0, 10, 25):
        s.append(f'<text x="{px(v):.1f}" y="{y1+14}" font-size="10" fill="{SUB}" text-anchor="middle">{v:+d}</text>')
    s.append("</svg>")
    return "chart_ladder_state_locked.svg", "\n".join(s)


def chart_79():
    """#79's damage vs attention share - why the per-tier fix failed (r=0.87)."""
    pts = [("DS-Lite Q4KM", 0.45, -0.4), ("Qwen3-30B", 0.51, 8.8), ("Coder-30B", 0.51, 11.1),
           ("3.5-35B APEX", 0.74, 12.8), ("3.6-35B Q2KXL", 0.75, 27.1), ("3.6-APEX-MTP", 0.77, 26.9)]
    x0, y0, x1, y1 = PAD + 22, 76, W - 60, H - 46
    def px(s_): return x0 + (s_ - 0.40) / 0.42 * (x1 - x0)
    def py(v): return y1 - (v + 5) / 40 * (y1 - y0)
    s = frame("Why per-tier pricing failed (prereg #79) - and what it revealed",
              "the damage tracks how much of the token we believe is ATTENTION (r = 0.87, n = 6): "
              "attention was priced at a bandwidth only FFN-shaped tensors reach",
              "prediction error added (points)", "attention share of the per-token read")
    s += axes(x0, y0, x1, y1)
    s.append(f'<line x1="{x0}" y1="{py(0):.1f}" x2="{x1}" y2="{py(0):.1f}" stroke="{LINE}" stroke-dasharray="3 3"/>')
    for name, sh, d in pts:
        col = GOOD if d < 5 else (B if d < 20 else WARN)
        s.append(f'<circle cx="{px(sh):.1f}" cy="{py(d):.1f}" r="6" fill="{col}" opacity="0.85"/>')
        s.append(f'<text x="{px(sh):.1f}" y="{py(d)-11:.1f}" font-size="9.5" fill="{INK}" text-anchor="middle">{name}</text>')
    s.append(f'<line x1="{px(0.44):.1f}" y1="{py(-1):.1f}" x2="{px(0.78):.1f}" y2="{py(28):.1f}" '
             f'stroke="{SUB}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    for v in (0.45, 0.55, 0.65, 0.75):
        s.append(f'<text x="{px(v):.1f}" y="{y1+14}" font-size="10" fill="{SUB}" text-anchor="middle">{v:.0%}</text>')
    for v in (0, 10, 20, 30):
        s.append(f'<text x="{x0-8}" y="{py(v)+4:.1f}" font-size="10" fill="{SUB}" text-anchor="end">{v:+d}</text>')
    s.append("</svg>")
    return "chart_79_attention_share.svg", "\n".join(s)


def chart_u33():
    """U-33: the op-count term refuted (left) and what survived (right)."""
    import math
    loo = [(0.00,20.9),(0.05,28.2),(0.10,29.7),(0.15,28.7),(0.20,27.7),(0.25,26.7),
           (0.30,25.7),(0.35,24.8),(0.40,23.8),(0.45,22.8),(0.50,21.8),(0.55,20.8),
           (0.60,19.8),(0.65,18.8),(0.70,17.8),(0.75,16.8),(0.80,15.9),(0.85,14.9),
           (0.90,14.0),(0.95,13.0),(1.00,12.1)]
    SHIP = 8.7
    pts = [("0.5B",0.708,0.889),("0.6B",0.591,0.878),("4B",0.821,0.961),
           ("7B IQ4",0.977,0.981),("7B Q4KM",1.017,0.981),("g12B",0.726,0.982)]
    s = frame("U-33: op count cannot price the overhead - tensor shape might",
              "left: every form of a launch-count term loses out of sample.  "
              "right: the L-20 knee curve, zero fitted parameters.",
              "", "")
    # ---- left panel: LOO vs exponent p
    x0,y0,x1,y1 = PAD+8, 84, 330, H-58
    s += axes(x0,y0,x1,y1)
    def lx(p_): return x0 + p_*(x1-x0)
    def ly(v):  return y1 - (v-5)/28*(y1-y0)
    s.append(f'<rect x="{x0}" y="{ly(SHIP):.1f}" width="{x1-x0}" height="{y1-ly(SHIP):.1f}" '
             f'fill="{GOOD}" opacity="0.07"/>')
    s.append(f'<line x1="{x0}" y1="{ly(SHIP):.1f}" x2="{x1}" y2="{ly(SHIP):.1f}" '
             f'stroke="{GOOD}" stroke-width="2" stroke-dasharray="5 3"/>')
    s.append(f'<text x="{x0+5}" y="{ly(SHIP)+13:.1f}" font-size="10" fill="{GOOD}">'
             f'shipped model: {SHIP}% - nothing below this line was reached</text>')
    d = " ".join(f"{'M' if i==0 else 'L'}{lx(a):.1f},{ly(b):.1f}" for i,(a,b) in enumerate(loo))
    s.append(f'<path d="{d}" fill="none" stroke="{WARN}" stroke-width="2.5"/>')
    for a,b in (loo[0], loo[-1]):
        s.append(f'<circle cx="{lx(a):.1f}" cy="{ly(b):.1f}" r="4" fill="{WARN}"/>')
    s.append(f'<text x="{lx(0)+4:.1f}" y="{ly(20.9)-9:.1f}" font-size="10" fill="{SUB}">'
             f'p=0: pure per-layer</text>')
    s.append(f'<text x="{lx(1.0)-4:.1f}" y="{ly(12.1)-9:.1f}" font-size="10" fill="{SUB}" '
             f'text-anchor="end">p=1: flat per-op (best, still loses)</text>')
    for v in (10,20,30):
        s.append(f'<text x="{x0-6}" y="{ly(v)+4:.1f}" font-size="10" fill="{SUB}" '
                 f'text-anchor="end">{v}%</text>')
    s.append(f'<text x="{(x0+x1)/2:.1f}" y="{y1+16:.1f}" font-size="10" fill="{SUB}" '
             f'text-anchor="middle">exponent p in  tax = c x layers x (ops/layer)^p</text>')
    s.append(f'<text x="{x0}" y="{y0-8}" font-size="11" font-weight="600" fill="{INK}">'
             f'leave-one-out median error (n=6, 2 params)</text>')
    # ---- right panel: needed vs shape-predicted
    a0,a1 = 420, W-46
    s += axes(a0,y0,a1,y1)
    def rx(v): return a0 + (v-0.55)/0.50*(a1-a0)
    def ry(v): return y1 - (v-0.55)/0.50*(y1-y0)
    s.append(f'<line x1="{rx(0.55):.1f}" y1="{ry(0.55):.1f}" x2="{rx(1.05):.1f}" '
             f'y2="{ry(1.05):.1f}" stroke="{SUB}" stroke-dasharray="4 4"/>')
    s.append(f'<text x="{rx(0.90):.1f}" y="{ry(0.86):.1f}" font-size="9" fill="{SUB}" '
             f'transform="rotate(-45 {rx(0.90):.1f} {ry(0.86):.1f})">perfect</text>')
    for nm,need,shp in pts:
        anchor = nm.startswith("7B")
        col = SUB if anchor else (WARN if nm=="g12B" else A)
        s.append(f'<circle cx="{rx(need):.1f}" cy="{ry(shp):.1f}" r="5" fill="{col}" '
                 f'opacity="0.9"/>')
        s.append(f'<text x="{rx(need):.1f}" y="{ry(shp)-10:.1f}" font-size="9" fill="{INK}" '
                 f'text-anchor="middle">{nm}</text>')
    s.append(f'<text x="{a0}" y="{y0-8}" font-size="11" font-weight="600" fill="{INK}">'
             f'shape-predicted vs actually needed   r = +0.77</text>')
    s.append(f'<text x="{(a0+a1)/2:.1f}" y="{y1+16:.1f}" font-size="10" fill="{SUB}" '
             f'text-anchor="middle">fraction of nominal FORMAT_EBW achieved</text>')
    s.append(f'<text x="{a0+4}" y="{y1-8}" font-size="9.5" fill="{SUB}">'
             f'grey = the 7B calibration anchor (residual ~0 by construction)</text>')
    s.append(f'<text x="{a0+4}" y="{y1-(-4)-24:.0f}" font-size="9.5" fill="{WARN}">'
             f'gemma: deficit is attention-shaped, not geometry</text>')
    s.append("</svg>")
    return "chart_U33_opcount_refuted.svg", "\n".join(s)


if __name__ == "__main__":
    out = os.path.join(HERE, "data")
    for fn, svg in (chart_shape(), chart_ladder(), chart_79(), chart_u33()):
        with open(os.path.join(out, fn), "w", encoding="utf-8") as f:
            f.write(svg + "\n")
        print("wrote", fn)
