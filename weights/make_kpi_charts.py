"""The three KPI charts, pure SVG (no matplotlib on this box), same house style as make_charts.py.

  python weights/make_kpi_charts.py   ->  weights/data/chart_kpi_*.svg

Every number here is MEASURED on the reference box and traceable to a pre-registration with its
verdict. Nothing is modelled, smoothed or interpolated - the curves are the measured points and
the lines only connect them. Where a denominator differs (the 0.6B's truncation quarantine) the
chart says so on its face rather than in a caption nobody reads.
"""
import os

W, H, PAD = 760, 400, 62
INK, SUB, LINE, GRID = "#16181d", "#5c6066", "#d8d7d2", "#eeedea"
TEAL, AMBER, RED, GREEN, SLATE = "#0f766e", "#b45309", "#b91c1c", "#15803d", "#475569"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")


def head(title, sub):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'font-family="Segoe UI,system-ui,sans-serif">',
            f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
            f'<text x="{PAD}" y="30" font-size="16" font-weight="650" fill="{INK}">{title}</text>',
            f'<text x="{PAD}" y="49" font-size="11.5" fill="{SUB}">{sub}</text>']


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ----------------------------------------------------------------------------------------
# 1. THE MODEL COMPARISON. Four models, one instrument, 52 executable predicates.
#    This is the chart people mean when they say "benchmark comparison" - except every bar
#    is a machine-checked pass rate, not a leaderboard score someone reported.
# ----------------------------------------------------------------------------------------
MODELS = [
    ("Qwen3-30B-A3B  2.95-bit", TEAL,  [(20, 20), (20, 20), (5, 6), (1, 6)], "recommended config"),
    ("Qwen2.5-7B  Q4_K_M",      GREEN, [(20, 20), (10, 20), (3, 6), (0, 6)], ""),
    ("Qwen2.5-7B  2-bit",       AMBER, [(18, 20), (9, 20), (4, 6), (0, 6)], "both quants identical"),
    ("Qwen3-0.6B  Q8_0",        RED,   [(15, 20), (7, 18), (3, 5), (1, 3)], "*truncations quarantined"),
]
TIERS = ["T1 routine", "T2 standard", "T3 hard", "T4 ceiling"]


def chart_ladder():
    s = head("Four models, one instrument: 52 executable predicates",
             "machine-checked pass rate - JSON to exact values, arithmetic to the cent, code that must "
             "run. Same box, same prompts, same checks.")
    x0, y0, x1, y1 = PAD + 8, 74, W - 168, H - 58
    gw = (x1 - x0) / len(TIERS)
    bw = gw / (len(MODELS) + 1.35)
    for pct in (0, 25, 50, 75, 100):
        y = y1 - (y1 - y0) * pct / 100
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-8}" y="{y+4:.1f}" font-size="10" fill="{SUB}" '
                 f'text-anchor="end">{pct}%</text>')
    for ti, tier in enumerate(TIERS):
        gx = x0 + gw * ti
        for mi, (name, col, scores, _) in enumerate(MODELS):
            got, tot = scores[ti]
            pct = 100.0 * got / tot
            bx = gx + bw * (mi + 0.7)
            bh = (y1 - y0) * pct / 100
            s.append(f'<rect x="{bx:.1f}" y="{y1-bh:.1f}" width="{bw*0.86:.1f}" height="{bh:.1f}" '
                     f'fill="{col}" rx="1.5"/>')
            s.append(f'<text x="{bx+bw*0.43:.1f}" y="{y1-bh-4:.1f}" font-size="9" fill="{SUB}" '
                     f'text-anchor="middle">{got}/{tot}</text>')
        s.append(f'<text x="{gx+gw/2:.1f}" y="{y1+16:.1f}" font-size="11" fill="{INK}" '
                 f'text-anchor="middle">{tier}</text>')
    # T4 is a ceiling by design - say so ON the chart, not in a caption
    t4x = x0 + gw * 3.5
    s.append(f'<text x="{t4x:.1f}" y="{y1+31:.1f}" font-size="9.5" fill="{SUB}" '
             f'text-anchor="middle">designed so today\'s models fail</text>')
    for mi, (name, col, _, note) in enumerate(MODELS):
        ly = 92 + mi * 26
        s.append(f'<rect x="{W-152}" y="{ly-9}" width="11" height="11" fill="{col}" rx="1.5"/>')
        s.append(f'<text x="{W-136}" y="{ly}" font-size="10.5" fill="{INK}">{esc(name)}</text>')
        if note:
            s.append(f'<text x="{W-136}" y="{ly+11}" font-size="8.5" fill="{SUB}">{esc(note)}</text>')
    s.append(f'<text x="{PAD}" y="{H-10}" font-size="9.5" fill="{SUB}">Staked pass bar was set '
             f'BEFORE any output existed. The 0.6B fires the suite\'s own kill rule (57.9% &lt; 60%).</text>')
    s.append("</svg>")
    return "chart_kpi_model_ladder.svg", "\n".join(s)


# ----------------------------------------------------------------------------------------
# 2. THE INVERSION. Aggregate throughput vs concurrent streams, two placements.
#    U-38 overturned our own C-06 here; U-39 confirmed the MoE cap as staked.
# ----------------------------------------------------------------------------------------
DENSE = [(1, 23.1), (2, 40.1), (4, 52.0), (8, 53.9), (9, 107.7), (12, 136.1), (16, 175.1), (32, 219.4)]
MOE = [(1, 19.7), (2, 28.5), (4, 31.4), (8, 37.5), (16, 37.1), (32, 40.0)]
XS = [1, 2, 4, 8, 9, 12, 16, 32]


def chart_batching():
    s = head("The best model inverts with user count (2016 GTX 1060, measured)",
             "aggregate decode across concurrent streams - dense 7B all-in-VRAM vs 30B MoE with experts in RAM")
    x0, y0, x1, y1 = PAD + 14, 76, W - 178, H - 56
    ymax = 240
    xpos = {n: x0 + (x1 - x0) * i / (len(XS) - 1) for i, n in enumerate(XS)}
    for v in (0, 60, 120, 180, 240):
        y = y1 - (y1 - y0) * v / ymax
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-8}" y="{y+4:.1f}" font-size="10" fill="{SUB}" text-anchor="end">{v}</text>')
    for n in XS:
        s.append(f'<text x="{xpos[n]:.1f}" y="{y1+16:.1f}" font-size="10" fill="{INK}" '
                 f'text-anchor="middle">{n}</text>')
    # the kernel boundary, drawn where it actually is
    bx = (xpos[8] + xpos[9]) / 2
    s.append(f'<line x1="{bx:.1f}" y1="{y0}" x2="{bx:.1f}" y2="{y1}" stroke="{SLATE}" '
             f'stroke-dasharray="3 3" opacity="0.55"/>')
    s.append(f'<text x="{bx+5:.1f}" y="{y0+12}" font-size="9.5" fill="{SLATE}">kernel switch (width 8-&gt;9)</text>')
    for pts, col, lab in ((DENSE, TEAL, "dense 7B Q4, all in VRAM"), (MOE, AMBER, "30B MoE, experts in RAM")):
        d = " ".join(f'{"M" if i == 0 else "L"}{xpos[n]:.1f},{y1-(y1-y0)*v/ymax:.1f}'
                     for i, (n, v) in enumerate(pts))
        s.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        for n, v in pts:
            s.append(f'<circle cx="{xpos[n]:.1f}" cy="{y1-(y1-y0)*v/ymax:.1f}" r="3.4" fill="{col}"/>')
    s.append(f'<text x="{xpos[32]-6:.1f}" y="{y1-(y1-y0)*219.4/ymax-9:.1f}" font-size="11" '
             f'font-weight="650" fill="{TEAL}" text-anchor="end">219.4</text>')
    s.append(f'<text x="{xpos[32]-6:.1f}" y="{y1-(y1-y0)*40.0/ymax-9:.1f}" font-size="11" '
             f'font-weight="650" fill="{AMBER}" text-anchor="end">40.0</text>')
    for i, (col, lab) in enumerate(((TEAL, "dense 7B Q4, all in VRAM"), (AMBER, "30B MoE, experts in RAM"))):
        ly = 96 + i * 22
        s.append(f'<rect x="{W-166}" y="{ly-9}" width="11" height="11" fill="{col}" rx="1.5"/>')
        s.append(f'<text x="{W-150}" y="{ly}" font-size="10.5" fill="{INK}">{esc(lab)}</text>')
    s.append(f'<text x="{W-166}" y="152" font-size="9.5" fill="{SUB}">1 user: the MoE is</text>')
    s.append(f'<text x="{W-166}" y="164" font-size="9.5" fill="{SUB}">the better model.</text>')
    s.append(f'<text x="{W-166}" y="182" font-size="9.5" fill="{SUB}">32 users: the dense</text>')
    s.append(f'<text x="{W-166}" y="194" font-size="9.5" fill="{SUB}">model wins by 5.5x.</text>')
    s.append(f'<text x="{W/2}" y="{H-8}" font-size="10.5" fill="{SUB}" text-anchor="middle">'
             f'concurrent streams</text>')
    s.append(f'<text x="14" y="{H/2}" font-size="10.5" fill="{SUB}" '
             f'transform="rotate(-90 14 {H/2})" text-anchor="middle">aggregate tok/s</text>')
    s.append("</svg>")
    return "chart_kpi_batching_inversion.svg", "\n".join(s)


# ----------------------------------------------------------------------------------------
# 3. THE DRAFT-LENGTH CLIFF (X-1). Single-stream speculation, same prompt, identical output.
# ----------------------------------------------------------------------------------------
DRAFTS = [(4, 48.9), (6, 51.2), (7, 48.2), (8, 88.5), (9, 98.2), (12, 124.3), (16, 122.5), (24, 132.1)]
BASE = 22.8


def chart_draft():
    s = head("Speculation draft length is a KERNEL decision, not an acceptance decision",
             "same 7B, same prompt, byte-identical output - only --spec-ngram-simple-size-m changes (X-1, staked)")
    x0, y0, x1, y1 = PAD + 14, 76, W - 150, H - 56
    ymax = 150
    xs = [d for d, _ in DRAFTS]
    xpos = {d: x0 + (x1 - x0) * i / (len(xs) - 1) for i, d in enumerate(xs)}
    for v in (0, 50, 100, 150):
        y = y1 - (y1 - y0) * v / ymax
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-8}" y="{y+4:.1f}" font-size="10" fill="{SUB}" text-anchor="end">{v}</text>')
    yb = y1 - (y1 - y0) * BASE / ymax
    s.append(f'<line x1="{x0}" y1="{yb:.1f}" x2="{x1}" y2="{yb:.1f}" stroke="{SLATE}" '
             f'stroke-dasharray="4 3"/>')
    s.append(f'<text x="{x1+5}" y="{yb+4:.1f}" font-size="9.5" fill="{SLATE}">off: {BASE}</text>')
    bx = (xpos[7] + xpos[8]) / 2
    s.append(f'<rect x="{x0}" y="{y0}" width="{bx-x0:.1f}" height="{y1-y0}" fill="{RED}" opacity="0.05"/>')
    s.append(f'<line x1="{bx:.1f}" y1="{y0}" x2="{bx:.1f}" y2="{y1}" stroke="{SLATE}" '
             f'stroke-dasharray="3 3" opacity="0.6"/>')
    s.append(f'<text x="{(x0+bx)/2:.1f}" y="{y0+14}" font-size="9.5" fill="{RED}" '
             f'text-anchor="middle">slow kernel - strictly dominated</text>')
    s.append(f'<text x="{bx+6:.1f}" y="{y0+14}" font-size="9.5" fill="{GREEN}">fast kernel (verify width &gt;= 9)</text>')
    d = " ".join(f'{"M" if i == 0 else "L"}{xpos[m]:.1f},{y1-(y1-y0)*v/ymax:.1f}'
                 for i, (m, v) in enumerate(DRAFTS))
    s.append(f'<path d="{d}" fill="none" stroke="{TEAL}" stroke-width="2.4"/>')
    for m, v in DRAFTS:
        s.append(f'<circle cx="{xpos[m]:.1f}" cy="{y1-(y1-y0)*v/ymax:.1f}" r="3.6" fill="{TEAL}"/>')
        s.append(f'<text x="{xpos[m]:.1f}" y="{y1+16:.1f}" font-size="10" fill="{INK}" '
                 f'text-anchor="middle">{m}</text>')
    s.append(f'<text x="{xpos[8]:.1f}" y="{y1-(y1-y0)*88.5/ymax-10:.1f}" font-size="11" '
             f'font-weight="650" fill="{GREEN}" text-anchor="middle">88.5</text>')
    s.append(f'<text x="{xpos[7]:.1f}" y="{y1-(y1-y0)*48.2/ymax-10:.1f}" font-size="11" '
             f'font-weight="650" fill="{RED}" text-anchor="middle">48.2</text>')
    s.append(f'<text x="{xpos[24]:.1f}" y="{y1-(y1-y0)*132.1/ymax-10:.1f}" font-size="11" '
             f'font-weight="650" fill="{TEAL}" text-anchor="middle">132.1 = 5.8x</text>')
    s.append(f'<text x="{W/2}" y="{H-8}" font-size="10.5" fill="{SUB}" text-anchor="middle">'
             f'draft length m (verify width = m + 1)</text>')
    s.append(f'<text x="14" y="{H/2}" font-size="10.5" fill="{SUB}" '
             f'transform="rotate(-90 14 {H/2})" text-anchor="middle">decode tok/s</text>')
    s.append("</svg>")
    return "chart_kpi_draft_cliff.svg", "\n".join(s)


def main():
    os.makedirs(OUT, exist_ok=True)
    for fn, svg in (chart_ladder(), chart_batching(), chart_draft()):
        with open(os.path.join(OUT, fn), "w", encoding="utf-8") as f:
            f.write(svg)
        print("wrote", os.path.join("weights", "data", fn))


if __name__ == "__main__":
    main()
