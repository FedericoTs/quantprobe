"""Size does not predict speed - active bytes and placement do.

Every point is a measured row on ONE machine (GTX 1060 6GB / 16GB DDR4). File sizes are read
off disk at render time; speeds come from the committed ladder result. The disk-tier point is
the prereg #66 anchor - included precisely because it is the one we over-promised on.

The headline is not "big models are slow". It is the INVERSION sitting in the middle of the
plot: a 14B dense at 8.99 GB decodes at 5.49 t/s while a 30B MoE at 11.26 GB decodes at 22.94
- 26% MORE bytes on disk, 4.2x FASTER - because a MoE token reads only its active experts.
"""
from __future__ import annotations
import json, math, os, sys
import brand as B

HERE = os.path.dirname(os.path.abspath(__file__))
GGUF = "D:/evo-compress-data/gguf"
LADDER = os.path.join(HERE, "data", "unattended_20260801_002809_ladder_result.json")

W, H = 1600, 1180
PX0, PY0, PW, PH = 150, 330, 810, 640
XMIN, XMAX = 0.4, 60.0        # GB
YMIN, YMAX = 0.4, 260.0       # tok/s

# The one row not in the ladder JSON: prereg #66's disk-tier anchor, quoted with its miss.
DISK_ROW = ("35B Q8_0, streamed from disk", 36.9, 0.66, "disk")


def px(v):
    return PX0 + (math.log10(v) - math.log10(XMIN)) / (math.log10(XMAX) - math.log10(XMIN)) * PW


def py(v):
    return PY0 + PH - (math.log10(v) - math.log10(YMIN)) / (math.log10(YMAX) - math.log10(YMIN)) * PH


def tier_of(placement):
    if "all in VRAM" in placement:
        return "VRAM"
    if "disk" in placement:
        return "disk"
    return "split"


COL = {"VRAM": B.VRAM, "split": B.TEAL, "disk": B.DISK}


def main():
    rows = []
    for r in json.load(open(LADDER, encoding="utf-8")):
        p = os.path.join(GGUF, r["file"])
        if not os.path.exists(p):
            continue
        rows.append((r["name"], os.path.getsize(p) / 1e9, r["measured"], tier_of(r["placement"])))
    rows.append(DISK_ROW)

    s = [B.svg_open(W, H),
         B.header(W, "MEASURED · ONE MACHINE · GTX 1060 6GB + 16GB DDR4",
                  "Size does not predict speed",
                  "Every point a measured row on the same box. What sets decode speed is active "
                  "bytes per token and where they live - not the file size.")]

    for t in (0.5, 1, 2, 5, 10, 20, 50):
        s.append(f'<line x1="{px(t)}" y1="{PY0}" x2="{px(t)}" y2="{PY0+PH}" stroke="{B.GRID}" '
                 f'stroke-width="1"/>')
        s.append(f'<text x="{px(t)}" y="{PY0+PH+32}" text-anchor="middle" fill="{B.MUT}" '
                 f'font-size="19">{t:g}</text>')
    for t in (0.5, 1, 2, 5, 10, 25, 50, 100, 200):
        s.append(f'<line x1="{PX0}" y1="{py(t)}" x2="{PX0+PW}" y2="{py(t)}" stroke="{B.GRID}" '
                 f'stroke-width="1"/>')
        s.append(f'<text x="{PX0-14}" y="{py(t)+7}" text-anchor="end" fill="{B.MUT}" '
                 f'font-size="19">{t:g}</text>')
    s.append(f'<text x="{PX0+PW/2}" y="{PY0+PH+70}" text-anchor="middle" fill="{B.SUB}" '
             f'font-size="22">file size on disk, GB (log)</text>')
    s.append(f'<text x="{PX0-96}" y="{PY0+PH/2}" fill="{B.SUB}" font-size="22" text-anchor="middle" '
             f'transform="rotate(-90 {PX0-96} {PY0+PH/2})">measured tok/s (log)</text>')

    # the inversion, drawn before the points so the arc sits underneath
    a = next(r for r in rows if r[0].startswith("Qwen2.5-14B"))
    b = next(r for r in rows if r[0].startswith("Qwen3-30B"))
    s.append(f'<path d="M {px(a[1])} {py(a[2])} Q {px(a[1])+70} {py(a[2])-150} {px(b[1])} {py(b[2])}" '
             f'fill="none" stroke="{B.INK}" stroke-width="2.5" stroke-dasharray="8 6" opacity="0.6"/>')

    for name, gb, tps, tier in rows:
        x, y = px(gb), py(tps)
        r = 9 + 7 * math.log10(1 + gb)
        s.append(f'<circle cx="{x}" cy="{y}" r="{r:.1f}" fill="{COL[tier]}" '
                 f'stroke="{B.BG}" stroke-width="2.5" opacity="0.92"/>')

    # labels: the extremes and the story points only, placed clear of the cloud
    lab = [("Qwen2.5-0.5B", 24, 6, "start"), ("Qwen2.5-14B", -22, 8, "end"),
           ("Qwen3-30B-A3B", 20, 34, "start"), ("35B Q8_0", -26, 6, "end")]
    for pre, dx, dy, an in lab:
        m = next((r for r in rows if r[0].startswith(pre)), None)
        if not m:
            continue
        s.append(f'<text x="{px(m[1])+dx}" y="{py(m[2])+dy}" fill="{B.SUB}" font-size="19" '
                 f'text-anchor="{an}">{m[0]}</text>')

    # the inversion callout - above the arc, right of the 30B cluster
    s.append(f'<text x="{PX0+PW}" y="{py(b[2])-96}" text-anchor="end" fill="{B.INK}" '
             f'font-size="25" font-weight="bold">+26% bytes, 4.2x faster</text>')
    s.append(f'<text x="{PX0+PW}" y="{py(b[2])-68}" text-anchor="end" fill="{B.SUB}" '
             f'font-size="19">14B dense 5.49 t/s &#8594; 30B MoE 22.94 t/s</text>')

    # legend: bottom-left, the one genuinely empty quadrant
    LX, LY = PX0 + 22, PY0 + PH - 152
    s.append(B.panel(LX, LY, 250, 128))
    for i, (k, txt) in enumerate((("VRAM", "all in VRAM"), ("split", "split VRAM+RAM"),
                                  ("disk", "streamed from disk"))):
        s.append(f'<circle cx="{LX+32}" cy="{LY+34+i*32}" r="10" fill="{COL[k]}"/>')
        s.append(f'<text x="{LX+56}" y="{LY+41+i*32}" fill="{B.SUB}" font-size="19">{txt}</text>')

    CX, CW = 1010, 500
    s.append(B.chip(CX, 380, CW, "SPEED RANGE, ONE BOX", "240x",
                    "158.7 t/s down to 0.66 - same machine", B.TEAL))
    s.append(B.chip(CX, 590, CW, "THE POINT THAT BREAKS THE TREND", "4.2x",
                    "a bigger MoE file, four times the speed", B.VRAM))
    s.append(B.chip(CX, 800, CW, "THE DISK ROW WE OVER-PROMISED", "-67%",
                    "predicted 2.0, measured 0.66 - shipped labelled", B.DISK))

    s.append(B.footer(W, H, "ladder 2026-08-01 quiesced run · sizes read from disk · prereg #66 disk anchor"))
    s.append("</svg>")
    B.save("size_vs_speed.svg", "".join(s))


if __name__ == "__main__":
    main()
