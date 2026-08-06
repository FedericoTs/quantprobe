"""Stuck-boost chart - "your GPU silently lost 28% and only a reboot fixes it".

Every number cited from committed measurements: prereg #60 (the degraded block + orphan
kill + not-thermal control) and prereg #61 (cold-boot A/B, weights/data/prereg61_coldboot.log).
Bars are zero-based BY THE AXIS RULE: magnitude of the loss IS the message here.

Two cuts from one data source:
  python make_stuck_boost.py            -> media/stuck_boost_state.svg   (portfolio cut)
  python make_stuck_boost.py --reddit   -> media/stuck_boost_reddit.svg  (r/LocalLLaMA cut:
      reader-centric headline, the self-check command ON the asset so a re-post without our
      comment still carries the action, and the n=1 scope stated in the subtitle rather than
      discovered in the replies.)
"""
from __future__ import annotations
import sys
import brand as B

W, H = 1600, 1200

# (label, tok/s, +-sd, bar color, sm MHz, mem MHz, note)
BARS = [
    ("DAY 1", 21.58, None, B.TEAL_DEEP, 1835, 4004, "calibration day"),
    ("DAY 2 - same box", 15.56, None, B.DISK, 1506, 3802, "stuck boost state"),
    ("AFTER REBOOT", 21.68, 0.20, B.TEAL, 1873, 4006, "position control"),
]
YMAX = 24.0

CH_X, CH_Y, CH_W, CH_H = 110, 360, 850, 620   # plot box (bars live here)
BAR_W = 170


def bar_x(i):
    gap = (CH_W - 3 * BAR_W) / 4
    return CH_X + gap + i * (BAR_W + gap)


def y_of(v):
    return CH_Y + CH_H - (v / YMAX) * CH_H


def main():
    s = [B.svg_open(W, H),
         B.header(W, "MEASURED · GTX 1060 6GB · SAME MODEL, SAME BINARY, SAME FLAGS",
                  "The stuck boost state",
                  "Qwen3-Coder-30B decode, tg128 - the box lost 28% overnight, and everything "
                  "obvious was ruled out before the fix.")]

    # gridlines + y labels
    for v in (0, 5, 10, 15, 20):
        y = y_of(v)
        s.append(f'<line x1="{CH_X}" y1="{y}" x2="{CH_X+CH_W}" y2="{y}" '
                 f'stroke="{B.GRID}" stroke-width="1.5"/>')
        s.append(f'<text x="{CH_X-14}" y="{y+7}" text-anchor="end" fill="{B.MUT}" '
                 f'font-size="19">{v}</text>')
    s.append(f'<text x="{CH_X-58}" y="{CH_Y-26}" fill="{B.SUB}" font-size="21">tok/s</text>')

    # day-1 reference line across the plot (dash = second cue beyond color)
    yref = y_of(21.58)
    s.append(f'<line x1="{CH_X}" y1="{yref}" x2="{CH_X+CH_W}" y2="{yref}" '
             f'stroke="{B.TEAL_DEEP}" stroke-width="2.5" stroke-dasharray="10 8" opacity="0.65"/>')

    for i, (label, v, sd, color, sm, mem, note) in enumerate(BARS):
        x, y = bar_x(i), y_of(v)
        s.append(f'<rect x="{x}" y="{y}" width="{BAR_W}" height="{CH_Y+CH_H-y}" rx="10" '
                 f'fill="{color}"/>')
        if i == 1:  # hatch = second cue on the pathological bar
            for hx in range(int(x) + 16, int(x + BAR_W), 26):
                s.append(f'<line x1="{hx}" y1="{y+8}" x2="{hx-18}" y2="{CH_Y+CH_H-6}" '
                         f'stroke="{B.BG}" stroke-width="3" opacity="0.35"/>')
        s.append(f'<text x="{x+BAR_W/2}" y="{y+58}" text-anchor="middle" fill="{B.BG}" '
                 f'font-size="44" font-weight="bold">{v:.2f}</text>')
        if sd:  # inside the bar - drawn in BG ink, so it must never sit on the canvas
            s.append(f'<text x="{x+BAR_W/2}" y="{y+90}" text-anchor="middle" fill="{B.BG}" '
                     f'font-size="21" opacity="0.75">± {sd}</text>')
        s.append(f'<text x="{x+BAR_W/2}" y="{CH_Y+CH_H+40}" text-anchor="middle" '
                 f'fill="{B.INK}" font-size="23" font-weight="bold">{label}</text>')
        s.append(f'<text x="{x+BAR_W/2}" y="{CH_Y+CH_H+72}" text-anchor="middle" '
                 f'fill="{B.MUT}" font-size="20">{note}</text>')
        cc = color if i == 1 else B.SUB
        s.append(f'<text x="{x+BAR_W/2}" y="{CH_Y+CH_H+106}" text-anchor="middle" '
                 f'fill="{cc}" font-size="19" font-weight="bold">SM {sm} · mem {mem}</text>')

    # hero: the loss, in the clean sky above the stuck bar
    hx = bar_x(1) + BAR_W / 2
    s.append(f'<text x="{hx}" y="{y_of(15.56)-26}" text-anchor="middle" fill="{B.DISK}" '
             f'font-size="104" font-weight="bold">-28%</text>')
    # recovery note over bar 3
    s.append(f'<text x="{bar_x(2)+BAR_W/2}" y="{yref-40}" text-anchor="middle" fill="{B.TEAL}" '
             f'font-size="22" font-weight="bold">day 1 reproduced to 0.5%</text>')

    # right panel: what it WASN'T (the diagnostic spine)
    PX, PW = 1010, 510
    s.append(B.panel(PX, CH_Y - 10, PW, 636))
    s.append(f'<text x="{PX+36}" y="{CH_Y+46}" fill="{B.INK}" font-size="26" '
             f'font-weight="bold">Ruled out, in order</text>')
    items = [
        ("✗", B.DISK, "Background load", "runaway orphan killed (16,285 CPU-s)", "still 15.3 tok/s"),
        ("✗", B.DISK, "Thermal", "box idled to 32 °C, GPU quiet", "still 15.56 tok/s"),
        ("✗", B.DISK, "Our instrumentation", "zero-patch pristine binary, same commit", "agrees within 1.4%"),
        ("✓", B.TEAL, "Stuck boost state", "SM pinned 1506 MHz under load, 38 °C", "reboot -> 1847-1898 MHz"),
    ]
    for j, (mark, mc, t, d, r) in enumerate(items):
        iy = CH_Y + 92 + j * 106
        s.append(f'<text x="{PX+36}" y="{iy+18}" fill="{mc}" font-size="34" '
                 f'font-weight="bold">{mark}</text>')
        s.append(f'<text x="{PX+92}" y="{iy+10}" fill="{B.INK}" font-size="23" '
                 f'font-weight="bold">{t}</text>')
        s.append(f'<text x="{PX+92}" y="{iy+42}" fill="{B.SUB}" font-size="20">{d}</text>')
        s.append(f'<text x="{PX+92}" y="{iy+72}" fill="{mc}" font-size="20" '
                 f'font-weight="bold">{r}</text>')
    s.append(f'<line x1="{PX+36}" y1="{CH_Y+512}" x2="{PX+PW-36}" y2="{CH_Y+512}" '
             f'stroke="{B.GRID}" stroke-width="1.5"/>')
    s.append(f'<text x="{PX+36}" y="{CH_Y+554}" fill="{B.INK}" font-size="22" '
             f'font-weight="bold">No software reset on consumer Pascal.</text>')
    s.append(f'<text x="{PX+36}" y="{CH_Y+588}" fill="{B.SUB}" font-size="21">If your numbers '
             f'sagged and nothing explains it:</text>')
    s.append(f'<text x="{PX+36}" y="{CH_Y+618}" fill="{B.SUB}" font-size="21">log clocks under '
             f'load, then reboot.</text>')

    s.append(B.footer(W, H, "preregs #60-#61 · staked before the reboot · weights/data/prereg61_coldboot.log"))
    s.append("</svg>")
    B.save("stuck_boost_state.svg", "".join(s))


CHECK_CMD = ("nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu "
             "--format=csv -l 1")


def reddit():
    """The r/LocalLLaMA cut. Same three bars, same ruled-out spine, but the reader is the
    protagonist: their symptom in the title, the check command on the canvas, the scope
    limit stated up front so 'doesn't repro on my 4090' arrives as a datapoint we asked
    for rather than a debunk."""
    CY, CH = 336, 440                 # shorter plot: the check strip owns the bottom
    CX, CW = 110, 850
    BW = 170

    def bx(i):
        gap = (CW - 3 * BW) / 4
        return CX + gap + i * (BW + gap)

    def yv(v):
        return CY + CH - (v / YMAX) * CH

    s = [B.svg_open(W, H),
         B.header(W, "SAME MODEL · SAME BINARY · SAME FLAGS · NOTHING IN THE LOGS SAID SO",
                  "Your GPU can quietly lose 28%",
                  "GTX 1060 6GB · Windows 10 · Qwen3-Coder-30B decode · n=1. Do Ampere/Ada "
                  "do this too? We don't know - please replicate.")]

    for v in (0, 5, 10, 15, 20):
        y = yv(v)
        s.append(f'<line x1="{CX}" y1="{y}" x2="{CX+CW}" y2="{y}" stroke="{B.GRID}" '
                 f'stroke-width="1.5"/>')
        s.append(f'<text x="{CX-14}" y="{y+7}" text-anchor="end" fill="{B.MUT}" '
                 f'font-size="19">{v}</text>')
    s.append(f'<text x="{CX-58}" y="{CY-22}" fill="{B.SUB}" font-size="21">tok/s</text>')
    yref = yv(21.58)
    s.append(f'<line x1="{CX}" y1="{yref}" x2="{CX+CW}" y2="{yref}" stroke="{B.TEAL_DEEP}" '
             f'stroke-width="2.5" stroke-dasharray="10 8" opacity="0.65"/>')

    for i, (label, v, sd, color, sm, mem, note) in enumerate(BARS):
        x, y = bx(i), yv(v)
        s.append(f'<rect x="{x}" y="{y}" width="{BW}" height="{CY+CH-y}" rx="10" fill="{color}"/>')
        if i == 1:
            for hx in range(int(x) + 16, int(x + BW), 26):
                s.append(f'<line x1="{hx}" y1="{y+8}" x2="{hx-18}" y2="{CY+CH-6}" '
                         f'stroke="{B.BG}" stroke-width="3" opacity="0.35"/>')
        s.append(f'<text x="{x+BW/2}" y="{y+54}" text-anchor="middle" fill="{B.BG}" '
                 f'font-size="42" font-weight="bold">{v:.2f}</text>')
        if sd:
            s.append(f'<text x="{x+BW/2}" y="{y+86}" text-anchor="middle" fill="{B.BG}" '
                     f'font-size="21" opacity="0.75">± {sd}</text>')
        s.append(f'<text x="{x+BW/2}" y="{CY+CH+38}" text-anchor="middle" fill="{B.INK}" '
                 f'font-size="23" font-weight="bold">{label}</text>')
        s.append(f'<text x="{x+BW/2}" y="{CY+CH+68}" text-anchor="middle" fill="{B.MUT}" '
                 f'font-size="20">{note}</text>')
        cc = color if i == 1 else B.SUB
        s.append(f'<text x="{x+BW/2}" y="{CY+CH+100}" text-anchor="middle" fill="{cc}" '
                 f'font-size="19" font-weight="bold">SM {sm} · mem {mem}</text>')

    s.append(f'<text x="{bx(1)+BW/2}" y="{yv(15.56)-24}" text-anchor="middle" fill="{B.DISK}" '
             f'font-size="96" font-weight="bold">-28%</text>')
    s.append(f'<text x="{bx(2)+BW/2}" y="{yref-34}" text-anchor="middle" fill="{B.TEAL}" '
             f'font-size="22" font-weight="bold">day 1 reproduced to 0.5%</text>')

    PX, PW = 1010, 510
    s.append(B.panel(PX, CY - 16, PW, 600))
    s.append(f'<text x="{PX+34}" y="{CY+34}" fill="{B.INK}" font-size="26" '
             f'font-weight="bold">Ruled out first, in order</text>')
    items = [
        ("✗", B.DISK, "Background load", "killed a runaway orphan (16,285 CPU-s)", "still 15.3 tok/s"),
        ("✗", B.DISK, "Thermal", "box idled down to 32 °C", "still 15.56 tok/s"),
        ("✗", B.DISK, "Our own patches", "zero-patch build, identical commit", "agrees within 1.4%"),
        ("✓", B.TEAL, "Stuck boost state", "SM pinned at 1506 MHz under load, 38 °C",
         "reboot -> 1847-1898 MHz"),
    ]
    for j, (mark, mc, t, d, r) in enumerate(items):
        iy = CY + 76 + j * 100
        s.append(f'<text x="{PX+34}" y="{iy+18}" fill="{mc}" font-size="34" '
                 f'font-weight="bold">{mark}</text>')
        s.append(f'<text x="{PX+90}" y="{iy+10}" fill="{B.INK}" font-size="23" '
                 f'font-weight="bold">{t}</text>')
        s.append(f'<text x="{PX+90}" y="{iy+42}" fill="{B.SUB}" font-size="20">{d}</text>')
        s.append(f'<text x="{PX+90}" y="{iy+72}" fill="{mc}" font-size="20" '
                 f'font-weight="bold">{r}</text>')
    s.append(f'<line x1="{PX+34}" y1="{CY+474}" x2="{PX+PW-34}" y2="{CY+474}" '
             f'stroke="{B.GRID}" stroke-width="1.5"/>')
    s.append(f'<text x="{PX+34}" y="{CY+516}" fill="{B.INK}" font-size="22" '
             f'font-weight="bold">No software reset on consumer Pascal.</text>')
    s.append(f'<text x="{PX+34}" y="{CY+550}" fill="{B.SUB}" font-size="21">nvidia-smi -rgc '
             f'is unsupported. Only a reboot clears it.</text>')

    # the check strip - the part that survives a screenshot with no comment attached
    SY = 946
    s.append(B.panel(80, SY, W - 160, 116, stroke=B.TEAL, sw=2))
    s.append(f'<text x="112" y="{SY+40}" fill="{B.TEAL}" font-size="19" '
             f'letter-spacing="3">CHECK YOUR OWN BOX - 10 SECONDS</text>')
    s.append(f'<text x="112" y="{SY+82}" fill="{B.INK}" font-size="23" '
             f'font-family="Consolas, Menlo, monospace">{CHECK_CMD}</text>')
    s.append(f'<text x="{W-112}" y="{SY+42}" text-anchor="end" fill="{B.INK}" font-size="21" '
             f'font-weight="bold">Run it WHILE a bench is mid-sweep.</text>')
    s.append(f'<text x="{W-112}" y="{SY+72}" text-anchor="end" fill="{B.SUB}" '
             f'font-size="20">Idle clocks read healthy in BOTH states -</text>')
    s.append(f'<text x="{W-112}" y="{SY+98}" text-anchor="end" fill="{B.SUB}" '
             f'font-size="20">which is exactly how this hides.</text>')

    s.append(B.footer(W, H, "preregs #60-#61 · staked before the reboot · full log in the repo"))
    s.append("</svg>")
    B.save("stuck_boost_reddit.svg", "".join(s))


if __name__ == "__main__":
    reddit() if "--reddit" in sys.argv else main()
