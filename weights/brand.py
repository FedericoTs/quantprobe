"""quantprobe media brand kit v2 - the share-asset design system.

Identity source of truth: assets/quantprobe-wordmark.svg (the CURRENT logo - pixel qp
letterforms sharing one bar coloured by the memory-tier gradient). The old pixel-Q icon is
retired from media. Palette derives from the wordmark plus a bright teal data ink.

Design rules (social-first):
- type ramp: nothing below 18px at 1600w; kicker 20 / title 58 / subtitle 24 / section 26 /
  body 22 / KPI 72 / hero 110. Numbers are the heroes; words are captions.
- one hero element per asset at 3-4x the scale of anything else.
- margins 80px; generous air beats extra panels; max ~3 receipt chips.
- data strokes >= 5px, points >= 12px radius - X compresses screenshots.
- dark single-theme by choice (screenshot objects), grid lifted enough to survive JPEG.
"""
from __future__ import annotations
import os, re

BG, PANEL, EDGE, GRID = "#14181f", "#1c222c", "#394355", "#2a3340"
INK, SUB, MUT = "#f5f6f8", "#b3bccb", "#7d879c"
TEAL, TEAL_DEEP = "#2dd4bf", "#14b8a6"
VRAM, RAM, DISK = "#e8a87c", "#d97757", "#a84b32"
FONT = "Segoe UI, Arial, sans-serif"

_WM = None


def wordmark(x, y, width):
    """The CURRENT logo, inlined whole from assets/quantprobe-wordmark.svg (348x216)."""
    global _WM
    if _WM is None:
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "assets", "quantprobe-wordmark.svg")
        src = open(p, encoding="utf-8").read()
        _WM = "".join(re.findall(r'<rect [^/]*?/>', src))
    k = width / 348
    return f'<g transform="translate({x},{y}) scale({k})">{_WM}</g>'


def tier_bar(x, y, w, h=8):
    third = w / 3
    return (f'<rect x="{x}" y="{y}" width="{third}" height="{h}" fill="{VRAM}"/>'
            f'<rect x="{x+third}" y="{y}" width="{third}" height="{h}" fill="{RAM}"/>'
            f'<rect x="{x+2*third}" y="{y}" width="{third}" height="{h}" fill="{DISK}"/>')


def header(W, kicker, title, subtitle=""):
    """Wordmark left, text block right of it, tier bar underneath. Height ~230."""
    s = [wordmark(80, 56, 150),
         f'<text x="266" y="92" fill="{MUT}" font-size="20" letter-spacing="4">{kicker}</text>',
         f'<text x="266" y="152" fill="{INK}" font-size="58" font-weight="bold">{title}</text>']
    if subtitle:
        s.append(f'<text x="80" y="216" fill="{SUB}" font-size="24">{subtitle}</text>')
    s.append(tier_bar(80, 238, W - 160, 5))
    return "".join(s)


HEADER_H = 270


def footer(W, H, receipt):
    return (tier_bar(80, H - 106, W - 160, 4)
            + f'<text x="80" y="{H-58}" fill="{INK}" font-size="24" font-weight="bold">quantprobe'
              f'  <tspan fill="{TEAL}" font-weight="normal">github.com/FedericoTs/quantprobe</tspan></text>'
            + f'<text x="{W-80}" y="{H-58}" text-anchor="end" fill="{MUT}" font-size="19">{receipt}</text>')


FOOTER_H = 140


def panel(x, y, w, h, stroke="#343d4d", sw=1.5, fill=None):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" '
            f'fill="{fill or PANEL}" stroke="{stroke}" stroke-width="{sw}"/>')


def chip(x, y, w, title, value, sub, color):
    """A receipt chip: label / big number / one-line source."""
    return (panel(x, y, w, 172, stroke=color, sw=2)
            + f'<text x="{x+w/2}" y="{y+44}" text-anchor="middle" fill="{color}" '
              f'font-size="19" letter-spacing="3">{title}</text>'
            + f'<text x="{x+w/2}" y="{y+108}" text-anchor="middle" fill="{INK}" '
              f'font-size="52" font-weight="bold">{value}</text>'
            + f'<text x="{x+w/2}" y="{y+146}" text-anchor="middle" fill="{SUB}" '
              f'font-size="18">{sub}</text>')


def svg_open(W, H):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'font-family="{FONT}"><rect width="{W}" height="{H}" fill="{BG}"/>')


MEDIA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")


def save(name, body_svg):
    os.makedirs(MEDIA, exist_ok=True)
    out = os.path.join(MEDIA, name)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(body_svg)
    print("media ->", out)
    return out
