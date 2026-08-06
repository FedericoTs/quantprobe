"""quantprobe media brand kit - one place for the share-asset look.

Palette is derived from the repo's own identity files, not invented:
- assets/quantprobe-icon.svg:   teal #14b8a6 on tile #14181f, near-white #e6fffa
- assets/quantprobe-wordmark.svg: the tier gradient - VRAM #e8a87c -> RAM #d97757 ->
  disk #a84b32 ("the bar is the probe: one column through the memory tiers it prices")
Every share asset gets: the pixel-Q logo, a kicker line, the tier-gradient hairline, and the
receipt footer. Dark single-theme by choice - these are X/Reddit screenshot objects.
"""
from __future__ import annotations
import os, re

BG, PANEL, EDGE = "#14181f", "#1b202a", "#2b3240"
INK, SUB, MUT = "#f2f3f5", "#a9b1bf", "#6e7684"
TEAL, TEAL_HI, TEAL_PALE = "#14b8a6", "#2dd4bf", "#e6fffa"
VRAM, RAM, DISK = "#e8a87c", "#d97757", "#a84b32"
FONT = "Segoe UI, Arial, sans-serif"

_ICON_CACHE = None


def logo(x, y, size):
    """The pixel-Q icon, inlined from assets/quantprobe-icon.svg (single source of truth)."""
    global _ICON_CACHE
    if _ICON_CACHE is None:
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "assets", "quantprobe-icon.svg")
        src = open(p, encoding="utf-8").read()
        _ICON_CACHE = "".join(re.findall(r'<rect x="[^/]*?/>', src))
    k = size / 128
    return (f'<g transform="translate({x},{y}) scale({k})">'
            f'<rect width="128" height="128" rx="20" fill="{PANEL}"/>{_ICON_CACHE}</g>')


def tier_bar(x, y, w, h=6):
    """The signature: one bar through the memory tiers."""
    third = w / 3
    return (f'<rect x="{x}" y="{y}" width="{third}" height="{h}" fill="{VRAM}"/>'
            f'<rect x="{x+third}" y="{y}" width="{third}" height="{h}" fill="{RAM}"/>'
            f'<rect x="{x+2*third}" y="{y}" width="{third}" height="{h}" fill="{DISK}"/>')


def header(W, kicker, title, subtitle=""):
    s = [logo(60, 42, 84),
         f'<text x="164" y="70" fill="{MUT}" font-size="15" letter-spacing="5">{kicker}</text>',
         f'<text x="164" y="112" fill="{INK}" font-size="42" font-weight="bold">{title}</text>']
    if subtitle:
        s.append(f'<text x="60" y="164" fill="{SUB}" font-size="17">{subtitle}</text>')
    s.append(tier_bar(60, 178, W - 120, 4))
    return "".join(s)


def footer(W, H, receipt):
    return (tier_bar(60, H - 92, W - 120, 3)
            + f'<text x="60" y="{H-52}" fill="{INK}" font-size="19" font-weight="bold">quantprobe'
              f'  <tspan fill="{TEAL_HI}">falsification-tested laws for local LLMs</tspan></text>'
            + f'<text x="{W-60}" y="{H-52}" text-anchor="end" fill="{MUT}" font-size="14">'
              f'github.com/FedericoTs/quantprobe</text>'
            + f'<text x="60" y="{H-26}" fill="{MUT}" font-size="13">{receipt}</text>')


def panel(x, y, w, h, stroke=EDGE, sw=1):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{PANEL}" stroke="{stroke}" stroke-width="{sw}"/>'


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
