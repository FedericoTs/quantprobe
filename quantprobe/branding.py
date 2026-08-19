"""The quantprobe mark: a Q that is a gauge - the dial is the Q's circle, the needle its tail.
One 12x12 pixel bitmap is the single source of truth: assets/quantprobe-icon.svg is generated
from it (python -m quantprobe.branding > assets/quantprobe-icon.svg), and the terminal banner
renders the same pixels as ANSI half-blocks. Old-style pixel art on purpose."""

import os
import sys

# '.' = background, 'o' = dial (teal), 'X' = needle/hub (white)
PIXELS = [
    "...oooooo...",
    "..oo....oo..",
    ".oo......oo.",
    ".o........o.",
    "oo........oo",
    "o....XX....o",
    "o....XX....o",
    "oo.....X....",
    ".o......X...",
    ".oo......X..",
    "..oo......X.",
    "...ooo.....X",
]
BG, DIAL, NEEDLE = (20, 24, 31), (20, 184, 166), (230, 255, 250)


def icon_svg(px=10, pad=4):
    w = len(PIXELS[0]) * px + 2 * pad
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {w} "'
        + 'shape-rendering="crispEdges" role="img" aria-label="quantprobe">',
        f'  <rect width="{w}" height="{w}" rx="{2 * px}" fill="#14181f"/>',
    ]
    for y, row in enumerate(PIXELS):
        for x, c in enumerate(row):
            if c == ".":
                continue
            color = "#14b8a6" if c == "o" else "#e6fffa"
            out.append(
                f'  <rect x="{pad + x * px}" y="{pad + y * px}" width="{px}"'
                f' height="{px}" fill="{color}"/>'
            )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _rgb(c):
    return {".": BG, "o": DIAL, "X": NEEDLE}[c]


def icon_ansi():
    """The same pixels as truecolor half-blocks, 12 wide x 6 lines tall."""
    lines = []
    for y in range(0, len(PIXELS), 2):
        top, bot = PIXELS[y], PIXELS[y + 1]
        line = []
        for t, b in zip(top, bot):
            (tr, tg, tb), (br, bg_, bb) = _rgb(t), _rgb(b)
            line.append(f"\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg_};{bb}m▀")
        lines.append("".join(line) + "\x1b[0m")
    return "\n".join(lines)


def banner(version=""):
    """Icon + wordmark for the CLI header. Plain-text fallback off-tty / NO_COLOR / dumb terms."""
    plain = (
        not sys.stdout.isatty() or os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb"
    )
    if not plain:
        try:  # legacy Windows consoles (cp1252) cannot encode the blocks
            "▀".encode(sys.stdout.encoding or "utf-8")
        except (UnicodeEncodeError, LookupError):
            plain = True
    if plain:
        return f"quantprobe {version} - placement beats budget"
    art = icon_ansi().split("\n")
    text = [
        "",
        "  quantprobe " + version,
        "  placement beats budget",
        "  predictions before downloads",
        "",
        "",
    ]
    return "\n".join(a + t for a, t in zip(art, text))


if __name__ == "__main__":
    sys.stdout.write(icon_svg())
