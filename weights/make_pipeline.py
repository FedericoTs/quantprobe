"""The pipeline: any model in, an optimally-served model out.

The asset the repo has never had. Every stage carries a MEASURED number, and the one stage
that is not shipped yet is drawn as not shipped - a roadmap box that looks like a feature box
is how a diagram starts lying.
"""
from __future__ import annotations
import brand as B

W, H = 1600, 1000
BX, BY, BW, BH, GAP = 88, 372, 228, 356, 15

# (n, verb, what it does (2 lines), command, proof, shipped)
STAGES = [
    ("1", "PROBE", ["find which layers", "actually break"], "quantprobe probe",
     "27x spread\nbetween bands", True),
    ("2", "REBUILD", ["quantize around", "what you found"], "quantprobe quantize",
     "-13.2% ppl\nat equal bytes", True),
    ("3", "PLACE", ["split across", "VRAM / RAM / disk"], "quantprobe plan",
     "9.0% median\nprediction error", True),
    ("4", "RUN", ["launch with the", "flags that fit"], "quantprobe run",
     "2.2x prefill\nvs naive -ngl", True),
    ("5", "SERVE", ["many sessions,", "an API, a container"], "quantprobe serve",
     "23 -> 219 tok/s\naggregate, measured", False),
    ("6", "PROVE", ["quality, speed and", "tasks re-measured"], "quantprobe bench",
     "misses published\nat full size", True),
]


def main():
    s = [B.svg_open(W, H),
         B.header(W, "PROBE · REBUILD · PLACE · RUN · SERVE · PROVE",
                  "Any model, rebuilt for your machine",
                  "Not a calculator you consult once - the path from a model you found to the "
                  "best version of it your hardware can serve.")]

    # input / output rails
    s.append(f'<text x="{BX}" y="{BY-46}" fill="{B.TEAL}" font-size="21" '
             f'letter-spacing="3">ANY MODEL IN</text>')
    s.append(f'<text x="{W-BX}" y="{BY-46}" text-anchor="end" fill="{B.TEAL}" font-size="21" '
             f'letter-spacing="3">SERVED, AND PROVEN</text>')
    s.append(f'<line x1="{BX}" y1="{BY-28}" x2="{W-BX}" y2="{BY-28}" stroke="{B.GRID}" '
             f'stroke-width="2"/>')

    for i, (n, verb, what, cmd, proof, shipped) in enumerate(STAGES):
        x = BX + i * (BW + GAP)
        col = B.TEAL if shipped else B.VRAM
        s.append(B.panel(x, BY, BW, BH, stroke=col, sw=2.5,
                         fill=B.PANEL if shipped else "#20242e"))
        # stage number in a disc
        s.append(f'<circle cx="{x+40}" cy="{BY+42}" r="21" fill="{col}"/>')
        s.append(f'<text x="{x+40}" y="{BY+51}" text-anchor="middle" fill="{B.BG}" '
                 f'font-size="24" font-weight="bold">{n}</text>')
        s.append(f'<text x="{x+74}" y="{BY+52}" fill="{B.INK}" font-size="27" '
                 f'font-weight="bold">{verb}</text>')

        for j, line in enumerate(what):
            s.append(f'<text x="{x+24}" y="{BY+106+j*28}" fill="{B.SUB}" '
                     f'font-size="20">{line}</text>')

        s.append(f'<rect x="{x+20}" y="{BY+178}" width="{BW-40}" height="40" rx="8" '
                 f'fill="{B.BG}" stroke="{B.GRID}"/>')
        s.append(f'<text x="{x+BW/2}" y="{BY+204}" text-anchor="middle" fill="{col}" '
                 f'font-size="17" font-family="Consolas, Menlo, monospace">{cmd}</text>')

        s.append(f'<line x1="{x+24}" y1="{BY+244}" x2="{x+BW-24}" y2="{BY+244}" '
                 f'stroke="{B.GRID}" stroke-width="1.5"/>')
        for j, line in enumerate(proof.split("\n")):
            s.append(f'<text x="{x+BW/2}" y="{BY+280+j*26}" text-anchor="middle" '
                     f'fill="{B.INK if j == 0 else B.MUT}" '
                     f'font-size="{20 if j == 0 else 17}" '
                     f'font-weight="{"bold" if j == 0 else "normal"}">{line}</text>')

        if not shipped:
            s.append(f'<text x="{x+BW/2}" y="{BY+BH-16}" text-anchor="middle" fill="{col}" '
                     f'font-size="16" letter-spacing="2" font-weight="bold">SHIPPING NEXT</text>')

        if i < len(STAGES) - 1:
            ax = x + BW + GAP / 2
            s.append(f'<path d="M {ax-6} {BY+BH/2-8} L {ax+5} {BY+BH/2} L {ax-6} {BY+BH/2+8}" '
                     f'fill="none" stroke="{B.MUT}" stroke-width="3"/>')

    FY = BY + BH + 36
    s.append(B.panel(BX, FY, W - 2 * BX, 92, stroke=B.TEAL_DEEP, sw=2))
    s.append(f'<text x="{BX+34}" y="{FY+38}" fill="{B.TEAL}" font-size="19" '
             f'letter-spacing="3">WHAT MAKES IT DIFFERENT</text>')
    s.append(f'<text x="{BX+34}" y="{FY+72}" fill="{B.INK}" font-size="23">'
             f'Other tools help you pick a quantization someone else built. This one measures '
             f'your model, then builds a file that exists only for your hardware.</text>')

    s.append(B.footer(W, H, "every stage number measured on a GTX 1060 6GB · sources in the repo"))
    s.append("</svg>")
    B.save("pipeline.svg", "".join(s))


if __name__ == "__main__":
    main()
