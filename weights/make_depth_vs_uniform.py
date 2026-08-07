"""Same bytes, better model - the A2A depth-aware vs uniform result.

Every number from preregistrations/2026-08-04-a2a-depth-aware-vs-uniform.md, scored the day
it was staked. The condition that makes it a claim rather than a trade: both files are the
same size (KR-2, +/-2% gate, actual +0.48%). Bytes are the budget; WHERE the protection goes
is the treatment.

The speed panel carries a staked MISS on purpose: P3 predicted invariance (+/-3%) and the
outcome was +6.6% - favourable, and still a failed prediction. It ships at the same size as
the wins.
"""
from __future__ import annotations
import brand as B

W, H = 1600, 1060

U_BYTES, D_BYTES = 3_015_940_800, 3_030_390_688
# (label, uniform, depth-aware, unit, lower_is_better, note, delta_in_points)
# delta_in_points: a metric already expressed in % gets its change in POINTS - reporting a
# percentage change of a percentage is how charts mislead without lying.
METRICS = [
    ("PERPLEXITY", 9.579, 8.319, "", True, "wikitext-2 held out", False),
    ("KL DIVERGENCE, median", 0.268, 0.162, "", True, "vs the f16 teacher", False),
    ("SAME TOP TOKEN", 70.34, 75.47, "%", False, "agreement with teacher", True),
    ("DECODE SPEED", 21.41, 22.82, " tok/s", False, "tg128 - a staked MISS, see below", False),
]

CX, CW, CGAP = 92, 348, 20
CY, CH = 424, 300


def main():
    size_delta = (D_BYTES - U_BYTES) / U_BYTES * 100
    s = [B.svg_open(W, H),
         B.header(W, "MEASURED · Qwen2.5-7B · ONE SESSION · SAME BOX, SAME FLAGS",
                  "Same bytes. Better model.",
                  "Uniform Q2_K against a quantization built from this model's own fragility "
                  "probe - at equal file size.")]

    # the condition, stated before any result
    s.append(B.panel(92, 300, W - 184, 96, stroke=B.TEAL, sw=2))
    s.append(f'<text x="126" y="{300+40}" fill="{B.TEAL}" font-size="19" '
             f'letter-spacing="3">THE CONDITION THAT MAKES THIS A CLAIM</text>')
    s.append(f'<text x="126" y="{300+76}" fill="{B.INK}" font-size="24">'
             f'{U_BYTES/1e9:.3f} GB uniform vs {D_BYTES/1e9:.3f} GB depth-aware = '
             f'<tspan fill="{B.TEAL}" font-weight="bold">+{size_delta:.2f}% size</tspan>. '
             f'Bytes are the budget; where the protection goes is the treatment.</text>')

    for i, (label, u, d, unit, lower, note, in_points) in enumerate(METRICS):
        x = CX + i * (CW + CGAP)
        won = (d < u) if lower else (d > u)
        col = B.TEAL if won else B.DISK
        delta = f"{d-u:+.2f} pts" if in_points else f"{(d-u)/u*100:+.1f}%"
        s.append(B.panel(x, CY, CW, CH, stroke=col, sw=2))
        s.append(f'<text x="{x+CW/2}" y="{CY+40}" text-anchor="middle" fill="{col}" '
                 f'font-size="18" letter-spacing="2">{label}</text>')
        s.append(f'<text x="{x+CW/2}" y="{CY+66}" text-anchor="middle" fill="{B.MUT}" '
                 f'font-size="17">{note}</text>')

        # uniform (grey, struck through by the arrow) -> depth-aware (hero)
        s.append(f'<text x="{x+CW/2}" y="{CY+118}" text-anchor="middle" fill="{B.MUT}" '
                 f'font-size="30">{u:g}{unit}</text>')
        s.append(f'<text x="{x+CW/2}" y="{CY+146}" text-anchor="middle" fill="{B.MUT}" '
                 f'font-size="16" letter-spacing="2">UNIFORM Q2_K</text>')
        s.append(f'<text x="{x+CW/2}" y="{CY+184}" text-anchor="middle" fill="{B.SUB}" '
                 f'font-size="26">&#8595;</text>')
        s.append(f'<text x="{x+CW/2}" y="{CY+232}" text-anchor="middle" fill="{B.INK}" '
                 f'font-size="48" font-weight="bold">{d:g}{unit}</text>')
        s.append(f'<text x="{x+CW/2}" y="{CY+260}" text-anchor="middle" fill="{col}" '
                 f'font-size="16" letter-spacing="2">DEPTH-AWARE</text>')
        s.append(f'<text x="{x+CW/2}" y="{CY+290}" text-anchor="middle" fill="{col}" '
                 f'font-size="24" font-weight="bold">{delta}</text>')

    # the miss, at the same size as the wins
    MY = CY + CH + 40
    s.append(B.panel(92, MY, W - 184, 150, stroke=B.DISK, sw=2))
    s.append(f'<text x="126" y="{MY+40}" fill="{B.DISK}" font-size="19" '
             f'letter-spacing="3">THE PREDICTION THAT FAILED</text>')
    s.append(f'<text x="126" y="{MY+78}" fill="{B.INK}" font-size="23">'
             f'P3 staked decode speed as INVARIANT (+/-3%). It came in +6.6% faster - a good '
             f'outcome and a wrong prediction.</text>')
    s.append(f'<text x="126" y="{MY+110}" fill="{B.SUB}" font-size="22">'
             f'Published as a miss because the band was staked symmetric. Also measured and '
             f'reported: 52/52 business-task verdicts were</text>')
    s.append(f'<text x="126" y="{MY+138}" fill="{B.SUB}" font-size="22">'
             f'IDENTICAL across both arms - at n=52 the task suite cannot see a difference '
             f'that KL divergence resolves easily.</text>')

    s.append(B.footer(W, H, "prereg 2026-08-04-a2a-depth-aware-vs-uniform · scored same day"))
    s.append("</svg>")
    B.save("depth_vs_uniform.svg", "".join(s))


if __name__ == "__main__":
    main()
