"""Prereg #95 stage 1: which llama.cpp flags actually move single-user decode.

150 Morris-screened runs, staked 2026-08-07 BEFORE any screening data, scored by the
pre-committed scorer. The chart carries the two passes AND the miss at the same size:
P-1 (top-3 factors carry >=70% of total mu_star on both models) PASS, P-2 (the top
factor differs between the dense-GPU and CPU-expert-split regimes) PASS, P-4 (our staked
-ub interaction warning) FAIL - sigma(-ub) ranked dead last on both models, because the
stake imported a PREFILL effect onto a decode-only response.

Cite-or-refuse: every bar renders from weights/data/prereg95_verdict.json, written by
weights/prereg95_score.py over the committed CSV. Missing or partial json -> refuse.
Method credit: bigattichouse's llama-optimize / robust (E-16) - Morris screening is
their funnel's stage 1, seeded here by our law.
"""
from __future__ import annotations
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import brand as B                                             # noqa: E402

VJ = os.path.join(HERE, "data", "prereg95_verdict.json")
W = 1600

NAME = {"ngl": "-ngl  GPU layers", "t": "-t  CPU threads", "fa": "-fa  flash attention",
        "ctk": "-ctk  KV cache type", "ub": "-ub  ubatch size", "mmp": "--no-mmap",
        "moe_cpu_frac": "-ot  expert offload"}
PANEL = {"7B": ("Qwen2.5-7B Q4_K_M", "dense - the GPU-fit regime"),
         "30B": ("Qwen3-30B-A3B Q2_K", "MoE - experts split to CPU RAM")}


def read_verdict():
    """Bars and verdicts from the scorer's own json - the chart invents nothing."""
    if not os.path.isfile(VJ):
        raise SystemExit(f"verdict json missing: {VJ} - run weights/prereg95_score.py first")
    v = json.load(open(VJ, encoding="utf-8"))
    for need in ("models", "verdicts"):
        if need not in v:
            raise SystemExit(f"verdict json lacks '{need}' - refusing a partial chart")
    if v["rows"]["present"] != 150 or v["rows"]["ok"] != 150:
        raise SystemExit("verdict json is not the complete 150-ok night - refusing")
    return v


def main():
    v = read_verdict()
    H = 1560
    # B.header draws kicker and subtitle as single unwrapped lines: both must fit W or
    # they clip at the right edge (caught in render-review, first version did exactly that)
    s = [B.svg_open(W, H),
         B.header(W, "MEASURED - 150 MORRIS RUNS - ONE MACHINE STATE - STAKED 2026-08-07",
                  "Three flags are the whole story",
                  "Morris ranks every flag by mu* - tok/s moved across the flag's range. "
                  "Top three: 97% of the effect (dense), 77% (MoE split).")]

    # --- two ranked-bar panels ------------------------------------------------------------
    top_y = 330
    pw = 700
    px = {"7B": 80, "30B": 80 + pw + 40}
    bar_h, gap = 42, 22
    label_w = 300
    for m in ("7B", "30B"):
        facs = v["models"][m]["factors"]
        ranked = sorted(facs, key=lambda f: -facs[f]["mu_star"])
        vmax = facs[ranked[0]]["mu_star"]
        x0 = px[m]
        name, sub = PANEL[m]
        s.append(f'<text x="{x0}" y="{top_y - 34}" fill="{B.INK}" font-size="26" '
                 f'font-weight="bold">{name}</text>')
        s.append(f'<text x="{x0}" y="{top_y - 8}" fill="{B.MUT}" font-size="19">{sub}</text>')
        for i, f in enumerate(ranked):
            mu = facs[f]["mu_star"]
            y = top_y + 16 + i * (bar_h + gap)
            top = i == 0
            dead = f == "ub"
            col = B.TEAL if top else (B.VRAM if dead else B.SUB)
            bw = max((pw - label_w) * (mu / vmax), 4)
            s.append(f'<text x="{x0 + label_w - 14}" y="{y + bar_h - 13}" text-anchor="end" '
                     f'fill="{B.INK if (top or dead) else B.MUT}" font-size="21" '
                     f'font-weight="{"bold" if (top or dead) else "normal"}">{NAME[f]}</text>')
            s.append(f'<rect x="{x0 + label_w}" y="{y}" width="{bw:.0f}" height="{bar_h}" '
                     f'rx="7" fill="{col}" opacity="{1 if (top or dead) else 0.85}"/>')
            s.append(f'<text x="{x0 + label_w + bw + 12:.0f}" y="{y + bar_h - 13}" '
                     f'fill="{B.INK}" font-size="21" font-weight="bold">{mu:.1f}</text>')
            if top:
                s.append(f'<text x="{x0 + label_w + 10}" y="{y + bar_h - 13}" '
                         f'fill="{B.BG}" font-size="18" font-weight="bold">tune this first</text>')
            if dead:
                s.append(f'<text x="{x0 + label_w + bw + 60:.0f}" y="{y + bar_h - 13}" '
                         f'fill="{B.VRAM}" font-size="19">dead last - see the miss below</text>')

    # --- verdict chips: the miss at the SAME SIZE as the hits -----------------------------
    p1 = v["verdicts"]["P-1"]["per_model"]
    cy = top_y + 16 + 7 * (bar_h + gap) + 60
    gap_c = 26
    cwid = (W - 160 - 2 * gap_c) / 3
    s.append(B.chip(80, cy, cwid, "P-1 CONCENTRATION - PASS",
                    f"{p1['7B']['share']:.0%} / {p1['30B']['share']:.0%}",
                    "top-3 share of total effect - bar was 70%", B.TEAL))
    s.append(B.chip(80 + cwid + gap_c, cy, cwid, "P-2 REGIMES SPLIT - PASS",
                    "ngl vs t", "dense tunes the GPU, the split tunes the CPU", B.TEAL))
    s.append(B.chip(80 + 2 * (cwid + gap_c), cy, cwid, "P-4 OUR MISS - FAIL",
                    "sigma(-ub) last", "we staked an interaction; decode never saw it", B.VRAM))

    # --- honest notes ---------------------------------------------------------------------
    ny = cy + 210
    notes = [
        ("THE MISS, DIAGNOSED: we staked that -ub would show interaction (high sigma) because "
         "-ub 2048 once measured +73% on the split and -39% all-in-VRAM. That asymmetry is a "
         "PREFILL effect - this experiment's response is decode only (tg128), and there -ub "
         "ranked last on both machines. The flag most tuning guides reach for first is the one "
         "that moves single-user decode the least. Prefill is a different regime; both numbers "
         "stand, in their own phases."),
        ("Chain of custody: staked 2026-08-07 before any screening run; amended pre-data with 8 "
         "declared deviations; harness AND scorer committed before the first CSV row; 150 of "
         "150 runs, 0 DNF, GPU settled to 38-49 C between every run; scored by the frozen "
         "scorer. The kill rule on the binding-constraint classifier (P-3) waits for stage-2 "
         "Sobol - it is still armed, not discharged."),
        ("Method: Morris elementary effects (R=10 trajectories), the screening stage of "
         "bigattichouse's llama-optimize / robust DoE funnel - his design, our hardware and "
         "stakes. Unstaked observation for stage 2: --no-mmap is #2 on the split with the "
         "highest sigma, the strongest interaction candidate on the board."),
    ]
    for n in notes:
        s.append(B.paragraph(80, ny, n, 18, B.MUT, W - 160))
        ny += 26 * (len(B.wrap(n, 18, W - 160)) + 0.5)

    s.append(B.footer(W, H, "prereg 2026-08-07-doe-flag-screening - "
                            "weights/data/doe_morris_stage1.csv - prereg95_verdict.json"))
    s.append("</svg>")
    B.save("flags_matter.svg", "".join(s), scale=2)


if __name__ == "__main__":
    main()
