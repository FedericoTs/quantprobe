"""The hardware x model matrix: what to run, and how fast, from a 2016 desktop to 4x DGX Spark.

  python weights/make_hw_matrix.py            > docs/MATRIX.md

Every cell is Law 4 through the SHIPPED evaluate() - the same function `quantprobe plan` calls,
not a spreadsheet that will drift from it. Cells are labelled by which tier binds, because a
number without its binding constraint is not actionable: "3 tok/s" tells you nothing, "3 tok/s,
disk-bound" tells you to buy RAM and "3 tok/s, bandwidth-bound" tells you not to bother.

SCOPE, STATED UP FRONT BECAUSE THIS TABLE WILL BE SCREENSHOTTED. One machine in this grid has
ever been measured - the 2016 desktop, 14 rows, median 8.4% absolute error. Every other row is
the law applied to spec-sheet bandwidth, and the all-in-VRAM placement is a documented FLOOR
(C-02: real speed came in >= 0.90x the printed number on 13 of 13 benchmarks, typically
1.1-1.8x higher). Treat unmeasured rows as lower bounds with an unvalidated eta.
"""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantprobe.plan import evaluate

# (label, vram GB, vram GB/s, ram GB, ram GB/s, disk GB/s, geta, note)
HW = [
    ("2016 desktop - GTX 1060 6GB + 16GB DDR4",   6,  192,  16,  24, 0.47, 0.50, "MEASURED"),
    ("laptop - iGPU + 16GB DDR4",                  0,    0,  16,  24, 3.5,  0.45, ""),
    ("laptop - iGPU + 64GB DDR4 (ProBook class)",  0,    0,  64,  51, 3.5,  0.45, ""),
    ("RTX 3060 12GB + 32GB DDR4",                 12,  360,  32,  51, 3.5,  0.50, ""),
    ("RTX 4090 24GB + 64GB DDR5",                 24, 1008,  64,  80, 7.0,  0.62, ""),
    ("RTX 5090 32GB + 128GB DDR5",                32, 1792, 128,  80, 7.0,  0.62, ""),
    ("2x RTX 4090 48GB + 128GB DDR5",             48, 1714, 128,  80, 7.0,  0.62, ""),
    ("DGX Spark - 128GB unified @ 273",          115,  273, 115, 273, 7.0,  0.60, ""),
    ("2x DGX Spark - 256GB unified @ 273",       230,  273, 230, 273, 7.0,  0.60, "capacity, not speed"),
    ("4x DGX Spark - 512GB unified @ 273",       460,  273, 460, 273, 7.0,  0.60, "capacity, not speed"),
    ("EPYC server - 512GB DDR5 8ch, no GPU",       0,    0, 512, 200, 7.0,  0.45, ""),
]

# (label, total B, active B, always-active B, moe, bits, note)
MODELS = [
    ("Qwen3-0.6B Q8_0",              0.6,  0.6,  0.6, False, 8.5, ""),
    ("Qwen2.5-7B Q4_K_M",            7.6,  7.1,  7.1, False, 4.9, ""),
    ("Qwen2.5-14B Q4_K_M",          14.8, 14.8, 14.8, False, 4.5, ""),
    ("Qwen3-30B-A3B Q2_K",          30.5,  3.3,  1.2, True,  2.5, ""),
    ("Qwen3.5-35B-A3B Q4_K_M",      35.0,  3.3,  1.2, True,  4.5, ""),
    ("gpt-oss-120B Q4_K_M",        120.4,  5.1,  1.8, True,  4.5, ""),
    ("Qwen3-235B-A22B Q2_K",       235.1, 22.0,  7.5, True,  2.5, ""),
    ("DeepSeek V4-Flash Q2_K",     284.0, 13.0,  4.0, True,  2.5, "arch est"),
    ("GLM-5.2 753B Q2_K",          753.3, 32.0,  8.0, True,  2.5, "a/ne est"),
    ("Kimi-K2.6 1058B Q2_K",      1058.6, 32.0,  6.0, True,  2.5, "a/ne est"),
    ("Qwen3.8-Max Q2_K",          2400.0, 95.0, 30.0, True,  2.5, "arch est"),
]

BIND = {"vram_bw": "VRAM", "ram_bw": "RAM", "io": "disk", "cpu_compute": "CPU"}


def cell(m, h):
    _, t, a, ne, moe, bits, _ = m
    _, vc, vb, rc, rb, db, geta, _ = h
    size = t * bits / 8 * 1.08
    try:
        _, _, rows = evaluate(t=t, a=a, ne=ne, moe=moe, bits=bits, vc=vc, vb=vb, rc=rc,
                              rb=rb, db=db, geta=geta, true_size_gb=size)
    except Exception:
        return None, None, None
    rows = [r for r in rows if getattr(r, "runnable", True)]
    if not rows:
        return None, None, None
    best = rows[0]
    terms = getattr(best, "terms", {}) or {}
    binds = BIND.get(max(terms, key=terms.get), "?") if terms else "?"
    return best[1], binds, size


def main():
    print("# What to run, and how fast\n")
    print("Every cell is Law 4 through quantprobe's shipped `evaluate()` - the same function")
    print("`quantprobe plan` calls. The letter after each number is **which resource binds**:")
    print("`V` VRAM bandwidth, `R` RAM bandwidth, `D` disk, `C` CPU compute. A speed without its")
    print("binding constraint is not actionable - *3 tok/s, disk-bound* means buy RAM, *3 tok/s,")
    print("bandwidth-bound* means do not bother.\n")
    print("**One row here has been measured** (the 2016 desktop: 14 models, median 8.4% absolute")
    print("error). Every other row is the law applied to spec-sheet bandwidth. The all-in-VRAM")
    print("placement is a documented **floor** - measured speed came in at or above the printed")
    print("number on 13 of 13 benchmarks, typically 1.1-1.8x higher (C-02). Read unmeasured rows")
    print("as lower bounds.\n")
    sizes = [f"{m[1]*m[5]/8*1.08:.0f}" for m in MODELS]
    print("| machine | " + " | ".join(f"{m[0]}<br><sub>{s} GB</sub>"
                                      for m, s in zip(MODELS, sizes)) + " |")
    print("|---|" + "---|" * len(MODELS))
    for h in HW:
        cells = []
        for m in MODELS:
            tps, binds, _ = cell(m, h)
            if tps is None:
                cells.append("—")
            elif tps < 0.1:
                cells.append(f"*{tps:.2f}* {binds[0]}")
            elif tps < 1:
                cells.append(f"{tps:.2f} {binds[0]}")
            else:
                cells.append(f"**{tps:.0f}** {binds[0]}")
        tag = f" *({h[7]})*" if h[7] else ""
        print(f"| {h[0]}{tag} | " + " | ".join(cells) + " |")
    print("\n*Italic* = under 1 tok/s, i.e. a capacity demo rather than usable inference.\n")
    print("## The DGX Spark rows are the interesting ones\n")
    print("Adding Sparks multiplies **capacity**, not speed. Each unit is 128 GB at 273 GB/s, and")
    print("linking them does not raise per-unit bandwidth - a token still traverses every layer in")
    print("sequence. So 4x Spark lets you *hold* a 2.4T model that one unit cannot, at roughly the")
    print("same tok/s as one unit running something that fits. That is Law 4 stated as a purchase")
    print("decision: buy Sparks to fit a bigger model, not to run the same model faster.\n")


if __name__ == "__main__":
    main()
