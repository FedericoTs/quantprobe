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
    # 1x Spark is CHECKED against third-party reports (see the note under the table): our law
    # lands 1.27x low on a 32B dense and 1.09x low on a 30B MoE - inside C-02's floor band.
    ("DGX Spark - 128GB unified @ 273",          115,  273, 115, 273, 7.0,  0.60, "checked vs reports"),
    # 2x/4x are UPPER BOUNDS and are labelled as such in the table. They model unified memory,
    # but real multi-node llama.cpp uses RPC, and the one public datapoint - GLM-5.2 UD-IQ1_S
    # on 2x Spark, 256K ctx - reports 8 tok/s where unified-memory arithmetic gives 23.7, i.e.
    # we are 3.0x optimistic. One datapoint is not a coefficient (C-02's standing warning), so
    # nothing is fitted; the gap is disclosed instead.
    ("2x DGX Spark - 256GB via RPC",             230,  273, 230, 273, 7.0,  0.60, "UPPER BOUND - RPC"),
    ("4x DGX Spark - 512GB via RPC",             460,  273, 460, 273, 7.0,  0.60, "UPPER BOUND - RPC"),
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
    print("## Scored against third-party DGX Spark reports\n")
    print("The Spark rows are the only ones anyone else has published numbers for, so they are the")
    print("closest thing we have to out-of-sample validation. On a single unit:\n")
    print("| model | our prediction | third-party report | ratio |")
    print("|---|---|---|---|")
    print("| 32B dense Q4 | 8.4 | 10.7 | **1.27x low** |")
    print("| 30B-A3B MoE Q4 | 81.7 | 89.0 | **1.09x low** |")
    print("| Gemma-4-26B A4B Q4 | 67.4 | 51.6 | **0.77x — we over-predict** |\n")
    print("The first two land inside C-02's floor band (real speed 1.1-1.8x above the printed")
    print("number). The third does **not**, and we are not hiding it: either our active-parameter")
    print("figure for that model is wrong, or the all-in-VRAM floor has an exception. It is logged")
    print("rather than explained away, and it is the next thing to check.\n")
    print("## Adding Sparks buys capacity, not speed\n")
    print("Each unit is 128 GB at 273 GB/s, and linking them does not raise per-unit bandwidth - a")
    print("token still traverses every layer in sequence. So 1x, 2x and 4x give identical tok/s on")
    print("anything that already fits one unit. What 4x buys is the first configuration where")
    print("GLM-5.2 (753B) and Kimi-K2.6 (1058B) become usable at ~12 tok/s instead of ~1.\n")
    print("**The 2x/4x rows are upper bounds.** They model unified memory; real multi-node")
    print("llama.cpp uses RPC. The one public datapoint - GLM-5.2 UD-IQ1_S on 2x Spark at 256K")
    print("context - reports 8 tok/s where unified-memory arithmetic gives 23.7. We are **3x")
    print("optimistic** there. One datapoint is not a coefficient, so nothing has been fitted to")
    print("it; the gap is disclosed and the rows are labelled.\n")
    print("## A number going around that cannot be true\n")
    print("\"DGX Spark runs 70B Q4 at 35-45 tok/s\" appears in several write-ups. A 70B dense model")
    print("at Q4 moves **42.5 GB per token**. At 273 GB/s the ceiling - perfect efficiency, eta =")
    print("1.0, no overhead of any kind - is **6.4 tok/s**. Reaching 35-45 would need 1,488-1,914")
    print("GB/s, i.e. **5.5-7x the bandwidth the hardware physically has**.\n")
    print("Whatever those benchmarks measured, it was not single-stream decode of a 70B dense")
    print("model: most likely batched throughput, prompt processing, or an MoE mislabelled as")
    print("dense. This is the sort of claim Law 4 is *for* - you do not need the machine to know")
    print("the number is impossible, only the bandwidth and the bytes.\n")


if __name__ == "__main__":
    main()
