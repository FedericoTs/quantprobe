"""U-33 / prereg #84, Q6: op count failed. Does TENSOR SHAPE (L-20) explain the same
residual better - using the ALREADY-MEASURED knee curve instead of a fitted parameter?

Each model needs its nominal FORMAT_EBW multiplied by some factor to match measurement:
    needed = byte_ms / measured_ms
L-20 (preregs #80/#81, kernelprobe shape.cu) measured that effective bandwidth is a
function of rows-per-tensor, rising to a knee at ~4096 rows. Small models have small
tensors, so they should sit low on that curve WITHOUT any new free parameter.

This is a zero-parameter prediction. It either tracks or it does not.

  python weights/shape_vs_ops.py
"""
import json
import os

from quantprobe import spec

MODELS = os.environ.get("QP_GGUF_DIR", r"D:\evo-compress-data\gguf")
HERE = os.path.dirname(os.path.abspath(__file__))
LADDER = os.path.join(HERE, "data", "ladder_state_locked.json")

# L-20 measured curve, K=2048 row width, prereg81_knee.log (same session, C-14 safe).
KNEE = [(128, 30.2), (256, 45.4), (512, 61.1), (1024, 76.5), (2048, 88.4),
        (4096, 94.2), (8192, 96.9), (16384, 98.7)]
PLATEAU = KNEE[-1][1]


def shape_bw(rows):
    """Interpolate the measured knee curve in log2(rows); clamp at the ends."""
    import math
    if rows <= KNEE[0][0]:
        return KNEE[0][1]
    if rows >= KNEE[-1][0]:
        return KNEE[-1][1]
    lr = math.log2(rows)
    for (r0, v0), (r1, v1) in zip(KNEE, KNEE[1:]):
        if r0 <= rows <= r1:
            f = (lr - math.log2(r0)) / (math.log2(r1) - math.log2(r0))
            return v0 + f * (v1 - v0)
    return PLATEAU


def dims(path):
    """hidden size and FFN intermediate size, straight from GGUF metadata."""
    from gguf import GGUFReader
    r = GGUFReader(path, "r")
    return spec._field(r, ".embedding_length"), spec._field(r, ".feed_forward_length")


def main():
    print("needed = how much of nominal FORMAT_EBW the model actually achieves")
    print("shape  = what the L-20 knee curve predicts from its tensor geometry alone\n")
    print(f"{'model':<24} {'hidden':>7} {'inter':>7} {'needed':>7} {'shape':>7} {'gap':>7}")
    rows_out = []
    for r in json.load(open(LADDER, encoding="utf-8")):
        if not r.get("measured") or r["placement"] != "all in VRAM":
            continue
        p = os.path.join(MODELS, r["file"])
        if not os.path.exists(p):
            continue
        s = spec.from_gguf(p)
        if not s.get("fmt_bw"):
            continue
        h, inter = dims(p)
        byte_ms = s["a"] * s["bits"] / 8 / s["fmt_bw"] * 1e3
        needed = byte_ms / (1000.0 / r["measured"])
        # decode reads gate+up (inter rows each) and down (h rows) per layer, plus
        # attention projections (h-ish rows). Weight by the 3 FFN tensors that dominate.
        bw_eff = (2 * shape_bw(inter) + shape_bw(h)) / 3
        shape_mult = bw_eff / PLATEAU
        rows_out.append((r["name"], h, inter, needed, shape_mult))
        print(f"{r['name'][:24]:<24} {h:>7} {inter:>7} {needed:>7.3f} {shape_mult:>7.3f} "
              f"{needed-shape_mult:>+7.3f}")
    n = [x[3] for x in rows_out]
    sm = [x[4] for x in rows_out]
    mn, ms_ = sum(n) / len(n), sum(sm) / len(sm)
    num = sum((a - mn) * (b - ms_) for a, b in zip(n, sm))
    den = (sum((a - mn) ** 2 for a in n) * sum((b - ms_) ** 2 for b in sm)) ** 0.5
    print(f"\n  correlation(needed, shape-predicted) r = {num/den:+.3f}  (n={len(n)})")


if __name__ == "__main__":
    main()
