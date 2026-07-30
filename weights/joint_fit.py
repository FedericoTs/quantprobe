"""U-33 / prereg #84, Q4: the JOINT re-derivation U-29 said was required.

The circularity that has been hiding this: spec.FORMAT_EBW was calibrated on dense 7B
files, so by construction the 7B rows have ~zero residual and every OTHER model carries
the whole mismatch. You cannot test a per-layer overhead term against a bandwidth that
already ate it. Both have to move at once:

    t_token  =  bytes/token / (k * fmt_bw)   +   n_ops * c

  x1 = 1/k  scales the calibrated bandwidth (one number, all formats)
  x2 = c    seconds per emitted non-matmul op   <- the new term

n_ops is COUNTED, not fitted: layers x ops-per-layer, taken from the #83 call structure
(calls/step/layer), which is exact. Two free parameters, six clean rows.

  python weights/joint_fit.py
"""
import json
import os

from quantprobe import spec

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("QP_GGUF_DIR", r"D:\evo-compress-data\gguf")
LADDER = os.path.join(HERE, "data", "ladder_state_locked.json")

# norms-per-layer counted from prereg83_perop.log: RMS_NORM calls / 33 steps / n_layer.
# qwen2 2.04, qwen3 4.04, qwen35 3.28, gemma4 7.02, deepseek2 3.04.
OPS_PER_LAYER = {"qwen2": 2.04, "qwen3": 4.04, "qwen35": 3.28, "gemma4": 7.02,
                 "gemma3": 7.02, "deepseek2": 3.04}


def lstsq2(rows):
    """Least squares for t = a*u + b*v, no intercept. Returns (a, b)."""
    suu = sum(u * u for u, v, t in rows)
    svv = sum(v * v for u, v, t in rows)
    suv = sum(u * v for u, v, t in rows)
    sut = sum(u * t for u, v, t in rows)
    svt = sum(v * t for u, v, t in rows)
    det = suu * svv - suv * suv
    return (sut * svv - svt * suv) / det, (svt * suu - sut * suv) / det


def main():
    ladder = json.load(open(LADDER, encoding="utf-8"))
    data = []
    for r in ladder:
        if not r.get("measured") or r["placement"] != "all in VRAM":
            continue
        p = os.path.join(MODELS, r["file"])
        if not os.path.exists(p):
            continue
        s = spec.from_gguf(p)
        opl = OPS_PER_LAYER.get(s.get("arch"))
        if not opl or not s.get("fmt_bw"):
            continue
        byte_ms = s["a"] * s["bits"] / 8 / s["fmt_bw"] * 1e3
        data.append({"n": r["name"], "L": s["n_layer"], "arch": s["arch"],
                     "byte_ms": byte_ms, "nops": s["n_layer"] * opl,
                     "t": 1000.0 / r["measured"], "pred_now": r["predicted"],
                     "meas": r["measured"]})

    print(f"{len(data)} clean all-in-VRAM rows\n")
    a, b = lstsq2([(d["byte_ms"], d["nops"], d["t"]) for d in data])
    print(f"joint fit:   t_ms = {a:.4f} * (bytes/fmt_bw)_ms  +  {b*1000:.1f} us * n_ops")
    print(f"  -> calibrated bandwidth is off by {(1/a - 1)*100:+.1f}% "
          f"(true BW = {1/a:.3f} x FORMAT_EBW)")
    print(f"  -> each emitted non-matmul op costs {b*1000:.1f} us\n")

    print(f"{'model':<24} {'L':>3} {'n_ops':>6} {'meas':>7} {'byte-only':>10} "
          f"{'joint':>7} {'err':>7} {'shipped':>8} {'err':>7}")
    e_j, e_s = [], []
    for d in data:
        t_joint = a * d["byte_ms"] + b * d["nops"]
        tok_joint, tok_byte = 1000.0 / t_joint, 1000.0 / d["byte_ms"]
        ej = (tok_joint / d["meas"] - 1) * 100
        es = (d["pred_now"] / d["meas"] - 1) * 100
        e_j.append(abs(ej)); e_s.append(abs(es))
        print(f"{d['n'][:24]:<24} {d['L']:>3} {d['nops']:>6.0f} {d['meas']:>7.1f} "
              f"{tok_byte:>10.1f} {tok_joint:>7.1f} {ej:>+6.1f}% "
              f"{d['pred_now']:>8.1f} {es:>+6.1f}%")
    med = lambda x: sorted(x)[len(x) // 2]
    print(f"\n  median |err|   joint fit {med(e_j):.1f}%   vs shipped {med(e_s):.1f}%")
    print(f"  max    |err|   joint fit {max(e_j):.1f}%   vs shipped {max(e_s):.1f}%")

    print("\nleave-one-out (fit on 5, predict the held-out row):")
    loo = []
    for i, d in enumerate(data):
        rest = [(x["byte_ms"], x["nops"], x["t"]) for j, x in enumerate(data) if j != i]
        aa, bb = lstsq2(rest)
        tk = 1000.0 / (aa * d["byte_ms"] + bb * d["nops"])
        e = (tk / d["meas"] - 1) * 100
        loo.append(abs(e))
        print(f"  {d['n'][:26]:<26} held out -> {tk:>6.1f} vs {d['meas']:>6.1f}  {e:>+6.1f}%")
    print(f"\n  LOO median |err| {med(loo):.1f}%   max {max(loo):.1f}%")
    print("  (2 params, 6 rows - LOO is the only honest read; in-sample fit is optimistic)")


if __name__ == "__main__":
    main()
