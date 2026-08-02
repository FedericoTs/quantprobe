"""U-33 / prereg #84, Q5: the flat per-op tax LOST out of sample (LOO 12.1% vs shipped
8.7%). Q1b says why: us/call FALLS as ops-per-layer rises (15.3 us at 2/layer -> 7.7 us
at 7/layer), because back-to-back tiny kernels pipeline and share their launch latency.
A flat per-op cost cannot express that, so it over-taxes the models with few ops/layer.

Physical form that CAN:
    layers are serialized (each depends on the last)          -> linear in L
    ops inside a layer pipeline                               -> sublinear in ops/layer

    t = a * (bytes/fmt_bw)  +  c * L * (ops_per_layer ** p)

p = 1 recovers the flat per-op tax (already refuted). p = 0 is a pure per-layer tax
(also refuted: -26..457 us/layer). Sweep p, judge ONLY on leave-one-out.

  python weights/sublinear_fit.py
"""
import json
import os

from quantprobe import spec

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("QP_GGUF_DIR", r"D:\evo-compress-data\gguf")
LADDER = os.path.join(HERE, "data", "ladder_state_locked.json")
OPS_PER_LAYER = {"qwen2": 2.04, "qwen3": 4.04, "qwen35": 3.28, "gemma4": 7.02,
                 "gemma3": 7.02, "deepseek2": 3.04}


def lstsq2(rows):
    suu = sum(u * u for u, v, t in rows); svv = sum(v * v for u, v, t in rows)
    suv = sum(u * v for u, v, t in rows); sut = sum(u * t for u, v, t in rows)
    svt = sum(v * t for u, v, t in rows)
    det = suu * svv - suv * suv
    return (sut * svv - svt * suv) / det, (svt * suu - sut * suv) / det


def load():
    data = []
    for r in json.load(open(LADDER, encoding="utf-8")):
        if not r.get("measured") or r["placement"] != "all in VRAM":
            continue
        p = os.path.join(MODELS, r["file"])
        if not os.path.exists(p):
            continue
        s = spec.from_gguf(p)
        opl = OPS_PER_LAYER.get(s.get("arch"))
        if not opl or not s.get("fmt_bw"):
            continue
        data.append({"n": r["name"], "L": s["n_layer"], "opl": opl,
                     "byte_ms": s["a"] * s["bits"] / 8 / s["fmt_bw"] * 1e3,
                     "t": 1000.0 / r["measured"], "meas": r["measured"],
                     "pred_now": r["predicted"]})
    return data


def loo(data, p):
    errs = []
    for i, d in enumerate(data):
        rest = [(x["byte_ms"], x["L"] * x["opl"] ** p, x["t"])
                for j, x in enumerate(data) if j != i]
        a, c = lstsq2(rest)
        t = a * d["byte_ms"] + c * d["L"] * d["opl"] ** p
        errs.append(abs((1000.0 / t / d["meas"] - 1) * 100) if t > 0 else 999.0)
    return errs


def main():
    data = load()
    med = lambda x: sorted(x)[len(x) // 2]
    ship = [abs((d["pred_now"] / d["meas"] - 1) * 100) for d in data]
    print(f"{len(data)} clean rows.  SHIPPED baseline: LOO-equivalent median "
          f"{med(ship):.1f}%  max {max(ship):.1f}%\n")
    print(f"{'p':>5} {'LOO median':>11} {'LOO max':>9}   verdict")
    best = None
    for p in [i / 20 for i in range(0, 21)]:
        e = loo(data, p)
        m, mx = med(e), max(e)
        if best is None or m < best[1]:
            best = (p, m, mx)
        mark = ""
        if p in (0.0, 1.0):
            mark = "  <- pure per-layer" if p == 0.0 else "  <- flat per-op (refuted above)"
        print(f"{p:>5.2f} {m:>10.1f}% {mx:>8.1f}%{mark}")
    p, m, mx = best
    print(f"\nbest p = {p:.2f}: LOO median {m:.1f}%  max {mx:.1f}%")
    print(f"shipped        : LOO median {med(ship):.1f}%  max {max(ship):.1f}%")
    if m < med(ship):
        print(f"-> sublinear form BEATS shipped by {med(ship)-m:.1f} points of median error")
    else:
        print("-> sublinear form does NOT beat shipped. The op-count axis, in every form")
        print("   tested, fails to improve out-of-sample prediction on this data.")
    a, c = lstsq2([(d["byte_ms"], d["L"] * d["opl"] ** p, d["t"]) for d in data])
    print(f"\nfull-data coefficients at p={p:.2f}:  BW x {1/a:.3f},  "
          f"tax = {c*1000:.1f} us * L * (ops/layer)^{p:.2f}")
    print(f"\n{'model':<24} {'L':>3} {'opl':>5} {'meas':>7} {'fit':>7} {'err':>7} {'shipped err':>12}")
    for d in data:
        t = a * d["byte_ms"] + c * d["L"] * d["opl"] ** p
        print(f"{d['n'][:24]:<24} {d['L']:>3} {d['opl']:>5.2f} {d['meas']:>7.1f} "
              f"{1000.0/t:>7.1f} {(1000.0/t/d['meas']-1)*100:>+6.1f}% "
              f"{(d['pred_now']/d['meas']-1)*100:>+11.1f}%")


if __name__ == "__main__":
    main()
