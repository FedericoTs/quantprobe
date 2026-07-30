"""U-33 / prereg #84, Q3: does a LAYER-COUNT term explain clean-build timing that the
BYTE term cannot?

Uses ONLY uninstrumented llama-bench numbers from the state-locked ladder (cal_id
a19aeee4). No profiler, so none of #83's event overhead is in here.

  t_token = bytes_per_token / BW_format   +   residual
                (Law 4 prices this)          (Law 4 prices NOTHING here)

If the residual is launch-bound scaffolding as the per-op probe suggests, it should
be ~constant per LAYER and independent of model size. That is a different axis from
bytes, and the two decouple exactly along small-vs-large.

  python weights/layer_tax.py
"""
import glob
import json
import os

from quantprobe import spec

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("QP_GGUF_DIR", r"D:\evo-compress-data\gguf")
LADDER = os.path.join(HERE, "data", "ladder_state_locked.json")

# ops-per-layer class, counted from the #83 call structure (calls/step/layer).
# Standard transformer = 2 norms/layer; QK-norm adds 2; gemma sandwich-norms 7.
# Hybrid (SSM/gated-delta) carries extra scaffolding: CONCAT/CPY/CONT/L2_NORM.
FAMILY_OPS = {"qwen2": 2.0, "qwen3": 4.0, "qwen35": 3.3, "gemma4": 7.0, "gemma3": 7.0}


def find(fn):
    p = os.path.join(MODELS, fn)
    if os.path.exists(p):
        return p
    g = glob.glob(os.path.join(MODELS, "**", fn), recursive=True)
    return g[0] if g else None


def main():
    rows = [r for r in json.load(open(LADDER, encoding="utf-8"))
            if r.get("measured") and r["placement"] == "all in VRAM"]
    print("all-in-VRAM rows only: no CPU tier, no split, no mmap - the cleanest regime\n")
    print(f"{'model':<24} {'L':>3} {'fmt':>7} {'GB/tok':>7} {'BW':>6} "
          f"{'byte ms':>8} {'meas ms':>8} {'resid ms':>9} {'us/layer':>9} {'share':>6}")
    out = []
    for r in rows:
        p = find(r["file"])
        if not p:
            print(f"{r['name'][:24]:<24}  -- gguf not found: {r['file']}")
            continue
        s = spec.from_gguf(p)
        L, arch = s["n_layer"], s.get("arch", "?")
        # active bytes read per decode token, and the blended format bandwidth the tool
        # already derives per model (q4_K + q6_K mix etc).
        gb_read = s["a"] * s["bits"] / 8   # a is in billions -> GB
        bw = s.get("fmt_bw")
        fmt = f"{s['bits']:.2f}b"
        if not bw:
            print(f"{r['name'][:24]:<24} {L:>3} {str(fmt):>7}  -- no blended BW")
            continue
        byte_ms = gb_read / bw * 1e3
        meas_ms = 1000.0 / r["measured"]
        resid = meas_ms - byte_ms
        out.append((r["name"], arch, L, resid, meas_ms))
        print(f"{r['name'][:24]:<24} {L:>3} {str(fmt):>7} {gb_read:>7.3f} {bw:>6.1f} "
              f"{byte_ms:>8.2f} {meas_ms:>8.2f} {resid:>+9.2f} {resid*1000/L:>9.1f} "
              f"{resid/meas_ms*100:>5.1f}%")

    if len(out) < 3:
        print("\nnot enough rows resolved")
        return
    print("\n" + "=" * 78)
    per = [(n, a, r * 1000 / L) for n, a, L, r, _ in out]
    vals = [v for _, _, v in per]
    print(f"residual per layer: {min(vals):.1f} - {max(vals):.1f} us   "
          f"(spread {max(vals)/min(vals) if min(vals) > 0 else float('nan'):.2f}x)")
    print(f"model size spans   : {min(m for *_ , m in out):.1f} - "
          f"{max(m for *_, m in out):.1f} ms/token ({max(m for *_, m in out)/min(m for *_, m in out):.1f}x)")
    print("\nresidual as a SHARE of the token is what drives prediction bias:")
    for n, a, L, r, m in sorted(out, key=lambda x: x[4]):
        print(f"  {n[:26]:<26} {r/m*100:>5.1f}% of a {m:>6.2f} ms token   ({L} layers)")


if __name__ == "__main__":
    main()
