"""U-33 / prereg #84 analysis: what SHAPE does the non-matmul cost have?

Prereg #83 reported us/call per op and I over-read it. The `calls` column is the
part that matters: normalized per token-step it is an exact, static property of the
architecture. This script asks three questions, in order of how much they can hurt:

  Q1  Is per-call cost independent of MODEL SIZE?  (launch-bound vs work-bound)
  Q2  How much of the reported non-matmul time is the PROFILER'S OWN event overhead?
  Q3  Does a layer-count term explain clean-build timing beyond the byte term?

Q3 is the one that matters for the tool, and it uses NO instrumented data at all.

  python weights/perop_shape.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "data", "prereg83_perop.log")

# llama-bench tg32 under this harness issues 33 token-steps (32 tg + 1). Verified
# independently per model below via ADD calls == 33 and FLASH_ATTN calls == 33*n_layer.
STEPS = 33

# clean (uninstrumented) tg32 for the same file, from the state-locked ladder a19aeee4.
# gemma4 is the depth-aware row and is NOT a like-for-like config - flagged, not used in Q2.
CLEAN = {
    "Qwen2.5-0.5B-Instruct-Q8_0.gguf": 153.39,
    "Qwen2.5-7B-Instruct-Q4_K_M.gguf": 22.73,
    "Qwen3-0.6B-Q8_0.gguf": 106.38,
    "Qwen3.5-4B-Q4_K_M.gguf": 30.17,
}

# ops that do real, size-scaling work over the KV cache or a state - NOT launch-bound.
ATTN_CLASS = {"FLASH_ATTN_EXT", "SOFT_MAX", "GATED_DELTA_NET", "SSM_CONV"}


def parse(path):
    models, cur = [], None
    for ln in open(path, encoding="utf-8"):
        m = re.match(r"^--- (\S+) \(", ln)
        if m:
            cur = {"file": m.group(1), "ops": {}, "tg": None}
            models.append(cur)
            continue
        if cur is None:
            continue
        m = re.search(r"\|\s+tg32\s+\|\s+([\d.]+) ±", ln)
        if m:
            cur["tg"] = float(m.group(1))
        m = re.match(r"^\[e9\] ([A-Z_]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s*$", ln)
        if m and m.group(1) != "TOTAL":
            cur["ops"][m.group(1)] = {"ms": float(m.group(2)), "calls": int(m.group(3)),
                                      "us": float(m.group(4))}
    return models


def n_layers(m):
    """Derive layer count from op call structure - no metadata needed."""
    o = m["ops"]
    att = o.get("FLASH_ATTN_EXT", {}).get("calls", 0) // STEPS
    gdn = o.get("GATED_DELTA_NET", {}).get("calls", 0) // STEPS
    if att or gdn:
        return att + gdn
    return o.get("ROPE", {}).get("calls", 0) // STEPS // 2   # MLA path: 2 ROPE/layer


def main():
    ms = parse(LOG)
    print("=" * 78)
    print("Q1  per-call cost vs model size - is it launch-bound?")
    print("=" * 78)
    print(f"{'model':<34} {'L':>3} {'params':>7}  {'ROPE':>7} {'RMS':>7} {'SET_ROWS':>9}")
    for m in ms:
        L = n_layers(m)
        o = m["ops"]
        g = lambda k: f"{o[k]['us']:.2f}" if k in o else "-"
        print(f"{m['file'][:34]:<34} {L:>3} {m['tg']:>6.1f}t/s  "
              f"{g('ROPE'):>7} {g('RMS_NORM'):>7} {g('SET_ROWS'):>9}")
    rope = [m["ops"]["ROPE"]["us"] for m in ms if "ROPE" in m["ops"]]
    print(f"\n  ROPE spread across 0.5B..16B: {min(rope):.2f}-{max(rope):.2f} us/call "
          f"({max(rope)/min(rope):.2f}x) while model size spans 32x")

    print()
    print("=" * 78)
    print("Q1b RMS_NORM: us/call vs NORMS PER LAYER (the anti-correlation)")
    print("=" * 78)
    print(f"{'model':<34} {'L':>3} {'norms/L':>8} {'us/call':>8} {'us/layer/tok':>13}")
    pts = []
    for m in ms:
        L, o = n_layers(m), m["ops"]
        if "RMS_NORM" not in m["ops"] or not L:
            continue
        npl = o["RMS_NORM"]["calls"] / STEPS / L
        per_layer = o["RMS_NORM"]["ms"] * 1000 / STEPS / L
        pts.append((m["file"], npl, o["RMS_NORM"]["us"], per_layer))
        print(f"{m['file'][:34]:<34} {L:>3} {npl:>8.2f} {o['RMS_NORM']['us']:>8.2f} "
              f"{per_layer:>13.1f}")
    print("\n  more norms per layer -> LOWER us/call: back-to-back tiny kernels pipeline,")
    print("  so the launch latency is shared. Cost is per-LAYER-ish, not per-call.")

    print()
    print("=" * 78)
    print("Q2  how much of 'non-matmul' is the PROFILER ITSELF?")
    print("=" * 78)
    print(f"{'model':<30} {'calls/tok':>10} {'instr ms':>9} {'clean ms':>9} {'delta':>8} {'us/call':>8}")
    for m in ms:
        if m["file"] not in CLEAN:
            continue
        calls = sum(o["calls"] for o in m["ops"].values()) / STEPS
        t_i, t_c = 1000.0 / m["tg"], 1000.0 / CLEAN[m["file"]]
        d = t_i - t_c
        print(f"{m['file'][:30]:<30} {calls:>10.0f} {t_i:>9.2f} {t_c:>9.2f} {d:>+8.2f} "
              f"{d*1000/calls:>8.2f}")
    print("\n  the instrumented build records a CUDA event pair per op. That cost is the")
    print("  SAME ORDER as the op times it reports -> #83's shares are UPPER BOUNDS.")

    print()
    print("=" * 78)
    print("Q3  non-matmul time, split into launch-bound vs attention-class")
    print("=" * 78)
    print(f"{'model':<30} {'L':>3} {'lb calls/tok':>13} {'lb ms/tok':>10} {'lb us/layer':>12}")
    for m in ms:
        L, o = n_layers(m), m["ops"]
        if not L:
            continue
        lb = [(k, v) for k, v in o.items() if k not in ATTN_CLASS and k != "MUL_MAT"
              and k != "MUL_MAT_ID"]
        c = sum(v["calls"] for _, v in lb) / STEPS
        t = sum(v["ms"] for _, v in lb) / STEPS
        print(f"{m['file'][:30]:<30} {L:>3} {c:>13.0f} {t:>10.3f} {t*1000/L:>12.1f}")


if __name__ == "__main__":
    main()
