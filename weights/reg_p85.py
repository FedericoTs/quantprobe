"""Register prereg #85 (uninstrumented launch probe). Run once.
  python weights/reg_p85.py
"""
import json

P = "findings/REGISTER.json"
d = json.load(open(P, encoding="utf-8"))

# ---- L-21: the magnitude is now measured, and it is 2-4x smaller than published.
L21 = [x for x in d["laws"] if x["id"] == "L-21"][0]
L21["claim"] += (
    " [THIRD CORRECTION, prereg #85, and this one has a number: the launch cost was "
    "measured WITHOUT per-op instrumentation - one CUDA event pair around N launches "
    "instead of N event pairs. A norm-shaped kernel at Qwen2.5-7B's hidden size costs "
    "3.74 us marginal (empty-kernel dispatch floor 2.10 us), against the 15.51 us/call "
    "#83 reported for the same model. Re-pricing the launch-bound bucket at the measured "
    "3.7 us gives 7.0% of the token on Qwen2.5-0.5B, 1.2% on Qwen2.5-7B and 2.7% on "
    "gemma4-12B - NOT the 27.0% / 8.4% / 14.2% published. Non-matmul overhead is real "
    "but roughly a quarter of what this law claimed, and it is far too small to be the "
    "size floor it was named after.]")
L21["status"] = "superseded"
L21["magnitude"] = (
    "SUPERSEDED: measured launch-bound overhead is 1.2%-7.0% of a decode token "
    "(prereg #85), not the 8.4%-27.0% originally published from instrumented data")
L21["wired_into"] = ("nothing, and it should stay that way - at 1-7% it cannot carry the "
                     "size floor. See D-24 (op-count refuted) and U-32 (shape survives).")
d["laws"] = [x if x["id"] != "L-21" else L21 for x in d["laws"]]

# ---- D-25: the OBSERVATION stands, my MECHANISM was wrong.
D25 = [x for x in d["dead_ends"] if x["id"] == "D-25"][0]
D25["claim"] += (
    " [MECHANISM RETRACTED by prereg #85 arm C/D, staked and scored the same day. I "
    "attributed the anti-correlation to back-to-back kernels PIPELINING and sharing "
    "launch latency. Measured: per-launch cost is FLAT across burst length (4.27 us at "
    "N=1, 3.77 us at N=1024 - no sharing), and a dependent chain where each kernel reads "
    "the previous kernel's output costs 1.00x an independent burst, not the >=1.3x I "
    "staked. There is no overlap to share, so pipelining cannot be the cause. The "
    "likeliest remaining explanation is GAP ATTRIBUTION inside the profiler: an isolated "
    "op absorbs more of the inter-kernel gap than a dense run of ops does - a property of "
    "the ruler, not the hardware, which is the same place prereg #85's P1 landed.]")
D25["magnitude"] += "; mechanism retracted - observation stands, explanation refuted"

# ---- D-24: now has a physical reason, not just a cross-validation result.
D24 = [x for x in d["dead_ends"] if x["id"] == "D-24"][0]
D24["claim"] += (
    " [STRENGTHENED by prereg #85: the joint fit put the op-count coefficient at 72.9 us "
    "per emitted op. The physically measured launch cost is 3.74 us. A term fitted at 20x "
    "the cost of the thing it claims to price is not pricing that thing - it is absorbing "
    "something else. This upgrades D-24 from 'loses out of sample' to 'cannot be the "
    "mechanism', and it says where to look instead: the residual is a BANDWIDTH mismatch, "
    "not an overhead one.]")
D24["magnitude"] += "; fitted 72.9 us/op vs measured 3.74 us/launch = 20x, so the term was absorbing bandwidth"

# ---- new law: the measured launch floor.
d["laws"].append({
    "id": "L-22",
    "kind": "law",
    "status": "established",
    "confidence": "measured",
    "claim": (
        "THE KERNEL-LAUNCH FLOOR ON THIS BOX IS 2.10 us, AND A REAL NORM COSTS 3.74 us. "
        "Measured with one CUDA event pair around N launches (prereg #85, tools/kernelprobe/"
        "launch.cu), so the harness contributes two events per measurement regardless of N - "
        "the opposite geometry to #83's per-op event pairs. An empty kernel costs 2.10 us/"
        "launch, flat from N=100 to N=10000. A single-block RMS-norm kernel costs 3.74 us "
        "marginal at hidden 3584. Per-launch cost does NOT fall with burst length (4.27 us "
        "at N=1, 3.77 us at N=1024) and a data dependency between consecutive kernels costs "
        "nothing extra (1.00x) - same-stream kernels are already serialized, so there is no "
        "overlap to lose. There IS a small work component on top of the dispatch floor: "
        "3.05-4.03 us over the 896-3840 hidden sizes this ladder actually uses (1.24x), "
        "widening to 7.04 us at 28672 floats (2.31x over the full sweep, which MISSED the "
        "<=1.5x I staked - though 28672 is an FFN width, not a norm width, so the staked "
        "range was my own spec error)."),
    "magnitude": "2.10 us dispatch floor; 3.74 us for a norm at hidden 3584; 1.24x spread over real norm widths",
    "evidence": "prereg #85, tools/kernelprobe/launch.cu (P1 held, P2 and P3 missed - all three staked in the source before compiling)",
    "scope": ("GTX 1060 6GB, Pascal cc 6.1, CUDA 12.9, single-block launches, one stream, "
              "no CUDA graphs. Synthetic kernels, NOT llama.cpp's own - so this bounds the "
              "GPU-side launch cost but does not measure ggml's CPU-side graph submission, "
              "which is a separate and unmeasured quantity."),
    "wired_into": ("nothing yet - its job was to falsify, and it did: it retired L-21's "
                   "magnitude, retracted D-25's mechanism and gave D-24 a physical cause."),
})

# ---- U-32 promoted: it is now the only surviving candidate.
U32 = [x for x in d["untried"] if x["id"] == "U-32"][0]
U32["claim"] += (
    " [PROMOTED by prereg #85: with launch overhead measured at 1.2%-7.0% of a token and "
    "the op-count coefficient exposed as 20x too large to be launch cost, tensor shape is "
    "no longer one candidate among several - it is the only one left standing for the "
    "residual Law 4 cannot explain.]")
U32["confidence"] = "suggestive, now sole surviving candidate"

json.dump(d, open(P, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("registered: L-21 superseded, L-22 added, D-24 strengthened, D-25 mechanism retracted, U-32 promoted")
