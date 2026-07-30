"""Register the U-33 result: op-count overhead REFUTED as a predictive term, plus the
second correction to L-21 (the profiler's own cost) and the zero-parameter shape signal.
Run once.  python weights/reg_u33.py
"""
import json

P = "findings/REGISTER.json"
d = json.load(open(P, encoding="utf-8"))

L21 = [x for x in d["laws"] if x["id"] == "L-21"][0]
L21["claim"] += (
    " [SECOND CORRECTION, same data, U-33: the overhead SHARES above are UPPER BOUNDS "
    "inflated by the profiler itself. The instrumented build records a CUDA event pair "
    "per op; comparing its tg32 against the clean ladder tg32 for the same four files "
    "gives 3.80-8.32 us of instrumentation per op call - the SAME ORDER as the 5-16 us "
    "op times it reports. With 124-373 launch-bound calls per token that is 0.5-3.1 ms, "
    "i.e. 30-100% of the non-matmul bucket it claims to measure. The 8.4%-27.0% range is "
    "therefore not a clean measurement of overhead, and the direction of the bias is "
    "worst exactly where call counts are highest - the small models the law is about.]")
L21["status"] = "qualified"
L21["magnitude"] = (
    "non-matmul share 8.4%-27.0% is an UPPER BOUND (profiler self-cost 3.8-8.3 us/call, "
    "same order as the op times); per-op costs vary 2x by architecture, no universal constant")
L21["wired_into"] = (
    "findings only, and U-33 refuted the obvious wiring. Any real magnitude needs an "
    "instrumentation-free probe (one event pair around N launches, not N event pairs).")

d["laws"] = [x if x["id"] != "L-21" else L21 for x in d["laws"]]

d["dead_ends"].append({
    "id": "D-24",
    "kind": "dead_end",
    "status": "refuted",
    "confidence": "measured",
    "claim": (
        "OP COUNT IS NOT A USABLE PREDICTIVE TERM. The launch-bound structure is real - "
        "ROPE costs 5.24-8.30 us/call across models spanning 32x in size (1.58x spread), "
        "which is definitive: these ops are latency-bound, not work-bound, so their cost "
        "tracks LAUNCHES (layers x ops-per-layer, countable from GGUF metadata with zero "
        "measurement) and not BYTES. That structure does NOT convert into better "
        "predictions. Fitting t = a*(bytes/fmt_bw) + c*L*(ops_per_layer^p) jointly on the "
        "six clean all-in-VRAM ladder rows and sweeping p over [0,1] in 0.05 steps: the "
        "BEST leave-one-out median error is 12.1% (at p=1.0) against the shipped model's "
        "8.7%. Every form tested loses. In-sample it looks like a win (3.3% median) - which "
        "is exactly why LOO is the only honest read at 2 parameters and 6 rows."),
    "magnitude": "LOO median 12.1% (best) vs shipped 8.7%; LOO max 34.6% vs shipped 18.6%",
    "evidence": ("weights/joint_fit.py, weights/sublinear_fit.py over "
                 "weights/data/ladder_state_locked.json (cal_id a19aeee4) + op counts from "
                 "weights/data/prereg83_perop.log"),
    "scope": ("Pascal, batch-1 decode, six GPU-resident models 0.5B-12B. NOT a claim that "
              "overhead is absent - it is a claim that op count cannot price it. The "
              "residual is NON-MONOTONE in op count (49 ops -> 1.90 ms, 57 -> 0.92 ms, "
              "108 -> 5.92 ms, 113 -> 3.84 ms, 337 -> 21.95 ms), and no monotone function "
              "of op count can fit a non-monotone target."),
    "wired_into": "nothing - refuted before wiring, which is the point of testing first",
})

d["dead_ends"].append({
    "id": "D-25",
    "kind": "dead_end",
    "status": "refuted",
    "confidence": "measured",
    "claim": (
        "THE 'RMS_NORM COST IS SIZE-INVARIANT' READING OF #83 WAS AN ARTIFACT OF AVERAGING. "
        "us/call ANTI-correlates with how many norms the architecture emits per layer: 2.04 "
        "norms/layer -> 15.27 and 15.51 us/call, 3.28 -> 8.26, 4.04 -> 9.30, 7.02 -> 7.72. "
        "Cost per LAYER moves far less than call count does: going from 2 to 7 norms per "
        "layer is 3.5x the calls but only 1.74x the time (31.2 -> 54.2 us/layer/token). "
        "Back-to-back tiny kernels pipeline and share their launch latency, so us/call is an "
        "average over a burst, not a per-call price. Reporting us/call as if it were a "
        "constant unit cost - which #83 and L-21 both did - reads the pipelining as physics."),
    "magnitude": "us/call spans 7.72-15.51 (2.0x) while us/layer/token spans 27.1-54.2 (2.0x) in the OPPOSITE direction",
    "evidence": "weights/perop_shape.py Q1b over weights/data/prereg83_perop.log",
    "scope": "Pascal, batch-1 decode, five valid GPU-resident profiles (DS-Lite excluded as streaming)",
    "wired_into": "nothing; corrects L-21's framing",
})

d["untried"].append({
    "id": "U-32",
    "kind": "untried",
    "status": "open",
    "confidence": "suggestive",
    "claim": (
        "TENSOR SHAPE (L-20), NOT OP COUNT, IS THE CANDIDATE THAT SURVIVED. Asking what "
        "fraction of its nominal FORMAT_EBW each clean model actually achieves, and "
        "predicting that fraction from the ALREADY-MEASURED L-20 knee curve using only the "
        "model's hidden and feed_forward_length - zero free parameters - gives r = +0.769 "
        "(n=6) and the correct direction on every row. It systematically UNDER-corrects "
        "(needed 0.591-1.017, shape-predicted 0.878-0.982), so it is a partial explanation, "
        "not the whole residual. Two rows say what the remainder is: the 7B pair sits at "
        "~1.00 because FORMAT_EBW was calibrated on dense 7B files, so their residual is "
        "~zero BY CONSTRUCTION and the whole mismatch is pushed onto every other model - "
        "that circularity is why U-29 could not be fixed by swapping one constant. And "
        "gemma4-12B needs 0.726 where shape predicts 0.982; its deficit is attention-shaped "
        "(FLASH_ATTN_EXT 134.71 us/call x 48 layers), not geometry."),
    "magnitude": "r = +0.769 with zero fitted parameters; under-corrects by 0.14-0.29 on the four non-anchor rows",
    "evidence": "weights/shape_vs_ops.py; L-20 curve from prereg81_knee.log (single session, C-14 safe)",
    "scope": "Pascal, six GPU-resident models. Correlation on n=6 with one custom model (gemma4-late12) in it - suggestive only.",
    "wired_into": ("nothing yet. The honest next step is a STAKED prereg: predict each "
                   "model's achieved-bandwidth fraction from geometry BEFORE measuring a "
                   "fresh ladder, with a kill rule on LOO beating 8.7%."),
})

json.dump(d, open(P, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("register updated: L-21 qualified, D-24, D-25, U-32 added")
