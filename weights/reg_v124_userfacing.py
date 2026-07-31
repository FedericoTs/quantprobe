"""Register the two v1.24.0 user-facing defects that shipped WITHOUT a register entry.

Both fixes landed inside commit 0782477, whose subject is about something else entirely
("L-24: prereg #91 scored"). The release audit could not trace either claim to an ID -
which is the point: a user-facing correction buried in an unrelated commit is invisible
to the very gate that exists to catch untraceable claims.

  python weights/reg_v124_userfacing.py
"""
import json

P = "findings/REGISTER.json"
d = json.load(open(P, encoding="utf-8"))
n = max(int(x["id"][2:]) for x in d["contradictions"] if x["id"].startswith("C-")) + 1

EV = ("weights/data/resweep340_20260731T1656Z_{STAKE,RESULT}.md plus _audit.json, _cells.log, "
      "_inject_upgrade.log, _inject_spec.log; auditor weights/resweep340_audit.py; "
      "driver sweep weights/data/resweep340_20260731T1656Z_exp54.json")

d["contradictions"].append({
    "id": f"C-{n}",
    "kind": "contradiction",
    "status": "resolved",
    "confidence": "measured",
    "claim": (
        "THE UPGRADE ADVISOR COULD ONLY EVER SUPPRESS GOOD UPGRADES, NEVER INVENT BAD ONES - a "
        "one-directional error, which is why it survived: every symptom looked like conservatism. "
        "upgrade_advisor() built its counterfactual by copying the baseline and overriding one "
        "resource key, but dropped n_layer and true_size_gb on the way in. evaluate() then "
        "re-derived both from presets, so the counterfactual was not the same model with more RAM "
        "- it was a DIFFERENT model. Two upgrades on unrelated resources could therefore land on "
        "the identical tok/s and both be reported, or a real gain could vanish into the "
        "re-derivation. Signature on the committed pre-fix artifact: llama-70b/colibri/ctx16384 "
        "printed '+16 GB RAM' and 'NVMe SSD' both at 0.5749513 tok/s. FIXED in 0782477 (identity "
        "carried at the function boundary). VERIFIED 2026-07-31 by a 340-cell re-sweep staked "
        "before it ran: 0 invented pairs against a live population of 24 cells where two "
        "different-resource upgrades both fire (so the check is not vacuous), 0 identity "
        "mismatches over 700 counterfactual evaluate() calls, 0 arithmetic mismatches over 131 "
        "fired upgrades. All 31 previously-affected cells named individually and clean. The "
        "auditor was falsified first: re-injecting the defect makes it exit 1 with 10 invented "
        "pairs, 700/700 identity mismatches and 51 arithmetic misses, and reproduces the "
        "0.5749513 signature to six decimals."),
    "magnitude": "31 of 340 grid cells carried altered upgrade advice; 0 remain after the fix",
    "evidence": EV,
    "scope": (
        "The 10-model x 17-machine x ctx{0,16384} grid at 2.5 bits - pure arithmetic, nothing "
        "timed, no cal_id, so C-14 does not bind. This proves the counterfactual is drawn from "
        "the right row menu and reproduces to 1e-9. It does NOT prove the advice is CORRECT about "
        "hardware: nobody has bought the recommended RAM and measured the printed tok/s. That "
        "needs an A/B on real hardware and has never been run."),
    "wired_into": (
        "quantprobe/plan.py upgrade_advisor; guarded by the staked auditor "
        "weights/resweep340_audit.py, which is re-runnable with --inject upgrade to prove it can "
        "still fail. Shipped in v1.24.0."),
})

d["contradictions"].append({
    "id": f"C-{n + 1}",
    "kind": "contradiction",
    "status": "resolved",
    "confidence": "measured",
    "claim": (
        "THE SPECULATION BLOCK PRINTED A SPEEDUP THE ROW IT WAS PRINTED ON COULD NOT REACH. "
        "speculation_advice() was called without the row, so it quoted its constant headline "
        "(2.10x dense / 4.7x n-gram-tuned) with no reference to the row's own terms. On any row "
        "carrying a CPU-attention term the arithmetic ceiling is total/(bw/R + cpu), which falls "
        "far below the constant - on the 17 affected cells the row's own bound was 1.025x-1.229x "
        "against a printed 2.10x, with CPU-attention shares of 64.5%-95.3%. FIXED in 0782477 (the "
        "row is passed; unreachable headlines print 'NOT REACHABLE ON THIS ROW' with the bound). "
        "VERIFIED 2026-07-31 by the same staked 340-cell re-sweep: 0 unreachable headlines printed "
        "unqualified, 0 spurious qualifiers, 0 printed-vs-recomputed bound mismatches, with the "
        "bound recomputed independently rather than by calling the shipped helper. All 17 "
        "previously-oversold cells now qualified, plus 5 MORE that the old ceiling(k=2)<1.5 metric "
        "never listed (llama-70b on 2016-xmp/2016/gaming/laptop-8gb/rtx-3060 at ctx16384) - the "
        "shipped gate is strictly wider than the metric that found the bug. Falsified first: "
        "re-injecting the pre-fix call (row=None) makes the auditor exit 1 with 22 unreachable "
        "headlines."),
    "magnitude": "17 of 127 speculation cells printed an unreachable multiplier; 22 now qualified, 0 unqualified",
    "evidence": EV,
    "scope": (
        "READ THIS BEFORE QUOTING THE CLEAN RESULT. All 22 qualified cells are the DENSE 2.10x "
        "headline. The 4.7x n-gram figure was never exercised by the sweep: the 25 cells that "
        "quote it all sit on the MoE 'split experts' placement, where attention stays on the GPU "
        "and cpu_compute is exactly 0, so bound == R and the qualifier is correctly silent BY "
        "CONSTRUCTION - only three placements in evaluate() carry a cpu_compute term and none is a "
        "split-experts placement. The 4.7x gate was checked directly rather than assumed (a "
        "synthetic split-experts row at 50% CPU attention prints 'at most 1.65x'; the same row "
        "without a CPU term prints no qualifier), but that is SYNTHETIC-PROBE coverage, not grid "
        "coverage. Separately: zero of the 127 cells fall inside the 1% display tolerance, so "
        "SPEC_HEADLINE_TOL is load-bearing nowhere on this grid and this run says nothing about "
        "whether 1% is the right threshold. Display rounding disclosed, not folded into tolerance: "
        "SPEC_X_NGRAM_TUNED = 4.634146 prints as '4.7x' (+1.42%)."),
    "wired_into": (
        "quantprobe/plan.py speculation_advice(row=...); guarded by weights/resweep340_audit.py, "
        "re-runnable with --inject spec. Shipped in v1.24.0."),
})

json.dump(d, open(P, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"C-{n} (upgrade advisor) and C-{n + 1} (speculation headline) registered")
