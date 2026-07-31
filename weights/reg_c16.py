"""Register the calibration-contract finding (C-16). Run once.
  python weights/reg_c16.py
"""
import json

P = "findings/REGISTER.json"
d = json.load(open(P, encoding="utf-8"))
n = max(int(x["id"][2:]) for x in d["contradictions"] if x["id"].startswith("C-")) + 1

d["contradictions"].append({
    "id": f"C-{n}",
    "kind": "contradiction",
    "status": "open",
    "confidence": "measured",
    "claim": (
        "CALIBRATION HAS NO COMPLETENESS CONTRACT, AND PARTIAL CALIBRATION IS WORSE THAN NONE "
        "FOR THE COMPONENTS YOU SKIPPED. Found via five full ladder runs on 2026-07-31 while "
        "chasing an unrelated (real, reboot-recoverable) machine slowdown; the GPU-only control "
        "row measured 153.39 / 153.74 / 153.00 / 152.95 / 152.08 across all five states, so the "
        "card is the fixed point throughout. THREE SHIPPED DEFECTS. (1) NOTHING EVER RE-MEASURES: "
        "calibration is written to ~/.quantprobe/calibration.json and read back forever; only an "
        "explicit 'quantprobe calibrate' re-measures, so a user who calibrates once during a "
        "degraded window - on battery, under load, mid-throttle - carries that state silently and "
        "permanently. Measured cost here: predictions frozen at degraded values against a "
        "recovered machine, median 8.8% -> 9.9%, every error flipping sign. (2) A MISSING FILE "
        "DEGRADES SILENTLY TO PRESETS rather than re-measuring, reporting 'machine state: "
        "uncalibrated (ram None GB/s, disk None GB/s)' in a line nobody must read, and produced a "
        "27.2% median on a perfectly healthy machine - while the calibrated case gets a prominent "
        "[calibrated] tag and the uncalibrated case gets no equivalent. (3) THE NON-OBVIOUS ONE: "
        "calibrating RAM alone made the RAM-bound rows the most accurate ever recorded on this "
        "ladder (Qwen3-30B-A3B -10.2% -> +1.6%, Qwen3-Coder-30B -10.0% -> +0.9%) while wrecking "
        "every GPU-bound row (Qwen2.5-0.5B -18.6% -> -28.1%, gemma4-12B 0.0% -> +12.8%, "
        "Qwen3.6-APEX-MTP +8.5% -> +42.0%), for a net median of 12.5% - WORSE than the 8.8% "
        "baseline it replaced. Calibration is a VECTOR whose preset components are mutually "
        "consistent, each compensating for the others' biases; measuring one component and leaving "
        "the rest on presets breaks that consistency, and the damage lands on exactly the "
        "components you did NOT measure. Restoring the full vector (ram 24.14, disk 2.99, decode "
        "anchor; cal c24a253b) gave a 7.9% median, beating the 8.8% baseline. SEPARATE FOURTH "
        "DEFECT: the disk-only probe read 0.46 GB/s where the full calibration measured 2.99 GB/s "
        "on the same drive in the same session (+550%); every disk-tier prediction under cal "
        "75eb1d48, which shipped 0.45, rested on it. Cause undiagnosed - cold-cache/first-touch "
        "suspected. WHAT WORKED: the drift detector fired unprompted on the state change, named "
        "both moved quantities, and quoted C-14 back at its own authors. The mechanism to catch "
        "this existed; what is missing is anything that forces it to run."),
    "magnitude": (
        "median |err| by calibration state on one idle machine: full 7.9%, baseline 8.8%, "
        "RAM-only 12.5%, uncalibrated 27.2%; disk probe wrong by 6.5x (0.46 vs 2.99 GB/s)"),
    "evidence": (
        "preregistrations/2026-07-31-calibration-contract.md; five retained ladders "
        "(weights/data/ladder_20260731_idle_prereboot_75eb1d48.json, _postreboot_stalecal.json, "
        "_uncalibrated.json, _ramonly_37a91948.json, plus the locked c24a253b ladder) and "
        "weights/data/calibration_75eb1d48_degraded.json"),
    "scope": (
        "One machine (GTX 1060 6GB + DDR4-3200), one OS, the 14-model ladder. The "
        "partial-calibration asymmetry is MEASURED here; the mechanism (mutually-consistent "
        "presets) is INFERRED from the row-direction pattern and not independently tested. The "
        "underlying slowdown was real and reboot-recoverable (ram 22.58 -> 25.23 GB/s, +11.7%) and "
        "is NOT the defect - it is how the defect became visible. Two staked calls scored at equal "
        "prominence: 'an idle machine will restore the MoE rows' REFUTED (30B measured 17.51 "
        "idle-gated, and the Qwen3.6 rows got SLOWER at idle than under load); 'a reboot will "
        "restore them' CONFIRMED (21.10 immediately post-reboot, 21.56 on the final full "
        "calibration, against a 21.71 baseline)."),
    "wired_into": (
        "nothing yet - the owed contract is five parts: (a) stamp every calibration with "
        "wall-clock time AND a boot-session id, warning loudly on reuse across a reboot or beyond "
        "an age threshold; (b) mark each component measured|preset individually and warn on any "
        "MIXED state, since partial calibration must be as visible as no calibration; (c) give the "
        "uncalibrated state the same prominence the [calibrated] tag gets; (d) re-measure by "
        "default in bench paths rather than on request; (e) diagnose the disk probe before any "
        "disk-tier number is quoted again."),
})

json.dump(d, open(P, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"C-{n} registered")
