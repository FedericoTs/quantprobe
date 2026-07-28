# Pre-registration #64: anchored predictions (v1.20) — the LOO gate is the ship condition

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the gate runs. **STAKED.**

## The design under test

Calibrate's anchors become tier-local correction ratios composed by the law — never a lookup:

```
ratio_cpu = measured_cpu_anchor / law_prediction_of_that_anchor_arm     (clamped 0.70–1.40)
ratio_gpu = measured_gpu_anchor / law_prediction_of_that_anchor_arm     (clamped 0.70–1.40)
plan then uses rb×ratio_cpu and vb×ratio_gpu, provenance printed:
  [anchored: CPU x1.08, GPU x0.93 from your <model> anchors]
```

The anchor arm can never be a target arm; the target's own measurement is never consulted.
The demonstration arithmetic that motivated this: the CPU anchor alone moves the #63 prediction
19.4 → ~20.8 vs 21.21 measured (−8.5% → ~−2%), from a different model on a different placement.

## The gate — same-state corpus only (post-reboot, clocks verified per arm)

Fresh measurements this session where needed, so no arm mixes boost states. Targets (never used
as anchors): 7B Q2_K all-in-VRAM, 7B Q4_0 all-in-VRAM, 0.6B Q8 all-in-VRAM, the flagship split
(#61/#63 arms), DS-Lite split (fresh re-run). Anchors: the 0.5B pure-CPU and 0.5B all-in-VRAM
runs from calibrate.

- **P-1 (SHIP CONDITION).** Median |error| of ANCHORED predictions across the target arms is
  **strictly lower** than the plain law's median |error| on the same arms.
- **P-2 (usefulness).** Anchored median |error| ≤ **12%**.
- **P-3 (the hard case, scoped in advance).** DS-Lite is predicted to REMAIN an outlier
  (#59 found its miss structural — the MLA graph term — and no tier ratio fixes a missing term).
  Its error is reported but the median protects the gate from it; if DS-Lite alone decides P-1,
  that is stated rather than hidden.

## KILL RULE

**If P-1 fails, anchored predictions ship BEHIND A FLAG (off by default) with the miss published
in the release notes** — the feature exists for those who want it, and the default stays the
plain law until the anchors earn it. If P-1 holds and P-2 fails, same treatment. No middle path
where the default changes on a partial result.

**Wired into:** pending the gate; v1.20.0 release notes will quote this document either way.
