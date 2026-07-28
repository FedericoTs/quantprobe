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

---

## Scored (2026-07-28, log: `weights/data/prereg64_gate.log`, clocks 1873-1898 MHz on every arm)

**Verdict: P-1 HIT decisively, P-2 HIT, P-3 held in spirit with its attribution corrected.
Per the staked rule, anchored predictions ship DEFAULT-ON in v1.20.**

| target (never an anchor) | anchored | plain law | measured | anch. err | plain err |
|---|---|---|---|---|---|
| 0.6B Q8 all-in-VRAM | 106.4 | 91.4 | 104.57 | **+1.8%** | -12.6% |
| 7B Q2_K all-in-VRAM | 20.7 | 17.8 | 21.98 | **-5.8%** | -19.0% |
| 7B Q4_0 all-in-VRAM | 15.3 | 13.2 | 27.07 | -43.5% | -51.2% |
| flagship split (plan-emitted config) | 21.5 | 18.8 | 21.16 | **+1.6%** | -11.2% |
| DS-Lite split (plan-emitted config) | 18.4 | 16.1 | 23.08 | -20.3% | -30.2% |

- **P-1 HIT: anchored median |error| 5.8% vs plain 19.0% — and every single arm improved.**
- **P-2 HIT** (5.8% <= 12%).
- **P-3:** the outliers are the KNOWN format blindness (a single GPU eta prices Q4_0 like Q2_K;
  L-16 measured them 1.8x apart), both safely in the UNDER-promise direction — the one-sided
  floor holds everywhere (worst case: measured 27.07 >= 0.9 x 15.3 by a mile). Per-format GPU
  anchoring is the obvious v1.21 refinement and is logged as U-15, not smuggled in now.

### Two instrument findings from running the gate, both fixed in the same commit

1. **The boost-state verdict false-positived**: calibrate's short tg64 GPU anchor gave the clock
   sampler only model-LOAD samples (1506 MHz mid-ramp) and printed a REBOOT alarm while the very
   next benchmarks sustained 1873-1898. Fixed: 1 s sampling, tg128 anchor, and a >=3-loaded-sample
   minimum before any verdict.
2. **#59's DS-Lite attribution is REVISED**: at healthy clocks DS-Lite measures **23.08 — inside
   #59's staked band [17.5, 25.5]** that it "missed" at 16.26. The miss was substantially the
   then-undiagnosed stuck-boost state, not an MLA structural term. The L-17 kill STANDS regardless
   (the 0.5B arm failed its band in any clock state), but the "missing term" language in #59 is
   withdrawn; what remains attributed to DS structure is the residual -20% anchored error here.

**Wired into:** `quantprobe/plan.py` (default-on anchoring, --no-anchors escape) ·
`quantprobe/calibrate.py` (GPU anchor + sampler guard) · `findings/REGISTER.json:V-19` (the
lever), `U-15` (per-format GPU anchors), L-17 note (#59 DS revision) · v1.20.0 release notes.
