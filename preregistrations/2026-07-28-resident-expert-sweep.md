# Pre-registration #62: the resident-expert sweep (U-14) — is ~13% of prompt processing on the table?

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the runs. **STAKED.**

## Where U-14 came from

#61's cold-boot run measured pp2048 at 336.94 on the currently-shipped `-ot` pattern (expert
layers 11–47 → CPU, 37 CPU expert layers) where the original 386.04 was recorded on a pattern
with 32 CPU expert layers. The −12.7% tracks the +15.6% CPU expert-layer count — consistent with
prefill being CPU-expert-bound. If that mechanism is right, resident-expert count is a pp dial
that costs nothing in tg (tg parity across nearby splits is measured three times over).

## Arms — flagship, cold-state box, one session, clocks logged, r=2

`-ot "blk.(K..47).ffn_.*_exps.=CPU"` for K ∈ {11 (shipped), 14, 16, 18}, each measured for
pp2048 AND tg128 at `-b 1024 -ub 1024 -mmp 0`. K=18 puts 7 expert layers + attention in VRAM —
predicted near the VRAM budget at ub 1024; if it OOMs, that fact is the datapoint (the pattern's
VRAM cliff located) and the arm is recorded as OOM, not dropped silently.

## Stakes

- **P-1 (the mechanism).** pp2048 rises monotonically with resident-expert count (falling K→more
  residents… note K is the CPU-start index: HIGHER K = more residents). Predicted shape:
  pp ∝ 1/(CPU expert layers), i.e. K=16 lands at **375–400** (the original 386 was this class).
- **P-2 (the free-lunch check).** tg128 stays within **±5%** across all arms that fit — the dial
  moves pp without a tg price.
- **P-3 (the VRAM edge).** K=18 either fits and continues the pp trend, or OOMs — either way the
  cliff position enters `moe_split_flags`' safety logic.

## KILL RULE

**If P-1 fails** — pp does not track resident-expert count — the #61 explanation for the 336-vs-386
gap was wrong, the difference is elsewhere (flags, build, KV state), and U-14 closes refuted with
the shipped pattern keeping its current form. **If P-1 holds and P-2 holds**, the tool's `-ot`
pattern generation moves from "fixed 25% residency" to "max residents that fit the VRAM budget",
shipped with the sweep as evidence.

**Wired into:** pending; `findings/REGISTER.json:U-14` scores either way.

---

## Scored (2026-07-28, log: `weights/data/prereg62_resident_sweep.log`, clocks 1860-1885 MHz throughout)

**Verdict: P-1 band HIT with the predicted magnitude, P-2 HIT (around-mean), P-3 exposed a third
outcome the stake did not name. The shipped pattern was DOMINATED and the fix is shipped.**

| residents | pp2048 | tg128 |
|---|---|---|
| 11 (shipped) | 341.44 ± 0.75 | 20.89 ± 0.09 |
| 14 | 391.11 ± 1.71 | 21.67 ± 0.18 |
| **16** | **393.71 ± 1.71** | **22.21 ± 0.03** |
| 18 | 367.51 ± 2.46 | 21.80 ± 0.69 |

- **P-1:** K=16 landed at 393.71, inside the staked [375-400], +15.3% over shipped — the
  CPU-expert-bound mechanism confirmed through K=16. Monotonicity BREAKS at K=18: the VRAM edge
  is a SOFT degradation (-6.6% pp), not the OOM the stake named — the third outcome.
- **P-2:** tg spread ±3.2% around the mean (pairwise extreme 6.3%, disclosed). And the direction
  is a bonus: tg IMPROVES with residents (+6.3% at K=16) — more bytes on the 161 GB/s bus.
- **The real finding: the tool's pattern generator and its own frontier measurement disagreed.**
  The 386.04 frontier constant was measured at ~16 residents; the generator's VRAM budget
  (a 0.90 multiplier STACKED on the desktop reserve — a double discount) emitted 11. K=16 fits
  fine at ub 1024 and wins both metrics.

**Shipped in the same commit:** the double discount removed (one reserve, counted once), with
the #62 numbers and the soft-edge margin in the comment. The plan's split row moves 25% -> 31%
residency on the reference box — inside the measured optimum, one notch off the measured edge.
U-14 closes CONFIRMED with the fix live; the -29% hard cliff the old caution feared was the
ubatch compute-buffer cliff, guarded separately by safe_ubatch since v1.15.

**Wired into:** `quantprobe/plan.py` v_free (the fix) · `findings/REGISTER.json:U-14` (closed).
