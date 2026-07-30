# Pre-registration #82: replace the fitted eta with three measured factors

**Author:** Federico Sciuca · **Date staked:** 2026-07-30, BEFORE computing the fit. **STAKED.**
**Baseline:** state-locked ladder `a19aeee4`, median |err| 8.8%.

Today established that a single `eta` absorbs at least three separate effects: the FORMAT of the
bytes (L-15/L-16/#70), the ROWS per tensor (L-20, occupancy floor at ~4096 rows) and the
BYTES PER ROW (L-20/#81, the ceiling scales ~sqrt). That is why the byte convention and eta are
coupled (U-29): eta is a free parameter silently compensating for whatever the byte model gets
wrong. Preliminary arithmetic on the six all-in-VRAM arms - exact per-tensor bytes, L-20's curve,
**no fitted eta at all** - reproduced measured bandwidth at r = 0.854 for the ranking, with the
level off by a ratio spanning 0.51-0.82.

## The form under test

    bw(tensor) = k_machine x FMT(format) x ceiling(bytes_per_row) x f(rows_per_tensor)
    time(model) = SUM over read tensors of  bytes / bw(tensor)

`FMT` = `FORMAT_EBW[fmt] / 106.4` (measured, #52/#70/#77). `ceiling` and `f` = measured (#80/#81).
**`k_machine` is the ONLY free parameter** and is fitted on the six all-in-VRAM arms.

## Stakes

- **P-1 (the form holds where it is fitted).** After fitting the single constant, all six
  all-in-VRAM arms land within **±20%** — i.e. the 1.6x ratio spread collapses. Three measured
  factors plus one constant must beat one constant doing all the work.
- **P-2 (OUT OF SAMPLE — the seven split arms are NOT in the fit).** Applying the same
  `k_machine` to the GPU side of the seven split/hybrid arms (their CPU side keeps today's
  model untouched) leaves their median |error| **no worse than the 8.8% baseline**, and their
  MoE-IQ / MoE-K class means both move toward zero from +9.1% / -7.0%.
- **P-3 (physical sanity).** `k_machine` lands in **0.3-1.0** — it must read as a machine
  efficiency against spec bandwidth, not as a fudge factor absorbing a modelling error. A value
  above 1.0 would mean the geometry model under-predicts the hardware, which would falsify the
  "measured factors are upper bounds" framing.

## KILL RULE

**If P-1 fails, the multiplicative form is refuted** and U-31 is scored dead: eta stays a fitted
constant, U-29 stays blocked, and the honest statement is that we cannot decompose it with the
data we have. If P-1 holds but P-2 fails, the form works only where fitted - recorded as a
scoped result and explicitly NOT wired, because a law that needs re-fitting per placement is a
lookup table wearing a law's clothes.

**Wired into:** pending; would replace `resolve_gpu_eta`'s constant with the three-factor product.
