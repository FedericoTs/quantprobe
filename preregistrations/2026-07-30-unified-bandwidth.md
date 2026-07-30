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

---

## SCORED — 2026-07-30. **P-1 FAILED. THE FORM IS REFUTED; U-31 IS DEAD AS FORMULATED.**

`k_machine` fitted on the six all-in-VRAM arms = **0.672** (per-arm 0.535–0.763).

| arm | measured | unified model | error | (today's model) |
|---|---|---|---|---|
| Qwen2.5-0.5B Q8_0 | 153.39 | 154.40 | **+0.7%** | −18.6% |
| Qwen3-0.6B Q8_0 | 106.38 | 133.46 | **+25.5%** | −2.6% |
| Qwen3.5-4B Q4_K_M | 30.17 | 31.71 | +5.1% | +8.7% |
| Qwen2.5-7B IQ4_NL | 25.36 | 22.55 | −11.1% | +8.8% |
| Qwen2.5-7B Q4_K_M | 22.73 | 20.01 | −12.0% | −5.0% |
| gemma4-12B | 12.50 | 12.53 | **+0.3%** | +0.0% |

- **P-3 HIT.** k = 0.672 is a physically sensible machine efficiency, not a fudge.
- **P-1 MISS.** Worst arm +25.5%, above the ±20% bar — and the deeper failure is that the
  spread barely moved: the per-arm constant still ranges 0.535–0.763 (1.4×) where the old ratio
  ranged 0.51–0.82 (1.6×). **Median |error| got WORSE, 6.9% → 8.1%.** Three measured factors
  plus one constant did *not* beat one constant doing all the work.
- **P-2 NOT RUN.** The kill rule makes P-1 the gate; running the holdout after the gate failed
  would be shopping for a number that flatters the form.

**What this means, plainly: we cannot decompose eta with the data we have.** The geometry curve
is real (#80/#81 are solid) but it was measured on a synthetic 4.5-bit kernel, and transferring
it to real files — different formats, mixed shapes, real graph overheads — loses whatever made
it predictive. `eta` remains a fitted constant, **U-29 remains blocked**, and L-20 stays
unwired. Two arms it fixed outright (0.5B −18.6% → +0.7%, gemma 0.0% → +0.3%) are a hint that
the form is not nonsense, but a hint is not a law and this one was staked to a bar it missed.

**The productive reading:** what is missing is not more theory but the *real* per-op measurement
— the instrumented build's profiler on an actual decode, which would give the shape/format
weights from inside llama.cpp instead of from a synthetic proxy. That is the next experiment, and
it is a measurement rather than another fit.
