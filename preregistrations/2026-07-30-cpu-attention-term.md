# Pre-registration #74: the CPU-attention term — out-of-sample, on a model it was not fitted to

**Author:** Federico Sciuca · **Date staked:** 2026-07-30, prediction computed BEFORE the run. **STAKED.**

## The claim under test

#73 measured a clean unexplained cost on dense splits at depth: **1.43 µs per position per CPU
layer per token** at d16384 and **1.42** at d32768 (two depths, 1% apart) — CPU-side attention
over the whole KV cache, which our bandwidth-shaped KV term does not price. That constant came
FROM the two arms that failed, so it cannot ship on its own evidence. This prereg tests it where
it was not fitted: a **different model, a different depth, a different CPU-layer count**.

## The staked number

Qwen2.5-14B-Instruct Q4_K_M, `plan --ctx 8192` emits `-ngl 18` of 48 layers → **30 CPU layers**
(the 7B arms had 8 and 11). Tool today predicts **3.30 tok/s**. The term adds
`30 × 8192 × 1.43e-6 = 351 ms/token` on top of its 303 ms:

> **CORRECTED PREDICTION: 1.53 tok/s** (band ±25%: **1.15 – 1.91**)

Note this is the term making the prediction MUCH WORSE (3.30 → 1.53) — it can only be validated
by the machine actually being that slow. A term that only ever excuses misses is not a term.

## Stakes

- **P-1 (THE TERM).** Measured tg at d8192 on the emitted config lands inside **1.15–1.91**.
- **P-2 (the term beats no term).** The corrected prediction's absolute error is smaller than
  today's uncorrected 3.30 — i.e. measured < 2.4 tok/s (the midpoint where both are equally
  wrong). If the machine runs faster than 2.4, the term is over-correcting and is worse than
  nothing on this arm.
- **P-3 (scaling, not coincidence).** The implied constant from THIS arm
  (`(1/measured − 1/3.30) / (30 × 8192)`) lands within **±30%** of 1.43 µs. This is the real
  test: the same constant across 7B/14B, 8/11/30 CPU layers, and 8k/16k/32k depths.

## KILL RULE

**If P-1 or P-3 fails, the term does NOT ship** and U-25 is scored refuted-as-formulated: the
dense-split-at-depth regime keeps the scope warning shipped in v1.23 (an honest "we don't know")
instead of a fitted number. A second failure mode worth naming in advance: if measured is far
BELOW 1.15, the cost is super-linear in CPU layers (cache-pressure effects), which is a different
law and gets its own prereg rather than a fudge factor.

**Wired into:** pending; `plan.evaluate` dense-split branch + `depth_scope_warning` (which the
term would REPLACE for this regime if it survives).

---

## SCORED — 2026-07-30, same session

Raw log: `weights/data/prereg74_cpu_attention.log`. Measured **1.36 ± 0.04 tok/s** (tg64 @ d8192,
`-ngl 18`, clocks healthy).

- **P-1 HIT.** Staked 1.53, band 1.15–1.91, measured **1.36** (+12.5% over). The term predicted a
  machine 2.4× slower than the shipped tool said, and the machine was.
- **P-2 HIT, overwhelmingly.** Corrected error 0.17 tok/s vs uncorrected 1.94 tok/s — the term
  cuts the error by **11×** on this arm.
- **P-3 HIT.** Implied constant **1.76 µs/pos/layer** vs the fitted 1.43 (+23%, inside ±30%).
  Across every arm now measured: 7B/8 CPU layers/d16k **1.43** · 7B/11/d32k **1.42** ·
  14B/30/d8k **1.76**. One constant, two models, 8→30 CPU layers, 8k→32k depth.

**Shipped:** `CPU_ATTN_S_PER_POS_LAYER = 1.55e-6` (the mean of the three measured arms, stated as
a mean rather than the value that flatters any single one), applied to dense-split arms only,
replacing the v1.23 scope warning for that regime with a number the evidence supports. The
warning's honest-uncertainty text is retained in shortened form because a three-point constant on
one CPU is still a three-point constant on one CPU. U-25 closes; **L-19 gains its term**.
