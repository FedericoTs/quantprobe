# Pre-registration #15: does Law 4 need a fixed per-token overhead term?

**Author:** Federico Sciuca · **Date staked:** 2026-07-26, BEFORE the discriminating measurement.
**Status: STAKED.**

## How this surfaced

A scenario matrix sweeping the tool's whole decision surface flagged an implausible cell (a 30B
MoE at 375 tok/s on a 24 GB card, where community reports run 100–160). Checking the all-in-VRAM
path against measurement found the law systematically **pessimistic**, not optimistic:

| model, all in VRAM | law predicts | measured (llama-bench, r=3) |
|---|---|---|
| qwen7b Q4_K_M | 12.5 | **19.98 ± 0.04** |
| Qwen3.5-4B Q4_K_M | 20.6 | **27.30 ± 0.08** |

Back-solving η from each gives **0.559** and **0.461** — not one constant. A single efficiency
cannot fit both points, which means the functional form is wrong, not just the constant.

## The proposed refinement

Law 4 currently prices decode as pure bandwidth: `t_token = active_bytes / (η · BW)`. The two
points above fit exactly if there is also a **fixed cost per token** that does not scale with
bytes — kernel launches, sampling, synchronisation, Python-side dispatch:

> **t_token = overhead + active_bytes / (η_bw · BW)**

Fitted on those two points: **η_bw · BW ≈ 159 GB/s (83% of this card's 192 GB/s spec)** and
**overhead ≈ 16.3 ms/token**. Note what this implies: the *bandwidth* side is far more efficient
than η = 0.35 suggested; the apparent inefficiency was a fixed cost being smeared into a
multiplicative constant.

This would also explain a community datapoint that has been sitting unresolved: a user measured
**300 tok/s** on a 5070 where pure bandwidth predicted 900+. A fixed overhead dominates exactly
when the model is small and fast — which is that case.

## The discriminating stake

`Qwen3-0.6B-Q8_0` (0.73 GB/token), all in VRAM on the reference box. The three models disagree
sharply, so one measurement settles it:

| model | prediction |
|---|---|
| **overhead model (staked)** | **47.8 tok/s** |
| pure bandwidth at η_bw 0.83 | 217.4 tok/s |
| current law at η 0.35 | 91.7 tok/s |

- **P-1 (the stake).** Measured lands in **40–58 tok/s**, i.e. within ±20% of the overhead
  model's 47.8 and nowhere near the alternatives. A result above ~90 refutes the fixed-overhead
  form outright and means something else explains the two-point discrepancy.
- **P-2 (no anchor moves).** Adopting the refined form must leave every published anchor
  retrodicted: the 30B hybrid 19.3, the 110B disk-stream 0.19, the Laguna 0.38, and the corrected
  18.35 baseline all stay inside their existing tolerances. If the refinement breaks an anchor,
  it is wrong or incomplete and does not ship.
- **P-3 (it explains the outlier).** Applied to a 5070-class card, the refined form predicts the
  community-reported ~300 tok/s within ±40% where the current law predicts 900+.

## Refuted if

P-1 outside 40–58. Misses publish with equal prominence — including that this whole line came
from my own tool being 30% wrong about its most common case.

## Why this matters

"All in VRAM" is the single most common configuration for anyone with adequate VRAM. If the law
is 30–37% pessimistic there, quantprobe has been telling those users a model is less usable than
it is. That is the opposite of the usual failure direction and just as wrong.

---

## Scored (2026-07-26, log: weights/data/geta_vram_recalibration.log)

**P-1: MISS, decisively. The fixed-overhead model is refuted.**

Staked 40–58 tok/s for `Qwen3-0.6B-Q8_0` all-in-VRAM. Measured **93.12 ± 4.41**. The overhead
model predicted a *small* model would be *slow* (fixed cost dominating a tiny byte budget); it is
in fact fast. Not a near-miss — the mechanism is wrong.

Worse for my reasoning: **the existing law predicted 91.7 and was right within 1.5%.** I had
concluded from two points that the law's functional form was broken. The third point says the
form is fine at the small end and the *constant* drifts at the large end.

| model, all in VRAM | GB/token | measured | law says | error | implied η |
|---|---|---|---|---|---|
| Qwen3-0.6B Q8_0 | 0.73 | 93.12 ± 4.41 | 91.7 | **−2%** | 0.354 |
| Qwen3.5-4B Q4_K_M | 3.24 | 27.30 ± 0.08 | 20.6 | −25% | 0.461 |
| qwen7b Q4_K_M | 5.38 | 19.98 ± 0.04 | 12.5 | −37% | 0.560 |

**What the data actually shows:** measured efficiency **rises monotonically with bytes per
token** — 0.354 → 0.461 → 0.560. The law's η = 0.35 is exactly right for the smallest model and
increasingly pessimistic as models grow. Larger tensors evidently use this GPU's memory system
better (coalescing, fewer launches per byte of work), but **three points do not identify a
mechanism** and I am not going to fit a curve to them and call it a law. That is precisely the
error I just made with the overhead model.

**P-2 and P-3 are not evaluated.** Both were conditional on adopting the refined form, and there
is no refined form to adopt. Recording them as unscoreable rather than quietly dropping them.

### What ships, and what does not

**Nothing ships.** A three-point trend on one GPU is a lead, not a calibration. Changing η would
move published anchors on the strength of a pattern I cannot yet explain — and the last time I
changed a constant on thin evidence (the +34.7% baseline) it cost a public correction.

**What is now known and worth stating plainly:** for models of ~4B and up sitting entirely in
VRAM, quantprobe under-predicts by 25–37% on this hardware. That is the most common configuration
for anyone with adequate VRAM, and it is the opposite of the usual failure direction — we are
telling those users a model is *less* usable than it is. The honest interim move is to say so in
the limitations rather than silently ship a number I know to be low.

### Next measurement, specified now

The pattern needs more points before any constant moves, across the axis that actually varies:
bytes per token, at fixed architecture and format, on one GPU. Candidates already on disk span
0.73 → 5.38 GB/token; adding 2–3 intermediate points would show whether η saturates, and where.
Only then is a size-dependent η (or a better-motivated form) worth staking.

**Method note for the record:** this entire line came from a scenario matrix that swept the
tool's own decision surface and flagged an implausible cell. The sweep found a real 37% error in
the most common configuration — and my first explanation for it was wrong. Both facts belong in
the record.

**Wired into:** nothing — correctly. The fixed-overhead form was refuted (staked 40-58 tok/s, measured 93.12) and the existing law was right within 1.5%. The residual all-in-VRAM pessimism it surfaced is not fitted but ratcheted, in `tests/smoke.py:VRAM_GAPS`, so it can shrink and never grow.
