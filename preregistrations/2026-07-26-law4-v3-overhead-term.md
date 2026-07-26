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
