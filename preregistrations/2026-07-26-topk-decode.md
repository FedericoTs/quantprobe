# Pre-registration #18: does top-k reduction speed up DECODE, not just prefill?

**Author:** Federico Sciuca · **Date staked:** 2026-07-26, BEFORE the measurement. **Status: STAKED.**

## Why this is open

Law 5's H6 measured `--override-kv qwen3moe.expert_used_count=int:4` on **prefill** only: CPU-pure
pp2048 rose to 47.2 tok/s, and the quality cost was measured at **+20.7% WikiText perplexity**
(8.31 → 10.03). The design it bounded was *asymmetric* top-k — k=4 for prefill, k=8 for decode —
so **decode at k=4 was never measured at all.** Every raw log in this repo shows
`expert_used_count = 8`.

That is a gap worth closing, because on the byte model it is the largest unshipped decode lever
there is. Qwen3-30B-A3B activates 3.3B parameters: 1.2B always-active plus 2.1B of routed
experts. Halving k halves only the routed half, cutting active parameters to 2.25B — a 32%
reduction, which on a bandwidth-bound tier should be worth about **1.47×**.

Measured today at k=8 on the split placement: **18.89 tok/s**. The byte model therefore projects
**27.7** at k=4.

## Stakes

Measured with `llama-cli` (llama-bench has no `--override-kv`, which is why H6 could not do this),
same file, same `-ot` split, same box, 128 generated tokens.

- **P-1 (magnitude).** k=4 decode lands in **24–30 tok/s**, i.e. within ±10% of the byte model's
  27.7. Landing below 24 means decode does not respond to expert-count the way bytes predict.
- **P-2 (the stake).** k=4 beats k=8 by **≥1.25×** on the same placement.
- **P-3 (harness control).** k=8 measured through `llama-cli` reproduces the llama-bench figure
  within ±10% (18.89 ± 1.9). If the harness disagrees with itself, P-1 and P-2 are void.

## Refuted if

P-2 fails — expert-count reduction does not transfer from prefill to decode, and the asymmetric
design is prefill-only as originally scoped.

## What ships if it holds

Nothing automatically. **This is a quality trade, not a free win**: +20.7% perplexity is far
outside what this project ships silently. If it holds it becomes an *offered* lever with the cost
stated next to it, in the same way pruning is offered but never ranked first without
`--allow-prune`. A 1.47× decode gain that costs a fifth of the model's quality is a choice the
user makes, not one the tool makes for them.
