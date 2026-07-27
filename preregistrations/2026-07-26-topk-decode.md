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

---

## Scored (2026-07-27, logs: `weights/data/prereg18_topk_decode.log`, `prereg18_topk_quality.log`)

**Verdict: all three speed stakes HIT — and the lever is a DEAD END anyway. It does not ship.**

Harness note: `llama-cli` never loaded the model and `llama-bench` has no `--override-kv`, so
`expert_used_count=4` was baked into a **copy** with `gguf_set_metadata.py` (a `u32→u32` edit, same
byte width). Original verified untouched at `k=8`, copy verified at `k=4`.

### Speed — Qwen3-30B-A3B Q2_K, split placement, tg128, r=3

| | tok/s |
|---|---|
| k=8 (control) | 20.32 ± 0.14 |
| k=4 | **27.13 ± 0.33** |

- **P-1 (24–30 tok/s): HIT.** 27.13. The byte model predicted **27.7** — accurate to 2%. Halving
  the routed experts cuts active parameters 3.3B → 2.25B and decode responds almost exactly as
  bandwidth arithmetic says it should.
- **P-2 (≥1.25×): HIT.** 1.335×.
- **P-3 (control within ±10% of 18.89): HIT.** 20.32.

### Quality — WikiText-2, measured on **this** pair, not cited

| | perplexity |
|---|---|
| k=8 | 9.2364 ± 0.358 |
| k=4 | **11.1411 ± 0.436** |

**+20.6%** — independently reproducing H6's +20.7% on a different file. The figure transfers.

### Why it does not ship: it is strictly dominated

The comparison that matters is not "is 1.335× real" but "is it the cheapest way to buy 1.335×".
It is not. On the same placement, with no metadata surgery at all:

| lever | speed | quality cost |
|---|---|---|
| top-k 8 → 4 | ×1.335 | **×1.206** |
| bits 2.95 → 2.0 | **×1.424** | **×1.048** |

**Quantizing further is faster AND four times cheaper in quality.** Top-k reduction costs more
quality than the entire 2-bit quantization of the model does, and on its own breaches the ×1.12
ceiling `optimize` already enforces. There is no operating point where a user should prefer it.

So the honest outcome of closing this dimension is a **negative result**: expert-count reduction
works exactly as the byte model predicts, and is still the wrong lever to reach for. Recorded as a
measured dead end alongside dynamic top-k, semantic paging and self-speculation — a law you only
ever confirm is a law you have not tested.

**Not shipped. The 11 GB modified copy is deleted; only the measurement remains.**

**Wired into:** nothing — deliberately. The result is a dead end, documented in `LAWS.md` and the
CHANGELOG so nobody re-derives it. `optimize`'s existing bit-ladder already dominates this lever,
so no code change makes the tool better here.
