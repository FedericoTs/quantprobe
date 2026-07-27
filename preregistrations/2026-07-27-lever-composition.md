# Pre-registration #35: do the shipped levers COMPOSE?

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## Why this is the last unmeasured thing on the recommended config

Three levers ship, each measured ALONE: ngram speculation (2.4-2.5x on copy-regime output, #28),
q8_0 KV (+37% decode at 16k depth, #25), and the split placement itself (#25). Nobody has run
them TOGETHER, and there is a real interference hypothesis: speculation verifies k tokens in ONE
forward pass, so a verify batch reads the KV cache differently than single-token decode. If the
q8_0 dequant cost is per-access rather than per-byte, the two levers could fight.

The barrier finding (#34) is now measured NOT to reach this config (KMP_BLOCKTIME A/B on the
split: 20.12 vs 20.56, bars overlap - libomp already spins), so composition is the remaining room.

## Arms (llama-server, split placement, ub 1024, edit task = copy regime, temp 0, r=2)

| arm | KV | spec |
|---|---|---|
| B | f16 | none |
| K | q8_0 | none |
| S | f16 | ngram-simple |
| KS | q8_0 | ngram-simple |

## Stakes

- **P-1 (speculation reproduces).** S >= 2.0x B. Below that the session is contaminated.
- **P-2 (KV quant is ~neutral at this depth).** K within +/-8% of B — the edit prompt is ~1k
  tokens, far from the 16k where #25 measured +37%.
- **P-3 (THE COMPOSITION TEST).** KS >= 0.90 x (S x K / B) — i.e. the levers compose to within
  10% of their product, no destructive interference.
- **P-4 (identity holds under the stack).** KS output is byte-identical to K's. Speculation is
  output-preserving by construction; q8_0 KV changes the cache, so K vs B may differ - but adding
  speculation on top of q8_0 must not change anything.

## Refuted if

**P-3 misses low.** Then the levers interfere, the tool must stop implying they add up, and the
recommendation becomes "pick one" with the measured pair table.

## What ships

The measured stack number in the plan output — what a user can actually get today on this box,
end to end, rather than three separate percentages they are left to multiply themselves.
