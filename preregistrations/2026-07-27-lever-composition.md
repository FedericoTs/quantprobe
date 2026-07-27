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

---

## Scored (2026-07-27, log: `weights/data/prereg35_stack.log`)

**Verdict: P-1 HIT, P-2 HIT, P-3 HIT at 1.038 (no interference — slightly super-multiplicative),
P-4 HIT. The levers compose.**

| arm | KV | spec | tok/s | vs B | output sha |
|---|---|---|---|---|---|
| B | f16 | — | 21.64 | 1.000× | `28a5c1e1c014` |
| K | q8_0 | — | 20.38 | 0.942× | `65985276565a` |
| S | f16 | ngram | **50.55** (89% acc.) | **2.336×** | `28a5c1e1c014` |
| KS | q8_0 | ngram | **49.42** (89% acc.) | **2.284×** | `65985276565a` |

- **P-1 (S ≥ 2.0×): HIT.** 2.336× — speculation reproduces cleanly on a fourth independent run.
- **P-2 (K within ±8%): HIT.** −5.8%. At ~1k context q8_0 KV is a small net *cost*, exactly as
  expected: the dequant work is paid every token while the byte saving is negligible until depth.
  #25's +37% is a 16k-depth result and must never be quoted at short context.
- **P-3 (KS ≥ 0.90 × product): HIT at 1.038.** Predicted 47.61 tok/s from the product, measured
  49.42. No destructive interference — the verify batch's multi-token KV access does not
  aggravate q8_0 dequant. If anything the batch amortises it slightly.
- **P-4 (identity under the stack): HIT.** KS and K produce byte-identical output (`65985276565a`),
  and S is byte-identical to B (`28a5c1e1c014`). Speculation preserves output exactly, at every KV
  precision; the only sha change comes from q8_0 KV, which genuinely alters the cache.

### The practical consequence

The levers are independent multipliers, so the right recommendation is depth-conditional rather
than "enable everything":

| workload | recommended stack | measured |
|---|---|---|
| copy-regime output, short/medium context | split + `ngram-simple`, **f16 KV** | **50.55 tok/s** |
| copy-regime at long context (≥8–16k) | split + `ngram-simple` + `q8_0` KV | 49.42 here, and q8_0's +37% at 16k applies on top |
| novel generation, any context | split alone | 21.6 (the wall, #30) |

Adding q8_0 KV at short context costs ~6%. That is a small, real, and previously unstated
trap: our own #25 headline (+37%) is a *depth* result, and a user who enables it for a chat-length
workload pays for it.

**Wired into:** `findings/REGISTER.json:V-08` (depth condition made explicit) · `V-04` ·
the plan output's stack advice.
