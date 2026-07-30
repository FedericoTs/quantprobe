# Where the v1.24 ladder's residuals come from — a decomposition, not a fit

**Date:** 2026-07-30 · **Data:** `weights/data/full_ladder_v124.json` (14 arms, predictions staked
in git before measurement) · **Status:** analysis; nothing shipped from it yet.

## The pattern is not size — it is format × placement

| class | n | mean error | arms |
|---|---|---|---|
| MoE split, IQ/APEX | 3 | **+19.7%** | DS-IQ2 +18, APEX-Mini −9, 3.6-APEX-MTP +50 |
| MoE split, K-quant | 4 | **−9.1%** | DS-Q4KM −22, 30B −11, Coder −12, 3.6-Q2KXL +8 |
| dense all-in-VRAM | 6 | +2.9% | −9 … +17, no trend |
| dense split | 1 | +0.6% | 14B (the L-19 term) |

Two opposite biases, each with a candidate mechanism that is *physically* identifiable rather
than fitted.

## Mechanism A — we charge for bytes the machine never reads

`spec.from_gguf` sets `ne = total − routed`, so **`token_embd` lands in the always-active set and
is priced at ≥4.5 bits every token**. It is a *gather*: batch-1 decode reads ONE row of a
150k-row matrix, i.e. ~zero bytes. When embeddings are **tied**, the same tensor is the output
projection and IS fully read — counting it once is correct. When they are **untied** (a separate
`output.weight`), we count it twice and over-charge by the whole embedding.

Measured share of active bytes, untied models: **4.2% – 17.2%**. Applying the correction:

| arm | now | embedding removed |
|---|---|---|
| Qwen3-Coder-30B | −11.6% | **−1.9%** |
| Qwen3.5-35B APEX-Mini | −9.0% | **+2.0%** |
| Qwen3-30B-A3B | −10.9% | **−5.5%** |
| Qwen2.5-7B Q4_K_M | −4.9% | **+0.9%** |
| Qwen2.5-14B split | +0.6% | +5.1% |
| DS-Lite 16B Q4_K_M | −22.4% | −17.6% |

It fixes the under-predicting family almost exactly and leaves the over-predictors worse — which
is the signature of a *correct* change revealing a second, opposite mechanism underneath.
**Note for discipline:** this must be judged on whether it is TRUE, not on whether it improves the
14-point aggregate. It is true; ship it behind a prereg, then read the new residuals.

## Mechanism B — "withhold rather than guess" backfires into an optimistic guess

Two nearly identical files, same architecture, same 3.52B active, opposite errors:

| file | format mix | fmt_bw | error |
|---|---|---|---|
| Qwen3.6-35B UD-Q2_K_XL | IQ2_XS 48% / IQ3_XXS 32% / Q5_K 9% | 65.1 (priced) | **+8.5%** |
| Qwen3.6-35B APEX-MTP-Nano | **IQ2_XXS 37% / Q3_K 34% / IQ2_S 22%** | **None** | **+49.5%** |

`IQ2_XXS` and `IQ2_S` are not in `FORMAT_EBW`. Known-format coverage falls under the 60% rule, so
we withhold `fmt_bw` — and the fallback is a *generic* eta that is far too optimistic for a file
that is 59% codebook. #70 already measured that codebook formats run 36–52% below K-quants per
byte; the withheld path silently assumes they don't. **Withholding a number is only honest if the
fallback is conservative; ours is optimistic.**

## Mechanism C — the residual after A and B

`DS-Lite 16B Q4_K_M` stays at −17.6% after A, the largest remaining under-prediction, and it is
the only **MLA** architecture in the set (kv_lora attention). Its IQ sibling over-predicts. Both
point at the DS-Lite active-byte accounting rather than the tiers. Unassigned; do not guess.

## The path to ±5%, in order

1. **Prereg A (embeddings):** subtract `token_embd` from always-active bytes for untied models
   only. Holdout: the tied models must not move at all (they have no double count) — a free
   falsification test built into the same run.
2. **Prereg B (codebook ladder):** measure IQ2_XXS, IQ2_S, IQ1_M against the #70 control, add the
   entries, and change the withheld-coverage fallback from optimistic-generic to
   the-worst-known-codebook. Holdout: the two Qwen3.6 siblings must converge.
3. **Then re-run the full 14** and read what is left. Only after A and B are scored does chasing
   Mechanism C make sense — with two known biases removed, the remaining residual is interpretable.

**What will not reach ±5% and should stop being expected to:** the all-in-VRAM dense arms
(−9…+17 with no trend) are the L-18 population-spread regime. Their spread is irreducible from
constants; only `calibrate`'s anchors collapse it, which is exactly what the anchored path does
for the machine in front of the user.
