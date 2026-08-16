# Pre-registration #102: does effective rank predict the fragile band? (U-46, the fork decider)

**Author:** Federico Sciuca · **Date staked:** 2026-08-16, **before a single singular value has
been computed.** **STAKED.**

## Why this is the decision experiment

The depth-aware recipe's cost is the probe: a quantize-and-perplexity loop that took 14 minutes on
a 4B and 7.3 hours on a 27B, per model. Law 3's standing claim is that fragility is *measurable,
not predictable* — architecture family fails (Mistral vs Qwen, near-twins, opposite ends) and
weight kurtosis points the wrong way (Gemma). But our own rank-robustness work suspects the state
variable of quantization is **effective rank**, a property computable from weights in seconds.

If rank predicts the band, the probe becomes optional: **instant depth-aware recipes for any model
on HuggingFace, day-one, automatable** — the capability that would change what this tool is. If it
does not, Law 3's negative is strengthened (now against rank too), and that is the honest evidence
for the maintain-vs-new-project fork. Either way the fork decision comes from data.

**We hold something that did not exist when Law 3 was written: five measured fragile bands as
ground truth,** all committed with their probe logs:

| model | layers | measured fragile band | effect size |
|---|---|---|---|
| Mistral-7B | 32 | **front** (first 8) | 27x median |
| Qwen2.5-7B | 28 | back | (recipes/) |
| Qwen3.5-4B | 32 | back 24–31 | 1.7x median (0.18/0.37/0.63/1.07) |
| Qwen3.5-35B-A3B | 40 | back 34–39 | (apex stageA) |
| Qwen3.8-27B (hybrid) | 64 | back 51–64 | 2.04x median (0.045/0.212/0.292/0.593) |

Mistral is the load-bearing row: any signal that only ever says "the back" fits 4 of 5 by luck.
A real predictor must turn around on Mistral.

## The signal, frozen now

**Per-layer weight effective rank via participation ratio of squared singular values,** computed
from each layer's FFN matrices (`ffn_down`, `ffn_gate`, `ffn_up` — the tensors the probe pushes to
2-bit), from the highest-precision file we hold for each model:

    erank(W) = (sum_i s_i^2)^2 / sum_i s_i^4          (participation ratio; s_i = singular values)
    layer score = mean of erank over the layer's FFN matrices, each normalized by min(rows, cols)
    band score  = mean layer score over the probe's own 4 bands (identical band boundaries)

Choices locked before computing, so none can be tuned to the answer:
- **Participation ratio**, not entropy-rank or a threshold rank — one formula, no cutoff parameter.
- **Normalization by min(rows, cols)** so bands of different-shaped matrices compare.
- **FFN matrices only** — they are what the probe quantizes; attention/SSM protection is a separate
  recipe decision.
- **MoE (35B):** routed-expert FFN tensors are included and averaged per layer like any other —
  they are what the probe's band regex pushed to 2-bit.
- The source files are the ones on disk today (BF16 where held, else the highest-precision quant);
  the file used per model is recorded in the score log. A quantized source perturbs singular values
  but the signal must survive it to be useful in practice — the tool's users hold quants, not BF16.

**Staked direction: LOWER effective rank → MORE fragile.** Rationale recorded before computing: a
low-rank layer concentrates its function in few directions, so quantization noise of fixed size
destroys a larger fraction of what the layer does; a high-rank layer spreads function across many
directions and degrades gracefully. If the truth is the opposite sign, that is scoreable and
useful — but it is P-B, not a confirmed P-A.

## Staked outcomes

Scored on **Spearman correlation between band score and the probe's measured band delta** (4 bands
per model, 5 models), by pre-written code committed before the first SVD:

- **P-A — RANK PREDICTS (the breakthrough):** correlation in the staked direction (negative: low
  rank ↔ high delta) with |rho| ≥ 0.8 in ≥ 4 of 5 models **including Mistral-7B** (the
  front-fragile control must turn around). Consequence: a `probe --fast` mode is built (rank-only
  band pick), validated against the measured bands, and the probe becomes its confirmation tool.
- **P-B — PREDICTIVE, WRONG SIGN:** same threshold, opposite sign, consistently. Still a
  breakthrough (a predictor is a predictor); reported as the staked direction being wrong.
- **P-C — RANK FAILS:** anything else — weak, inconsistent, or Mistral refuses to turn around.
  Law 3's "measurable, not predictable" is strengthened (rank joins kurtosis and family on the
  refuted-predictor list), the probe remains the only honest instrument, and the fork decision
  input is: no cheap breakthrough here.

Between-band outcomes are reported as between bands (the #99 rule). n=4 points per model is tiny
and is why the bar is "≥4 of 5 models consistent", not a p-value ritual.

## Kill rules

- **KR-1 FROZEN FORMULA:** the signal above is computed exactly once per model. Any variant
  (activation-based rank, different normalization, attention tensors) is a NEW prereg, not a retry.
- **KR-2 GROUND TRUTH IS READ-ONLY:** the five bands come from the committed probe logs; no probe
  is re-run to make a row agree.
- **KR-3 SCORED BY CODE:** prereg102_score.py committed before the first singular value, printing
  per-model rho, direction, and the P-A/P-B/P-C verdict mechanically.
- **KR-4 MISTRAL IS NOT OPTIONAL:** if the Mistral weights cannot be obtained, the experiment
  waits; scoring 4 back-fragile models alone would be confirmation theater.

## What this cannot show

Why rank would or would not track fragility (mechanism), whether an activation-based rank does
better (named follow-up, separate stake), or anything about attention/SSM-path fragility — the
probe's FFN scope bounds the claim. And a P-A here validates a *screen*, not a replacement: the
staked consequence is probe-becomes-confirmation, not probe-deleted.


---

## AMENDMENT (2026-08-16, BEFORE any singular value was computed - rank_bands.json does not exist)

Four changes, all declared while the measurement is still unrun. The adversarial verify pass over
the (uncommitted) harness and scorer forced three of them; the fourth adds test power.

**1. A SIXTH ground-truth model is declared: qwen3-30b (plain Qwen3-30B-A3B).** Its full 4-band
curve has been committed since 2026-06 in `quantprobe/recipes/qwen3-30b.json` (deltas 0.5907 /
0.0165 / 0.1352 / 1.3374, bands (0,11)(12,23)(24,35)(36,47), back-fragile 3.68x) - same 4-band
probe method as every other row. The bar moves **">= 4 of 5" -> ">= 5 of 6", Mistral still
mandatory**. Declared now because nothing has been computed; adding a row after the first SVD
would be scope-shopping.

**2. The verify pass caught a cross-model join before it corrupted the row.** The harness's first
draft pointed qwen3-30b at the CODER finetune's file (the only 30B on disk by that name-shape),
but the recipe's ground truth was probed on the PLAIN Qwen3-30B-A3B - the repo's own ladders list
the two as separate models. Rank of one model against band deltas of another is wrong numbers,
and under the 5-of-6 bar a single corrupted row can flip the verdict. Repointed at
`Qwen3-30B-A3B-Q2_K.gguf`, the plain variant held.

**3. Source files fixed to honour the frozen "highest-precision held" clause, and recorded:**

| model | source file | note |
|---|---|---|
| mistral-7b | Mistral-7B-Instruct-v0.3-Q4_K_M.gguf | highest held (downloaded 2026-08-16, KR-4) |
| qwen2.5-7b | qwen2.5-7b-instruct-fp16 (4-part split) | was Q4_K_M in the draft; fp16 held since 08-04 |
| qwen3.5-4b | Qwen3.5-4B-BF16.gguf | true original |
| qwen3.5-35b | Qwen3.5-35B-A3B-Q8_0.gguf | highest held |
| qwen3-30b | Qwen3-30B-A3B-Q2_K.gguf | **highest held for the PLAIN model is 2-bit** - the roughest source in the set; if this row alone misses, the source is the named suspect, and re-sourcing a higher-precision file would be a new prereg, not a retry |
| qwen3.8-27b | Qwen3.8-27B-Q4_K_M.gguf | highest held (BF16 deleted for disk, prereg #101) |

**4. Two errors in this document's own ground-truth table, corrected against the sources:** the
35B fragile band is **30-39** (recipe `fragile_band: [30, 39]`), not the "34-39" first written
(that number came from a later resize build's log line, not the probe); and qwen3.8-27b has **65
blocks** (the probe quantized blk.64, the MTP block, as part of band 51-64), not "64 layers".
Neither changes any staked threshold; both are quote corrections.

The harness/scorer contract was also aligned in the same pre-measurement commit (the draft
harness emitted a row shape the scorer rejects - KR-3 would have been voided by a
post-measurement edit; instead the fix lands before the first SVD, which is the only legal
moment). KR-1/KR-2/KR-3/KR-4 all still hold.
