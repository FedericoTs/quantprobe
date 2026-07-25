# Pre-registration #14: the fair APEX rematch — out-of-domain, KL divergence

**Author:** Federico Sciuca · **Date staked:** 2026-07-26, BEFORE running any of it.
**Scoring: same day.**

## Why this exists

Pre-registration #12 ended with our build beating APEX Mini by 1.07% on wikitext perplexity —
and with an explicit warning, written before measuring, that **the result was in-domain and
should not be read as a general win**: our importance matrix was calibrated on wikitext *train*
and scored on wikitext *test*, while APEX's i-variants deliberately calibrate on a diverse
non-Wikipedia corpus and optimise for real-world accuracy and KL divergence rather than wikitext
ppl. This pre-registration runs the comparison the way APEX's own design would prefer, so the
verdict is not decided by a metric that flatters us.

## Method — the two changes that make it fair

1. **Out-of-domain corpus.** Evaluation on **source code + technical prose** (this repo's Python,
   LAWS.md, README, CHANGELOG — 183 KB), a domain our imatrix never saw. If the #12 win was
   Wikipedia memorisation, it evaporates here.
2. **KL divergence against the full-precision source**, not just perplexity. KL measures how far
   a quantized model's *whole output distribution* drifts from the Q8_0 parent — the metric APEX
   optimises for, and a much sharper instrument than a single perplexity scalar. Reference logits
   are generated once from `Qwen3.5-35B-A3B-Q8_0.gguf`, then every contender is scored against
   the same file.

Contenders, all at matched ~13.3 GB unless noted, identical settings, hybrid placement:

| build | what it is |
|---|---|
| APEX Mini (13.25 GB) | mudler's MoE-aware recipe, diverse-imatrix i-variant |
| ours, imatrix + shexp (13.35 GB) | the v1.7/v1.8 recipe, wikitext-calibrated |
| ours, no imatrix (13.21 GB) | isolates how much of our result is calibration |
| UD-IQ2_M (11.4 GB) | community dynamic quant, smaller — context only |

## Stakes

- **P-1 (the honest prior — our win narrows or reverses).** Our +1.07% wikitext win does NOT
  survive out-of-domain: on OOD perplexity we land **between −1% and +6%** relative to APEX
  Mini. Stated plainly: **I expect to lose this one**, because APEX calibrated for exactly this
  case and we did not. A win here would be a genuine surprise and would publish as such.
- **P-2 (KL divergence — the sharper test).** APEX Mini shows **lower** (better) KL vs the Q8
  parent than our build, by **2–15%**. Same reasoning: diverse calibration should generalise
  better than in-domain calibration.
- **P-3 (calibration still pays, even out-of-domain).** Our imatrix build beats our
  no-imatrix build on OOD perplexity by **≥ 3%**. This is the load-bearing claim for the v1.7
  feature: if calibration only helped on the corpus it was calibrated on, then shipping it as a
  default was wrong and the tool must say so. **This is the stake I most want to survive**, and
  the one whose failure would cost us the most.
- **P-4 (the ranking is metric-dependent, and that is the real finding).** The three-way order
  differs between wikitext ppl (#12) and OOD KL (here) for at least one pair. If instead the
  ranking is identical across both, "which recipe is better" is simpler than this project has
  been claiming, and that publishes too.

## Refuted if

Any band missed. Misses publish with equal prominence — including P-1 and P-2, where a
*better-than-staked* result is still a miss of my stated prior and will be labelled one.

## What changes downstream

If P-3 fails, `auto --custom` stops generating an imatrix by default and the CHANGELOG claim is
corrected. If P-1/P-2 confirm APEX generalises better, the honest recommendation becomes
**diverse calibration**, and quantprobe should ship a diverse corpus rather than defaulting to
wikitext — a concrete product change driven by losing.
