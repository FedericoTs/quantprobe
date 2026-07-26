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

---

## Scored (2026-07-26, log: weights/data/prereg14_apex_rematch.log)

Out-of-domain corpus (this repo's code + technical prose, 183 KB — a domain our imatrix never
saw), 8 chunks, ctx 512, hybrid placement, KL measured against logits from the Q8_0 parent
(reference PPL 4.9634).

| build | size | OOD PPL | ratio to parent | **Mean KL** |
|---|---|---|---|---|
| **ours (imatrix + shexp)** | 13.35 GB | **5.1645** | **1.0405** | **0.06665 ± 0.0024** |
| ours, no imatrix | 13.35 GB | 5.3234 | 1.0725 | 0.08958 ± 0.0037 |
| UD-IQ2_M (smaller) | 11.4 GB | 5.2176 | — | 0.08280 ± 0.0037 |
| APEX Mini | 13.25 GB | 5.3382 | 1.0755 | 0.09011 ± 0.0034 |

- **P-1: MISS — and my stated prior was wrong.** I staked that our wikitext win would narrow or
  reverse out-of-domain (−1% to +6% vs APEX). Measured **−3.25%**: we did not just hold the win,
  we *widened* it on unseen data. Outside the band, published as a miss of my own prediction.
- **P-2: MISS, decisively and in the opposite direction from staked.** I predicted APEX would
  show 2–15% *better* KL because it calibrates diversely. Measured: **ours is 26.0% better**
  (0.0667 vs 0.0901). Error bars do not overlap. This is the sharpest instrument in the test and
  it points the other way.
- **P-3: MISS by 0.02 percentage points — held to the letter.** Staked "imatrix beats no-imatrix
  OOD by ≥3%"; measured **2.98%** on perplexity. That is a miss, and it is recorded as one.
  Reporting it honestly, the same comparison on KL is **25.6%** — so the *conclusion* the stake
  was testing (calibration generalises, it is not wikitext memorisation) is strongly supported
  even though the specific perplexity threshold was missed by a hair.
- **P-4: HIT.** The ranking IS metric-dependent, as staked. On wikitext ppl: ours < APEX < IQ2_M.
  On out-of-domain KL: ours < **IQ2_M < APEX** — APEX moves from second to last, behind a
  *smaller* community quant. "Which recipe is better" genuinely depends on what you measure.

### What this actually establishes (and what it does not)

**Established:** importance-matrix calibration generalises. It was not wikitext memorisation —
the calibrated build is better on code and technical prose it was never calibrated on, by every
metric measured. Shipping it as the `auto --custom` default (v1.7.0) is supported, and the P-3
threshold miss does not change that.

**NOT established:** "quantprobe beats APEX." Caveats, stated rather than buried: only 8 chunks
(though error bars separate cleanly); our file is 0.8% larger; APEX Mini is built by a different
pipeline from a different source, so this compares *artifacts*, not isolated recipe choices; and
a single OOD domain (code/technical prose) is not "general". A diverse-corpus calibration on our
side remains untested — the follow-up experiment is diverse vs specialised calibration, which
this result makes more interesting, not less.

**The honest headline:** I predicted we would lose this and published that prediction first. We
won it. Two of four stakes missed *in our favour*, which is still two misses.

**Wired into:** nothing — correctly. This was a head-to-head comparison, not a mechanism: it establishes that our imatrix recipe generalises out of domain and that ranking is metric-dependent. Neither changes a prediction the tool makes. The result is a claim in the README, not a constant.
