# LLM-in-the-Loop Quantization Codec Discovery — Honest Results

**Date:** 2026-06-06 · **Model:** Qwen2.5-0.5B · **Hardware:** CPU laptop (Intel Core 7 150U, 16 GB, no GPU)
**Mutation operator:** Claude (Opus) writing new codec code each round · **Verifier:** held-out perplexity + bits/weight

## What this is

An AlphaEvolve-style loop where an LLM (Claude) *writes new quantization-codec code* each round,
an automated arena scores it against a cheap held-out verifier (perplexity + storage bits), and the
best survive. Goal: test whether code-level mutation can discover codecs beyond hand-design — and how far.

- `weights/quant_arena.py` — evaluator + leaderboard, **crash-resilient + resumable** (per-codec JSON checkpoint).
- `weights/codec_zoo.py` — the registry of codec functions (the evolvable artifacts).
- `weights/quant_lab.py` — calibration (activation hooks) + model load + perplexity.

## The honest result — scalar entropy-constrained quantization (ECVQ) is the champion

Dominant bits-vs-quality frontier (fp16 held-out ppl = 3.943):

| storage bits | perplexity | codec |
|---|---|---|
| 3.13 | 4.483 | ECVQ (entropy-constrained) |
| 3.52 | 4.287 | ECVQ |
| 3.91 | 4.169 | ECVQ |
| 4.24 | 4.109 | entropy-coded 16-level |
| 5.27 | **3.971** | entropy-coded 32-level (near-lossless, beats int8's 8 bits) |

**Headline:** the loop improved the hand-designed champion from **4.934 → 4.483 ppl at ~3.1 bits**
(a ~45% cut in the gap to fp16), and reached **near-lossless at ~5.3 bits**. Real, but modest.

## The discovery arc (12 rounds) — wins, failures, and a caught bug

| round | strategy (LLM-written) | outcome |
|---|---|---|
| 0 | hand champ: rotation + activation-aware + NF + outlier | 4.934 @ 3.28b (recombination baseline) |
| 1 | additive 2-bit + 1-bit residual | ❌ broke (1-bit residual has no zero → amplifies noise). Diagnosed. |
| 2 | learned Lloyd-Max levels | ✅ 4.636 @ 3.30b — first to beat the wall |
| 3 | sensitivity mixed-precision | ❌ 2-bit floor too damaging |
| 4 | entropy-coded indices | revealed: distortion-optimal levels flatten the index distribution → little to entropy-code |
| 5 | **ECVQ** (entropy-constrained levels) | ✅ **4.483 @ 3.125b** — the champion; jointly optimal for distortion+rate |
| 6 | additive ECVQ; sigma-delta | sigma-delta ❌ (carry blow-up) |
| 7 | cross-layer adaptive-λ; sigma-delta leaky | both ❌ (crude sensitivity; leaky carry still unstable) |
| 8 | ECVQ + low-rank residual | ❌ all worse — **insight: rotation already whitens the error → low-rank redundant** |
| 9 | D4 lattice + entropy | partial: won at aggressive bits, lost mid-range |
| 10–11 | E8 lattice + entropy | *appeared* to be a breakthrough (near-fp16 @ 2.5b) |
| 12 | **honest re-count** | ❌ **E8 "breakthrough" was a bug** — see below |

### The bug we caught (autonomously)

Rounds 10–11 reported E8 lattice quantization at **near-fp16 quality (4.02) for ~2.5 bits** — too good
(better than published QuIP# 2-bit). A **suspicious plateau** (q=.08/.10/.14 all reporting 2.475 bits)
gave it away: the bit count used the *joint empirical entropy* of the 8-D lattice points, which
**saturates at log₂(#vectors)** when points are near-unique at fine resolution → it *under-counts*
storage badly. Fixed with **honest per-coordinate entropy** (a valid achievable rate). Re-counted, E8's
bits **more than doubled** (2.48 → 5.36) and **the lattices became dominated by scalar ECVQ** at every
budget. The flashy result evaporated; the honest scalar frontier stands.

## Honest conclusions

1. **The discovery mechanism works** — an LLM in the loop, writing new codec code, *self-corrected*
   (round-1 bug → round-4 analysis → round-5 ECVQ) and produced a genuine, deployable improvement.
2. **We are at the rate-distortion floor** for *per-weight, rotated, entropy-coded* quantization — which
   is precisely *why* nothing (low-rank, lattices, sigma-delta) could beat scalar ECVQ.
3. **ECVQ is novel *for LLM quantization*** (no deployed quantizer uses entropy-constrained design) but is
   a *known classical technique* (Chou-Lookabaugh-Gray 1989) — a novel *application*, not invention ex nihilo.
4. **Honest failures recorded:** additive-1bit, sigma-delta (×2), cross-layer allocation, low-rank residual,
   and the E8 entropy-accounting bug.

## Caveats (do not overstate)

- **0.5B model + perplexity proxy** — not real downstream tasks, not 7B+.
- **Storage win, not compute win** — ECVQ/entropy decode is "high" cost; needs a fast entropy-decode kernel.
- **No comparison to *real* GPTQ/AQLM at scale** — GPTQ's O(in³) Cholesky is infeasible on this CPU.

## What would actually move the needle next (GPU-gated)

The only axis with real remaining headroom is **data-aware quantization** (minimize *output* error, not
weight error — GPTQ/AQLM-style). That needs a GPU. The concrete scale plan: validate ECVQ + entropy on
7B+ vs *real* GPTQ/AQLM on *real* tasks, add proper Hessian-based allocation, and prototype a fast
entropy/lattice decode kernel (the realizability gate).
