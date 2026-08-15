# Does depth-aware quantization preserve *capability*, or just perplexity?

Every quality claim quantization tools make is perplexity or KL-divergence against the
full-precision model. That is a proxy. This page is the campaign that asked whether the depth-aware
recipe survives contact with the benchmarks people actually quote — MATH-500, GSM8K, IFEval — under
one machine state, temp 0, full sets, scored by code that refuses to publish an undiagnosed metric.

Four pre-registrations, each staked **before** the arms ran. Every number below regenerates from a
committed `results_*.json` and its probe log; the reasoning, including two places we got it wrong
and corrected in public, is in the linked prereg.

## 1. The recipe beats the naive default, on a 35B — and by a lot

`prereg 2026-08-09` (#98). Same Q8_0 parent, same quantizer binary, same box, same day; the only
variable is which layers got which bits. **NAIVE** is `llama-quantize ... Q2_K` (what a normal
person types). **OURS** is the depth-aware recipe (fragile band at `q4_k`, rest `q2_k`, attention
and token-embedding protected).

| Qwen3.5-35B-A3B | NAIVE Q2_K | OURS Q2_K |
|---|---|---|
| MATH-500 | 57.0 | **81.0**  (+24.0) |
| GSM8K | 75.8 | **84.9**  (+9.1) |
| IFEval | 72.5 | **84.7**  (+12.2) |

P1 confirmed at 12x its staked bar. KR-1 stands: OURS carries +2.57% bytes, so read every win as
"the recipe plus 2.5% more bytes."

## 2. The win is *capability*, not answer-format — and we had to prove it against ourselves

`prereg 2026-08-11` (#99). The naive arm emitted a `\boxed{}` on only 64.4% of MATH-500 items vs
86.4% for ours, and the first reading (ours, publicly) was that the +24 was mostly lost *formatting*.
A pre-staked re-grade with lm-eval's own format-blind extractor refuted it: the gap moved **+24.0 →
+23.8**. Of the naive arm's 178 unboxed items, 100 had a numeric gold a last-number rule could have
caught; it caught 2. The responses are long (median 6,787 chars) and untruncated — reasoning that
never converges. **Naive 2-bit does not make the model worse at maths; it makes it stop arriving at
an answer.** The mechanism is the always-active path (SSM / attention) left at 2 bits. See C-30.

## 3. But 2-bit viability is *size-dependent* — the 4B proves it

`prereg 2026-08-11` (#100). The 35B ceiling (BF16 original) is infeasible on a 6GB/16GB box, so the
full three-arm chain ran on **Qwen3.5-4B**, where the true original fits in VRAM.

| Qwen3.5-4B | BF16 (original) | OURS Q2_K | NAIVE Q2_K | Q4_K_M ref |
|---|---|---|---|---|
| MATH-500 | 81.0 | 50.2 | 2.6 | 77.6 |
| GSM8K | 82.9 | 71.0 | 0.4 | 81.7 |
| IFEval | 83.7 | 61.7 | 17.0 | 80.6 |

Naive 2-bit on a 4B is **catastrophic** (2.6 on MATH-500 — 22% of items even produce a box, 11.6%
right given one). The recipe **rescues it enormously** (+47.6). But even rescued, it loses **30.8
points** to the original — and unlike the 35B, on the 4B it loses on *both* format and conditional
reasoning. **A 4B has too few parameters to survive 2-bit even with perfect placement.** The
sensible quant for a small model is Q4, which is near-lossless here (77.6 vs 81.0).

**The law:** aggressive quant + the recipe pays on *large* models; on small ones use Q4 — 2-bit is a
false economy the recipe rescues from catastrophe but not from mediocrity. Federico's staked
expectation was a *modest* loss; the prereg's P-C3 ("the size class binds") is the outcome that
shipped.

## 4. And the fragility law generalizes — it survives hybrid linear attention

`prereg 2026-08-14` (#101), staked the day Qwen3.8-27B's weights landed. It is the first **hybrid**
model we have probed: 48 of 64 layers are linear attention, the 16 full-attention layers spread
evenly across depth. Our fragility probe assumes fragility is localized by *depth*; a hybrid could
have refuted that (if fragility tracked attention *type*, the 4-band depth profile would be flat).

| depth band | Δ perplexity at 2-bit |
|---|---|
| layers 0-16 | 0.045 |
| layers 17-33 | 0.212 |
| layers 34-50 | 0.292 |
| **layers 51-64** | **0.593** (fragile, 2.04x median) |

Monotone, back-heavy — the same shape as every full-attention Qwen. **Depth-localized fragility
survives linear attention; the recipe transfers to hybrids unchanged.** Measured from a Q4 source
(the BF16 27B probe is infeasible here); the declared risk that a quantized source could fake a flat
profile did not fire. ([chart](../media/qwen38_fragility.png))

## What this campaign does NOT yet claim

- **The BF16 ceilings for the 35B and 27B** — weeks per arm on a 16GB box, deferred to rented
  hardware. The 4B is the only full ceiling measured.
- **Placement vs bundle.** The recipe protects the fragile band *and* spends more bits on
  attention/SSM/embedding. Separating "the right layers" from "more bits on the always-active path"
  needs a byte-identical position-swapped arm, not yet built (the method exists — the Gemma-4-12B
  control in the [deep dive](DEEP-DIVE.md)).
- **Speed.** Wall-clock is logged but licenses no tok/s claim — GPU clock drifted across the runs and
  temp-0 scoring is clock-independent by design.

Preregs: `preregistrations/2026-08-09`, `-08-11-format-vs-capability`, `-08-11-4b-ceiling-chain`,
`-08-14-qwen38-27b-hybrid-fragility`. Findings: U-48/U-49/U-50, C-30.
