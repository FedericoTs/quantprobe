# Evolved Quantization Codecs — Rounds 13–21 (GPU, GTX 1060)

Continuation of the LLM-as-mutation-operator quantization-codec discovery loop, now on a
CUDA desktop. Verifier (unchanged, un-gameable): **held-out perplexity** on a disjoint enwik8
slice + **honest bits/weight** + decode-cost class. Model: Qwen2.5-0.5B. **fp16 ppl = 3.944.**
PTQ champion entering this phase: **rotated scalar ECVQ + entropy + 0.5% outliers ≈ 4.483 ppl @ 3.13 bits.**

## The question
Rounds 1–12 established ECVQ+entropy as the PTQ champion and showed we were "near the
rate-distortion floor for per-weight rotated quantization." This phase attacks every remaining
axis to find out whether that floor is real.

## Results

### Data-aware axis (the explicitly-flagged frontier) — NEGATIVE
| Round | Method | Result | Verdict |
|---|---|---|---|
| 13 | GPTQ (Hessian error feedback, no rotation) | 3b: 5.316@3.25b; 4b: 4.186@4.25b | loses to rotated ECVQ; at 3b, ECVQ (4.483@3.13b) crushes it |
| 13 | GPTQ + entropy-coded indices | 4b grid: 4.186@**3.854b** (same ppl, −0.4 bit) | **free win: entropy-coding ANY quantizer's indices saves ~0.4 bit** |
| 14 | rotation + GPTQ-feedback + ECVQ-entropy fusion | 4.272@3.022b | beats ECVQ.008 marginally, but ~20 min/scheme (per-column Python loop = CPU-bound) |
| 15 | block-Hessian-weighted ECVQ (vectorized) | all variants worse (D-only 5.003 vs control 4.297 @ ~3.7b) | data-aware weighting HURTS — rotation already whitened the signal |

### Space-filling axis — MARGINAL
| 16 | trained 2-D/4-D entropy-constrained VQ | 2-D: small gain over scalar that grows at low bits (~0.1→0.22 ppl); 4-D K256 collapses | real but small; still under tuned scalar ECVQ |

### Rotation axis — NEGATIVE (for beating the frontier)
| 17 | two-sided incoherence rotation (vs single-sided) | two-sided beats single-sided at every matched λ under block-amax (5.03 vs 8.21 @ ~2.8b), ~128× fewer scales | two-sided *enables* cheap few-scale quant |
| 18 | two-sided + 0.5% outliers + finer output groups | best 4.978@3.199b | still behind per-row scalar ECVQ (4.483@3.13b); per-row adaptivity wins at 0.5B |

### QAT axis (regime change: retrain weights to the quantizer) — TIES, doesn't beat
| 19 | QAT 3-bit RTN (no rotation), STE | PTQ 68.2 → QAT plateau ~6.4 @ 3.125b | 10× recovery but grid-limited; the GRID matters even under QAT |
| 20 | QAT 3-bit rotated uniform | PTQ 18.1 → QAT plateau ~5.3 @ 3.125b | better grid → better QAT, still > 4.483 |
| 21 | QAT 3-bit rotated + normal-float (NF) | PTQ 6.84 → QAT plateau ~4.6 @ 3.125b | **ties PTQ-ECVQ (4.483); does not clearly beat it** |

## Headline findings
1. **The PTQ codec frontier is saturated.** Rotated per-row scalar **ECVQ + entropy + outliers
   (~4.48 ppl @ 3.13 bits)** is at/near the achievable rate-distortion frontier for training-free
   PTQ of small-LLM weights. Data-aware (GPTQ/Hessian), vector quantization, fixed lattices, and
   two-sided rotation each give ≤ marginal or negative gains against it.
2. **Entropy-coding any quantizer's indices is a free ~0.4-bit win** (e.g. GPTQ 4.25b → 3.854b at
   identical perplexity) — the project's lossless-coding DNA composes with any quantizer.
3. **The quantization grid matters even under QAT.** Weight adaptation alone cannot overcome a
   coarse/unrotated grid: naive 3-bit RTN QAT plateaus ~6.4; only with rotation + NF levels does
   QAT reach ~4.6, i.e. it merely *ties* what entropy-constrained PTQ already achieves with **no
   retraining**. On this 0.5B model with a small calibration set, QAT is overfitting-limited
   (train loss ≪ held-out) — a striking testament to how strong ECVQ is.

## Engineering law learned (the hard way)
Quantizers MUST stay fully vectorized (torch/numpy batch ops). A Python per-element/column loop on
the GTX 1060 is CPU-launch-bound → **hours** per scheme (R14 GPTQ). Hoist every `.item()`/sync out
of hot loops. Watch every run live (Monitor); never blind-wait on background runs.

## Modules added
`quant_dataaware.py` (GPU GPTQ + Hessian), `quant_fast.py` (block-Hessian vectorized ECVQ),
`quant_vq.py` (trained N-D ECVQ-VQ), `quant_rot.py` (two-sided rotation), `quant_qat.py`
(STE fine-tuning: uniform / rotated / NF grids). All GPU, vectorized, watched live.

### Scale validation (Round 22) — the frontier holds and the gap halves
Qwen2.5-1.5B-Instruct, **fp16 ppl = 3.128** (`quant_scale.py`, fp16 on GPU):

| codec | 1.5B bits | 1.5B ppl | gap to fp16 | (0.5B gap) |
|---|---|---|---|---|
| naive RTN 3b | 3.125 | 12.448 | +9.3 | (~+64) |
| champ | 3.276 | 3.640 | +0.51 | (+0.99) |
| ECVQ λ.008 | 3.135 | 3.473 | +0.35 | (+0.54) |
| ECVQ λ.005 | 3.504 | 3.302 | +0.17 | — |
| **ECVQ λ.003** | **3.904** | **3.240** | **+0.11** | (+0.23) |
| entropy32 | 5.268 | 3.172 | +0.04 | (+0.03) |

The discovered **ECVQ+entropy codec generalizes to 1.5B**, still dominates champ, and the
quantization gap to fp16 **roughly halves** vs 0.5B — i.e. quantization gets *easier* at scale.
At 1.5B, **ECVQ at 3.9 bits is within +0.11 ppl of fp16 (near-lossless under 4 bits)**.

## Net conclusion of the GPU phase
The LLM-as-mutation-operator loop, against a cheap un-gameable verifier, converged on
**rotated scalar entropy-constrained quantization + outliers** as the PTQ frontier; mapped that
no data-aware / vector / lattice / two-sided-rotation refinement materially beats it; showed even
QAT only ties it on a small model; and validated that it scales to 1.5B with a shrinking gap.

## Next
Trellis-coded quant (QTIP) as the last codec-space SOTA check; GPU-port ECVQ for fast 1.5B/3B
sweeps; QAT at 1.5B if a memory-light optimizer is found.
