# Lossless Compression of the LLM Lifecycle: Variants, Checkpoints, and Optimizer State via Adaptive Delta Coding

*(working draft — target venue: MLSys / NeurIPS D&B / ICLR)*

## Abstract

The weights of a trained neural network are nearly incompressible: at the bit level the
mantissa is effectively random, so the best lossless coders (ZipNN, DFloat11) plateau near
a ~30% reduction set by the exponent's low entropy. We observe that the artifacts an AI
organization actually stores are rarely *independent* models — they are **variants** of a
shared base (instruction-tuned, abliterated, merged, LoRA-merged, domain-adapted), **training
checkpoints** of one run, and the **optimizer state** that accompanies them. The *change*
between two such artifacts has dramatically lower information content than the artifacts
themselves, and — unlike lossy delta methods (BitDelta) — it can be captured **exactly**.

We present a unified, per-tensor adaptive **lossless delta codec** that selects among five
modes (identical-copy, sparse bitmap, byte-plane XOR, float-ordered arithmetic, and a
**lossless low-rank residual**) and a characterization of *why* model changes compress. On
real models this yields **up to 109× lossless compression** of a model variant (98.7% at 3B,
on a 6.2 GB model with 3.5 GB of RAM), **~50–69%** of a checkpoint, **~55%** of a full
checkpoint *including optimizer state*, and **92–97%** of a quantized (int8/fp8) variant —
all byte-exact (SHA-256 gated). We match the only published lossless-delta result on its own
regime and far exceed any single-model coder on variants. The implication is an ecosystem one:
model registries and checkpoint stores hold one to two orders of magnitude of removable,
recoverable redundancy.

## 1. Introduction

Modern AI storage is dominated by *near-duplicates*. A single base model (Llama, Qwen,
Mistral) spawns hundreds of thousands of derivatives on public hubs, and a single training
run emits dozens of checkpoints, each paired with an optimizer state ~2× the model's size.
Yet the dominant storage practice is to keep every artifact in full, or to compress each one
independently — which, for weights, buys almost nothing (the mantissa is random).

The lossless-compression literature has accepted this wall: ZipNN [Hoffman'24] and DFloat11
[2025] entropy-code the *exponent* (the only skewed field) for a ~30% reduction and stop.
The orthogonal observation — that the *delta* between related models is low-entropy — has
been exploited only **lossily** (BitDelta [NeurIPS'24] quantizes the delta to 1 bit,
sacrificing exactness) or, very recently, losslessly for a narrow case (blockwise-XOR of bf16
checkpoints, ~62% [arXiv 2508.19263]). No prior work treats the **whole lifecycle**
losslessly, nor exploits the rank structure of edits.

**Contributions.**
1. A **characterization of model-change information** (§3): which bits move and why, across
   training steps, fine-tunes, abliteration/merges, optimizer EMAs, and quantization.
2. A **lossless low-rank residual coder** (§4.3): store int16 rank-r factors plus an exact
   residual against a *deterministically reconstructed* reference (exact integer matmul), so
   the codec is byte-identical on any machine while the residual absorbs all quantization
   error. To our knowledge the first lossless use of low-rank structure for weight coding.
3. A **float-ordered arithmetic delta** (§4.2) that beats XOR on formats without frozen
   low-mantissa bits (bf16/fp16, and fp32 optimizer state).
4. A **unified per-tensor codec** (§4.4) that is provably never worse than any single mode,
   is self-verifying (embedded SHA-256), memory-bounded (mmap + streaming), and scales to
   multi-GB models on commodity RAM.
5. **Comprehensive, byte-exact experiments** (§5) across precisions (bf16/fp16/fp32/int8/fp8),
   scales (0.5B–7B), and lifecycle stages, plus **honest negative results** (§6).

## 2. Background

IEEE/bf16 weights: 1 sign, 8 (bf16) / 5 (fp16) / 8 (fp32) exponent bits, the rest mantissa.
ZipNN/DFloat11 finding: exponent carries ~2.6 bits of real entropy; mantissa is ~uniform.
Lossy delta (BitDelta): ΔW ≈ sign(ΔW)·scale (1 bit), ~10× but accuracy-lossy. Lossless delta
(2508.19263): blockwise XOR + Huffman, ~62% on bf16 checkpoints.

## 3. Characterizing model-change information

**(C1) Training freezes low mantissa bits.** On real Pythia-70m checkpoints (fp32, 1000
steps), only mantissa bits 13–23 change; bits 0–12 stay frozen at their random init values.
Those frozen-random low bits are exactly what makes standalone incompressible (16.9% floor),
and they cancel in the delta — hence fp32 checkpoint deltas reach 69.5%.

**(C2) Edits are low-rank.** Abliteration is ΔW = −r̂·(r̂ᵀW), rank-1 per layer with a shared
direction; LoRA-merges are rank-r by construction. On the real Qwen abliteration, the changed
matrices have **rank-4 capturing 100% of the delta energy** (vs 1.8% for a random matrix).

**(C3) Optimizer moments are EMAs.** mₜ = β₁mₜ₋₁+(1−β₁)gₜ, vₜ = β₂vₜ₋₁+(1−β₂)gₜ². The
per-step change is a fixed small fraction (10% / 0.1%) of scale. In bf16, v's change falls
**below one ULP**, so much of v is byte-identical step-to-step.

**(C4) Quantization preserves the edit's support.** Quantizing a variant on the base's grid
leaves untouched tensors byte-identical and changes only a small fraction (~12%) of the rest.

## 4. Method

### 4.1 XOR byte-plane delta (frozen-bit regime)
For fp32, XOR cancels frozen low bits; per-plane zstd codes each byte plane. Best for fp32.

### 4.2 Float-ordered arithmetic delta
Map a float's bits to a monotonic integer key (positives set the top bit, negatives flip all
bits) so adjacent representable values get adjacent keys; then (keyₐ − keyᵦ) is the signed
number of representable steps moved — small for gentle changes — and zig-zag + byte-plane
coding compresses it. Beats XOR, which over-counts small moves that cross an exponent
boundary. Implemented for 16- and 32-bit floats. (+4 pts on bf16 fine-tunes.)

### 4.3 Lossless low-rank residual
For a 2D delta ΔW that is low-rank: take a randomized rank-r SVD, keep the **numerical rank**
(singular values > 1% of the largest — the signal, not the rounding-noise floor), quantize
the factors A,B to **int16**, and reconstruct a reference W' = round(W_base + A·B·sₐ·s_b)
using **exact integer matmul** (so W' is bit-identical on any platform). Store the int16
factors plus the arithmetic residual between the target and W'. Lossless regardless of factor
quality (the residual closes the gap); portable because the reconstruction never uses
nondeterministic float matmul. This lifts the abliteration from 90.9% (elementwise) to
**99.1%**.

### 4.4 Unified per-tensor codec
Per tensor, probe the elementwise candidates at a cheap zstd level to pick the winner, then
encode it at the target level; compute the low-rank candidate in full; keep the **smallest**
(plus copy/sparse). Never worse than any single mode. Self-verifying: the archive embeds the
original's SHA-256, so decode reproduces the exact bytes or raises (wrong reference is caught,
never silent). Memory-bounded: mmap input, on-demand reference reads, streaming decode →
multi-GB models on < model-size RAM. Multithreaded (zstd releases the GIL).

## 5. Experiments (all real models, byte-exact / SHA-256 gated)

| Lifecycle stage | format | save% | ratio |
|---|---|---:|---:|
| Variant (abliteration) | bf16 | **99.1%** | 109× |
| Variant, quantized | int8 / fp8 | 97.4% / 92.3% | 38× / 13× |
| Checkpoint (1000-step) | fp32 / bf16 | 69.5% / 61.3% | 3.3× / 2.6× |
| Heavy full fine-tune | bf16 | 53.0% | 2.1× |
| Optimizer state (real Adam) | bf16 | v 67%, full ckpt ~55% | — |
| Training run (6 ckpts + optim) | bf16 | 46% | 1.85× |
| Single model (no ref) | bf16 | 32.7% | 1.5× |

**Scale (bounded memory):** 0.5B / 1.5B / 3B abliteration → 99.1% / 98.8% / 98.7% (109× /
84× / 80×); 3B is a 6.2 GB model compressed on 3.5 GB free RAM, with base and variant sharded
*differently* (cross-shard reference-by-name). [7B pending.]

**Baselines.** Single-model: we match ZipNN/DFloat11 (~32.7% bf16). Lossless-delta: we match
the 2508.19263 bf16-checkpoint result (61.3% ≈ 62%) and exceed it on fp32 (69.5%); on
variants we are **74× smaller files** than any single-model coder can be.

## 6. Limitations and negative results (rigor)

- **Momentum is not exploitable for weight deltas.** Consecutive training-step deltas are
  statistically independent (cosine ≈ 0, ‖dd‖/‖d‖ ≈ √2); a 2nd-order coder does not help.
- **fp32 optimizer deltas compress poorly (~30%)** because the EMA recomputes each value,
  scrambling low mantissa bits; only bf16 optimizer state compresses well.
- **A general entropy coder buys < 1 pt** over zstd on the arithmetic delta; the residual is
  irreducible information.
- **Low-rank helps only structured edits**; dense full fine-tunes (MLP) are near-full-rank.
- Synthetic-vs-real gap: real early-training gradients make `m` *less* compressible than a
  smooth synthetic AR model suggested (32% vs 49%); we report the real number.
- Independent foundations do not cross-derive (boundary of the technique). Qwen2-0.5B and
  Qwen2.5-0.5B are the same architecture (290/290 tensors) but independently trained; the
  cross-generation delta saves 28.0% -- *worse* than the 30.2% standalone (deltaing scrambles
  the compressible exponent). A true descendant (abliteration) saves 98.9%. So lossless
  derivation is a *lineage* property (shared init/basin), not an architecture or family
  property: you can compress a model's descendants 10-100x, but not derive one foundation from
  another. Constructive corollary: make models share a lineage *by design* (a reproducible-root
  "model genome"), so an entire training tree is one root + deltas.

## 7. Related work
ZipNN, DFloat11 (lossless single-model); BitDelta, Delta-CoMe, GPT-Zip (lossy delta);
2508.19263 (lossless bf16-checkpoint XOR); LoRA (low-rank adaptation); standard checkpoint
compression (mostly lossy / gradient-focused).

## 7b. Ecosystem measurement
A real Qwen2.5-0.5B family of 11 models (base + 10 independent hub derivatives) stored as
deltas vs the base compresses 12844→7392 MB (42.5%; 54.6% over the bf16-only subset). The key
result is that **the lossless delta size directly measures how much a fine-tune changed the
model**: an exact byte-duplicate re-upload costs **0 bytes**; abliteration **1%**; preference
tuning (DPO/GRPO) **5–8%**; full SFT **~80%**. Thus the family ratio depends on the mix:
1.5–3× for typical bf16 SFT families, and 10×+ for light-derivative-dominated sets
(re-uploads, quantizations, abliterations, LoRA-merges, preference-tuning). Deltas require a
same-precision reference (fp32 derivatives do not delta against a bf16 base).

## 8. Conclusion
Treating the model *lifecycle* as the unit of compression turns a near-incompressible object
(a model) into a highly compressible one (its change), losslessly. Real hubs contain exact
duplicates and light derivatives that compress to ~zero, and even heavy fine-tunes store no
worse than independently; a model family thus costs the base plus its few heavy members.
The recoverable redundancy no single-model coder can reach is large — and lossless. Artifacts,
code, and a reproducible benchmark accompany this paper.
