# 2. Method

## 2.1 The evolutionary loop

We treat post-training quantization (PTQ) codec design as a program-synthesis search in which a large language model (Claude) is the *mutation operator*. Each round, the LLM does not toggle hyperparameters; it writes new codec *code* into an evolvable registry (`codec_zoo.py`). A codec is a Python function `fn(W, key, calib) -> (W_hat, total_bits)`: it consumes a weight tensor and a calibration record, returns a dequantized tensor and an honest bit count, and carries a decode-cost tag (`low`/`med`/`high`) that acts as a realizability gate. The orchestration is an AlphaEvolve-style loop: the LLM proposes one or more mutations, a verifier scores every registered codec, and the resulting leaderboard conditions the next round of mutation. Over the course of the project this produced genuinely distinct strategies — data-aware/GPTQ-Hessian error feedback, trained 2-D/4-D entropy-constrained vector quantization, D4/E8 lattice quantization, fixed-rate and entropy-constrained trellis coding, two-sided incoherence rotation, RMSNorm-gain absorption, and quantization-aware training — rather than variations on one template.

The scoring harness (`quant_arena.py`) is the *arena*. It is deterministic, crash-resilient, and resumable: each codec is evaluated in isolation behind a `try/except` with a content hash of its source, checkpointed to disk immediately, so an out-of-memory failure loses only the current codec and a re-run skips finished ones. The arena reports a held-out leaderboard sorted by perplexity and flags any codec that breaks the standing "wall." This persistence is what makes a multi-round, multi-day search with an LLM in the loop tractable on a single GPU.

## 2.2 The verifier

The selection signal is a cheap, *un-gameable* triple: (i) held-out perplexity, (ii) honest bits/weight, and (iii) a decode-cost class.

**Held-out perplexity.** Quality is measured as perplexity on a *disjoint* prose slice of enwik8 — a fixed slice held out from calibration. We deliberately keep the arena paired and deterministic (fixed rotation seed, fixed prose slice) so that codec-vs-codec margins on that slice are decidable. We report perplexity for the Qwen2.5 family, whose fp16 references on the eval slice are 3.944 (0.5B), 3.128 (1.5B), and 2.947 (3B).

**Honest bits.** The bit count is not a formula evaluated on an idealized stream; it is the byte size of an *actually decoded container*. The champion's side information is serialized into five streams — high/low bytes of each fp16 group-amax, delta-gap varint outlier positions, and high/low bytes of each fp16 outlier value — compressed with real decodable coders (zstd-19 and a context-mixing `cmcore`, best-of), and then *decoded and checked for byte-identical reconstruction*. Bits/weight equals container bytes divided by weight count, with no hidden side information.

**The round-trip gate.** No codec is credited unless its container round-trips: the verifier reconstructs every plane from the decompressed streams and asserts byte-identity before any rate is reported (`scale_plane_codec.py`). This single gate is what makes the metric un-gameable, because it forecloses the two accounting-bug classes the loop encountered historically. The first is the *joint-entropy undercount*: charging the empirical joint entropy of a high-dimensional code, which is optimistic when points are near-unique at fine resolution and which no causal decoder can realize. The second is the *per-coordinate overcount*: summing per-coordinate entropies, which over-charges structured codes — the R12 lattice metric overcharged D4/E8 by roughly 0.7–0.8 bits/weight. A round-trip-verified container cannot exhibit either bug: its size is a number of bytes a real decoder consumed. A complementary conditional-entropy oracle with a train/code layer split (`ecvq_idx_oracle.py`) confirms the residual exposure is nil — incoherence rotation whitens the index stream to near-iid, with triple-verified D_idx = 0 (no positive conditional-coding bank found, not literal exact independence).

**Decode cost.** Each codec also declares a decode-cost class, so a rate win that depends on an unrealistic decoder is visible rather than hidden in the headline number.

## 2.3 The champion codec pipeline

The unbeaten codec (it survived roughly 30 attack rounds and multiple hostile audits) is a recombination of classical information-theory primitives, applied step by step per weight tensor:

1. **Outlier extraction.** The largest-magnitude 0.5% of weights are removed and stored in fp16; they capture the heavy tail that otherwise dominates distortion.
2. **AWQ activation scaling.** Columns are scaled by an activation-aware factor derived from calibration before quantization, then unscaled on dequant.
3. **Per-group incoherence rotation.** Within each group of g = 128 weights, a signed-Hadamard transform (a random sign flip followed by a fast Walsh–Hadamard transform, normalized by √g) rotates the weights into an incoherent basis. After this rotation the per-coordinate sensitivities are equalized (|R[j,c]|² = 1/g), which is precisely why data-aware (Hessian/GPTQ-style) weighting *hurts* rather than helps, and why the index stream whitens to near-iid.
4. **Scalar entropy-constrained quantization (ECVQ).** Levels and probabilities are learned per layer from a large pool (K = 64) using the Chou–Lookabaugh–Gray (1989) objective, assigning each rotated value x to

   k\*(x) = argmin_k [ (x − c_k)² − λ · log₂ p_k ],

   so the quantizer concentrates probability (low entropy) at controlled distortion; λ is the rate knob and unused levels are pruned. This decouples reconstruction quality (number of levels) from storage (entropy).
5. **Entropy-coded indices + honest side-info container.** The ECVQ indices are entropy-coded, and the group-amax and outlier planes are packed into the A1 honest container (Section 2.2).

ECVQ is a known 1989 technique; its novelty here is as an application to LLM quantization, not as an invention. At the 0.5B operating point the champion sits near 3.00 bits/weight honest at perplexity ~4.58 ± 0.07. The ±0.07 is load-bearing: an iso-bit positive control (varying only the rotation seed) shows a selection noise floor of std 0.067 ppl (spread 0.21), so the historical headline "4.483 ppl @ 3.13 b/w" must be reported as a single rotation-seed instance rather than a sharp value; bits fall from 3.13 to 3.00 after the A1 lossless re-coding. Measured frontier points on the same 0.5B model (honest b/w) include 3.957 ppl @ 4.277 b/w and 4.044 ppl @ 4.069 b/w, with a near-lossless entropy-32 point at ~3.97 ppl @ ~5.14 b/w. A deployable "simple-decode" variant (rotation + ECVQ levels, no entropy coder, LUT-decodable) trades up to roughly 3.5 bits/weight for hardware realizability today.
