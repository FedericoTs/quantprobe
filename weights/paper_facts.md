# Authoritative corrected record (use THESE numbers; do not invent or round differently)

PROJECT: an AlphaEvolve-style loop where an LLM (Claude) is the MUTATION OPERATOR that writes
quantization-codec CODE each round, scored by a cheap UN-GAMEABLE verifier: held-out perplexity on
a disjoint enwik8 slice + HONEST bits/weight (= bytes of an ACTUALLY-DECODED container, no hidden
side info; both joint-entropy UNDERcount and per-coordinate OVERcount bugs were caught historically)
+ decode-cost class. Models: Qwen2.5-0.5B (fp16 ppl 3.944 on the eval slice), 1.5B (3.128), 3B (2.947).

## Champion (unbeaten across ~30 attack rounds + multiple hostile audits)
per-group (g=128) signed-Hadamard incoherence rotation + AWQ activation scaling + scalar
ENTROPY-CONSTRAINED quantization (ECVQ; Chou-Lookabaugh-Gray 1989: assign x to argmin (x-c)^2 minus
lambda*log2 p_k, with levels and probabilities learned per layer) + entropy-coded indices + 0.5%
largest-magnitude weights kept fp16 (outliers) + the A1 honest side-info container.
- Operating point (0.5B): ~3.00 bits/weight honest, ppl ~4.58 +/- 0.07 (see seed-noise below).
- Measured frontier points (0.5B, honest b/w): 3.957 ppl @ 4.277 b/w; 4.044 @ 4.069; near-lossless
  entropy32 ~3.97 @ ~5.14 b/w.
- HONESTY NOTE: the historical headline "4.483 ppl @ 3.13 b/w" was ONE rotation-seed instance; R2
  showed seed alone moves ppl by std 0.067 (spread 0.21) at iso-bit, so the honest champion ppl is
  ~4.58 +/- 0.07 and "4.483" must be reported as a seed instance, not a sharp value. Bits 3.13 ->
  3.00 after the A1 lossless re-coding.

## Key result A1 (the one NEW positive result; lossless, round-trip byte-verified)
The champion paid side info RAW (16 bits/group amax + 32 bits/outlier = 0.2876 b/w). A real
decodable container (byte-split + zstd-19/cmcore best-of) compresses it to 0.1608 b/w. BANK = 0.127
b/w at IDENTICAL ppl (it is lossless side-info recompression). This is a LOWER bound (the AWQ
per-column scale stream, ~0.011 b/w, is still uncoded). Caveat: ~0.07 of the 0.127 is the amax bank,
available to ANY per-group codec; only ~0.077 (the outlier bank) is champion-exclusive.

## The map of closed axes (every alternative fails to beat the champion under honest symmetric accounting; this robustness IS the main scientific contribution)
- Data-aware (GPTQ/Hessian error feedback; block-Hessian-weighted ECVQ): negative; after incoherence
  rotation the per-coordinate sensitivities are ~equal (|R[j,c]|^2 = 1/g), so data-aware weighting hurts.
- Vector quantization (trained 2-D/4-D ECVQ codebooks): a small space-filling gain WITHOUT outliers,
  but LOSES once outliers are added (outliers capture the same heavy tail).
- Lattices D4/E8: CLOSED. The historical R12 'per-coordinate entropy' metric OVERCHARGED them by
  ~0.7-0.8 b/w (a real accounting bug, fixed with real decodable containers); but even after the fix,
  a champion point MEASURED at matched ppl (ECVQ lam.0015 = 3.957 ppl @ 4.277 b/w) STRICTLY
  Pareto-dominates E8 q0.08 (4.020 ppl @ 4.640 b/w) on both axes.
- Trellis-coded quantization, BOTH fixed-rate (lost, paid 5 b/w) AND entropy-constrained (ECTCQ, rate
  inside the Viterbi metric): CLOSED. ECTCQ initially APPEARED to beat the envelope (3.978 ppl @ 4.140
  'b/w'), was self-flagged as suspicious, and sent to a 6-agent audit (3 independent rate derivations
  + 2 code audits + a brute-force DP check). Verdict: the reconstruction was CORRECT but the rate
  OMITTED H(coset | current-state) = the branch/path bit a real decoder must receive (~0.95 b/w,
  measured). Corrected honest rate ~5.13 b/w -> LOSES by ~0.85 b/w. STRUCTURAL LESSON: a near-balanced
  4-state trellis pays ~0.95 b/w of path entropy to buy at most ~0.25 b/w of space-filling gain on a
  whitened-iid source = a guaranteed loss; pricing the branch INSIDE the Viterbi metric does NOT make
  the decoder's branch bit free.
- Two-sided incoherence rotation: enables cheap few-scale block quant but loses to per-row scales.
- RMSNorm-gain absorption (a function-preserving canonicalization): KILLED, -0.144 ppl -- folding the
  gamma into weight columns breaks the column-scale homogeneity that rotation + per-group-amax need.
- Index context-coding / micro entropy models / permutation gauges: DEAD. The ECVQ index stream is
  iid AFTER rotation -- triple-verified D_idx = 0 (a conditional-entropy oracle with a train/code
  layer split: ALL contexts give negative bank; plus zstd-19 and cmcore on the FULL stream both do
  WORSE than order-0). Clean citable: "incoherence rotation whitens the index stream to near-iid".
- QAT (straight-through fine-tuning; uniform/rotated/NF grids): only TIES the PTQ codec, never beats
  it; the GRID matters more than retraining (naive-RTN QAT plateaus ~6.4 ppl, rotated+NF ~4.6;
  entropy-constrained PTQ is already there with no training). Overfitting-limited at 0.5B.
- amax-snapping (lossy): just trades along the existing R-D frontier (the same trade lambda gives);
  no free win beyond A1's lossless bank.

## Verifier characterization (A2/C1/R2 -- the methodology defense)
- A2: enwik8 slices are wildly content-heterogeneous (fp16 ppl 2.2-17.2; some are markup). The arena
  is deterministic + PAIRED (fixed seed, fixed prose slice), so codec-vs-codec margins are decidable
  on that slice; generalization across content is content-dependent.
- C1 (divergence battery): Spearman(single-slice-ppl rank, worst-of-6-domain-KL rank) = 1.000 over 8
  principled codecs (2.4-5.3 b/w) -> single-slice ppl is a FAITHFUL rate-distortion proxy among
  principled codecs; even the 2.4-bit point degrades uniformly across domains.
- R2 (positive control): iso-bit, vary ONLY the rotation seed -> selection NOISE FLOOR std 0.067 ppl
  (spread 0.21); Spearman(worked-slice, prose-escrow) = 0.324 -> single-slice seed selection does NOT
  transfer (mostly noise). NUANCE: this is an UPPER bound (unpaired); historical comparisons used a
  FIXED seed (paired, smaller noise). Implication: headline margins (>0.2 ppl: the champion, the
  lattice Pareto 0.19-0.36 b/w, QAT ties, ECTCQ -0.85 b/w) are SAFE; sub-0.1 ppl orderings are within
  noise and must be reported as ties.

## Scale law (the external-validity result)
The champion's gap-to-fp16 at ~3.13 b/w SHRINKS MONOTONICALLY with model size: 0.5B +0.539 -> 1.5B
+0.345 -> 3B +0.244 (~30-35% per step; extrapolates near-lossless at ~7B). At 3B the codec is
NEAR-LOSSLESS (+0.12 ppl) by ~3.5 bits, and entropy32 is lossless within noise (gap -0.004 @ 5.27
b/w). Secondary (report honestly as model-specific, NOT a clean law): naive unrotated/no-outlier
RTN-3b COLLAPSES at 3B (46036 ppl) where the champion stays robust -- consistent with outlier
features growing with scale -- but naive's trend is NON-monotonic (0.5B 68 -> 1.5B 12 -> 3B 46k).

## Methodological novelty (the actual paper claim)
First demonstration of an LLM as the in-loop MUTATION OPERATOR discovering PTQ codecs against a cheap
un-gameable verifier, INCLUDING documented SELF-CORRECTION: the loop caught TWO of its own
bit-accounting bugs (the E8/lattice metric overcharge and the ECTCQ branch-bit undercount) via
hostile multi-agent audit and reversed its own conclusions. ECVQ is a known 1989 technique that is
novel FOR LLM quantization (a novel application, not an invention).

## Honest limitations
0.5B-3B scale (no 7B+); a perplexity-proxy verifier (validated against multi-domain KL but not a full
downstream task suite); no production fused entropy-decode GPU kernel (the 3.00 b/w needs one to be
deployable; a "simple-decode" ~3.5 b/w variant -- rotation + ECVQ levels, no entropy coder,
LUT-decodable -- is deployable today); a single model family (Qwen2.5); the champion is a
classical-information-theory recombination, not a new primitive.

## Files (real artifacts to cite)
weights/codec_zoo.py (the evolvable registry), quant_arena.py (resumable verifier),
scale_plane_codec.py (A1 container), lattice_rescore.py, ecvq_idx_oracle.py (D_idx), ectcq.py,
divergence_battery.py, overfit_control.py, quant_scale.py / quant_scale3b.py (scale),
ROADMAP_R25_v2.md, EVOLUTION_R13-21.md.
