# An LLM as the Mutation Operator: Evolving an Honestly-Accounted Rate-Distortion Frontier for ~3-Bit LLM Weight Quantization

## Abstract

We study post-training quantization (PTQ) of large-language-model weights as a
rate-distortion problem and ask whether an LLM, used as the in-loop *mutation
operator* of an AlphaEvolve-style evolutionary search, can discover a competitive
quantization codec when scored by a cheap, un-gameable verifier. The verifier
combines held-out perplexity on a disjoint enwik8 slice with an *honest*
bits/weight measured from an actually-decoded container (no hidden side
information) and a decode-cost class. Across roughly 30 attack rounds and multiple
hostile multi-agent audits, the search converges to and cannot dethrone a single
champion: per-group signed-Hadamard incoherence rotation plus AWQ activation
scaling plus entropy-constrained scalar quantization (ECVQ), with entropy-coded
indices and 0.5% fp16 outliers. On Qwen2.5-0.5B (fp16 perplexity 3.944) the
champion operates at approximately 3.00 honest bits/weight with perplexity
roughly 4.58 +/- 0.07. We contribute (1) a verified rate-distortion frontier for
~3-bit PTQ in which entropy-constrained scalar quantization plus incoherence
rotation closes the lattice and trellis-coded axes under symmetric accounting --- a
champion point strictly Pareto-dominates the best measured E8 point on both axes,
and ECTCQ loses by ~0.85 b/w once its path entropy is paid; (2) an
honest-accounting methodology that recovered at least 0.127 b/w of recompressible
side information inside the champion (a lower bound; only ~0.077 b/w
champion-exclusive) and caught two of the loop's own bit-accounting bugs via audit;
and (3) a monotonic scale law in which the gap to fp16 at ~3.13 bits/weight shrinks
0.539 -> 0.345 -> 0.244 perplexity from 0.5B to 1.5B to 3B.

## 1. Introduction

Post-training quantization compresses the weights of a trained network without
gradient-based retraining, and it has become the dominant deployment lever for
large language models: a model that fits in fewer bits per weight is cheaper to
store, to move, and to serve. The central difficulty is that LLM weight matrices
are not benign i.i.d. sources. They contain heavy-tailed outlier coordinates and
strongly correlated columns, so naive round-to-nearest at low bit rates degrades
quality catastrophically. Modern PTQ pipelines therefore compose several
ingredients --- incoherence-inducing rotations, activation-aware scaling,
non-uniform grids, and outlier handling --- and the design space of such
compositions is large, combinatorial, and easy to evaluate *dishonestly*.

This paper treats PTQ as an explicit rate-distortion search and drives that search
with an LLM as the *mutation operator*. Following the AlphaEvolve template, each
round the LLM (Claude) writes new quantization-codec code into an evolvable
registry; a resumable arena then scores the candidate. The crucial design choice
is the **verifier**, which must be cheap enough to run every round yet hard to
game. Ours has three parts. First, distortion is held-out perplexity on a prose
slice of enwik8 that is disjoint from any data the codec is allowed to see.
Second, rate is *honest bits/weight*: the byte size of a container that an
independent decoder actually reconstructs, with no side information smuggled
outside the count. This matters because both directions of mis-accounting are
real failure modes --- a joint-entropy estimate can *undercount* the true cost,
and a per-coordinate entropy estimate can *overcount* it --- and we caught
instances of both historically. Third, a decode-cost class prevents the search
from buying rate with an undeployable decoder. Models are Qwen2.5 at 0.5B, 1.5B,
and 3B parameters (fp16 eval-slice perplexities 3.944, 3.128, and 2.947).

Run under this verifier for roughly 30 attack rounds and several hostile audits,
the loop converges on one champion and repeatedly fails to beat it. The champion
combines per-group (g = 128) signed-Hadamard incoherence rotation, AWQ activation
scaling, scalar entropy-constrained quantization (ECVQ, in the
Chou-Lookabaugh-Gray sense, with per-layer learned levels and probabilities),
entropy-coded indices, and the 0.5% largest-magnitude weights retained in fp16.
On 0.5B it operates at approximately 3.00 honest bits/weight at perplexity
roughly 4.58 +/- 0.07. We stress the hedge: an earlier headline of 4.483
perplexity at 3.13 b/w was a *single rotation-seed instance*, and a positive
control shows that varying only the rotation seed moves iso-bit perplexity with
standard deviation 0.067 (spread 0.21), so 4.483 must be read as one draw from
that distribution rather than a sharp value. Measured frontier points on 0.5B
include 3.957 perplexity at 4.277 b/w and 4.044 at 4.069 b/w, with a near-lossless
entropy-32 variant near 3.97 at ~5.14 b/w.

We make three contributions, stated numerically.

**(1) A verified rate-distortion frontier for ~3-bit PTQ.** Entropy-constrained
scalar quantization plus incoherence rotation resists roughly 30 attack rounds.
Under honest, symmetric accounting it Pareto-dominates the natural alternatives:
a champion point measured at 3.957 perplexity / 4.277 b/w strictly dominates an E8
lattice point at 4.020 / 4.640 on both axes, and entropy-constrained trellis
coding (ECTCQ), once its path entropy is paid, loses by about 0.85 b/w. The robustness
of this frontier --- the map of *closed* axes (data-aware weighting, vector
quantization, lattices, trellises, QAT) --- is itself a central scientific result.

**(2) An honest-accounting methodology.** The champion originally paid its side
information raw (0.2876 b/w); a real decodable container recompresses it to 0.1608
b/w, banking 0.127 b/w at identical perplexity. This is a lower bound (an ~0.011
b/w scale stream remains uncoded), and we report honestly that only ~0.077 of the
0.127 is champion-exclusive. The same discipline caught two of the loop's *own*
bugs --- a lattice metric that overcharged D4/E8 by ~0.7-0.8 b/w, and an ECTCQ
rate that undercounted the decoder's branch bit (~0.95 b/w) --- each surfaced by
multi-agent audit and reversed by the loop.

**(3) A monotonic scale law.** The champion's gap to fp16 at ~3.13 b/w shrinks
monotonically with model size: +0.539 perplexity at 0.5B, +0.345 at 1.5B, and
+0.244 at 3B (~30-35% per step). By 3B the codec is near-lossless (+0.12
perplexity by ~3.5 bits), and the entropy-32 variant is lossless within noise
(gap -0.004 at 5.27 b/w).

Beyond these results, the methodological claim is that this is, to our knowledge,
the first demonstration of an LLM serving as the in-loop mutation operator that
discovers PTQ codecs against a cheap un-gameable verifier --- including documented
self-correction, in which the loop reversed its own conclusions after audit. We
are explicit about scope: ECVQ is a known 1989 technique applied in a novel
setting rather than a new primitive, the study spans 0.5B-3B of a single model
family, and the verifier is a perplexity proxy validated against multi-domain KL
but not a full downstream-task suite.

## 2. Method

### 2.1 The evolutionary loop

We treat post-training quantization (PTQ) codec design as a program-synthesis search in which a large language model (Claude) is the *mutation operator*. Each round, the LLM does not toggle hyperparameters; it writes new codec *code* into an evolvable registry (`codec_zoo.py`). A codec is a Python function `fn(W, key, calib) -> (W_hat, total_bits)`: it consumes a weight tensor and a calibration record, returns a dequantized tensor and an honest bit count, and carries a decode-cost tag (`low`/`med`/`high`) that acts as a realizability gate. The orchestration is an AlphaEvolve-style loop: the LLM proposes one or more mutations, a verifier scores every registered codec, and the resulting leaderboard conditions the next round of mutation. Over the course of the project this produced genuinely distinct strategies — data-aware/GPTQ-Hessian error feedback, trained 2-D/4-D entropy-constrained vector quantization, D4/E8 lattice quantization, fixed-rate and entropy-constrained trellis coding, two-sided incoherence rotation, RMSNorm-gain absorption, and quantization-aware training — rather than variations on one template.

The scoring harness (`quant_arena.py`) is the *arena*. It is deterministic, crash-resilient, and resumable: each codec is evaluated in isolation behind a `try/except` with a content hash of its source, checkpointed to disk immediately, so an out-of-memory failure loses only the current codec and a re-run skips finished ones. The arena reports a held-out leaderboard sorted by perplexity and flags any codec that breaks the standing "wall." This persistence is what makes a multi-round, multi-day search with an LLM in the loop tractable on a single GPU.

### 2.2 The verifier

The selection signal is a cheap, *un-gameable* triple: (i) held-out perplexity, (ii) honest bits/weight, and (iii) a decode-cost class.

**Held-out perplexity.** Quality is measured as perplexity on a *disjoint* prose slice of enwik8 — a fixed slice held out from calibration. We deliberately keep the arena paired and deterministic (fixed rotation seed, fixed prose slice) so that codec-vs-codec margins on that slice are decidable. We report perplexity for the Qwen2.5 family, whose fp16 references on the eval slice are 3.944 (0.5B), 3.128 (1.5B), and 2.947 (3B).

**Honest bits.** The bit count is not a formula evaluated on an idealized stream; it is the byte size of an *actually decoded container*. The champion's side information is serialized into five streams — high/low bytes of each fp16 group-amax, delta-gap varint outlier positions, and high/low bytes of each fp16 outlier value — compressed with real decodable coders (zstd-19 and a context-mixing `cmcore`, best-of), and then *decoded and checked for byte-identical reconstruction*. Bits/weight equals container bytes divided by weight count, with no hidden side information.

**The round-trip gate.** No codec is credited unless its container round-trips: the verifier reconstructs every plane from the decompressed streams and asserts byte-identity before any rate is reported (`scale_plane_codec.py`). This single gate is what makes the metric un-gameable, because it forecloses the two accounting-bug classes the loop encountered historically. The first is the *joint-entropy undercount*: charging the empirical joint entropy of a high-dimensional code, which is optimistic when points are near-unique at fine resolution and which no causal decoder can realize. The second is the *per-coordinate overcount*: summing per-coordinate entropies, which over-charges structured codes — the R12 lattice metric overcharged D4/E8 by roughly 0.7–0.8 bits/weight. A round-trip-verified container cannot exhibit either bug: its size is a number of bytes a real decoder consumed. A complementary conditional-entropy oracle with a train/code layer split (`ecvq_idx_oracle.py`) confirms the residual exposure is nil — incoherence rotation whitens the index stream to near-iid, with triple-verified D_idx = 0 (no positive conditional-coding bank found, not literal exact independence).

**Decode cost.** Each codec also declares a decode-cost class, so a rate win that depends on an unrealistic decoder is visible rather than hidden in the headline number.

### 2.3 The champion codec pipeline

The unbeaten codec (it survived roughly 30 attack rounds and multiple hostile audits) is a recombination of classical information-theory primitives, applied step by step per weight tensor:

1. **Outlier extraction.** The largest-magnitude 0.5% of weights are removed and stored in fp16; they capture the heavy tail that otherwise dominates distortion.
2. **AWQ activation scaling.** Columns are scaled by an activation-aware factor derived from calibration before quantization, then unscaled on dequant.
3. **Per-group incoherence rotation.** Within each group of g = 128 weights, a signed-Hadamard transform (a random sign flip followed by a fast Walsh–Hadamard transform, normalized by √g) rotates the weights into an incoherent basis. After this rotation the per-coordinate sensitivities are equalized (|R[j,c]|² = 1/g), which is precisely why data-aware (Hessian/GPTQ-style) weighting *hurts* rather than helps, and why the index stream whitens to near-iid.
4. **Scalar entropy-constrained quantization (ECVQ).** Levels and probabilities are learned per layer from a large pool (K = 64) using the Chou–Lookabaugh–Gray (1989) objective, assigning each rotated value x to

   k\*(x) = argmin_k [ (x − c_k)² − λ · log₂ p_k ],

   so the quantizer concentrates probability (low entropy) at controlled distortion; λ is the rate knob and unused levels are pruned. This decouples reconstruction quality (number of levels) from storage (entropy).
5. **Entropy-coded indices + honest side-info container.** The ECVQ indices are entropy-coded, and the group-amax and outlier planes are packed into the A1 honest container (Section 2.2).

ECVQ is a known 1989 technique; its novelty here is as an application to LLM quantization, not as an invention. At the 0.5B operating point the champion sits near 3.00 bits/weight honest at perplexity ~4.58 ± 0.07. The ±0.07 is load-bearing: an iso-bit positive control (varying only the rotation seed) shows a selection noise floor of std 0.067 ppl (spread 0.21), so the historical headline "4.483 ppl @ 3.13 b/w" must be reported as a single rotation-seed instance rather than a sharp value; bits fall from 3.13 to 3.00 after the A1 lossless re-coding. Measured frontier points on the same 0.5B model (honest b/w) include 3.957 ppl @ 4.277 b/w and 4.044 ppl @ 4.069 b/w, with a near-lossless entropy-32 point at ~3.97 ppl @ ~5.14 b/w. A deployable "simple-decode" variant (rotation + ECVQ levels, no entropy coder, LUT-decodable) trades up to roughly 3.5 bits/weight for hardware realizability today.

## 3. Results: The Operating Point and the Measured Frontier

### 3.1 The champion operating point

On Qwen2.5-0.5B (fp16 eval-slice perplexity 3.944), the champion codec — per-group (g = 128) signed-Hadamard incoherence rotation, AWQ activation scaling, scalar entropy-constrained quantization (ECVQ), entropy-coded indices, and the 0.5% largest-magnitude weights retained in fp16 — operates at approximately **3.00 honest bits/weight at perplexity roughly 4.58 ± 0.07**. This codec is the unique fixed point of the search: it survived roughly 30 attack rounds and multiple hostile multi-agent audits without being dethroned.

We are deliberate about the hedge on perplexity, because it is the single number most likely to be over-read. An earlier headline of **4.483 perplexity at 3.13 b/w** is a *single rotation-seed instance*, not a sharp operating value. A positive control (R2; Section 5) that varies only the rotation seed at iso-bit measures a selection noise floor of standard deviation 0.067 perplexity, with a spread of 0.21 across seeds. The 4.483 figure is therefore best read as one favorable draw from that distribution; the honest champion perplexity is **4.58 ± 0.07**. The rate moved from 3.13 to 3.00 b/w not by re-tuning the lossy codec but by losslessly re-coding the side information (Section 3.3), at identical perplexity.

### 3.2 The measured rate–distortion frontier

Sweeping the ECVQ rate knob λ traces a frontier on the same 0.5B model, with every point reported in honest bits/weight (container bytes ÷ weight count) and held-out perplexity. Two representative interior points are **3.957 perplexity at 4.277 b/w** (λ = 0.0015) and **4.044 perplexity at 4.069 b/w**. A near-lossless **entropy-32** variant sits at approximately **3.97 perplexity at ~5.14 b/w**, recovering essentially the fp16 quality of 3.944 at the cost of rate. Table 1 collects these points alongside the headline operating point and the two leading closed alternatives.

The frontier is what licenses the Pareto claims of Section 4 under honest, symmetric accounting. The champion point at 3.957 / 4.277 b/w *strictly* dominates an E8 lattice point at 4.020 / 4.640 b/w on both axes — better perplexity at fewer bits — and entropy-constrained trellis coding (ECTCQ), once its omitted ~0.95 b/w of decoder path entropy is paid, lands near 5.13 b/w and loses by roughly 0.85 b/w at matched quality. Because each of these margins exceeds the 0.067-ppl / sub-0.1-ppl noise floor and is measured in bits rather than perplexity, the orderings are safe rather than artifacts of seed noise.

### 3.3 A1: lossless re-coding of the champion's side information

The one *new positive* result is a lossless recompression of the champion's side-information streams, verified by byte-identical round trip. As originally formulated, the champion paid its side information **raw**: 16 bits per group for the fp16 group-amax plus 32 bits per outlier (position and fp16 value), totaling **0.2876 b/w**. Serializing these planes into byte-split streams and compressing them with real decodable coders (zstd-19 and a context-mixing `cmcore`, best-of), then decoding and asserting byte-identity, compresses the same information to **0.1608 b/w**. This banks **0.127 b/w at identical perplexity** — it is purely a re-coding of side info, so distortion is unchanged — and is what carries the operating point from 3.13 to 3.00 b/w.

Two caveats keep this honest. First, the **0.127 b/w saving is a lower bound** (equivalently, 0.1608 b/w is an upper bound on the achievable side-info rate): the AWQ per-column scale stream (~0.011 b/w) is still uncoded, so the realizable saving is at least this large. Second, the 0.127 b/w **splits** into an amax bank and an outlier bank. Roughly 0.07 b/w of the saving comes from recompressing the group-amax stream, which is available to *any* per-group codec and is therefore not champion-specific; only the remaining **~0.077 b/w (the outlier bank)** is exclusive to the champion's 0.5% fp16-outlier design. We report the split rather than the headline 0.127 so the credit is not overstated.

### 3.4 Results table

**Table 1.** Measured 0.5B operating points and frontier (Qwen2.5-0.5B, fp16 eval-slice perplexity 3.944). Bits/weight are honest (round-trip-verified container bytes per weight); perplexity is on the held-out enwik8 prose slice.

| Codec / point | Perplexity | Honest b/w | Notes |
|---|---:|---:|---|
| fp16 reference | 3.944 | 16.000 | eval-slice baseline |
| Champion (operating point) | 4.58 ± 0.07 | ~3.00 | seed-noise floor σ = 0.067; "4.483" is one seed instance |
| Champion (frontier, λ = 0.0015) | 3.957 | 4.277 | Pareto-dominates E8 below |
| Champion (frontier) | 4.044 | 4.069 | interior λ point |
| Champion (entropy-32) | ~3.97 | ~5.14 | near-lossless variant |
| E8 lattice (q0.08) | 4.020 | 4.640 | strictly dominated on both axes |
| ECTCQ (honest rate) | ~3.978 | ~5.13 | +0.95 b/w path entropy restored; loses ~0.85 b/w |

**Side-information re-coding (A1), at identical perplexity:** raw **0.2876 b/w** → decodable container **0.1608 b/w**, banking **0.127 b/w** (lower bound; ~0.011 b/w AWQ scale stream still uncoded). Split: ~0.07 b/w amax bank (available to any per-group codec) + ~0.077 b/w outlier bank (champion-exclusive).

## 4. The Map of Closed Axes

The central scientific result of this work is not the champion codec but the *map
of what fails to beat it*. A single positive result is weak evidence; a champion
that survives roughly thirty attack rounds and several hostile audits, each probing
a structurally distinct axis, is strong evidence that the operating point sits near
a genuine rate-distortion frontier rather than a local artifact of insufficient
search. This section catalogs every axis the loop opened and closed and --- the part
that makes the map scientific rather than anecdotal --- states the *mechanism* by
which each loses under honest, *symmetric* accounting: every candidate and the
champion are charged identically, as the byte size of a container an independent
decoder actually reconstructs and round-trips (Section 2.2), so no axis can win by
being measured more generously than the incumbent.

### 4.1 Data-aware weighting (GPTQ / Hessian error feedback)

Making quantization *data-aware* --- weighting the per-coordinate squared error by
its impact on the output, GPTQ/block-Hessian style --- is the most natural attack.
It is negative, and the mechanism is a direct consequence of the champion's own
front end. After per-group signed-Hadamard incoherence rotation the per-coordinate
sensitivities are *equalized*: each rotated coordinate is a near-uniform mixture of
the column, so |R[j,c]|^2 = 1/g for a group of size g. A weighting scheme can only
help when sensitivities are *unequal*; applied to a whitened source whose
sensitivities are flat, it fits estimation noise in the Hessian rather than real
structure, and the data-aware variant performs *worse* than the data-agnostic ECVQ
it was meant to improve (`quant_dataaware.py`). Rotation does not merely tolerate
the absence of data-aware weighting --- it removes the signal such weighting needs.

### 4.2 Vector quantization (trained 2-D / 4-D ECVQ codebooks)

Replacing scalar ECVQ with a trained multi-dimensional codebook is the textbook way
to capture space-filling gains. The outcome is *conditional* and instructive: a
trained 2-D/4-D codebook posts a small space-filling gain over the scalar quantizer
*when no outliers are used*, but *loses* the moment the champion's 0.5% fp16 outlier
branch is added (`quant_vq.py`). The two ingredients are not additive --- they
compete for the same job. A vector quantizer's advantage comes almost entirely from
how it tessellates the heavy tail; but the outlier branch has already excised that
tail into fp16. On the residual whitened, light-tailed bulk the cell geometry buys
little while its codebook overhead and decode cost go unamortized, so the champion
captures the same gain more cheaply with a scalar grid plus a sparse exact branch.

### 4.3 Lattices (D4 / E8)

Structured lattices (D4, E8) are the canonical high-dimensional grids and were
treated as a first-class threat. They are closed --- but the path to that verdict is
itself a result, because the loop's *first* measurement was wrong in the lattice's
*favor*. The historical R12 metric summed per-coordinate entropies, over-charging
D4/E8 by roughly 0.7-0.8 b/w (the per-coordinate overcount class of Section 2.2): a
structured code packs information jointly across its dimension, so the sum of
marginal entropies exceeds the joint entropy a real decoder pays. We fixed this not
by re-deriving a formula but by routing lattices through the same
round-trip-verified container as everything else (`lattice_rescore.py`). The honest
verdict survives the fix and is unambiguous: a champion point at matched perplexity
--- ECVQ at lambda = 0.0015, **3.957 ppl @ 4.277 b/w** --- *strictly
Pareto-dominates* the best lattice point, E8 at q = 0.08, **4.020 ppl @ 4.640 b/w**,
on *both* axes at once. The mechanism: incoherence rotation has already whitened the
source to near-i.i.d., and a lattice's space-filling gain over a well-shaped scalar
quantizer on a whitened i.i.d. source is small --- smaller than the rate its fixed
cell structure costs relative to an entropy-constrained scalar grid that places
probability adaptively.

### 4.4 Trellis-coded quantization (fixed-rate and ECTCQ)

Trellis-coded quantization is the most sophisticated axis attacked and produced the
sharpest self-correction, so we treat it in two stages. In the *fixed-rate* form
(QTIP/TCQ-style) it lost outright, paying about **5 b/w** for quality the champion
reaches near 3, because a fixed-rate trellis cannot exploit the source's low
entropy. The interesting case is the *entropy-constrained* form (ECTCQ), where the
rate term is folded *inside* the Viterbi branch metric so the search trades
distortion against bits along the trellis. ECTCQ initially *appeared to beat the
envelope*, posting **3.978 ppl @ 4.140 "b/w"** --- a point that would have dominated
the champion. We give its audit and reversal in Section 4.8; the verdict is that
ECTCQ is closed, losing by about 0.85 b/w once charged honestly, for the structural
reason in Section 4.9.

### 4.5 Two-sided incoherence rotation

A two-sided rotation (both row and column bases) enables cheap quantization with
only a few scales per block, proposed to shrink side-information cost. It loses to
the champion's per-row scaling. The trade is rate-allocation: two-sided rotation
buys cheaper scale metadata by discarding the per-row dynamic-range adaptivity that
per-group amax scaling provides. On matrices whose rows differ substantially in
scale, the coarse few-scale-per-block quantization it permits raises distortion by
more than the saved metadata is worth, and per-row scales win the net trade.

### 4.6 RMSNorm-gain absorption

This axis is a *function-preserving canonicalization*: the RMSNorm gain folds
algebraically into the next weight matrix's columns without changing the output ---
exactly invertible, "free," and seemingly a friendlier distribution for a
quantizer. It is killed: folding the gamma *worsens* perplexity by **0.144**
(`rmsnorm_fold.py`). The mechanism is a destructive interaction with the front end.
Per-group rotation and per-group amax scaling both rely on *column-scale
homogeneity* --- groups whose coordinates share a comparable dynamic range.
Absorbing the gain multiplies each column by an arbitrary positive constant,
deliberately *breaking* that homogeneity, so one group-amax can no longer tightly
bound a group now spanning gains of very different magnitude. A transform that is
function-preserving in floating point is not quantization-preserving; the
canonicalization that helps a naive quantizer harms a rotation-plus-amax pipeline.

### 4.7 Index context-coding, QAT, and amax-snapping

Three further axes close briefly. **Index context-coding** (micro entropy models,
permutation gauges, conditional coders over the index stream) is dead because the
indices are *already* near-i.i.d. after rotation: a conditional-entropy oracle with a
train/code layer split finds every context yields a *negative* bank (D_idx = 0,
triple-verified — meaning no positive conditional-coding bank was found, not literal
exact independence), and both zstd-19 and the context-mixing `cmcore` on the full
stream do *worse* than an order-0 coder (`ecvq_idx_oracle.py`). The citable
statement: incoherence rotation whitens the index stream to near-i.i.d., leaving no
conditional structure to exploit. **QAT** (straight-through fine-tuning on uniform,
rotated, and NF grids) only *ties* the PTQ codec; the operative finding is that the
*grid* matters more than retraining --- naive RTN QAT plateaus near **6.4 ppl**, a
rotated NF grid reaches **~4.6 ppl**, which entropy-constrained PTQ already attains
with *no* training --- and at 0.5B, QAT is overfitting-limited (`quant_qat.py`).
**Amax-snapping** (lossy rounding of the group-amax stream) merely slides along the
*existing* frontier, reproducing the trade the ECVQ lambda knob already exposes, and
yields no free win beyond the lossless bank the A1 container already recovered
(`amax_snap.py`).

### 4.8 Self-correction: two accounting bugs the loop caught on itself

The methodology's strongest evidence is that the loop twice reversed its *own*
favorable conclusions after its accounting was found wrong. Both bugs were
asymmetries in rate measurement --- the exact failure mode the round-trip gate
exists to prevent --- and both were caught by hostile audit, not by the proposer.

The **first** is the lattice overcharge of Section 4.3, an *overcount*: the R12
per-coordinate-entropy metric charged D4/E8 about 0.7-0.8 b/w *too much*. This
biased *against* a candidate, so the correction was to *amnesty* the lattices,
re-score them honestly, and only then confirm they still lose. We flag it because
honest accounting must cut both ways: a verifier that catches only optimistic bugs
would itself be a biased referee.

The **second** is the ECTCQ branch-bit *undercount* of Section 4.4, the sharper
episode. The apparent envelope-beating point (**3.978 ppl @ 4.140 "b/w"**) was
*self-flagged as suspicious* and sent to a **six-agent audit**: three independent
rate derivations, two independent code audits, and a brute-force dynamic-programming
check of the achievable rate. The verdict was subtle, which is why the bug was
non-obvious: the *reconstruction was correct* --- the trellis really did achieve
that distortion --- but the reported rate *omitted H(coset | current-state)*, the
branch/path bit a real decoder must receive to follow the Viterbi path. That omitted
term was *measured* at about **0.95 b/w**. Adding it back gives a corrected honest
rate of about **5.13 b/w**, so ECTCQ in fact *loses* by about **0.85 b/w**. The loop
reversed its conclusion on the audit's strength, not the original proposal
(`ectcq.py`).

### 4.9 The structural trellis lesson

The ECTCQ reversal generalizes into a structural principle that explains *why* the
entire trellis axis is closed, not merely this instance. A near-balanced four-state
trellis must pay roughly **0.95 b/w of path entropy** --- telling the decoder which
branch was taken at each step --- to purchase at most about **0.25 b/w of
space-filling gain** on a whitened-i.i.d. source. The trade is a *guaranteed* loss
by construction, independent of implementation: the gain is bounded by the small
shaping advantage available on an already-whitened source, while the cost is path
entropy the decoder *cannot* avoid. The deeper lesson, the one that retired the
axis, is that pricing the branch *inside* the Viterbi metric --- the very move that
made ECTCQ look like a winner --- does *not* make the decoder's branch bit free.
Folding rate into the search objective changes which path the encoder selects; it
does not change the information the decoder must be sent to reconstruct that path.
Any honest container carries that bit, and once it does, the trellis is dominated.
This is the same insight, in a different costume, that closes the lattice axis
(Section 4.3) and the vector-quantization axis (Section 4.2): once incoherence
rotation has whitened the source, the structural gains that motivate
high-dimensional and stateful quantizers collapse to almost nothing while their
structural *costs* remain --- so an entropy-constrained scalar quantizer with a
sparse outlier branch is, on this source, hard to beat.

## 5. Verifier Characterization and the Scale Law

The credibility of every Pareto claim in this paper rests on two questions that
are easy to assume away and hard to answer honestly. First: *is the verifier
measuring what it purports to measure* — does a single held-out perplexity number
faithfully order codecs by rate–distortion quality, and how large is the noise on
that number? Second: *does the result survive a change of model* — is the
champion an artifact of one small network, or does its advantage behave lawfully
with scale? This section answers both with measurements. Subsection 5.1
characterizes the verifier through three controls (A2, C1, R2) that bound its
heterogeneity, its faithfulness, and its noise floor. Subsection 5.2 reports the
scale law and is candid about the one place where a *secondary* signal does not
behave monotonically.

### 5.1 Verifier characterization: heterogeneity, faithfulness, and the noise floor

**A2 — slice heterogeneity and the paired-deterministic design.** The held-out
signal is perplexity on a fixed prose slice of enwik8, but enwik8 is not a
homogeneous corpus. Across its slices the fp16 perplexity ranges from roughly
**2.2 to 17.2**, with some slices dominated by markup rather than prose. A naive
reading would treat this spread as a defect of the verifier; it is instead the
reason the arena is built the way it is. Because the slice and the rotation seed
are *fixed*, the arena is **deterministic and paired**: when we compare two
codecs we compare them on the *same* slice with the *same* seed, so the
comparison is a within-pair difference and the large between-slice variance
cancels. Codec-vs-codec margins are therefore *decidable on that slice*. What the
pairing does **not** buy is content generalization: because content
heterogeneity is real, a margin established on one prose slice is only guaranteed
on that content, and cross-content generalization is content-dependent. A2 thus
fixes the scope of every comparison in the paper — margins are claims about a
fixed paired condition, not unconditional claims across all text.

**C1 — the divergence battery (single-slice perplexity is a faithful proxy).**
A2 makes margins decidable but raises a fair worry: a single slice might rank
codecs differently from a broader, multi-domain distortion measure. C1 tests this
directly. We took **8 principled codecs** spanning **2.4 to 5.3 bits/weight** and
correlated, across them, the rank induced by single-slice perplexity against the
rank induced by a *worst-of-six-domain* KL-divergence battery (the adversarial
summary statistic — each codec judged by its worst domain). The rank correlation
is **Spearman = 1.000**: the single-slice perplexity ordering and the
worst-domain KL ordering agree perfectly over these codecs. We read this
narrowly and correctly — *among principled codecs*, single-slice perplexity is a
**faithful rate–distortion proxy**, and even the aggressive 2.4-bit point
degrades *uniformly* across domains rather than collapsing on one. C1 does not
claim faithfulness for pathological or adversarial codecs outside this set; it
licenses the proxy for the kind of principled codecs the loop actually compares.

**R2 — the seed-noise floor and the limits of single-slice selection.** The
sharpest control is a *positive* one: hold everything fixed at iso-bit and vary
**only the rotation seed**. This isolates selection noise, because every such
variant is the same codec at the same rate. The measured **noise floor is
standard deviation 0.067 perplexity, with a spread of 0.21** across seeds. This
single number is load-bearing throughout the paper: it is why the historical
headline "4.483 perplexity" is reported as one seed instance rather than a sharp
value (Sections 2–3), and it sets the **decidable margin**: differences above
roughly **0.2 perplexity** are safe, while **sub-0.1-perplexity orderings are
within noise and must be reported as ties**. By this rule the load-bearing
results are all safe — the champion's gap, the lattice Pareto margin (0.19–0.36
b/w), and ECTCQ's −0.85 b/w loss exceed it; the QAT-vs-PTQ comparison is reported
as a *tie* precisely because it does not.

R2 also delivers a cautionary result about transferability. Correlating
worked-slice perplexity against a *separate* prose-escrow slice gives only
**Spearman = 0.324**: selecting a rotation seed by its single-slice score does
**not transfer** to held-out prose — seed selection is mostly noise, not signal.
This carries an important **nuance**. The 0.324 figure is an **upper bound on the
noise**, because it is *unpaired*: it compares performance across two different
slices, so it absorbs both seed noise and content shift. The historical
codec-vs-codec comparisons that the paper relies on were instead run with a
**fixed seed (paired)**, where the seed term cancels and the effective noise is
*smaller* than R2's unpaired estimate. The implication is twofold and
asymmetric: one must **never** chase a seed-level perplexity advantage (it does
not generalize), yet the paired margins used for the actual codec rankings are
*more* trustworthy than the unpaired 0.324 would suggest. R2 therefore both
disciplines the headline (no sharp seed values) and protects the comparisons (the
real margins clear the decidable threshold under the tighter paired noise).

### 5.2 The scale law

The external-validity question is whether the champion's advantage is specific to
a tiny model. It is not. Measuring the champion's **gap to fp16 at ~3.13
bits/weight** across the Qwen2.5 family, the gap **shrinks monotonically with
model size**:

**Table 2.** Champion gap to fp16 at ~3.13 b/w across model scale.

| Model | fp16 eval-slice ppl | Champion gap to fp16 (ppl) |
|---|---:|---:|
| Qwen2.5-0.5B | 3.944 | **+0.539** |
| Qwen2.5-1.5B | 3.128 | **+0.345** |
| Qwen2.5-3B | 2.947 | **+0.244** |

The gap contracts by roughly **30–35% per step** (0.539 → 0.345 → 0.244), a
consistent rate that extrapolates to **near-lossless quantization at ~7B** — an
extrapolation we flag as such, since the study does not include a 7B point. By
**3B the codec is near-lossless**: the gap is only **+0.12 perplexity at ~3.5
bits/weight**, and the near-lossless **entropy-32** variant is **lossless within
noise**, with a measured gap of **−0.004 perplexity at 5.27 b/w** (a negative
sign that is itself inside the 0.067 noise floor and should be read as "lossless,"
not "better than fp16"). The mechanism is consistent with the standard picture
that quantization error is increasingly forgiven as representational redundancy
grows with scale, and it is the strongest piece of external validity we have for
the claim that the champion is a genuine codec rather than a 0.5B curiosity.

**An honest caveat — naive RTN is non-monotonic.** We are careful to separate
this clean law from a *secondary* observation that does **not** form a clean law
and must be reported as model-specific. A naive baseline — unrotated,
no-outlier round-to-nearest at 3 bits (RTN-3b) — **collapses at 3B**, reaching a
perplexity of **46,036**, exactly where the champion stays robust. This is
*consistent* with outlier features growing in importance with scale (the champion
handles them by construction; naive RTN does not), and it is a useful contrast
because it shows the champion's robustness is doing real work at scale. But the
naive baseline's own trend across sizes is **non-monotonic** — its perplexity
goes **68 → 12 → 46,036** from 0.5B to 1.5B to 3B — so we explicitly decline to
present it as a scaling law in its own right. The disciplined reading is: the
*champion's* gap-to-fp16 shrinks monotonically and lawfully (the headline
result), while the naive-RTN collapse at 3B is a model-specific data point that
motivates outlier handling rather than a second trend line. Reporting it any other
way would over-read three points of a pathological baseline.

## 6. Related Work

Our work sits at the intersection of two literatures that rarely meet: modern
post-training quantization (PTQ) of large language models, and LLM-driven program
search. We position the champion codec against the first and the *method* --- an
LLM as the in-loop mutation operator --- against the second. Throughout, we are
careful to claim novelty only for the application and the verifier discipline, not
for the underlying quantization primitives, which are classical.

### 6.1 Incoherence-rotation and codebook PTQ

The closest technical lineage is the family of methods that precondition weight
matrices with an incoherence-inducing rotation before quantizing. **QuIP** and
**QuIP#** establish that random orthogonal (and later structured Hadamard)
transforms make weight matrices *incoherent* --- spreading magnitude evenly so no
single coordinate dominates --- which is precisely the property our champion's
per-group signed-Hadamard rotation exploits, and which underlies the central
mechanism in our negative map: once the source is whitened to near-i.i.d., the
per-coordinate sensitivities equalize and the structural gains of higher-order
quantizers collapse (Section 4). QuIP# additionally couples Hadamard incoherence
with an E8-lattice codebook; this is exactly the design our lattice axis evaluates
and closes under symmetric accounting, where a champion point at **3.957 perplexity
/ 4.277 b/w** strictly Pareto-dominates an E8 point at **4.020 / 4.640 b/w** on
both axes (Sections 3-4). **AQLM** (additive quantization) and **VPTQ**
(vector PTQ) push the codebook direction further, representing weights as sums or
products of learned vector-quantizer codewords and reaching aggressive sub-3-bit
rates. **GPTVQ** generalizes GPTQ-style error feedback to vector quantization.
These methods are strong and, at scale, competitive; our contribution is *not* to
claim the scalar champion beats them universally, but to show that *on a whitened
i.i.d. source* the marginal space-filling advantage of vector and lattice
quantizers is small relative to their structural cost, so an entropy-constrained
scalar quantizer with a sparse outlier branch is hard to beat under honest
accounting --- a finding consistent with, not contradictory to, the broader VQ
literature, since our trained 2-D/4-D ECVQ codebooks *do* post a small
space-filling gain until the outlier branch absorbs the same heavy tail
(Section 4.2).

**QTIP** is the most direct point of contact for our trellis axis: it pairs
incoherence processing with a *trellis-coded* quantizer, using the trellis to
amortize a high-dimensional codebook cheaply. Our work evaluates both fixed-rate
trellis quantization (which lost outright at about **5 b/w**) and an
entropy-constrained variant (ECTCQ), and the ECTCQ episode is where our accounting
discipline did its sharpest work: the method initially *appeared* to beat the
envelope at **3.978 perplexity / 4.140 "b/w"** before a six-agent audit found that
the reported rate omitted the decoder's branch bit, **H(coset | state) ≈ 0.95 b/w**,
restoring an honest rate near **5.13 b/w** and a loss of about **0.85 b/w**
(Sections 4.4, 4.8-4.9). We read this not as a refutation of QTIP --- which
reports its own rates honestly --- but as a cautionary structural result: a
near-balanced four-state trellis pays roughly **0.95 b/w** of unavoidable path
entropy to buy at most about **0.25 b/w** of shaping gain on an already-whitened
source, so on *this* source the trade is a guaranteed loss, and pricing the branch
inside the Viterbi metric does not make the decoder's branch bit free.

### 6.2 Activation-aware and error-feedback PTQ

A second cluster makes quantization *data-aware*. **GPTQ** quantizes columns
greedily while propagating the induced error through an inverse-Hessian update, and
**AWQ** rescales weight channels by activation statistics so that salient channels
survive quantization; our champion borrows AWQ's per-column activation scaling
directly as a front-end ingredient. **SpQR** isolates a sparse set of outlier
weights at high precision and quantizes the dense remainder, a decomposition our
0.5%-fp16-outlier branch echoes in spirit. The instructive divergence is with the
*Hessian* half of this lineage: we find that after incoherence rotation, data-aware
error weighting --- GPTQ-style or block-Hessian-weighted ECVQ --- is *negative*,
because the rotation equalizes per-coordinate sensitivities (|R[j,c]|² = 1/g) and
leaves no unequal weighting for the Hessian to exploit, so data-aware weighting
fits estimation noise rather than structure (Section 4.1). This is not a claim that
GPTQ is wrong; it is the observation that GPTQ's mechanism and aggressive
incoherence rotation are partial substitutes, and composing both adds little once
the source is whitened. AWQ's activation scaling, by contrast, *does* survive in
the champion, because it addresses a different problem (channel salience) than the
one rotation solves (coordinate coherence).

### 6.3 Entropy-constrained quantization

The champion's distortion-rate core is **entropy-constrained scalar quantization**
in the **Chou-Lookabaugh-Gray (1989)** sense: assign each coordinate to the level
minimizing (x − cₖ)² − λ·log₂ pₖ, with the levels cₖ and probabilities pₖ learned
per layer and the resulting indices entropy-coded. This is a classical
information-theory technique --- the ECVQ Lagrangian is more than three decades old,
and the lattice and trellis comparators (D4/E8 packing, trellis-coded quantization,
ECTCQ) are likewise textbook constructions. We are explicit that *we did not invent
ECVQ*; our claim is that entropy-constrained scalar quantization, composed with
incoherence rotation and a sparse outlier branch, is a strong and *novel-for-LLM*
codec, and that the rotation is what makes the scalar form competitive by whitening
the index stream to near-i.i.d. --- a property we verify directly, finding the
index-conditional bank is non-positive across all tested contexts (D_idx = 0,
triple-verified; both zstd-19 and a context-mixing coder do *worse* than order-0 on
the full stream; Section 4.7).

### 6.4 LLM-driven program search

The *method* descends from **FunSearch** and **AlphaEvolve**, which use an LLM to
propose program mutations inside an evolutionary loop guided by an automatic
evaluator, and which have produced novel constructions in mathematics and
algorithm design. We adopt the same template --- the LLM (Claude) is the mutation
operator that writes new codec code each round into an evolvable registry
(`codec_zoo.py`), scored by a resumable arena (`quant_arena.py`) --- and apply it
to a domain those works did not target: discovering PTQ codecs. The decisive
difference from a generic AlphaEvolve setup is the **verifier**. Because the search
objective is *compression*, the evaluator is adversarially easy to game (hidden
side information, optimistic joint-entropy estimates, undeployable decoders), so our
verifier scores rate as the byte size of a container an independent decoder
actually round-trips, with no side information outside the count, plus a decode-cost
class. This un-gameable verifier is what makes the negative map trustworthy: every
axis is charged identically to the incumbent, and the loop even caught two of its
*own* bit-accounting bugs --- the lattice overcount (~0.7-0.8 b/w) and the ECTCQ
branch-bit undercount (~0.95 b/w) --- via hostile multi-agent audit, reversing its
own conclusions (Section 4.8). To our knowledge this is the first demonstration of
an LLM as the in-loop mutation operator discovering PTQ codecs against a cheap,
un-gameable verifier, with documented self-correction.

## 7. Limitations

We are deliberately candid about scope; several of these limitations bound exactly
how far the results should be read.

**Scale ceiling.** The study spans **Qwen2.5 at 0.5B, 1.5B, and 3B** only; we have
no 7B-or-larger point. The scale law is therefore an *extrapolation* beyond its
support: the champion's gap to fp16 at ~3.13 b/w shrinks monotonically (**+0.539 →
+0.345 → +0.244** perplexity from 0.5B to 1.5B to 3B, roughly 30-35% per step) and
*extrapolates* to near-lossless near 7B, but we flag the near-lossless-at-7B claim
as a projection, not a measurement (Section 5.2). Whether the frontier orderings
themselves --- not just the gap magnitude --- hold at larger scale is untested.

**The perplexity proxy.** Distortion is held-out perplexity on a disjoint enwik8
prose slice. We validate this proxy against a multi-domain KL battery, where it is a
*faithful* rate-distortion ranking among principled codecs (Spearman = 1.000 over 8
codecs spanning 2.4-5.3 b/w; Section 5.1), but it is *not* a full downstream-task
suite (e.g., zero-shot reasoning, code, or long-context benchmarks). A codec that is
near-lossless in held-out perplexity could still shift downstream task accuracy in
ways perplexity does not surface; our claims are scoped to the rate-distortion proxy
we measured, on the paired, deterministic slice on which margins are decidable.

**No fused decode kernel.** The headline **~3.00 b/w** operating point requires
entropy decoding of the index stream, and we provide *no* production fused
entropy-decode GPU kernel. The ~3.00 b/w figure is therefore an honest *storage*
rate, not a demonstrated end-to-end serving speedup; making it deployable at speed
needs kernel work we did not do. We stress the mitigating fact: a **"simple-decode"
variant at ~3.5 b/w** --- rotation plus ECVQ levels, with *no* entropy coder, so
the index stream is plain LUT-decodable --- is deployable *today* on standard
dequantization paths. At 3B the champion codec is already near-lossless at this rate
(**+0.12 perplexity by ~3.5 bits**; Section 5.2); the simple-decode variant shares
that rate, but its 3B gap was not separately measured. The entropy-coded champion buys
roughly half a bit per weight over this deployable variant, and realizing that gain
in a serving system remains future work.

**Single model family.** All results are on **Qwen2.5**. The incoherence-rotation +
AWQ + ECVQ + outlier recipe is architecture-agnostic in principle, but we have not
verified it on a second family (e.g., Llama, Mistral, or a mixture-of-experts
model), and family-specific weight statistics could shift the operating point or the
relative ordering of closed axes.

**Recombination, not a new primitive.** The champion is a *recombination of
classical information-theory components* --- Hadamard incoherence, AWQ scaling,
Chou-Lookabaugh-Gray ECVQ (1989), entropy-coded indices, sparse fp16 outliers ---
not a new quantization primitive. ECVQ is a known technique applied in a novel
setting (LLM weight quantization); the scientific contribution is the *map* of which
compositions survive honest accounting and the demonstration of an LLM discovering
that map, not the invention of a codec. Relatedly, the single *new positive* result
(A1) is a **lossless side-information re-coding** that banks **0.127 b/w at
identical perplexity**, of which only **~0.077 b/w** (the outlier bank) is
champion-exclusive --- the remaining ~0.07 b/w (the amax bank) is available to any
per-group codec, and we report the split rather than the headline so the credit is
not overstated (Section 3.3).

## 8. Conclusion

We asked whether an LLM, used as the in-loop mutation operator of an
AlphaEvolve-style search, can discover a competitive PTQ codec when scored by a
cheap, un-gameable verifier --- and what the *structure* of the resulting frontier
is. The answer is a qualified yes with an unusually well-characterized boundary.
Across roughly 30 attack rounds and multiple hostile multi-agent audits, the loop
converges on a single champion --- per-group signed-Hadamard incoherence rotation,
AWQ activation scaling, entropy-constrained scalar quantization (ECVQ),
entropy-coded indices, and 0.5% fp16 outliers --- that operates on Qwen2.5-0.5B at
approximately **3.00 honest bits/weight at perplexity ~4.58 ± 0.07**, and that no
attacked axis dethrones under symmetric accounting.

The central scientific result is not the champion but the *map of closed axes* and
the *mechanism* unifying them. Data-aware weighting, vector quantization, lattices
(D4/E8), and trellis coding (fixed-rate and ECTCQ) all fail to beat the champion for
one recurring reason: incoherence rotation whitens the weight source to
near-i.i.d., after which the per-coordinate sensitivities equalize and the
structural gains that motivate higher-order quantizers collapse to almost nothing
while their structural costs --- codebook overhead, lattice cell rate, ~0.95 b/w of
trellis path entropy --- remain. A champion point at **3.957 perplexity / 4.277
b/w** strictly Pareto-dominates the best E8 lattice point (**4.020 / 4.640 b/w**),
and ECTCQ, once its branch bit is paid, loses by about **0.85 b/w**. Equally
important is what the methodology guarantees: because rate is measured as the byte
size of a round-trip-verified container, the loop caught *two of its own*
accounting bugs --- a lattice overcount and an ECTCQ branch-bit undercount --- and
reversed its own conclusions on hostile audit, which is the strongest evidence we
can offer that the frontier is real rather than an artifact of generous bookkeeping.
The accounting also yielded one new positive result, a lossless side-information
re-coding banking **0.127 b/w at identical perplexity** (a lower bound; only
~0.077 b/w champion-exclusive). And the advantage behaves
lawfully with scale: the gap to fp16 shrinks monotonically (**+0.539 → +0.345 →
+0.244** perplexity from 0.5B to 1.5B to 3B), reaching near-lossless (**+0.12
perplexity by ~3.5 bits**) at 3B.

We close on the discipline rather than the codec. The recurring lesson of this work
is that in compression research the *verifier*, not the proposer, is where rigor
lives: an objective that can be gamed will be gamed, including by a well-meaning
search that fools itself, and the only durable defense is to charge every candidate
the bytes a real decoder must receive. The champion is a recombination of
thirty-year-old components, and ECVQ is applied here in a novel setting rather than
invented; the lasting contributions are the demonstration that an LLM can drive such
a search with documented self-correction, and the honestly-measured map of where the
rate-distortion frontier for ~3-bit LLM weights actually lies. The natural next steps
follow directly from the limitations: a fused entropy-decode kernel to make the
~3.00 b/w point deployable at speed (the ~3.5 b/w simple-decode variant is already
deployable), a 7B-and-larger scale point to test the extrapolation, a second model
family, and a downstream-task evaluation to corroborate the perplexity proxy. We
release the evolvable registry, the resumable arena, and the round-trip-verified
containers (`codec_zoo.py`, `quant_arena.py`, `scale_plane_codec.py`,
`lattice_rescore.py`, `ecvq_idx_oracle.py`, `ectcq.py`) so that the map --- and the
honest-accounting gate that makes it trustworthy --- can be extended and attacked
further.
