# 4. The Map of Closed Axes

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

## 4.1 Data-aware weighting (GPTQ / Hessian error feedback)

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

## 4.2 Vector quantization (trained 2-D / 4-D ECVQ codebooks)

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

## 4.3 Lattices (D4 / E8)

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

## 4.4 Trellis-coded quantization (fixed-rate and ECTCQ)

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

## 4.5 Two-sided incoherence rotation

A two-sided rotation (both row and column bases) enables cheap quantization with
only a few scales per block, proposed to shrink side-information cost. It loses to
the champion's per-row scaling. The trade is rate-allocation: two-sided rotation
buys cheaper scale metadata by discarding the per-row dynamic-range adaptivity that
per-group amax scaling provides. On matrices whose rows differ substantially in
scale, the coarse few-scale-per-block quantization it permits raises distortion by
more than the saved metadata is worth, and per-row scales win the net trade.

## 4.6 RMSNorm-gain absorption

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

## 4.7 Index context-coding, QAT, and amax-snapping

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

## 4.8 Self-correction: two accounting bugs the loop caught on itself

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

## 4.9 The structural trellis lesson

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
