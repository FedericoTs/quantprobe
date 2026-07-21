# 6. Related Work

Our work sits at the intersection of two literatures that rarely meet: modern
post-training quantization (PTQ) of large language models, and LLM-driven program
search. We position the champion codec against the first and the *method* --- an
LLM as the in-loop mutation operator --- against the second. Throughout, we are
careful to claim novelty only for the application and the verifier discipline, not
for the underlying quantization primitives, which are classical.

## 6.1 Incoherence-rotation and codebook PTQ

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

## 6.2 Activation-aware and error-feedback PTQ

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

## 6.3 Entropy-constrained quantization

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

## 6.4 LLM-driven program search

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

# 7. Limitations

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

# 8. Conclusion

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
re-coding banking **0.127 b/w at identical perplexity**. And the advantage behaves
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
