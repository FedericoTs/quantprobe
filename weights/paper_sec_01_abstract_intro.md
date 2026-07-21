# Abstract

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

# Introduction

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
