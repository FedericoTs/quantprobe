# 5. Verifier Characterization and the Scale Law

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

## 5.1 Verifier characterization: heterogeneity, faithfulness, and the noise floor

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

## 5.2 The scale law

The external-validity question is whether the champion's advantage is specific to
a tiny model. It is not. Measuring the champion's **gap to fp16 at ~3.13
bits/weight** across the Qwen2.5 family, the gap **shrinks monotonically with
model size**:

**Table 5.** Champion gap to fp16 at ~3.13 b/w across model scale.

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
