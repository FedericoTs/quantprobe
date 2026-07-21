# ROADMAP R25 — Consolidated Research Plan for the evo-compress Weight-Codec Loop

Date: 2026-06-10. Status: post-R24 consolidation. Champion: per-group(g=128) signed-Hadamard +
AWQ scaling + scalar ECVQ + entropy-coded indices + 0.5% fp16 outliers
(0.5B: 4.483 ppl @ 3.13 b/w, 4.169 @ 3.91; 1.5B: 3.473 @ 3.135, 3.240 @ 3.904).

## Context preamble

**Where the loop stands.** Twenty-four rounds of LLM-mutated codec evolution have produced a
champion that resisted 12 distinct attack families (lattices, VQ, TCQ, GPTQ fusion, QAT,
sensitivity allocation, noise shaping, low-rank correction). The established facts are sharp:
post-rotation weights are near-iid Gaussian per group; entropy coding is ~0.4 free bits on any
quantizer; the outlier lever captures what VQ space-filling would; quantization gets easier with
scale; grids beat retraining. But three blind spots have emerged from hostile review of 30 new
candidates: (1) the champion's own *accounting* is leaky — ~0.30 b/w of side info (amax fields,
outlier positions, AWQ scales) is paid raw, and at least one historical negative (E8 lattices) was
rendered under a rate metric that overcharges by ~1.0 b/w; (2) the verifier is a single-slice
teacher-forced ppl monoculture, exactly the configuration the 2024-26 literature
(Dutta et al., LLM-KICK) shows can hide divergence — and 24 rounds of hill-climbing on one slice
have never been audited for selection overfitting; (3) the loop has been single-plane (weights)
and single-model, while our unique assets (cmcore context mixer, wcodec delta codec, three model
variants on disk) point at unexploited adjacent planes.

**Why these 20.** Selection maximized expected information gain weighted by the hostile-review
scores (all 19 ideas scoring >= 6.0 are kept, plus the Sinkhorn ablation at 5.5 for its 1-hour
gate and decode-cost axis), under three hard constraints: coverage of seven distinct axes
(context/entropy-coding fusion; meta-evolution of policies/transforms; symmetry canonicalization;
verifier/objective redesign; hybrid lossless guarantees; adjacent-domain transfer; decode-aware),
eight ideas tagged novel-to-science with surviving partial novelty vs literature, and more than
half immediately runnable on the GTX 1060 harness (all of Tier A is hours-to-one-day, mostly
CPU). Ten candidates were cut, but their salvageable levers were absorbed: the entropy-coded
outlier mask from Outlier Herding lives in EscapeRules/Scale-Plane; LambdaForge's closed-form
Stage-0 allocator becomes the Role-Dispatch oracle pre-test; the Wyner-Ziv conditional-gain
measurement becomes a deliverable of Family Superdictionary; the Bits-Back/REC hypothesis is
tested for free by the Dither A/B; CellMap survives as a frozen 3-class decode-cost reporting
column plus a one-round LL-Huffman-ECVQ candidate inside the composer's op library; GaugeZip and
Balanced-Factorization collapse to the closed-form pre-checks embedded in Anti-iid Sculpting and
Sinkhorn.

**How they were generated and verified.** Each candidate was produced by domain scouts over
2024-2026 literature, then subjected to an independent hostile review that (a) checked
distinctness against the 24-round tested list, (b) searched for refuting prior art with specific
arXiv ids, (c) attacked the verifier for gaming channels (the project has caught both
joint-entropy undercounting and fixed-rate overcounting before — this culture is a moat), and
(d) re-derived the feasibility arithmetic for the 1060. Every spec below incorporates its
verdict's mandated improvement verbatim: pre-registered gates before builds, real-bitstream
round-trip rates instead of formulas, lever-matched and noise-floor-aware comparisons, and
degenerate-optimum patches to fitness functions. House rules apply throughout: codecs fully
vectorized torch/numpy; all side info counted; honest bits = bytes of an actually-decoded stream.

## Summary table

| # | Name | Axis | Novelty | Expected gain | Tier |
|---|------|------|---------|---------------|------|
| 1 | Scale-Plane Side-Info Codec | context/entropy fusion + hybrid lossless | novel-application | solid (near-certain ~0.08-0.15 b/w) | A |
| 2 | EscapeRules (encoding fix, then selection) | meta-evolution + entropy fusion | novel-application | marginal-but-certain (~0.055 b/w + upside) | A |
| 3 | Lattice Rate Amnesty | entropy fusion / rate-metric audit | novel-application | breakthrough-possible | A |
| 4 | TrellisForge (ECTCQ) | context/entropy fusion (frontier) | untried-published | solid (+0.08-0.15 b/w possible) | A |
| 5 | Evolved-Context Micro Entropy Model | context/entropy fusion + meta-evolution | novel-to-science (automation) | solid | A |
| 6 | Partial-Mixing Incoherence (butterfly depth) | decode-aware + meta-evolution | novel-to-science (measurement) | marginal + decisive curve | A |
| 7 | Zipf-Weighted Embedding ECVQ | objective redesign (whole-model rate) | novel-application | solid (coverage) | A |
| 8 | Seeded Subtractive Dither + matched-MSE A/B | objective science (distortion validity) | novel-application | marginal, decisive either way | A |
| 9 | KV-Cache ECVQ + cmcore | adjacent-domain transfer | novel-to-science (partial) | solid (second track) | B |
| 10 | Family Superdictionary | adjacent-domain + hybrid lossless-lossy | novel-to-science (partial) | solid | B |
| 11 | RecoveryForge | meta-evolution of objectives | novel-application | solid | B |
| 12 | Role-Dispatch Pipeline Composer | meta-evolution (pipeline structure) | novel-application | solid | B |
| 13 | RotForge (rate-objective transforms) | meta-evolution of transforms | novel-to-science (objective) | marginal | B |
| 14 | Embedded ECVQ (successive refinement) | context/entropy fusion (capability) | novel-application | solid (comparative claim) | B |
| 15 | Anti-iid Gauge Sculpting | symmetry canonicalization + entropy fusion | novel-to-science (objective) | marginal | B |
| 16 | DeltaFunSearch | adjacent-domain + hybrid lossless | novel-to-science (combination) | marginal, zero GPU | B |
| 17 | Variance-Field Factorization (Sinkhorn) | symmetry/canonicalization + decode-aware | novel-application | marginal + 1-h decisive gate | B |
| 18 | Divergence Battery Verifier + Sealed Escrow | verifier redesign | untried-published | solid (near-mandatory) | C |
| 19 | Red-Team Coevolution | verifier redesign (adversarial) | novel-to-science (combination) | solid (for sub-3-bit era) | C |
| 20 | Free-Running Drift Verifier | verifier redesign (autoregressive) | novel-application | marginal, nearly free | C |

---

# TIER A — run first, highest expected gain

## A1. Scale-Plane Side-Info Codec (non-contiguous variance clustering + context-coded side info)

**Mechanism.** The champion pays 16 raw bits per g=128 amax (0.125 b/w) and 32 bits per scattered
outlier (0.16 b/w) — ~0.29 b/w of side info coded at zero compression. The hostile review
*measured* the real amax statistics on Qwen2.5-0.5B (the originally cited 5.0-bit entropy figure
was circular): per-tensor order-0 entropy of the fp16 amax field is ~10.2 bits, so pure lossless
re-coding banks ~0.046 b/w; an arithmetic-coded outlier bitmap at H2(0.005) ~ 9.1 b/outlier plus
a fitted heavy-tail value prior banks another ~0.035-0.06 b/w — a zero-risk floor of ~0.08-0.10
b/w at provably identical ppl. The lossy upside: cluster every group's amax into per-tensor
log-domain k-means bins (V=16-64) and snap **up to the bin ceiling, never down** — quantile bins
have mean snap error of 1.6-2.8% but MAX error up to 700% in the log-scale tails, and snapping
down pushes normalized values outside the level grid (clipping blowups in rare sensitive groups);
an ESCAPE code keeps exact fp16 amax for the ~0.1% of groups beyond the last centroid. Snapping
happens BEFORE ECVQ level fitting so reconstruction is byte-consistent. The id plane is then
context-coded with cmcore: the UNROTATED scale field retains the spatial structure Hadamard never
touched (row neighbor, column position, layer role, previous id) — the one place a context mixer
profits where indices cannot. Clustering also homogenizes group variance, slightly improving the
ECVQ level fit.

**What is evolved.** The scale-plane codec in `codec_zoo.py`: clustering method (log-domain
k-means, V), the cmcore context configuration for the id stream, the outlier-bitmap and
value-prior coders, escape-code policy. The weight-index codec is untouched; snap direction is
fixed (up) by the harness.

**Verifier (+anti-gaming).** Bits = byte length of the fully serialized, round-trip-DECODED
container (bin centers, context-program description, priors, all tables counted) — never
component-wise estimates, never model-estimated cross-entropy from a possibly non-causal context.
The pure re-coding variant must pass a byte-identical reconstruction check (ppl provably
unchanged); the snapping variant is forced through the held-out ppl gate.

**Experiment plan.** Module `weights/scale_plane.py` + registry entry in `codec_zoo.py`.
1-2 rounds. Runtime: one afternoon of CPU (k-means on ~3.9M scalars is seconds; cmcore is the
existing Rust coder) + 1-2 confirmation arena evals (30-60 s each).

**Success criterion.** >= 0.08 b/w saved at identical (re-coding) or statistically unchanged
(snapping) ppl; stretch 0.10-0.15 b/w (champion 4.483@3.13 -> ~4.483@3.00).

**Kill criterion.** Realized saving < 0.06 b/w after implementation, or any reconstruction
mismatch — then ship only the trivially-safe parts (bitmap + value coding) and close.

**Prior art note.** SpQR (bilevel quantized scales), QLoRA double quantization, llama.cpp
k-quants 6-bit scales, DeepCABAC/MPEG-NNR significance maps, LO-BCQ. Every ingredient exists; the
combination (non-contiguous log-scale clustering + context-mixing coder on the unrotated scale
field + arithmetic outlier bitmap inside an honest-rate frontier) appears unpublished.
Partly removes self-imposed accounting inefficiency — necessary for fair comparison to published
baselines.

---

## A2. EscapeRules (evolved outlier selection + escape encoding — encoding fix lands first)

**Mechanism.** The 0.5% global top-|w| fp16 escape at flat 32 b/outlier is the one champion knob
never swept in 24 rounds; all tested negatives changed the quantizer, never the escape set or its
encoding, and outliers escape PRE-rotation so the R15 whitening negative does not transfer. Per
the verdict, this is split into two stages. Stage 1 (deterministic, round-1 re-baseline): replace
flat 32-bit storage with an actually-arithmetic-coded position bitmap at its true iid cost
(H2(p)/p ~ 9.1 b/outlier at p=0.005 — note log2(1/p) analytic accounting would re-introduce the
historical undercount bug class) plus tail-conditional value coding swept at 8/10/12 bits — a
near-deterministic ~0.055 b/w (32 -> ~21 b/outlier). Stage 2: evolve ONLY the selection policy
against that strengthened baseline — scoring rules over (|w|, activation energy, output-row
saliency, predicted post-rotation residual) and structured variants, with output-row saliency
explicitly legal because rotation whitens input coordinates, not row importance. Whole-row fp16
escape cells are dropped from the archive (14 kbits/row can never win at iso-bits). Before
trusting the 0.02-ppl decision margin, the held-out ppl noise floor is measured (seed/calib-shuffle
variance) so the kill threshold is decidable.

**What is evolved.** `select_outliers(W, act, rot, budget_bits) -> (mask, encoder_spec)` in
`weights/escape_zoo.py` — the scoring rule and encoder_spec only; budget_bits caps escape spend
so all candidates compete at iso-bits; position/value coding machinery is fixed audited harness
code after Stage 1.

**Verifier (+anti-gaming).** Held-out ppl at matched honest TOTAL bits (escapes change the dense
stream entropy, so totals are compared, not the escape plane alone); decoder reconstructs the
mask purely from shipped bits (round-trip), so policies cannot hide information in the mask;
decode order enforced when position coding conditions on dense weights.

**Experiment plan.** Module `weights/escape_zoo.py`. 1 round (Stage 1 fix + noise-floor
measurement) + up to 6 rounds (Stage 2 evolution). 30-60 s per candidate; Stage 1 is an
afternoon.

**Success criterion.** Stage 1: ~0.055 b/w banked at identical ppl. Stage 2: > 0.02 b/w at
iso-ppl or > 0.02 ppl at iso-bits (above measured noise floor) from selection.

**Kill criterion.** After 6 Stage-2 rounds, best policy improves < 0.02 b/w AND < 0.02 ppl vs
top-|w|+fp16 over the Stage-1 baseline — record the knob as saturated.

**Prior art note.** arXiv 2505.18758 anticipates the joint selection-vs-rate economics on CV
nets; SpQR, SqueezeLLM, CLAQ, OWQ cover components. The Stage-1 split exists precisely so the
evolution loop is not credited with rediscovering textbook coding theory.

---

## A3. Lattice Rate Amnesty (coset-conditional re-scoring of the E8/D4 family)

**Mechanism.** E8 = D8 ∪ (D8 + 1/2): every coordinate of a decoded point re-reveals the
integer-vs-half-integer coset, and D8's even-sum parity determines the last coordinate; the
stored `_coord_entropy` marginal-sum rate therefore pays the coset flag 8 times plus an
unexploited parity bit. The hostile review *confirmed this empirically on a Gaussian surrogate:
exactly 1.000 b/w of overcharge* at q=0.08-0.28 — the exact dual of the R12 joint-entropy
undercount the project caught in the other direction. Step 1 is pure re-scoring: regenerate the
deterministic (seeded) R10-R12 index arrays and push them through an actually-decodable
sequential conditional coder (1 coset bit per 8-vector, 7 coordinates on coset-conditioned
alphabets, parity-constrained 8th); reconstruction is unchanged so no ppl re-eval is needed.
Per the verdict's mandate, accounting is made SYMMETRIC before declaring any flip: the champion
ECVQ index stream goes through the identical real-bitstream round-trip coder (adaptive coding, or
probability-table bits charged in the header) at the same time — this is delivered by A1/A2's
re-baselined container format. If the corrected frontier flips (surrogate projections: E8 q.14 ->
~3.56 b/w @ 4.151 ppl vs champion 4.169 @ 3.91), Step 2 re-opens lattices properly and
LEVER-MATCHED: nested-coset enumeration for rate, per-group lambda for entropy-constrained point
selection, the champion's 0.5% fp16 outlier lever added to the E8 path, and optional shared-seed
subtractive dither. Note the honest theory bound: E8 granular gain over ideal ECSQ is ~0.109 b/w
(G=0.0717 vs 1/12), so the realistic lever-matched prize is ~0.1-0.2 b/w, not the raw 0.35
surrogate flip.

**What is evolved.** A lattice rate module `weights/lattice_rate.py` replacing `_coord_entropy`
(codec_zoo.py:438) for lattice codecs, plus (Step 2 only) the lattice codec variants the mutator
iterates: scale q, lambda, dither seed, nesting depth, outlier-fraction.

**Verifier (+anti-gaming).** Lattice rate = byte length of a REAL encoded stream from the
sequential conditional coder, round-tripped to the exact indices (SHA-checked) — a formula cannot
undercount because the bits are literally produced and decoded. Static probability tables are
either adaptive or charged. ppl side unchanged and disjoint from calibration.

**Experiment plan.** Step 1: minutes of CPU, zero GPU — one afternoon including the symmetric
champion re-code. Step 2 (conditional): reuses vectorized `_e8_nearest`/`_d4` GPU code at
30-60 s/scheme, 1-2 rounds.

**Success criterion.** Lever-matched E8 beats the re-baselined ECVQ champion by >= 0.05 b/w at
iso-ppl -> lattice family re-opened, likely new champion direction. Either outcome
hardens/flips the single most consequential closed branch in the loop.

**Kill criterion.** Re-scored, lever-matched E8 still >= 0.05 b/w worse at iso-ppl -> permanently
close the lattice family after one afternoon; do not proceed to Step 2.

**Prior art note.** NestQuant (2502.09720) validates nested-coset lattice coding on Llama-3;
Leech Lattice VQ (2603.11021) already shows lattices win under fair indexing — so the negative
branch is no longer paper-grade on its own; residual novelty is the ECVQ head-to-head at
real-bitstream honest rates plus the metric-audit methodology. Coset-conditional accounting is
textbook (Zamir-Feder ECDQ, Conway-Sloane).

---

## A4. TrellisForge (entropy-constrained TCQ with rate-in-metric Viterbi)

**Mechanism.** R23's negative was an accounting artifact, not a trellis failure: the 4-state
Marcellin-Fischer toy was scored at FIXED R bits with distortion-only Viterbi. ECTCQ
(Fischer-Wang 1992) puts the rate term inside the branch metric — cost = (x-c)^2 +
lambda*(-log2 p(branch|state)) — re-estimates decision probabilities ECVQ-style each iteration,
and entropy-codes the decision stream conditioned on trellis state (fully decodable: the decoder
follows the same table sequentially). Classically this lands within ~0.5 dB of the Gaussian R-D
bound at 8 states vs ~1.53 dB for entropy-constrained scalar — ~0.15-0.2 b/w of theoretically
sanctioned headroom on exactly our post-rotation near-iid Gaussian source, attacking the
granular/memory axis that the 12 failed data-awareness attacks never touched. Per the verdict's
mandate, a 1-day near-zero-GPU pre-flight gate runs first: (a) measure the champion ECVQ's actual
per-group distortion gap to the Gaussian R-D bound on real rotated weight statistics — if the
measured gap is already < 0.08 b/w the prize does not exist and the build is killed; (b) validate
the batched Viterbi + arithmetic round-trip pipeline on synthetic iid N(0,1) against the
published Fischer-Wang curve as a unit test. Encode is batched Viterbi vectorized over thousands
of g=128 groups (states x branches as gather+min per step); p(branch|state) re-estimation is
subsampled (~5% of groups, 1-2 full passes) to keep encode under ~1 h/candidate; groups are
chunked (~50k) to dodge the naive traceback-memory blowup. States capped at 256 (Pascal-feasible;
captures most of the classical gain). Run AFTER A3's coset recount — both compete for the same
space-filling budget.

**What is evolved.** `make_tcq(n_states, tstats) -> (next_state, level_gen, p_init)` in
`weights/tcq_zoo.py`: trellis topology (bitshift, permuted-shift, sparse tables), 1MAD/3INST-style
computed-codebook programs, branch-metric lambda variants. Seeded with the Fischer-Wang design
(per-state subset codebooks, Ungerboeck partitions). Expect state count to dominate and topology
to be marginal — the evolution garnish is secondary to the ECTCQ cell itself.

**Verifier (+anti-gaming).** Honest rate = ACTUAL bits of an arithmetic round-trip encode of the
decision stream (claimed entropy must match emitted bits within 0.5% — directly blocks the
undercount bug class), with ALL decodable artifacts counted: per-layer per-state probability
tables, per-state codebooks, outliers (~0.045 b/w), scales. Round-trip decode enforces context
causality. Decode-cost class: sequential (QTIP class), reported.

**Experiment plan.** Pre-flight: 1 day, near-zero GPU. Then `weights/tcq_zoo.py`, max 4 rounds,
~1 h encode per candidate after subsampling, standard 30-60 s eval.

**Success criterion.** +0.08-0.15 b/w at iso-ppl over the re-baselined champion at 64 or 256
states -> new champion. A clean negative ("entropy-coded scalar matches 256-state ECTCQ on
rotated LLM weights") fills the comparison cell that arXiv 2510.11234 explicitly left empty.

**Kill criterion.** Pre-flight gap < 0.08 b/w -> kill before building. Otherwise: both 64- and
256-state ECTCQ gain < 0.05 b/w at iso-ppl within 4 rounds -> declare the scalar verdict final.

**Prior art note.** QTIP (2406.11235) is fixed-rate TCQ + Hadamard on LLM weights; NWC
(2510.11234) benchmarks fixed-rate TCQ vs entropy-constrained scalar and leaves the ECTCQ cell
empty; mechanism is 34-year-old textbook (Fischer-Wang 1992). ECTCQ-on-LLM-weights unpublished as
of 2026-06.

---

## A5. Evolved-Context Micro Entropy Model (attack iid index entropy where it can actually lose)

**Mechanism.** The loop entropy-codes indices at order-0 only and pays side info raw; cmcore (our
1.67-bpc Hutter-grade mixer) has never been pointed at quantization artifacts. Per the verdict's
mandate, the static side-info diet is the ROUND-0 BASELINE INSIDE THE VERIFIER, not a fallback:
A1/A2 land double-quantized amax + order-0 arithmetic coding of all side planes + enumerative
outlier-position coding first, and every evolved candidate is scored as delta-vs-static-baseline —
otherwise the loop burns rounds "discovering" QLoRA double quantization and the evolution gets
credited with 1989-2023 gains. The 30-minute cmcore gate is extended to EVERY stream separately
(indices, amax plane, outlier values, positions, AWQ scales), yielding per-stream redundancy
bounds before any build. The contrarian bet: the exploitable conditioning is not neighbor indices
(near-iid by construction; predicted 0.02-0.08 b/w) but CROSS-STREAM features — quantized group
amax, pegged-coordinate position within each normalized group, tensor role/depth — feeding one
tiny (<= 50k-param, counted-in-bits, ~0.004 b/w amortized) shared logistic-mixture or 2-layer-MLP
coder. Train on half the layers, CODE the other half (split fixed by hash) so overfit-flattered
rates die. Either outcome has paper value: a win stacks toward ~2.95 b/w at byte-identical
reconstruction; a null upgrades "rotation whitens indices to near-iid" to a Hutter-grade-mixer
bound.

**What is evolved.** The causal context-feature extractor program (FunSearch-style: only the
feature code mutates) in `weights/ctx_entropy_zoo.py`; the micro-coder architecture is fixed.

**Verifier (+anti-gaming).** Exact bits of a real encode + byte-exact decode round trip (decoder
recomputes the same causal features — peeking at future symbols breaks decode and auto-fails);
model bits included; train/code split hash-fixed. Screening may use parallel teacher-forced exact
cross-entropy; FINALISTS require the real arithmetic-coded round trip (the screening-vs-final gap
is where the 5x runtime optimism hides). Rotate splits / confirm on 1.5B against multi-round
adaptive overfitting to the fixed coded half.

**Experiment plan.** Gate: 30-60 min CPU (cmcore over all streams). Then 2 rounds,
~5-10 min/candidate screening, full round trips for finalists. Mostly CPU; runs concurrently with
GPU tracks.

**Success criterion.** >= 0.04 b/w on side planes or >= 0.03 b/w on indices BEYOND the static
baseline at byte-identical reconstruction.

**Kill criterion.** Pre-registered: cmcore gate < 0.03 b/w on indices AND evolved model < 0.04
b/w on side planes after 2 rounds -> kill the neural model, ship only the static recode (A1/A2).

**Prior art note.** DeepCABAC (hand-built contexts on raw weights), QLoRA double quantization
(the amax lever is 2023-vintage), EntroLLM/rANS-ECQ (order-0 only), FunSearch (method). The
LLM-evolved causal context-program + counted-in-bits micro-model combination over rotated indices
and side planes is unpublished; components are all established.

---

## A6. Partial-Mixing Incoherence (butterfly depth as a rate-distortion knob)

**Mechanism.** The loop only ever ran k=7 (full 128-Hadamard, champion) and k=0 under codecs that
lost for codec reasons (RTN/NF/Lloyd-Max, GPTQ) — notably, **k=0 + ECVQ + outliers (the champion
minus rotation) has never been run**: the cleanest single falsifier of the whole axis is a
missing 1-minute eval, and it runs first. A depth-k signed butterfly mixes 2^k coordinates
(block-diagonal Hadamard with block 2^k — a reshape plus small matmul, same cost class); CLT
Gaussianization and hence index entropy at fixed variance rise monotonically with k, but so do
outlier suppression and amax stability. Since the fp16 escape already eats the extreme tail, an
intermediate k could dominate both endpoints. Per the verdict's mandate the build order is
inverted: before any depth-policy or generalized-Gaussian-family code, run a confound-controlled
GLOBAL sweep — k in {0,2,4,7} applied to all tensors, ECVQ refit per k, outlier fraction co-swept
over {0.5%, 1%, 2%} (at low k either the outlier budget grows or ppl blows; the interior optimum
is confounded unless co-swept) — and gate on POST-TRIM kurtosis (after removing fp16 outliers),
not the raw 3.6-9.4, since the escape already removes the tail that motivates the idea. Honest
sizing: the Gaussian-vs-Laplacian entropy gap is ~0.10 b/w; post-trim recoverable tax is
~0.05-0.1 b/w. The per-tensor evolved depth policy is built only if some global k<7 wins at
iso-ppl. This sweep doubles as RotForge's (B5) go/no-go gate — one experiment, two decisions.

**What is evolved.** (Only if the sweep passes) a depth-policy `stats -> k in {0..7}` plus
matched generalized-Gaussian ECVQ level-family fitting in `codec_zoo.py`; the butterfly transform
itself is fixed audited vectorized torch.

**Verifier (+anti-gaming).** Standard arena: held-out ppl + honest entropy-counted b/w including
3 bits/tensor for k, per-(tensor,k) GG-shape/levels/probs side info, and outlier position bits
when the fraction floats. Decode-cost class can only improve (fewer butterfly stages). Per-tensor
depth selection by full-model ppl is infeasible (~168x8 evals), so the policy uses local
weight-space proxies — the arena verifier stays the ground truth.

**Experiment plan.** Step 0: the missing k=0 baseline (1 minute). Sweep: 4 depths x 3 outlier
fractions = 12 arena evals (<1 GPU-day with refits). Policy evolution: <= 2 rounds if gated in.
Extends the rotation op in `codec_zoo.py`.

**Success criterion.** Some global k<7 beats the champion by >= 0.03 b/w at iso-ppl -> open the
per-tensor policy; expected composable 0.05-0.1 b/w if the rate-tax hypothesis holds.

**Kill criterion.** Iso-ppl rate monotone non-increasing in k across the round-1 sweep -> close
the axis permanently and publish the measured "incoherence is rate-free at ~3 b/w with outlier
escape" curve (workshop-grade; unreported in the QuIP/QTIP/PolarQuant line).

**Prior art note.** ButterflyQuant (full-depth learnable butterflies, W4A4 objective), HARP,
LRQ-DiT adaptive rotation strength, EntroLLM (leptokurtosis helps Huffman — never connected to
rotation depth), PolarQuant/QuIP#/QTIP treat Gaussianization as free. The
iso-ppl-rate-vs-mixing-depth curve appears unpublished; "novel recombination + novel measurement"
is the accurate class.

---

## A7. Zipf-Weighted Embedding ECVQ (whole-model honest coverage)

**Mechanism.** The tied embedding (151936x896, ~28% of 0.5B params) is fp16 and excluded from the
denominator in all 24 rounds — the biggest standing reviewer objection. Quantize it with
rotation-free ECVQ (row norms are tight, p1-p99 0.30-0.60; no incoherence problem), code the
per-row gain with A1's scale-plane machinery over frequency-sorted rows (sorting is pure storage
order, not a function change), and switch the verifier denominator to ALL parameters. Because the
tied matrix serves both lookup and logits, a row's distortion matters roughly in proportion to
its unigram frequency as input and as target — so per-row lambda follows an evolvable monotone
map f(freq_rank). Per the verdict's mandate, round 1 is UNIFORM-lambda embedding ECVQ (coverage
lands immediately as the new honest whole-model baseline) and Zipf-lambda runs strictly as an
iso-rate ablation against it. The verifier is fixed BEFORE the axis opens: calib and held-out
slices share enwik8 unigram statistics, so an evolved f(freq) that zeroes rate on Qwen's huge
non-enwik8 (multilingual/code) vocabulary would be rewarded invisibly — a tiny out-of-domain
guard slice (~1MB code + multilingual, disjoint from calibration) is reported alongside, with a
hard regression bound (guard ppl delta < 2x enwik8 ppl delta). Expect ~6-bit embeddings (the
lm_head side is the sensitive end) -> realistic whole-model ~3.9 b/w, not the optimistic 3.2-3.5.

**What is evolved.** Embedding codec + parametric f(freq_rank; theta) map + frequency-banded vs
single codebook choice, in `codec_zoo.py` / `weights/embed_codec.py`.

**Verifier (+anti-gaming).** Held-out enwik8 ppl + OOD guard-slice ppl with hard bound +
whole-model honest b/w (embedding bits, row-gain plane, and shipped frequency table all counted).
Calibration/held-out disjointness as usual.

**Experiment plan.** 2 rounds (uniform coverage, then Zipf ablation). ECVQ fit on 136M scalars =
minutes; one arena + one guard eval per candidate. 545MB fp32 fits the 1060.

**Success criterion.** Whole-model honest rate at near-unchanged ppl (within ~0.02) and guard
bound green; Zipf-lambda beats uniform at iso-rate by a measurable margin.

**Kill criterion.** Frequency-aware never beats uniform at iso-rate -> keep uniform (coverage
still ships). Any sub-5-bit embedding costing > 0.05 ppl -> report embeddings at 6-8 bits
honestly and close the axis.

**Prior art note.** GroupReduce (NeurIPS 2018) did Zipf-weighted vocab-matrix allocation;
llama.cpp imatrix does frequency-weighted embedding quantization in production; Mixed-Precision
Embeddings, RSQ adjacent. Surviving contribution: tied-matrix per-row ECVQ-lambda under honest
whole-model rates + the guard-slice protocol. Coverage hardening, not new science — but
near-mandatory for publication.

---

## A8. Seeded Subtractive Dither + matched-MSE A/B (the cheap test of MSE sufficiency)

**Mechanism.** Every idea in this roadmap is priced under the assumption that per-tensor MSE is a
sufficient distortion proxy; the 2-bit cliff is the regime where error STATISTICS might dominate
instead. Per the verdict's mandate the order is inverted and de-scoped: run the matched-MSE
error-injection A/B FIRST as a pure synthetic-perturbation experiment — no entropy coder, no
codec. Three arms at identical per-tensor MSE (numerically verified), at the 2.2 b/w-equivalent
distortion level: (a) iid uniform error, (b) iid Gaussian error, (c) actual ECVQ residuals
rescaled — the three-arm design decomposes signal-dependence from error-shape (a two-arm version
conflates them). Three pre-registered seeds (Philox counter-based PRNG for cross-device
reproducibility); pre-registration closes the seed-shopping side channel by construction. Only if
independent error beats ECVQ-shaped error by >= 0.1 ppl does the full codec get built:
shared-seed subtractive dither (decoder regenerates U from the pinned seed, zero rate) + the
closed-form Gaussian+uniform posterior-mean shrinkage at the decoder — with saturation explicitly
modeled (Schuchman's exactness guarantee fails in the clipped region, which at 2 bits is most of
the grid): report the clipped-coordinate fraction and exclude/handle it in the shrinkage. A null
kills BOTH cliff-as-error-statistics hypotheses (this and the dropped Bits-Back/REC idea, which
is dominated a fortiori since REC adds ~0.25-1.0 b/w overhead on the same error statistics) in
one afternoon and strengthens the ECVQ paper.

**What is evolved.** (Only if the A/B passes) per-tensor step/shrinkage schedules and the
dithered-index entropy coder in `codec_zoo.py`; the A/B itself is a fixed experiment.

**Verifier (+anti-gaming).** A/B: identical measured MSE by construction, held-out ppl, seeds
pre-registered before any eval. Codec (if built): honest bits = real coder's H(K|U) conditioned
on decoder-known dither; seed in the artifact hash; shrinkage is decoder-side and cannot touch
rate accounting.

**Experiment plan.** `weights/dither_ab.py`. A/B: ~2 hours (elementwise vectorized torch,
3 arms x 3 seeds x 1 eval). Optional codec round: 1 round, 30-60 s/candidate.

**Success criterion.** A/B: independent error >= 0.1 ppl better than ECVQ-shaped at matched MSE
-> build the dither codec and sweep 1.8-2.6 b/w; benchmark the 2-bit point against WaterSIC
(2603.04956, public code), now the published near-IT-limit baseline on Qwen.

**Kill criterion.** A/B gap < 0.1 ppl -> both stochastic-error hypotheses die in one afternoon;
record the negative as a distortion-metric validity result. If built: codec fails to beat ECVQ by
> 0.3 ppl at 2.2 b/w -> kill.

**Prior art note.** Agustsson-Theis 2020 (universal quantization for neural compression; reports
dither is MSE-inferior at low rates), Choi 2018 (universal NN compression), NestQuant MMSE
scaling, HIGGS linearity theorem (ppl ~ linear in L2 error — published evidence for the null,
unverified at the cliff where ParetoQ claims a regime change). The specific package + the
three-arm matched-MSE causal test appears nowhere.

---

# TIER B — high-variance moonshots

## B1. KV-Cache ECVQ with sequence-axis cmcore entropy modeling

**Mechanism.** Weight indices are near-iid by construction; KV indices are a genuine time series
— the cleanest match between the cmcore asset and an open gap. Port the loop to a
`kv_codec_zoo.py`: per-head rotation, per-layer/head ECVQ levels+lambda on a calibration cache,
fp16 recent-window and sink-token escapes, and cmcore context features spanning the sequence axis.
The champion recipe transfers wholesale; evolution attacks KV-specific structure. Per the
verdict's mandates: (1) the verifier is hardened BEFORE round 1 against eviction-mimicry — a
random passphrase is planted early in the compressed prefix and teacher-forced ppl on its
verbatim re-occurrence in the continuation is scored (one cheap forward pass; un-gameable by
middle-token bit starvation, the StreamingLLM failure mode); (2) the baseline set is seeded with
a CacheGen-style adjacent-token-delta + arithmetic-coding codec, since THAT — not order-0 — is
the published sequence-axis frontier; (3) keys are intercepted PRE-RoPE (post-RoPE keys have
oscillating temporal correlation) and serialization is channel-major so cmcore's contexts see the
sequence axis; (4) the decode-cost story is told honestly: CM decode at MB/s does not compete
with NVMe — the regime is cold archival / redundancy ceiling. Qwen2.5-0.5B GQA gives only 2 KV
heads, which coarsens per-head learning — accepted.

**What is evolved.** The full codec fn(K,V) -> (K_hat, V_hat, honest_bits): transform, quantizer,
escapes, and the cmcore context-feature set; mutator rewrites all three each round.

**Verifier (+anti-gaming).** Continuation held-out ppl after compressed-cache resume + passphrase
teacher-forced ppl + honest bits/KV-element (real coded bytes; recent-window and sink escapes
charged) + decode-cost class. Continuation slice disjoint from calibration.

**Experiment plan.** `weights/kv_codec_zoo.py`. 3 rounds to the gate. 0.5B prefill of 2-4k tokens
(~25-50MB cache) fits 6GB; 30-60 s/candidate + 10-60 s cmcore CPU pass.

**Success criterion.** Context mixing >= 0.15 bits/element over the CacheGen-style baseline at
iso-ppl with passphrase intact (expect more on V than post-RoPE K); beating published
KVTC/TurboQuant operating points at matched honest bits.

**Kill criterion.** < 0.15 bits/element over the sequence-axis baseline at iso-ppl, or the seeded
stack cannot match published KV points within 3 rounds -> retreat to the weight track.

**Prior art note.** CacheGen (SIGCOMM'24) already does adjacent-token delta + arithmetic coding
of quantized KV — the published frontier to beat; KVTC, RateQuant, TurboQuant crowd the
transform/allocation space; EvolKV evolves budgets only. Residual novelty: Lagrangian ECVQ on KV
+ adaptive context MIXING (CacheGen's distributions are static) + LLM-evolved KV codecs.

---

## B2. Family Superdictionary (lossy joint coding of base + fine-tunes, ship-as-patch)

**Mechanism.** Code {base, instruct, abliterated} Qwen2.5-0.5B as ONE artifact: champion-quantize
the base, freeze its rotation seeds/codebooks/probability tables as the family superdictionary,
then code each member conditionally on the QUANTIZED base (true conditional coding — encoder has
the side info, so for near-Gaussian deltas this matches Wyner-Ziv; the dropped WZ idea's
conditional-vs-independent gain measurement is delivered here as a reported number). Per-group
policy chooses: reuse base indices verbatim (amortized flag bit), code the index-space delta via
conditional entropy tables, or independent re-code; the lambda split between base and member
streams is part of the search (Gray-Wyner framing: common stream + per-model refinements). Per
the verdict's mandate, two fixes are non-negotiable: (1) absolute per-member ppl gates are
replaced with DISCRIMINATIVE gap-preservation gates — instruct: fp16 instruct-vs-base ppl gap on
chat-templated held-out text, decoded member must reproduce >= X% of that gap and sit closer to
fp16-member than fp16-base; abliterated: refusal-logit probe on ~50 harmful prompts (single
forward pass each, no generation) — killing the degenerate "member := base, zero bits, all gates
green" optimum that otherwise makes the headline unfalsifiable; (2) every candidate must beat the
trivial-glue control: champion-on-base + champion independently applied to the residual
(member - dequant(base)). Conditional tables are static p(k_member | k_base) per candidate;
cmcore reserved for finals (2-15 min/member at CM speeds). Per-member tables, outlier maps, and
refit scales are charged to the member stream.

**What is evolved.** The family codec policy program (per-group decision code + rate-split
policy) in `weights/family_codec.py`, building on `wcodec.py`/`ecosystem.py`.

**Verifier (+anti-gaming).** Summed REAL coded bits across all streams; star decode graph rooted
at base (members never read each other's streams, harness-checked); discriminative gates as
above; denominators count all members' parameters.

**Experiment plan.** 3-4 rounds. One model in VRAM at a time; ~3-5 min/candidate (3 members).
Models already on disk.

**Success criterion.** Members at <= 0.7-1.5 b/w incremental with gap-preservation gates green
and the trivial-glue control beaten -> "N models for the bits of one plus a little" with honest
totals (the unclaimed accounting+framing delta over BitDelta/DeltaZip/ZipLLM).

**Kill criterion.** Conditional coding cannot get below 2.0 b/w incremental at gate-passing
fidelity -> fall back to the lossless-delta-only story (wcodec) and kill the joint-lossy claim.

**Prior art note.** BitDelta (owns the headline at fixed 1-bit vs fp16 base), DeltaZip,
Delta-CoMe, DeltaDQ, AdaMix, ME-Switch; lossless: ZipLLM/BitX, ZipNN. Unclaimed: quantized-base
side information, joint base-vs-member rate split, honest entropy-coded family totals, Gray-Wyner
framing.

---

## B3. RecoveryForge (recovery fine-tuning of continuous side parameters; evolved objectives gated)

**Mechanism.** R19-21 QAT retrained WEIGHTS end-to-end (millions of params, STE, overfit wall at
train 1.7 vs held-out 4.6); the standard QuIP#/QTIP final stage — refining only continuous side
parameters with indices FROZEN (rate-exact, no STE needed since w = scale*level[idx] is
differentiable in scale/level) — was never run. Per the verdict's mandate the structure is
inverted: a one-day hand-written QuIP#-style baseline spike runs BEFORE any evolution loop, under
a hardened harness: (a) lm_head/embeddings BANNED from the trainable set (tied, ~136M params —
otherwise the run becomes free-rate enwik8 domain fine-tuning the gap monitor cannot flag), with
a whitelist + hard trainable-param cap in HARNESS code, distinguishing the truly-small set
(RMSNorm ~21k + level tables ~10k) from the 3.9M per-group scales — scales gated behind a
low-dimensional reparam (one learned multiplier per row or per layer) to keep capacity honest;
(b) a small out-of-domain held-out probe (code/C4 slice) reported alongside enwik8 ppl to expose
domain-adaptation flattery; (c) the eval noise floor measured first (champion eval re-run with 3
calib shuffles) so the minimum detectable effect is known. The evolution loop (evolved loss
mixes: block-output MSE vs KL-to-fp16 vs CE, regularizers, schedules) launches only if the
baseline recovers >= 0.05 ppl at iso-bits AND the noise floor is < 0.02 — otherwise the baseline
ships as a free composable stage and the GPU-week goes elsewhere.

**What is evolved.** (Gated) `recovery_step(model_q, fp16_act_cache, batch, t) -> (loss,
trainable)` in `weights/recovery_zoo.py` — trainable-set choice within the whitelist, loss
program, schedules. Evaluator cascade: 100-step proxy (~3-4 min) gates; survivors 500-1000 steps
(~15-20 min) then full held-out ppl.

**Verifier (+anti-gaming).** Held-out enwik8 ppl + OOD probe + honest bits including
side-parameter deltas (thousands of params = negligible rate, verified not asserted);
proxy/full runs on disjoint calib shuffles; generalization-gap monitor; held-out slice never in
the training loop; whitelist/cap enforced by harness, not evolved code.

**Experiment plan.** Baseline spike: 1 day. Evolution: <= 6 rounds if gated in. ~1 s/step at 0.5B
in 6GB with frozen weights.

**Success criterion.** Baseline: >= 0.05 ppl recovered at iso-bits (0.5B@3.13: 4.483 -> ~4.43 or
better), OOD probe non-regressing. Evolution: > 0.02 ppl over the hand-written baseline.

**Kill criterion.** Plain block-MSE baseline captures >= 90% of the gain and 6 evolved rounds add
< 0.02 ppl -> keep the plain version, stop evolving. Baseline < 0.05 ppl or noise floor >= 0.02
-> never launch the loop.

**Prior art note.** QuIP# stage-2, EfficientQAT E2E-QP, AQLM+PV-Tuning (codebook >> RMSNorm
ablation already published), arXiv 2604.08118; evolved-loss near-misses DiscoPOP/ShinkaEvolve/
CLAMP-ViT. The LLM-evolved recovery-loss wrapper is unpublished but thin; most information
arrives on day 1 from the non-novel baseline — priced accordingly.

---

## B4. Role-Dispatch Pipeline Composer (per-role codec pipelines as an evolvable DSL program)

**Mechanism.** All 24 rounds applied one identical pipeline to every linear tensor; the harness
already passes `key` to codecs, so per-role dispatch was always possible and never used. A
composer program maps (role in {q,k,v,o,gate,up,down}, depth, shape, cheap stats) -> op sequence
+ knobs from a fixed audited op library — {butterfly(k) from A6, AWQ-scale, lambda, outlier%,
group size, level family, scale-plane coder from A1, LL-Huffman-ECVQ decode variant inherited
from the cut CellMap idea} — under a single global honest budget enforced by the verifier, so the
program must internally trade roles off. Per the verdict's mandate, a one-afternoon ORACLE
HEADROOM pre-test runs first: with all other roles frozen at champion settings, sweep one role at
a time over {lambda, outlier%, group size, rotation on/off} (~8 roles x 5 settings = 40 arena
evals, a few GPU-hours), then greedily combine per-role argmins — this directly upper-bounds the
additive role-heterogeneity gain (this is also where the cut LambdaForge's closed-form
measured-slope allocation gets its one-day test). Simultaneously: a second disjoint enwik8
confirmation slice is added and the ppl noise floor measured, so the 0.03 b/w kill criterion is
actually decidable; the decode-cost class is derived automatically from the composed op set (no
hand labels the DSL can under-declare); the 'embed' role is excluded (handled by A7; keeps the
b/w denominator comparable with all 24 rounds). Caution priors: ECVQ already learns
levels+probs per layer, and two adjacent levers (heuristic per-layer lambda, sensitivity
mixed-precision) failed in-loop — realistic headroom is 0.03-0.08 b/w.

**What is evolved.** The dispatch program in `weights/codec_pipeline.py`, written in a
constrained mini-DSL; op implementations are fixed audited code. Archive keyed by (rate bucket,
decode-cost class) so partial wins survive as splice material.

**Verifier (+anti-gaming).** Existing arena: held-out ppl on two disjoint slices + honest b/w
summed over ALL tensors (accounting lives in fixed ops; composer cannot touch the denominator) +
auto-derived decode-cost class. Global budget enforced by harness with inner lambda bisection
(3-5x evals).

**Experiment plan.** Pre-test: 1 afternoon (~40 evals). Composer: 3 rounds, ~20 candidates/round
at 30-60 s each (~20 min GPU/round).

**Success criterion.** Composed dispatch beats the uniform champion by >= 0.03 b/w at iso-ppl,
confirmed on the second slice and above the noise floor.

**Kill criterion.** Oracle headroom < 0.05 b/w -> kill before round 1 and bank "role-uniformity
holds post-rotation" as an established fact. Otherwise: best composed < 0.03 b/w after 3 rounds
-> fold the DSL back to scalar allocation.

**Prior art note.** AMQ (EMNLP 2025), SliM-LLM, HAWQ-V3/HAQ, ScaleBITS, RAMP (allocation);
FlatQuant/SpinQuant (per-layer learned transforms). No published work evolves heterogeneous
per-module preprocessing PIPELINE STRUCTURE as an interpretable program — the
FunSearch-configuration centerpiece for the "LLM-evolved quantization codecs" paper.

---

## B5. RotForge (rate-objective rotation-generator programs, gated by the A6 sweep)

**Mechanism.** The champion's transform was never an evolution target, and nobody in the
QuIP/QuaRot/QTIP line optimizes a transform for the RATE term of an entropy-coded weight codec —
the "price of incoherence" is unquantified. Per the verdict's mandate, the go/no-go pre-test runs
BEFORE any DSL is built: the A6 block-size x outlier-fraction sweep traces the suspected
one-dimensional Gaussianization-vs-outlier tradeoff directly; if rate-at-iso-ppl is monotone
toward full g=128 Hadamard, the tax is intrinsic — write the quantification and skip RotForge
entirely; if an interior block size wins by > 0.03 b/w, the DSL search is justified. If it
proceeds: generators emit a DECLARATIVE AST (compositions of signed permutations,
block-Hadamards 4..g, Givens cascades, Cayley transforms of sparse skew patterns, Kronecker
products; identity/partial/per-tensor-selective legal) that the HARNESS compiles, serializes, and
bit-counts itself — no closures, no self-reported param_bits; orthogonality is verified by
random-vector norm preservation (round-trip invertibility alone does NOT enforce it and would
break iso-MSE bookkeeping). Two-stage evaluator: Stage A (seconds, no GPU model pass) scores
honest coded bits at iso-MSE on 8 cached tensors — using ACTIVATION-SECOND-MOMENT-WEIGHTED
distortion from the existing calib hooks, because unweighted weight-MSE systematically favors
weak-mixing transforms whose structured error hurts ppl (equi-sensitivity only holds under full
mixing); Stage B for survivors: full held-out ppl. Honest sizing: realistic reclaim 0.02-0.06 b/w.

**What is evolved.** `make_transform(g, tstats)` AST generators in `weights/rot_zoo.py`; the DSL
compiler enforces the no-Python-loops law; per-transform lambda bisection for iso-MSE.

**Verifier (+anti-gaming).** Stage-B held-out ppl + honest bits including harness-computed
transform param_bits; norm-preservation check; distortion measured post-inverse.

**Experiment plan.** Gate: shared with A6 (1 afternoon). DSL track: <= 6 rounds; Stage A
ms/tensor, Stage B standard 30-60 s.

**Success criterion.** > 0.03 b/w over signed-Hadamard at activation-weighted iso-MSE in Stage A,
confirmed at iso-ppl in Stage B.

**Kill criterion.** A6 sweep monotone toward full Hadamard -> never build. Else: Stage A never
shows > 0.03 b/w across 6 rounds -> the tax is intrinsic; publish the quantification.

**Prior art note.** ButterflyQuant (the exact Givens parameterization, different objective),
KurTail (statistic-targeted rotation, opposite sign), SpinQuant (Cayley), FlatQuant (Kronecker);
classical coding-gain/ICA objective (Sezer-Guleryuz). A reviewer will demand a
ButterflyQuant-objective baseline; the rate-objective intersection and the tax quantification are
the open residue.

---

## B6. Embedded ECVQ (successive refinement with conditional inter-stage coding)

**Mechanism.** One truncatable bitstream decoding at ~2.0/3.1/3.9 b/w. R6's additive ECVQ lacked
exactly the piece where the zero-penalty theorem lives: CONDITIONAL inter-stage coding
(refinement indices coded with per-coarse-cell tables) — structurally, a two-stage scalar
quantizer with per-cell conditional refinement and conditional entropy coding IS a one-shot
quantizer over the refined partition at rate H(s1)+H(s2|s1), so the only true penalty is the
nesting constraint plus table overhead; <= 0.05 b/w is credible. Fit coarse ECVQ at ~2 b/w,
within each coarse cell fit refinement levels on the conditional residual, lay stages
contiguously, truncate. Per the verdict's mandate, two upgrades convert this from theorem
confirmation into a comparative claim: (1) a head-to-head NESTED-INT baseline inside the same
harness — rotated+AWQ-scaled sliced-bit nested grid (MatQuant/Any-Precision-style, no entropy
coding) under identical honest-rate and prefix-decode rules at the same truncation points — so
the deliverable becomes "embedded ECVQ pays <= 0.05 b/w where nested-int pays ~0.3-0.5 b/w at
matched honest rates"; (2) the one-shot comparator is REFIT at the exact achieved truncation
rates (no frontier interpolation, which understates the penalty). Outlier accounting (0.5% fp16 +
positions ~ 0.14 b/w — nontrivial at the 2.0 point) is pinned in the stage-1 prefix. Framed
honestly as storage/distribution capability (entropy-coded streams need a decode pass), not
multi-precision serving.

**What is evolved.** The refinement partition code in `codec_zoo.py`: coarse codebook,
per-cell conditional refinement codebooks + lambdas, inter-stage entropy model; stream
layout/truncation is fixed harness code.

**Verifier (+anti-gaming).** Each truncation decoded INDEPENDENTLY by the harness from the prefix
alone (hidden later-byte dependencies fail round-trip); honest rate includes all per-cell tables;
penalty measured against the refit one-shot at the same achieved rate.

**Experiment plan.** 2 rounds. ~3-5 min/candidate (3 truncation evals); per-cell fits are small
numpy. ~1-2 GPU-hours total.

**Success criterion.** Embedding penalty <= 0.05 b/w at every operating point AND the nested-int
baseline pays >= 0.3 b/w -> citable comparative capability claim ("one artifact, three rates").

**Kill criterion.** Penalty > 0.1 b/w at any point after 2 rounds of conditional-coding fixes ->
record the measured penalty, ship the negative.

**Prior art note.** CEC-RVQ (IEEE TIP 1996) is the mechanism on images; MatQuant/MatGPTQ,
Any-Precision, AnyBCQ, BitStack are the LLM rivals — none entropy-code or report an embedding
penalty vs a one-shot entropy-constrained baseline at real coded bytes. By construction this can
only tie the champion per-rate; it is a capability add-on.

---

## B7. Anti-iid Gauge Sculpting (orbit search scored by cmcore bytes — oracle-gated)

**Mechanism.** The champion destroys structure (Hadamard -> iid indices -> order-0); the
scientific object is the minimum of context-coded size over the exact-symmetry orbit. Per the
verdict's mandate, a 10-MINUTE ORACLE UPPER BOUND runs before any machinery: on stored champion
streams, compute (vectorized numpy) the conditional entropy of indices given shipped contexts —
group scale, row statistics, neighboring indices — versus order-0, on the 3 largest tensors.
Conditioning on shipped side info dominates nearly everything a permutation can expose (sorting
is a degraded substitute for conditioning; only encoder-only sort keys escape the bound), so if
H(idx | context) - H0 < 0.02 b/w, the entire idea dies without writing the mutation loop. If it
passes: the transform family is RESTRICTED to permutations and sign flips only (diagonal scalings
are NOT exact symmetries through SwiGLU — the original proposal's scaling half is broken), e.g.
norm-sorted MLP channels and energy-sorted heads that make the scale field monotone and index
statistics drift slowly — exactly what an adaptive mixer exploits; and the PREFERRED route is
extending cmcore to take the scale plane as an explicit side-channel context rather than orbit
search. Caveat budgeted: down_proj's intermediate channels are its INPUT dim, so permuting
crosses g=128 group/Hadamard boundaries — that tensor needs requantization, not stream-only
re-scoring.

**What is evolved.** Transform-proposal + measurement script pair in `weights/gauge_sculpt.py`:
candidate exact-symmetry re-orderings and the cmcore context configuration; fitness = real
compressed bytes on stored champion index+scale streams.

**Verifier (+anti-gaming).** Actual cmcore output bytes (round-trip decoded; per-tensor config
bytes counted) — definitionally achievable; unchanged held-out ppl via tolerance-based
probe-logit gate (row permutation changes fp accumulation order); decode-cost class reports CM
speed honestly.

**Experiment plan.** Oracle: 10 minutes CPU. If passed: 1 measurement round (~30 min/batch
through the Rust coder, 5.8-13.7 MB/s), GPU only for one confirmation eval.

**Success criterion.** Sculpted gauge > 0.03 b/w over order-0 on the 3 largest tensors -> build
the codec path; upgrades "indices are near-iid" toward orbit-minimized evidence.

**Kill criterion.** Oracle < 0.02 b/w -> kill in 10 minutes. Else: best gauge < 0.03 b/w after
one measurement round -> hard abort.

**Prior art note.** PQF (CVPR 2021) searches the permutation orbit for compressibility (VQ
distortion objective); DuQuant, PermuQuant, RPTQ use permutation gauges; DeepCABAC codes as-given
layouts. New objective (actual context-mixed bytes) in a known search space; "null = theorem"
claims are out.

---

## B8. DeltaFunSearch (evolved predictor programs for related-model ULP-delta coding — audit-gated)

**Mechanism.** wcodec's hand-built best-of (XOR/arith-ULP/low-rank/sparse) leaves 31-47% of bits
on the SFT/checkpoint pairs; the context/prediction model was never an evolution target. But the
in-loop record cuts against easy wins (a Golomb/magnitude-class coder gained only +0.6pt over
zstd on heavy-SFT; dense-delta order-0 is at floor), so per the verdict's mandate, round 1 of
evolution is REPLACED by a one-day conditional-entropy audit that bounds the idea before any LLM
calls: directly measure H(zigzag symbol | base-exponent bucket, byte plane, neighboring
decoded-delta magnitude, tensor role) on the SFT and fp32-checkpoint streams (the one measured
headroom pocket: per-plane floor ~77% vs 68.7-69.5% achieved), and run cmcore as-is over the
per-plane byte streams as a ceiling probe. If the measured gap over wcodec best-of is < 3%
relative on the held-out pair, kill at a cost of hours. If it passes, evolution happens in the
Rust cmcore EVOLVE-BLOCK (compiled predictor) or a vectorized two-pass context-bucketing DSL —
NEVER per-symbol Python callbacks (the project's own forbidden anti-pattern; the original
"millisecond fitness" claim was off by 2-4 orders of magnitude). Generalization is forced by the
split: evolve on one model pair and tensor subset, score on disjoint tensors AND a disjoint pair.

**What is evolved.** The predictor/context program `prob_model(ctx) -> distribution` plugged into
wcodec/cmcore arithmetic coding, in `weights/delta_funsearch.py` + cmcore EVOLVE-BLOCK.

**Verifier (+anti-gaming).** Metric = TOTAL archive bytes including fitted tables; mandatory
round-trip decode equality executed in a directory that provably lacks the target safetensors
file (I/O sandbox — otherwise a program can re-open the target, predict perfectly, and still pass
round-trip); decode-causality enforced by the harness API.

**Experiment plan.** Audit: 1 day CPU. Evolution (gated): <= 6 rounds, seconds-to-minutes per
candidate, fully concurrent with all GPU tracks (zero GPU).

**Success criterion.** Audit gap >= 3% relative -> proceed; then 2-5%+ relative over wcodec
best-of on the held-out pair, with interpretable discovered structure (delta-magnitude laws by
base exponent / layer role).

**Kill criterion.** Audit gap < 3% relative (hours, not 6 rounds); or < 2% improvement over
best-of on the held-out pair after 6 rounds.

**Prior art note.** Task side is active (FM-Delta, ZipLLM/BitX, IBM snapshot compression,
TStore) — "no prior art" was false for the task; the narrow combination (evolved context programs
for arithmetic ULP-delta coding of related-model weights) has no hit. Cheap, well-instrumented
side-science using the idle cmcore asset.

---

## B9. Variance-Field Factorization (Sinkhorn biscaling — three-arm controlled ablation)

**Mechanism.** Tests whether pre-rotation heavy tails (kurtosis 3.6-9.4) are a rank-1
row-by-column variance-MIXTURE artifact: if W ~ D_r A D_c with A homogeneous, equilibration
(alternating row/col RMS normalization, a few vectorized passes) homogenizes the core, shrinks
the outlier set, and could cut decode cost to two diagonal multiplies. Per the verdict's mandate
this is reframed from either/or into a THREE-ARM controlled ablation with the outlier budget
PINNED across arms (moving the escape before/after equilibration silently changes effective
rate): (A) champion rotated-ECVQ; (B) Sinkhorn-equilibrated UNROTATED ECVQ; (C) STACKED
equilibrate -> rotate -> ECVQ — diagonals and rotation compose, and the most plausible real win
is equilibration shrinking the outlier set while rotation handles residual tails, which the
dodge-rotation framing would miss. The kurtosis-only kill gate is replaced with a DIRECT RATE
PROXY: ECVQ index entropy at matched MSE plus outlier fraction at fixed threshold, on the
equilibrated cores of the 5 heaviest tensors — same 1-hour budget, gating on the quantity
actually scored. The two smooth gain fields are entropy-coded with A1's machinery
((m+n)x16 bits/tensor ~ 0.02 b/w raw, counted). SINQ (2509.22944) is cited as the explicit
baseline; any writeup is positioned as its honest-rate entropy-coded extension. Honest priors:
SINQ's own ablation has Hadamard beating Sinkhorn on weight reconstruction; the recoverable
budget is mostly the ~0.06 b/w outlier bill.

**What is evolved.** The factorization codec in `codec_zoo.py`: equilibration
iterations/damping, field representation (raw vs low-order parametric), residual codec choice,
escape placement (pinned budget).

**Verifier (+anti-gaming).** Standard arena: held-out ppl + honest b/w with both gain fields
fully counted; round-trip decoding catches spline-fit residual cheating; decode-cost class
reported (drops if rotation removed).

**Experiment plan.** Gate: 1 hour (rate proxy on 5 heavy tensors). Ablation: 3 arms x 2 rate
points = ~6 arena evals; 1-2 rounds total. Equilibration is seconds/tensor on GPU.

**Success criterion.** Arm B or C >= 0.03 b/w at iso-ppl over arm A, OR iso-rate at a strictly
lower decode-cost class (no Hadamard at decode) — a valuable Pareto point either way; plus a
decisive answer to the scale-mixture question for the whole loop.

**Kill criterion.** Rate-proxy gate fails on the 5 heavy tensors (1 hour) -> never build the
codec. Else: no arm beats the champion after 2 rounds -> close.

**Prior art note.** SINQ (Huawei, ICML 2026) is the central mechanism, fixed-rate and without
entropy coding/honest rates; D^2Quant, Nagel 2019 CLE, NWC. The honest-rate entropy-coded
extension + the scale-mixture diagnostic is the surviving (ablation-grade) delta.

---

# TIER C — strategic / infrastructure

## C1. Divergence Battery Verifier with Sealed Escrow

**Mechanism.** The loop has selected on a single enwik8 slice for 24 rounds — post-Dutta ("ppl-
matched compressed models hide large KL/flip divergence") and LLM-KICK, that will not survive
review, and selection-induced overfitting has never been quantified. Build a frozen battery:
(a) 64 prompts x 4 domains (prose, Python, multi-digit arithmetic chains, Chinese) scoring
mean and p95 token-level KL(quantized || fp16) over 128-token teacher-forced continuations;
(b) ~200 cloze/copy/induction canaries scored exact-match; (c) existing held-out ppl; (d) a
SEALED escrow enwik8 slice scored only every 5th round. Per the verdict's mandates: reference
caching uses TOP-K=128 logprobs + tail bucket (~25 MB) — the full-logit cache is ~10 GB, a 250x
error in the original spec; and the retro-audit is extended from champion+3 to ALL 24 per-round
champions (re-quantized from the codec_zoo registry — quant_arena checkpoints store metrics JSON
only — ~2-3 min each), plotting worked-slice ppl vs escrow ppl vs mean-KL per round: the loop's
full selection-overfitting curve, which is the one publishable-novel result here and a first for
AlphaEvolve-style loops. Escrow/battery scores are WRITE-ONLY (never enter the LLM-mutator's
context — a slow Dwork-style leak otherwise) and battery prompts are hash-excluded from any
calibration data candidate codec code can read (candidate code has filesystem access).

**What is evolved.** Nothing in the metric — `weights/verify_battery.py` is hand-written, FROZEN,
hash-checked each round, excluded from the mutable file set. Codecs are re-selected under the
augmented fixed fitness (ppl gate AND no-regression on mean KL; p95 as catastrophe sentinel).

**Verifier (+anti-gaming).** KL against hash-pinned pre-computed references (fixed before any
evolution); binary exact-match canaries are entropy-trick-proof; escrow detects loop-level
overfitting per-candidate honesty cannot.

**Experiment plan.** Build + full 24-champion retro-audit: ~1 day. Ongoing: +1-3 min/candidate
(fp32 forwards on the 1060) on top of the 30-60 s ppl eval.

**Success criterion.** Either a re-ranking of the frontier (redirects the next 10 rounds) or a
confirmed-robust champion (free paper robustness section) — PLUS the selection-overfitting curve
either way.

**Kill criterion.** All 24 champions rank identically under the full battery AND escrow-vs-worked
delta-ppl < 0.02 -> the richer verifier adds no information; revert to the cheap ppl gate.

**Prior art note.** Dutta et al. NeurIPS 2024 (KL/flips), LLM-KICK, LLMC, Dwork reusable
holdout / Blum-Hardt ladder. Only the in-loop selection-overfitting audit of an LLM-mutator loop
is unpublished. Cannot itself improve compression; near-mandatory for publishing.

---

## C2. Red-Team Coevolution (adversarial prompt-miner species)

**Mechanism.** A second evolving population: sandboxed pure-Python prompt-miner programs whose
fitness is the divergence they induce between the current champion and fp16; top-k novel
attackers enter a persistent archive; codecs must then pass a worst-case gate over the frozen
archive snapshot while improving ppl@rate. Per the verdict's central mandate, raw mean-KL fitness
is REPLACED with CONFIDENT-FLIP fitness: score only positions where the fp16 oracle is confident
(top-1 prob > ~0.7) and reward the quantized top-1 flip rate (or KL restricted to those
positions), normalized by the same statistic on a matched random-corpus baseline — this kills the
degenerate attractor where the archive fills with rare-unicode/garbage text whose high KL is
amplified noise, makes every archived witness a real, human-readable behavioral failure, and
makes the 2x kill criterion diagnostic. fp16 logits are cached once per archive snapshot;
attacker prompts are batched per forward. The claim is framed honestly as an "adversarially-mined
stress archive / empirical LOWER bound on worst-case divergence" (a max over 16 heuristic
generators is not a bound). Per the verdict, the species is SCHEDULED to coincide with the first
sub-3-bit codec campaign — the regime where narrow circuit breakage (induction heads, arithmetic
carries) plausibly exists; at the current near-lossless operating points its yield is low.

**What is evolved.** Two species: codecs (as usual) and attacker programs over a pinned corpus +
tokenizer, no access to codec internals beyond black-box logits; embedding-novelty dedup.

**Verifier (+anti-gaming).** Attacker fitness = measured confident-flip rate against the fixed
fp16 oracle on its emitted prompts (fixed physics); codec fitness = honest bits + ppl + max
confident-flip over a hashed frozen archive snapshot per round (no mid-round moving target).

**Experiment plan.** `weights/redteam_zoo.py` + arena gate. <= 5 attacker generations; seconds
per attacker eval (batched), ~1 min added per codec eval with cached fp16 logits. Deploy with the
sub-3-bit campaign.

**Success criterion.** An archive of genuine behavioral-failure witnesses that re-ranks or gates
sub-3-bit candidates; the first PTQ result with an explicit empirical worst-case stress archive.

**Kill criterion.** After ~5 generations, archive max confident-flip stays within 2x of the
matched random-corpus baseline -> no adversarial structure for this codec class; drop the species
and bank "rotated ECVQ has no narrow failure modes" as a robustness claim.

**Prior art note.** Rainbow Teaming / RainbowPlus (the attacker machinery, aimed at safety),
Dutta flips metric, Hillis 1990 host-parasite coevolution, PTQ worst-case 2303.13003,
capability-breakage line (2504.04823). The composition (attacker programs vs quantization codecs
inside a codec-evolution loop) is novel.

---

## C3. Free-Running Drift Verifier (on-policy KL slope, retro-tested first)

**Mechanism.** Teacher forcing resets to the fp16-consistent prefix at every position, so
compounding autoregressive error is structurally invisible to the loop's only metric. Per the
verdict's mandates, the original drift-gap metric (fp16 log-prob of quantized-generated text) is
DROPPED ENTIRELY — judge log-prob rewards degenerate repetition (Holtzman effect), inverting the
metric exactly where it matters. What remains: per-position KL(fp16 || quant) slope along the
QUANTIZED model's own rollouts (fp16 as fixed judge), with the horizon extended from 64 to 256
tokens (16 prompts x 256 = same ~4k-token budget) since compounding divergence in the
quantized-rollout literature only emerges at long horizon, plus a free distinct-n / mean-entropy
tripwire to catch repetition collapse directly. Flat slope = errors stay local; positive slope =
compounding catastrophe risk at 2-3 b/w even when slice ppl looks fine. Greedy + one pinned-seed
sampled run per prompt. Before wiring anything into selection, a one-day retro-scoring kill test
runs over the 24-round archive INCLUDING the sub-3-bit losers, where drift is most likely to
appear. Self-recovery literature (Schmidt) predicts a null at the champion's precision — the
test is cheap enough that either answer is worth having, and the metric's real home is the
sub-3-bit campaign alongside C2.

**What is evolved.** Nothing in the metric — a frozen drift module inside `verify_battery.py`
(prompts hashed into the frozen verifier set); codecs are re-selected under ppl@rate + a
no-regression gate on drift slope if adopted.

**Verifier (+anti-gaming).** Fixed prompts, pinned seeds, immutable fp16 judge; slope + tripwire
only (no gameable log-prob-of-own-text term); models loaded serially, fp16 rollouts/baselines
cached once.

**Experiment plan.** Retro-test: ~1 day (2-4 min per archived codec including weight
patch/unpatch — the original 1-2 min claim ignored swap overhead). Ongoing if adopted: 2-4
min/candidate.

**Success criterion.** Drift slope re-ranks sub-3-bit codecs invisible to ppl -> a
"generation-faithful selection" selling point and a standing gate for the low-bit campaign.

**Kill criterion.** Spearman(drift-slope rank, held-out-ppl rank) > 0.95 across the 24-round
archive -> metric redundant at these operating points; shelve until sub-3-bit.

**Prior art note.** Dutta 2024 (teacher-forced KL/flips), llama.cpp --kl-divergence, ExAccErr
(exposure bias), MiniLLM (teacher-judged on-policy rollouts), 2402.18158 Gen-vs-PPL; counter-
result: Schmidt 1905.10617 (LMs self-recover). The unclaimed sliver is on-policy drift SLOPE as a
frozen gate inside an automated codec-search loop.

---

# Recommended first-5 execution sequence

1. **A1 Scale-Plane Side-Info Codec** (afternoon, CPU, zero ppl risk). The near-certain ~0.08-0.15
   b/w win, and the prerequisite for everything else: it converts the champion's rate from
   formula-counted to real-container-decoded bytes, which is exactly the symmetric accounting
   that A3's verdict demands before any lattice "flip" can be declared.
2. **A2 EscapeRules Stage 1** (same week, deterministic). Banks ~0.055 b/w more and measures the
   held-out ppl noise floor — the number that decides whether half the kill criteria in this
   roadmap (0.02-0.03 ppl margins) are even decidable. After steps 1-2 the champion sits at
   ~3.00 b/w / 4.483 ppl with fully honest accounting: the new baseline every later idea must beat.
3. **A3 Lattice Rate Amnesty Step 1** (minutes of CPU). The highest-scored item on the board
   (8.5, breakthrough-possible): re-score stored E8/D4 arrays under the decodable coset-
   conditional coder against the freshly re-baselined champion. One afternoon either re-opens the
   most consequential closed branch or closes it permanently with a SHA-checked negative.
4. **C1 Divergence Battery retro-audit** (~1 day). Before spending the next 10 rounds of mutator
   budget, learn whether 24 rounds of single-slice hill-climbing overfit the verifier and whether
   the champion ranking survives a KL/canary lens — this protects every subsequent selection
   decision, and the selection-overfitting curve is independently publishable.
5. **A4 TrellisForge pre-flight, then first ECTCQ round** (1 day gate + first round). The
   best-grounded remaining frontier lever (theorem-sanctioned ~0.15-0.2 b/w on exactly our
   source): measure the champion's true gap to the Gaussian R-D bound first — if < 0.08 b/w the
   prize doesn't exist and the loop pivots to the A6 sweep (which simultaneously gates B5) at
   zero sunk cost; if the gap is real, ECTCQ is the most likely path to a new champion.

Rationale for the ordering: certain wins and accounting symmetry first (1-2), because every
later iso-ppl/iso-bits comparison in Tiers A-B is meaningless against a leaky baseline; then the
cheapest possible shot at the largest closed-branch reversal (3); then verifier hardening before
committing rounds (4); then the highest-expected-gain frontier attack, itself gated by a free
pre-flight (5). Total cost of the full sequence: roughly one week, almost all CPU, with three
separate chances of a frontier-moving result and two permanent-knowledge results guaranteed.
