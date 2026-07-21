# The KV-Latent is the Bottleneck: A Rank-Robustness Law for Data-Free 2-Bit Mixture-of-Experts Quantization on a 6 GB GPU

*Working draft — target MLSys / NeurIPS ENLSP. All numbers are measured.*

## Abstract

Mixture-of-Experts (MoE) language models are large on disk but activate only a small slice of their
parameters per token, which makes them attractive for memory-constrained inference — provided they can
be quantized aggressively without collapsing. We study 2-bit post-training quantization of
DeepSeek-V2-Lite (15.7 B parameters, 64 routed + 2 shared experts per layer, multi-head latent
attention), the regime required to make a 16 B MoE fully resident on a 6 GB consumer GPU. Uniform 2-bit
quantization (every tensor at 2-bit) inflates WikiText-2 perplexity from 6.31 to 18.31 (+12.01) on the full test set, and we show the damage is *not*
located in the experts. Using a per-tensor, **data-free** trellis codec (no calibration set, no
fine-tuning, no Hessian), protecting only the attention and shared-expert tensors at 4 bits — a small
fraction of parameters — while leaving the 64 routed experts at 2 bits collapses the gap **more than fifteenfold**, to
+0.66 ppl (6.96) at 2.49 bits per quantized weight, with a **better gap-ratio than MxMoE** — the one prior 2-bit result on this model —
(1.10× vs 1.18×), without using any data. We pin the mechanism with a causal
decomposition (keep one tensor group at fp16, leave the rest at 2-bit, measure what is recovered): the
dominant low-bit sensitivity is the **internal projections** — attention q/kv and MLP gate/up — not the
residual writers and not routing. For DeepSeek's multi-head *latent* attention, **two tiny tensors, the
low-rank KV-latent projections, carry 87% of the collapse by themselves** — the very compression that
makes MLA efficient is what is most fragile at 2 bits. (Expert routing does diverge under quantization,
but *forcing* it back to fp16's choices recovers only ~21% — a secondary symptom of the same
hidden-state perturbation.) The finding holds across both families we test: on Qwen1.5-MoE (standard attention, no
latent) the internal projections still dominate, more mildly and distributed across q/k/v/gate/up. Our
2-bit weights are fully resident at 5.64 GB on a 2016-era GTX 1060, and we show a same-architecture model
generating text resident on that card (a generative runtime for our own packed weights is future work); by
a sequence of controls we show its single-stream speed is bounded not by the silicon but by the MoE
batch-1 memory access pattern. Probing *why* the KV-latent is fragile yields a law: it is a low-rank
information bottleneck, and **incoherence rotation — the foundation of QuIP#/QTIP/QuaRot/SpinQuant — is
rank-conditional**, helping full-rank tensors but catastrophic (+1623 ppl) on the bottleneck, which only
native-basis precision repairs. A systematic search for further data-free gains returns a unifying
negative — a load-balance-trained MoE is information-dense on weights (experts at the Gaussian
rate-distortion floor), routing (flat), and depth — so **2-bit is the floor, not a step**, and the residual
"fitting *and* fast" prize is a decode-utilization kernel, not a better code. Applied to dense models —
including Gemma 4 12B, weeks after release — the framework generalizes with one inversion: dense 2-bit
costs more (1.91×; no sparsity buffer, super-additive depth compounding), its fragility sits in the *late*
layers (the opposite of the MoE, invisible to weight statistics, found by a 30-minute functional probe),
and protecting just those 12 layers at 4-bit **halves the gap to 1.45×** at ~4.5 GB resident.

## 1. Introduction

Mixture-of-Experts architectures (DeepSeek-V2/V3, Mixtral, Qwen-MoE) decouple model capacity from
per-token compute: a 16 B-parameter model may activate only ~2.4 B parameters for any given token. For
*memory*-bound deployment — consumer GPUs, edge devices, single-card serving — this asymmetry is the
whole game. The model's quality lives in its full parameter count, which must be *stored*, while its
speed depends only on the active slice. The binding constraint is therefore memory: a 16 B MoE needs
~29 GB in bf16 and ~7.3 GB even at 4-bit, both of which overflow a 6 GB card. Only ~2-bit quantization,
at roughly 2.5 effective bits per weight (~5.6 GB), crosses the line.

The difficulty is that naive 2-bit quantization destroys MoE models. We measure WikiText-2 perplexity
rising from 6.31 to 18.31, and the literature reports outright divergence (RTN/AWQ/GPTQ at uniform
2-bit land at 10⁴–10⁵). The methods that recover quality at this bit-width — MC-MoE, EAQuant, MxMoE —
all rely on **calibration data**. This is doubly unfortunate for MoEs: a data-dependent method observes
each expert only on the tokens that happen to route to it, so rarely-routed experts are starved of
calibration signal exactly when they most need careful quantization.

We show that a **data-free** per-tensor codec sidesteps this starvation entirely — every expert is
quantized to the same fidelity regardless of routing frequency — and that a single mixed-precision rule
recovers near-full quality and beats the calibrated state of the art. The rule is to protect the
**attention and shared-expert** tensors at 4 bits (a small parameter fraction), leaving the 64 routed
experts at 2 bits. We then explain *why* by a causal decomposition that pinpoints, within those protected
blocks, the tensors that actually carry the error — for MLA, the low-rank KV-latent projections.

**Contributions.**
1. **The mechanism (our central finding):** a causal decomposition showing low-bit MoE quality is
   bottlenecked not by the routed experts but by the **internal projections** of attention and the shared
   experts — and for multi-head *latent* attention, two tiny low-rank KV-latent tensors carry **87% of the
   collapse alone**. Expert-routing divergence, the intuitive culprit, is only a 21% symptom (forcing fp16
   routing recovers just 21%).
2. **A data-free recipe that applies it:** protect attention + shared at 4 bits, leave the 64 routed
   experts at 2 bits → 6.96 ppl (+0.66 over fp16) at 2.49 quantized b/w, a **better gap-ratio than
   calibrated MxMoE (1.10× vs 1.18×)** with no calibration data. The win is not a bit-budget artifact: at a
   near-matched 2.44 b/w (vs MxMoE's 2.25) our light-AWQ variant still wins (6.93, 1.099×). Fully
   resident in 5.64 GB.
3. **Cross-architecture generality:** the recipe and mechanism replicate on Qwen1.5-MoE (standard MHA,
   top-4-of-60 routing) — uniform collapse (13.99) and carve-out recovery (7.75, gap-ratio 1.074×), with
   the internal projections again dominant and the routing collapse replicating (Figure 5).
4. **A capacity demonstration and a measured speed characterization:** our 2-bit weights are resident at
   5.64 GB on a 6 GB GTX 1060, and a same-architecture model generates text resident on it (generative
   decode from our own packed weights is future work); single-stream speed is bounded by the MoE batch-1
   access pattern, not the silicon — a memory win, not a speed win.
5. **The rank-robustness law (§5):** the KV-latent's fragility is a property of its low effective rank, and
   incoherence rotation — the dominant data-free tool — is **rank-conditional**: an exact gauge between
   kv_a and kv_b shows every rotation *worsens* the 2-bit bottleneck (Hadamard +1623, SVD +2779) while the
   *same* rotation is benign on a high-rank tensor (+0.006), a ~270,000× swing on effective rank alone.
   Only native-basis precision repairs it; keeping the KV-latent fp16 (≈2% of memory) is the best operating
   point (6.80). The same channel governs weight-, gauge-, and KV-cache-quantization fragility.
6. **The density result (§6):** a systematic eight-lens search finds a load-balance-trained MoE is
   near-optimally information-dense — experts at the Gaussian RD floor (1-bit collapses, +253), routing flat
   (dynamic top-k refuted), early layers ~40× more fragile than late — so data-free 2-bit is the floor, and
   the remaining prize is a kernel/systems problem, not a better code.

## 2. Method

**Codec (data-free).** We quantize each target weight matrix with a per-group (group size 128)
signed-Hadamard incoherence rotation followed by a QTIP-style bitshift-trellis scalar quantizer
(trellis length L = 12), with int8 per-group affine side-information. No activations, no Hessian, and no
fine-tuning enter the codec; the only inputs are the weights. Reconstruction is byte-structural and
quality is measured as WikiText-2 perplexity at sequence length 2048.

**Streaming evaluation.** To quantize and score a 16 B model on a 6 GB card, we never hold the dense
model. We embed all evaluation tokens once, then process one decoder layer at a time: materialize its
bf16 weights to fp32 on CPU, quantize each target linear in place, move the layer to the GPU, forward
every evaluation window at batch size 1 (which bounds the MLA attention memory), and free it. The
fully-resident size is computed analytically from the packed payload plus the fp16-kept tensors; we
verify it with a live generation run (§3.3).

**The mixed-precision rule.** Each target linear receives a bit-width by its role. The MLA attention
projections (q, kv_a, kv_b, o), the shared experts, and the dense layer-0 MLP are quantized at 4 bits;
the 64 routed experts' gate/up projections at 2 bits and their down projection (a residual writer) at 3
bits; the router (`mlp.gate`), token embeddings, output head, and all norms are kept in fp16. This
yields ≈2.49 b/w over quantized parameters and 2.87 b/w over the whole model. The rule mirrors the
mixed-precision idea of calibrated methods (MC-MoE, EAQuant, MxMoE) but is set data-free; §4 shows by a
causal decomposition that the operative tensors within the protected blocks are the *internal*
projections — for MLA, the low-rank KV-latent — which carry the dominant low-bit error, while
expert-routing divergence is only a secondary symptom. An optional one-pass static AWQ scaling (data-light) further closes the gap to fp16 (§3.1);
lower-bit operating points form the rate-distortion sweep.

## 3. Results

### 3.1 Quality

| Configuration | b/w (whole) | WikiText-2 ppl | gap vs own fp16 | gap-ratio |
|---|---:|---:|---:|---:|
| fp16 (ours, full test set) | 16.0 | 6.307 | — | 1.00 |
| Uniform 2-bit (all tensors at 2-bit) | 2.46 | 18.315 | +12.01 | 2.904× |
| **Carve-out 2-bit (ours)** | **2.87** | **6.962** | **+0.66** | **1.104×** |
| **+ KV-latent at fp16 (§5, +~2% mem)** | **2.92** | **6.805** | **+0.50** | **1.079×** |
| **+ static-AWQ (data-light)** | **2.88** | **6.768** | **+0.46** | **1.073×** |
| + AWQ at lower bits (2.44 b/w quant) | 2.83 | 6.933 | +0.63 | 1.099× |
| + drop writers (lowest-memory, 4.98 GB) | 2.54 | 7.603 | +1.30 | 1.206× |
| MxMoE (calibrated; their fp16 5.92, seqlen 4096) | 2.25 | 7.01 | +1.09 | 1.184× |

All perplexities use the standard GPTQ protocol (the full `wikitext-2-raw-v1` test split, the model's own
tokenizer, non-overlapping windows, PPL = exp(ΣNLL/Σtokens)) at sequence length 2048 over the full test
set (~151 windows). Our full-precision baseline, **6.31**, matches SINQ's independently-reported BF16 of
6.31 on this model under the identical protocol — external corroboration that our evaluation is the
standard one. To our knowledge DeepSeek-V2-Lite is quantized to ~2-bit and evaluated on WikiText-2 by
exactly one prior work, **MxMoE**; the other recent MoE-quantization papers either do not evaluate this
model (MC-MoE evaluates Mixtral; EAQuant evaluates the distinct DeepSeek-MoE-16B) or report no ~2-bit
point for it (SINQ reports only 3- and 4-bit). We therefore benchmark against MxMoE.

Absolute perplexity is not directly comparable across papers: **MxMoE evaluates at sequence length 4096**
(the default of their public harness) while we use the GPTQ-standard 2048, and longer context lowers
absolute perplexity. We thus compare via the within-paper **gap-ratio** (quantized PPL ÷ that paper's own
full-precision PPL), which largely cancels the context-length difference. Our data-free 2-bit carve-out
attains **1.10×** (6.96/6.31) versus **1.18×** for MxMoE at 2.25-bit (7.01/5.92) — a better ratio with no
calibration data, though at a slightly higher quantized budget (2.49 vs 2.25-bit weights). This is not a
bit-budget artifact: our light-AWQ variant at a near-matched **2.44 b/w** still reaches 1.099× (6.93),
below MxMoE's 1.184×.

The gap-ratio is **eval-length-invariant**: re-evaluating from the 16–32k-token subset of earlier drafts
to the full test set raises both our fp16 (5.66→6.31) and carve-out (6.25→6.96) but leaves the ratio
unchanged (1.105×→1.104×) — the penalty we report does not depend on evaluation length. Among genuinely
data-free methods on this model class, the closest published points are 3-bit (SINQ/HQQ/RTN =
7.45/8.36/7.94) and 4-bit (6.49–6.61); uniform 2-bit RTN/AWQ/GPTQ diverge. A one-pass static AWQ scaling
(512 calibration tokens, no fine-tuning — making the method data-*light*) closes a further part of the
gap, to **6.77** (gap-ratio 1.073×); we stress the MxMoE comparison is already won by the **data-free**
6.96, and the light-AWQ point is an optional further gain, not the basis of the data-free claim. The run
had zero quantization fallbacks, confirming that the static formulation (capture all activation statistics
in a single fp16 pass) eliminates the divergence the sequential AWQ variant suffered. At a lower bit
budget (attention/shared at 3-bit, 2.44 b/w quant) AWQ reaches **6.93** — Pareto-improving on the
data-free carve-out and shifting the rate-distortion frontier rather than trading along it (Figure 4).

### 3.2 The lever: protect attention and shared, not the experts

A single-layer ablation (one MoE layer at 2-bit, the rest fp16) costs only +0.143 ppl, yet the full
uniform model costs +12.01 — error accumulates super-linearly across the 26 MoE layers. The carve-out
collapses this gap more than fifteenfold (18.31 → 6.96; Figure 1) by spending its extra bits on the attention, shared,
and dense-layer-0 tensors. These are a small fraction of the parameters but the large fraction of the error.
The routed experts compress essentially for free at 2-bit: their data-free reconstruction rel-MSE is
6.8 × 10⁻², identical across every sampled expert, layer, and position, and equal to the Gaussian
rate-distortion bound D(R=2) = 0.069 — there is no per-expert quality variance to exploit, which is the
data-free advantage stated quantitatively.

### 3.3 Capacity demonstration

A 16 B MoE at 2-bit generates coherent text, GPU-resident, peaking at 5873 / 6144 MiB on a GTX 1060
(2016, no tensor cores) at ~5–6 tok/s (Figure 3). We demonstrate this with a llama.cpp IQ2 GGUF of the same
architecture family (the *capacity* claim — that a 16 B model runs resident on a 6 GB card — is what
this shows; community IQ2 files exist, so capacity itself is not the novel part). Our carve-out at
5.64 GB is smaller than that IQ2_XS file (5.97 GB), so it fits with more headroom. A live demo from our
own packed weights requires the deployable runtime (future work, §7).

### 3.4 Speed: a measured negative, and what actually bounds it

This is a memory/capacity result, not a speed result — but we pin *why* by measurement rather than
asserting a hardware limit. Single-stream generation of the 16 B MoE at 2-bit runs at **5–6 tok/s**
(llama-bench tg128, IQ2_XS, full offload; ~20% run-to-run variance over 5.3–6.3). A sequence of controls, all on the same GTX 1060, isolates the
cause:

| measurement | tok/s | bandwidth util |
|---|---:|---:|
| MoE 16 B IQ2, **prompt** (batched) | 145 | (amortized) |
| MoE 16 B IQ2, **generation** (batch-1) | 5.3–6.3 | ~2% |
| dense 7 B Q4, generation | 22.4 | ~41% |
| dense 7 B IQ3 (IQ-quant), generation | 19.6 | ~33% |

Each candidate cause is ruled out by data: **not the silicon** (the card sustains 22 tok/s at ~41%
bandwidth utilization on a dense model); **not bandwidth** (the MoE reads ~6× *fewer* bytes/token than
the dense model yet runs 4× slower); **not the IQ-quant codebook** (dense IQ3 ≈ dense Q4 — a ~20% gap,
not 20×); **not memory spill** (freeing VRAM left generation unchanged, 6.33→6.34); and **not
amortizable by batching** (parallel generation stays flat, 5→7 tok/s from 1 to 8 streams). The residual
is the **MoE batch-1 access pattern itself**: dynamically-routed experts produce scattered,
latency-bound weight reads and tiny one-token-per-expert GEMVs that use a sliver of the GPU, leaving it
at ~2% utilization where a dense model reaches ~40%.

The statement is therefore sharper than "low-bit is slow on Pascal": **MoE sparsity — the property that
lets a 16 B model fit in 6 GB — is a throughput optimization that becomes a single-stream *latency*
pessimization, because dynamic routing defeats both memory coalescing and batch amortization.** The same
sparsity that wins memory loses single-stream speed; batched throughput (145 tok/s) is unaffected.
Speculative decoding, CUDA graphs, and a faster quant kernel are each refuted as remedies by the
controls above. Whether this tension can be *broken* — fitting *and* fast — is the question for future
work (§7).

### 3.5 Generality: the recipe transfers to a second architecture

To test whether "protect attention + shared" is a general rule or a DeepSeek-V2-Lite artifact, we apply
the *identical* data-free carve-out to **Qwen1.5-MoE-A2.7B** — a deliberately different MoE: standard
multi-head attention (no MLA), 60 routed + 1 shared expert per layer, every layer sparse, plain top-4
softmax routing. The quantization code is identical; only the model wiring differs.

| model | attention | fp16 | carve-out 2-bit | gap | gap-ratio |
|---|---|---:|---:|---:|---:|
| DeepSeek-V2-Lite | MLA | 6.307 | 6.962 | +0.66 | 1.104× |
| **Qwen1.5-MoE-A2.7B** | **MHA** | **7.217** | **7.749** | **+0.53** | **1.074×** |

The carve-out holds quality on Qwen-MoE just as well — gap-ratio 1.074×, even slightly *tighter* than
DeepSeek's 1.104× — at 2.55 b/w quant, fully resident in 5.6 GB. The carve-out rule is
not MLA-specific (it holds on standard-attention Qwen too). **Uniform 2-bit collapses both architectures** (DeepSeek 18.31,
+12.01, gap-ratio 2.904×; Qwen 13.99, +6.77, 1.939×) — and DeepSeek's MLA collapses *harder*, consistent
with its uniquely fragile KV-latent (§4) — while the carve-out recovers both to near-fp16, an 18× / 13×
gap reduction respectively. On Qwen the routing also collapses under
uniform 2-bit (overlap falls to 0.67 vs the carve-out's ≥0.86) while the carve-out holds it —
replicating the routing *phenomenon* on a non-MLA architecture (Figure 5) — a secondary symptom; the
dominant lever is the internal projections (§4).

## 4. Mechanism: the attention/MLP internals — for MLA, the KV-latent — not routing or the writers

We isolate *what* the carve-out protects with a battery of cheap causal interventions on the uniform-2bit
model (ppl 18.31, fp16 6.31, full test set): keep one group of tensors at fp16, leave the rest at 2-bit, and measure how
much of the collapse is recovered. Routing is tested separately by *forcing* the quantized model to
replay fp16's expert selection (off the cache, no re-quant).

| keep at fp16 (rest 2-bit) | collapse recovered (marginal) |
|---|---:|
| force fp16 routing | 21% |
| residual writers (o_proj, down_proj) | 49% |
| **internal projections (q/kv, gate/up)** | **95%** |
| **— MLA KV-latent alone (kv_a, kv_b)** | **87%** |
| gate/up | 32% |
| q_proj | 23% |

Each row is a *marginal* intervention (one group restored to fp16 and measured independently against the
uniform baseline), so the rows overlap and are not meant to sum to 100% — the internals and the writers
share a single error pathway. The percentages are computed against the full-set baseline (uniform 18.31,
fp16 6.31) and are stable across evaluation length — the KV-latent recovers 86% on the 16k-token subset
and 87% on the full set — and the load-bearing claim below does not depend on this marginal decomposition
at all. Even so the ranking is unambiguous: the dominant low-bit sensitivity is the
**internal projections** (attention q/kv and MLP gate/up), not the residual writers and not routing. For DeepSeek's multi-head
*latent* attention, **two tiny tensors — the low-rank KV-latent projections (kv_a, kv_b) — carry 87% of
the entire collapse.** MLA compresses K/V through a low-rank latent for efficiency, and that compression
is precisely what is most fragile at 2 bits. Expert routing *does* diverge under quantization (Figures
2, 5: overlap collapses from ≥0.85 under the carve-out to 0.58 under uniform), but forcing the quantized
model to use fp16's exact routing recovers only **21%** of the collapse — because the routing changes and
the direct error share an upstream cause: the hidden-state perturbation from quantizing these internal
projections. Routing divergence is a measurable symptom, not the lever.

**Generality.** On Qwen1.5-MoE (standard MHA, no latent) the internal projections still dominate
(internal-fp16 recovers 82% vs the writers' 64%), but more mildly and spread across q/k/v/gate/up — there
is no single 87% tensor. So "protect the internals, not the writers or routing" holds on both families we test;
the *extreme* concentration in two tensors is MLA-specific.

**A caveat on the decomposition — and why the carve-out is well-tuned.** These percentages are measured
against the *uniform* baseline, where everything is already broken, so they do **not** transfer linearly
to the low-error carve-out regime — but the *ranking* does, which we confirm with two interventions in the
operating regime itself. Dropping the residual writers to 2-bit in the carve-out costs only +0.64 ppl for
−0.35 b/w (§3.1, the 4.98 GB point), a modest rate-distortion trade. Dropping just the **two KV-latent
tensors** (kv_a, kv_b) to 2-bit, by contrast, costs **+5.27 ppl** (6.96 → 12.23) — roughly 8× more, from
far fewer parameters. And the effect is specific to the KV-latent, not to small or low-rank tensors in
general: dropping an *equal parameter count* — a 3.28M-parameter row-slice of the full-rank q_proj — to
2-bit costs only **+0.12 ppl**, a 43× smaller penalty from the same number of bits. So while the precise 87% does not carry over (dropping the KV-latent recovers ≈46%
of the carve-out's advantage over uniform, not 87%), the qualitative finding is emphatic: even in the
deployed carve-out, those two tiny low-rank tensors are by far the most valuable bits, and the carve-out
spends its bits where the error concentrates.

## 5. The Rank-Robustness Law: why the KV-latent is fragile, and why rotation cannot fix it

§4 *locates* the fragility in the KV-latent. This section explains *why* it is fragile and shows the
explanation generalizes into a law about the dominant tool in modern data-free quantization — incoherence
rotation — which we find is not universal but **rank-conditional**.

**The KV-latent is a genuine low-rank bottleneck.** Computing the effective rank (entropy of the singular
spectrum) of the map each tensor implements, the composed KV map (kv_b·kv_a through the 512-latent) is the
most spectrally concentrated tensor in the layer — effective rank ≈394 against a 2048 nominal dimension
(flatness 0.19) — while the routed experts are the flattest (0.57–0.63), and the attention q_proj sits
between (0.36). The same ordering appears in a purely local statistic: the excess kurtosis of the weights
is +0.2 for the experts (near-Gaussian, maximal-entropy) but +1.3 for kv_a (structured, heavy-tailed).
Effective rank and kurtosis are thus **data-free predictors of low-bit fragility**, and the KV-latent is
the extreme of the axis.

**Fragility is gauge-sensitive but basis-*privileged*: rotation cannot relocate it.** MLA's latent admits
an exact gauge freedom — an invertible R inserted between kv_a and kv_b (kv_a ← R·kv_a, kv_b ← kv_b·R⁻¹)
leaves the fp16 function bit-identical (we verify max|Δ| ≈ 10⁻⁷). It does *not* leave the 2-bit function
identical, so the gauge relocates quantization error for free. We sweep R and find the **opposite of the
intuition that whitening helps**: every rotation makes the 2-bit KV-latent *worse*, and orthogonal
rotations are catastrophic.

| gauge R between kv_a, kv_b (2-bit) | Δppl vs carve-out |
|---|---:|
| identity (native basis) | +5.27 |
| diagonal balance (SmoothQuant-style) | +6.23 |
| signed-Hadamard incoherence | **+1623** |
| SVD (rotate to singular basis) | **+2779** |
| **keep KV-latent at fp16 (no quantization)** | **−0.16** (6.80) |

The fp16 row is the practical consequence: keeping the two KV-latent tensors at 16 bits — ~88 M parameters,
≈2% of the resident model — yields **6.80 ppl, the best non-trivial operating point**, below the 4-bit
carve-out (6.96). The bottleneck's only remedy is precision in its *native* basis; no gauge helps.

**The dichotomy: incoherence is rank-conditional.** The result is not that rotation is bad — it is that
rotation's effect is governed by rank. We apply the *same* random-orthogonal-rotation-then-2-bit operation
to a **high-rank** intermediate (the shared-expert MLP, effective rank ≈1168) and to the **low-rank**
KV-latent (≈394). The high-rank tensor is unharmed (+0.006 ppl); the low-rank bottleneck is destroyed
(+1623). A 3× difference in effective rank produces a ~270,000× difference in rotation sensitivity. This is
the law:

> **Incoherence processing — the foundation of QuIP#, QTIP, QuaRot, and SpinQuant — is rank-conditional,
> not universal.** It whitens full-rank tensors (spreading outliers across many directions that average
> out — which is *why* our routed experts reach the Gaussian RD floor under the codec's per-group Hadamard)
> but is catastrophic on low-rank bottlenecks, where it smears the few high-variance directions across all
> coordinates that a coarse 2-bit grid then shreds. The data-free discriminator is effective rank × kurtosis.

The field's universal tool inverts sign on exactly the structures efficient architectures (MLA, and by
extension LoRA, GQA, adapters) deliberately build. This has gone unseen because incoherence methods predate
MLA's bottlenecks and are benchmarked on dense, full-rank Llama, where rotation always helps.

**The fragility is the compression *write*.** Decomposing the bottleneck below the tensor: quantizing only
kv_a (the hidden→latent squeeze, the act of compression) to 2-bit costs +2.95 ppl, while quantizing only
kv_b (the latent→heads expansion) costs +0.25 — the damage lives in the compression step, not the storage
or expansion, nailing the "information bottleneck" reading to a specific operation.

**The same channel, in cache form.** In MLA the low-rank latent *is* the KV cache. Quantizing the c_KV
*activation* (rather than the weights that write it) shows the identical signature with a sharp
critical-bit-width: 8-bit is free (+0.018 ppl), then it cliffs — 4-bit +4.87, 3-bit +350, 2-bit collapse.
So the bottleneck appears in three forms — weight-quant (+5.27), gauge-rotation (+1623), cache-quant
(collapse below 8-bit) — all the same low-rank channel, none rescuable by incoherence, all fixed only by
precision. The 8-bit-cache point is a free, context-scaling memory harvest that weight-quantization SOTA
never touches.

## 6. The Density of Trained MoEs

Having found one fragile channel and one free harvest, we asked systematically whether *any* further
data-free gain remains, sweeping the axes a static codec cannot normally see. The answer is a unifying
negative: **a load-balance-trained MoE is near-optimally information-dense on every axis we can reach.**

- **Weights.** The routed experts sit exactly at the Gaussian rate-distortion floor (rel-MSE = D(R=2) =
  0.069, identical across all 64) and are SVD-irreducible (a stacked-expert decomposition leaves only
  1.6–2% in a shared backbone; the remaining variation is spread flat across all 64 modes — no
  backbone+low-rank factorization exists). Forcing them below 2 bits collapses quality: a valid binary
  (1-bit) code, at the textbook Gaussian limit rel-MSE = 1−2/π ≈ 0.365, gives +253 ppl, codec-independently
  (even an optimal 1-bit code's D(R=1)=0.25 is 3.6× the 2-bit distortion). **2-bit is the genuine functional
  floor for the experts**, not a conservative choice.
- **Routing.** The renormalized top-6 routing mass is flat — top-1 ≈0.33, top-2 ≈0.53; a token needs ~5.3
  of its 6 experts to cover 90% of the mass, and 0% of tokens are top-2-dominated. Dynamic top-k therefore
  cannot reduce bytes-per-token, and the bandwidth ceiling (§3.4) holds. The flatness is also
  *domain-conditional*: routing 2048 tokens of prose vs. 2048 of source code, both domains use ~50 of 64
  experts for 90% of routings, zero experts go unused, and the used sets are **identical** (Jaccard 1.00)
  — there are no task-specific "brain regions" to trim. Load-balancing fills the routing, marginally and
  conditionally, just as training fills the weights.
- **Depth.** Fragility is depth-graded: confining 1-bit experts to the early half costs +102 ppl but to
  the late half only +5.6 (the late *quarter* +2.4) — early-layer error propagates through all downstream
  layers, late-layer error does not. Bit allocation thus has a second variable beyond per-tensor rank
  (error-propagation depth), but the early layers anchor the 2-bit floor, so the net memory headroom is
  small.
- **Time.** The KV cache is a trajectory, not i.i.d.: consecutive c_KV latents are weakly correlated
  (lag-1 autocorrelation 0.19–0.46, strongest mid-network), giving ~0.2–0.4 b/w to predictive/delta-coding
  on the cache — a modest, genuinely-new axis on the context-growing term.

An exhaustive search across eight lenses (information-theoretic, systems, architecture-surgery,
dynamic-computation, numerical, activation-sparsity, cross-domain analogy, joint-3-axis) produced 30
candidate levers; under adversarial filtering against the findings above, exactly one survived — and on
measurement it turns out to be a *rate–distortion trade*, not even the free harvest it promised. The two
token-indexed matrices (embed_tokens, lm_head) are the only weight tensors no mechanism touches; quantizing
them to 4-bit per-128 frees ~0.63 GB (≈13% of resident) but costs **+0.45 ppl** (6.96 → 7.41), and 3-bit
cliffs to +2.92 — the weight error is tiny (rel-MSE 0.015) but lm_head feeds the softmax directly, so it
amplifies into the logits. Their "clever" version (entropy-coding the embedding's Zipfian structure) is
measured dead outright (k-means removes only 6–10% of variance; the rows are near-Gaussian and
token-frequency does not transfer to row geometry). So even the single surviving axis yields no free bits —
which sharpens rather than softens the conclusion.

The density result reframes the contribution. **Data-free 2-bit is not a step on a ladder we can keep
descending — it is the floor**, because load-balanced training drives weights, routing, and depth to
capacity simultaneously. What remains is therefore not compression but *engineering*: a decode-utilization
kernel to reach the bandwidth ceiling (§3.4, §9), and the modest harvest (8-bit cache + depth-grade +
temporal + 4-bit embed/lm_head).

## 7. Dense models: the law transfers, the sparsity buffer does not

To test whether the framework survives outside MoE entirely, we applied it to two dense models — including
**Gemma 4 12B**, released weeks before this experiment (48 layers, GQA, GeGLU with a 4× intermediate,
262K tied embeddings, alternating sliding-window/global attention) — streamed and quantized on the same
6 GB GPU.

**Incoherence rescues dense 2-bit (third architecture).** On Qwen2.5-7B, 2-bit MLPs with a plain
Lloyd-Max codec collapse (ppl 28,119); the *same* bit-width under the incoherence-trellis codec holds at
6.52 (fp16 5.13) — a ~4,300× rescue, confirming the rank-conditional law on full-rank dense MLPs.

**Dense has no sparsity buffer.** Where the MoE's routed experts absorb 2-bit noise (only ~6/64 fire per
token), every dense MLP weight fires every token. The compounding is directly measurable and strongly
super-additive: on Gemma, three 12-layer bands costing +2.1/+3.2/+3.2 ppl individually cost **+20.8
jointly**. The uniform carve-out lands at **1.91×** (7.37 → 14.06) — the honest price of dense 2-bit.

**The depth-fragility direction inverts — and only functional tests see it.** A per-band functional probe
(each 12-layer band's MLPs to 2-bit, rest fp16, identical windows) shows Gemma's **late** layers are ~4×
more fragile (+7.98) than its early ones (+2.14) — the *opposite* of the MoE's early-layer fragility, and
the opposite of what the data-free kurtosis signature predicts (early layers are the heavy-tailed ones).
The regimes differ: at 1-bit (catastrophic), early error compounds through depth; at 2-bit (moderate),
late error feeds the tied-embedding logit head with no downstream washout. **Depth-fragility direction is
architecture- and regime-dependent, and weight statistics do not predict it** — a 30-minute functional
probe does.

**Depth-aware allocation halves the gap — and transfers to a second codec.** Promoting only the 12
late-layer MLPs to 4-bit (+0.53 GB) yields **1.45×** (10.71), cutting the dense penalty from +6.69 to
+3.34 at ~4.5 GB resident. The finding is codec-independent: reproduced in stock llama.cpp k-quants
(`--tensor-type`, no code changes, no imatrix), uniform Q2_K FFNs give 14.41; protecting the *first* 12
layers gives 12.27; protecting the *last* 12 gives **10.02 — from a byte-identical file**. The same extra
bytes buy 2× more quality when placed on the measured-fragile band:

| model | 2-bit config | gap-ratio |
|---|---|---:|
| DeepSeek-V2-Lite 16B (MoE) | carve-out | **1.10×** |
| Qwen2.5-7B (dense) | carve-out | 1.39× |
| Gemma 4 12B (dense) | uniform carve-out | 1.91× |
| Gemma 4 12B (dense) | **+ depth-aware (late-12 @ 4-bit)** | **1.45×** |

**The fragility atlas: the direction is model-specific, so the probe is the method.** Repeating the
band probe across dense families overturns any universal placement rule:

| model | head | MLP | fragile end | band Δppl (early → late) |
|---|---|---|---|---|
| Gemma 4 12B | tied | GeGLU | late (~4×) | +2.14 / +3.22 / +3.16 / **+7.98** |
| Qwen2.5-7B | untied | SwiGLU | late (~2–3×) | +0.88 / +0.54 / +0.58 / **+1.85** |
| Mistral-7B-v0.3 | untied | SwiGLU | **early (~25×)** | **+6.53** / +0.15 / +0.22 / +0.26 |

Qwen's late spike on an *untied* head refutes tied-embedding amplification as the primary mechanism
(output-proximity survives); Mistral — an architectural near-twin of Qwen — then inverts the direction
entirely. **No configuration flag, architecture family, or weight statistic predicts where a model is
fragile; a ~30-minute data-free functional band probe measures it**, and the stakes of guessing wrong
range up to a 25× fragility differential. Probe-then-quantize, not rules-of-thumb, is the transferable
method.

**The density law extends to activations.** Every structural lever is measured-dead on Gemma too: MLPs
spectrally flat (top-10% SV energy 0.22–0.41), no outlier concentration (top-0.1% of weights ≈ 3% of
energy), element stream white (0 dB vector-quantization coding gain). Beyond weights: GeGLU activation
energy is *diffuse* — 72–84% of the 15360 intermediate neurons are needed for 90% of the energy in every
domain we probed — killing contextual precision-paging (interestingly, domains *do* activate distinct
top-neuron sets, Jaccard ≈ 0.3: semantic differentiation exists, but without concentration it yields no
compression leverage). Weights at the RD floor, routing flat, activations diffuse: **trained networks are
dense on every axis a data-free method can reach**, and the remaining freedom is only *where* the forced
bits go — which the depth-aware result shows is worth a halving of the gap.

(Gemma evaluations use 8×2048-token WikiText-2 windows under the identical streaming protocol as its own
fp16 baseline; the gap-ratios are within-protocol and consistent with the full-set MoE results.)

## 8. The tiered decode law: placement across memory tiers

The placement principle extends from *which layers get bits* to *which memory tier serves them*. On
commodity hardware, batch-1 decode throughput obeys a single measured law:

$$\text{tok/s} = \eta(\text{tier}) \cdot \frac{BW_{\text{tier}}}{\text{active bytes per token}}$$

with the utilization constant η collapsing per tier across every configuration we measured — and,
notably, across **colibri**'s independently published 744B-MoE tiers:

| tier | η range (ours) | colibri's published points |
|---|---|---|
| VRAM (GTX 1060) | 0.56 | — |
| RAM/CPU | 0.29–0.68 (dense ≈0.65, MoE ≈0.35) | 128 GB desktop: **0.48** |
| disk (SATA/NVMe) | 1.00 | 25 GB cold tier: **0.88** |

One equation spans 7B→744B and both projects' hardware. Its terms are each independently measured:
the **1/bytes** numerator (dense 7B at Q2 vs Q4, same CPU tier: 1.41× measured vs 1.56× pure-bandwidth
bound — the residual is the codec's decode cost, η(Q2)<η(Q4)); the **MoE scatter penalty** (η ≈0.35 vs
dense ≈0.65 on the same tier); **soft tier boundaries** (a model near RAM capacity enters a mixed
RAM+paging regime with unstable throughput); and the **disk tier's η≈1** (bandwidth-saturated, codec-free).

Three practical inversions follow, each measured: on Pascal-class GPUs, **serving MoE experts from CPU
RAM beats VRAM** (+54%, one llama.cpp flag — poor low-bit decode utilization makes the GPU tier
latency-bound); for a 30B-A3B MoE on a 16 GB box, **the CPU alone (12.6 tok/s) beats the CPU+GPU
hybrid** (RAM contention); and **batch scaling returns** once experts live on the CPU tier (4.9→22.1
aggregate tok/s at batch 8, where the GPU-resident MoE was flat).

The law also prices the *streaming* frontier: GLM-4.5-Air (110B-A12B, 2.7× our RAM) runs from a SATA
drive at **0.19 tok/s — inside the band the law predicted before the download finished** (0.2–0.3).
Its strongest test was a *hardware intervention predicted in advance*: raising DRAM from 2133 to
3000 MT/s (enabling XMP, a +41% bandwidth change) was pre-registered to scale in-RAM decode by ~×1.41;
the dense model delivered **×1.52** (7.18→10.88, the excess from the accompanying latency gain) and the
in-RAM MoE ×1.32 (11.81→15.58). A model whose working set sat at the RAM-capacity boundary instead
became *unstable* — the bottleneck **migrated** from bandwidth to capacity, exactly the behavior a
law-governed tier system must exhibit. A further corollary: speculative decoding is *antagonistic* to
MoE sparsity on bandwidth-bound tiers — a draft model made the 30B **2.3× slower** (4.61 vs 10.4),
because verification batches read the *union* of the drafted tokens' experts (~40 per layer instead
of 8), destroying the per-token sparsity that made the tier fast; the magnitude is predicted by the
same batch-throughput measurements.
Two colibri-relevant corollaries: router lookahead is architecture-general and stronger than reported
(91.2% top-6 predictability one layer ahead on DeepSeek-V2-Lite vs 71.6% on GLM-5.2; a top-12 prefetch
covers 98.9%), and converting a streamed int4 expert tier to 2-bit — quality held by probing for the
fragile band first — should yield ≈2× throughput on disk-bound tiers (η≈1: bytes are everything) and
1.4–1.7× on RAM tiers (decode-cost discount).

## 9. Related Work

**MoE quantization.** MxMoE allocates mixed precision with GPTQ-style calibration and is, to our
knowledge, the only prior work reporting a ~2-bit WikiText-2 result on DeepSeek-V2-Lite (7.01 at 2.25
b/w, at sequence length 4096); MC-MoE allocates bits by per-expert importance (evaluated on Mixtral), and
EAQuant aligns router behavior (evaluated on the distinct DeepSeek-MoE-16B). All are data-dependent. We match the mixed-precision *idea* but derive the
allocation from a role-based prior (protect attention and shared) and validate it data-free — with §4
supplying the mechanism (the internal projections, for MLA the KV-latent).

**Data-free PTQ.** QTIP and QuIP# use incoherence rotation and trellis/lattice codes but ship
tensor-core kernels (Ampere+); SINQ, HQQ, and RTN are data-free but reported at ≥3-bit on this class.
Our codec is QTIP-class but runs (slowly) on pre-tensor-core Pascal.

**Fitting large models on small GPUs.** llama.cpp's IQ-quants already produce sub-6 GB files for this
architecture, so the *capacity* claim is not novel; what is new is a *measured-quality*, data-free,
fully-resident 2-bit MoE on a 6 GB consumer GPU that achieves a better gap-ratio than MxMoE — the one prior 2-bit result on this
model — with a mechanistic account of why the allocation works.

## 10. Limitations

We evaluate two MoE families — DeepSeek-V2-Lite (MLA) and Qwen1.5-MoE (standard attention); generality to
larger and structurally different MoEs (Mixtral, DeepSeek-V3) is untested. Quality is measured by
perplexity on the full WikiText-2 test set (we verified the penalty is invariant from a 16k–32k-token
subset to the full set), itself a proxy for downstream task accuracy. The result is a memory win, not a single-stream-speed
win (§3.4); we have not yet wired a fused 2-bit decode kernel to the MoE, so the capacity demonstration
uses a third-party codec. The mixed-precision rule is a physically-motivated recombination of
known priors, validated empirically, rather than a new primitive. The rank-robustness law (§5) is
demonstrated on MLA's KV-latent — the one architecture in our study with an *extreme* low-rank bottleneck —
via the gauge sweep and the high-vs-low-rank dichotomy; the *trend* (rank predicts rotation sensitivity)
should be checked on more bottleneck families (LoRA-merged, GQA, SSM states), which we conjecture are
progressively more fragile. The density result (§6) bounds *data-free post-training* compression
specifically: quantization-aware training or distillation are not subject to it and remain the route to a
genuinely smaller model. The depth and temporal axes (§6) are characterized but not yet harvested into the
packed container, and the decode-utilization kernel (§10) is designed but unbuilt. The Gemma 4 results
(§7) use 8×2048-token WikiText-2 windows (a ~16k-token subset, consistent with its own fp16 baseline under
the identical protocol) rather than the full test set, and the depth-band curve was mapped with a fast
conservative codec whose rankings — but not magnitudes — transfer to the trellis; the final 1.45× is a full
trellis measurement.

## 11. Future Work: fitting *and* fast?

The speed result (§3.4) has a precise, measured cause — MoE sparsity defeats single-stream memory
coalescing — yet the byte counts say the tension *need not* be fundamental: the MoE reads ~6× *fewer*
bytes per token than an equivalent dense model, so at the dense model's memory efficiency it would run
*faster* than dense. We probed the most natural lever — temporal routing *locality*, which would let a
hot-expert cache coalesce the reads — and **measured it away**: consecutive tokens reuse experts only
2.5× above chance, and the working set reaches ~51 of 64 routed experts within 32 tokens and ~62 within
128. A small hot cache is therefore not viable; over any useful window a stream touches nearly every
expert. What remains is not a caching insight but a kernel problem — a fused, coalesced batch-1
expert-GEMV that reads the (few) active-expert bytes at bandwidth, with a ceiling of the active-byte
bandwidth bound (~100–250 tok/s) and the cost of a substantial Pascal kernel build. We report the
negative because it redirects the effort precisely: the prize (the model that fits being the fastest) is
real and bounded above the current rate, but it is a systems effort, not a free lunch from locality.

The codec choice is part of the kernel problem, and it cuts against the rate-optimal instinct: on Pascal,
`tok/s = bandwidth × utilization / resident_bytes`, so **decode-utilization, not bit-count, is the speed
variable**. The rate-optimal trellis (which saturates the expert RD floor) is decode-*pessimal* — measured
at ~12 tok/s at ~10% utilization — whereas a coarser 4-bit register-LUT reaches ~22 tok/s at ~44% even
though it moves *more* bytes. The 6 GB budget forbids simply spending more bits (a 4-bit 16 B model is
8 GB), so the target is a **2-bit code that is itself gather/LUT-decodable** (decode-as-gather, DP4A
accumulation) — and the experts' maximal entropy is an asset here, since equiprobable 2-bit levels make the
LUT buckets balanced. The kernel that reaches the ceiling is therefore not the most compressive codec but
the most decode-coalesced one, exactly inverting the datacenter codec priority.

## 12. Conclusion

Aggressive low-bit quantization of Mixture-of-Experts models fails not in the experts but in the
attention and shared-expert blocks — and, by causal decomposition, in their *internal* projections
specifically: for multi-head latent attention, two tiny KV-latent tensors carry 87% of the collapse.
Protecting attention and shared at 4 bits while leaving the 64 routed experts at 2 bits — a small
parameter fraction — recovers near-full quality at ~2.5 bits, data-free, achieves a better gap-ratio than calibrated MxMoE,
and fits a 16 B model on a 6 GB consumer GPU. The mechanism is *measured*, not assumed: expert-routing
divergence, the intuitive culprit, is only a 21% symptom.

Pressing on the KV-latent yields a law: its fragility is not a bit-budget to be reallocated but a property
of its low effective rank, and **incoherence rotation — the foundation of the dominant data-free codecs — is
rank-conditional**, helping full-rank tensors but catastrophic (+1623 ppl) on the low-rank bottleneck, which
only native-basis precision repairs. Searching for any further gain returns a unifying negative: a
load-balance-trained MoE is information-dense on weights (experts at the Gaussian RD floor), routing (flat),
and depth — so **data-free 2-bit is the floor, not a step**, and the remaining prize (fitting *and* fast) is
a decode-utilization kernel, not a better code. The framework then generalizes to dense models — Gemma 4
12B, quantized weeks after its release, runs on the same 6 GB card at 1.45× after a functional depth probe
revealed its fragility sits in the *late* layers, the opposite of the MoE and invisible to weight
statistics. The work overturns four intuitions — that the experts are the weak point, that low-bit MoE
quality is about routing, that incoherence always helps, and that depth-fragility has a universal
direction — and replaces them with a rank-robustness account of where the bits, and the bottlenecks,
actually are.

## Figures / artifacts
- F1: gap-collapse bar (uniform 18.31 → carve-out 6.96 → +AWQ 6.77 → fp16 6.31; MxMoE 7.01 at seqlen 4096), full test set — fig_gap_collapse.png.
- F2: routing overlap & top-1 agreement vs depth, carve-out vs uniform — fig_routing_divergence.png.
- F3: VRAM trace, 16 B MoE generating at 5873/6144 MiB — fig_vram.png.
- F4: rate-distortion frontier as gap-ratio (data-free + light-AWQ below MxMoE; lowest-memory drop-writers point) — fig_rate_distortion.png.
- F5: routing overlap vs depth on both architectures (DeepSeek MLA, Qwen MHA) — fig_generality_routing.png.
- F6 (§5): the rank-conditional dichotomy — Δppl of the same orthogonal-rotation+2-bit op vs effective rank (KV-latent eff 394 → +1623; shared-MLP eff 1168 → +0.006), with the gauge sweep (identity/diag/Hadamard/SVD/fp16) — fig_rank_robustness.png.
- F7 (§5): the bottleneck in three forms — weight-quant (+5.27), gauge-rotation (+1623), KV-cache precision ladder (8-bit free → cliff) — fig_bottleneck_forms.png.
- F8 (§6): density panel — expert 1/2/3-bit floor (1-bit +253), routing top-k mass concentration (mean k≈5.3/6), depth-selective 1-bit gradient (early +102 vs late +5.6) — fig_density.png.
- Data (round 1): moe_results.txt, route_diverge{,_uniform,_qwen_*}.txt, forced_output/forced_routing logs, route_locality.txt, MOE_2BIT_RESULTS.md, MOE_GEN_DEMO.md.
- Data (round 2, §5–6): gauge_*.log + m1_gauge.py (gauge sweep), dichotomy_*.log + m1_dichotomy.py (rank-conditional rotation), kvcache_*.log + kv_cache_quant.py (cache ladder), kvprobe_*.log + m1_kvprobe.py (down-vs-up), expert_bits.py + expdepth_*.log (expert floor + depth), router_confidence.py, cache_temporal.py, rank_fragility.py, embed_probe.py + embedharvest_*.log.
- Data (round 3, §7 dense/Gemma): dense_2bit_gate.py + dense_gate_*.log (Qwen dense gates), evoq_gemma.py + gemma_fp16*.log / gemma_carveout.log / gemma_flipped.log (fp16 7.37 / uniform 14.06 / depth-aware 10.71), gemma_band_*.log (inverted depth curve + flip validation), gemma_lever_probe.py + gemma_lever.log (structure kills), gemma_semantic_probe.py + gemma_semantic.log (diffuse activations), x_compression_chart.png (cross-model scoreboard).
