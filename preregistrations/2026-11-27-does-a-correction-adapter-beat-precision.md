# Pre-registration #114 — does a low-rank correction buy more quality per byte than precision?

**Staked 2026-11-27, before any error spectrum of the model below was measured.**

## Why this one

The November memory-efficiency analysis promoted **correction adapters**: keep the quantized
weights, and add a small low-rank matrix that cancels part of their error. llama.cpp can already
load one at runtime (`--lora`) over a quantized base. Published methods report large recoveries
(LQER, QERA, EoRA; "Align, Then Correct", arXiv 2610.08164, recovers 51–84% of a 2-bit gap on
Qwen3). But they correct quantizers that do not see activations. No one has measured them against
what this project ships: llama.cpp K-quants built **with an importance matrix**, which already spend
precision where activations are large.

The question that decides whether this becomes a lever: **at equal bytes, does a rank-r correction
remove more error than spending the same bytes on a better format?** This stage answers it in
llama.cpp's own error metric, in about an hour of CPU, before any adapter file is built or any
perplexity run is spent.

## First principles

- **Metric.** llama-quantize minimises the per-weight squared error weighted by the importance
  of its input column, `d_j = in_sum2_j / count`, the mean squared activation of channel j. Here
  `ε = Σ_ij d_j (w_ij − ŵ_ij)²`, reported relative to the tensor's weighted energy.
- **The adapter's best case is a spectrum.** A rank-r correction `B·A` fitted to that metric
  removes exactly the top r squared singular values of `E·D^(1/2)` (Eckart–Young). So the
  eigen-spectrum of the weighted error *is* the best adapter at every rank. Nothing has to be
  trained.
- **Cost.** r·(rows + cols) numbers, stored as llama.cpp allows: Q8_0 (8.5 bits) for a row length
  that is a multiple of 32, F16 otherwise. On a 5120 × 17408 FFN matrix that is about 0.0021
  bits per weight per rank.
- **The white-noise null.** If the weighted error were i.i.d., rank r would remove about
  `r·(1+√c)²/min(m,n)` of it (Marchenko–Pastur edge): 0.74% at rank 16 and 3.0% at rank 64 for
  these shapes. By the usual ~6 dB per bit, spending +0.25 bits per weight to move a Q2_K tensor
  toward Q3_K removes roughly 20%. That is an estimate; the measured ladder replaces it in the
  score. So an adapter can only win if the weighted error is **roughly six times more concentrated
  than white noise**, for example in the few input channels where activations are huge.
- **The precision alternative.** The same bytes spent promoting tensors up the K-quant ladder. Mixing
  whole tensors interpolates linearly in ε, so the alternative is the lower convex hull of
  (bits per weight, ε) over Q2_K, Q3_K, Q4_K, Q5_K and Q6_K.

## Design

- **Source:** `Qwen3.8-27B-Q8_0.gguf`, the #112/#113 source.
- **Importance matrix:** #113's (128 chunks of WikiText-2 train), unchanged.
- **Quantizer:** llama.cpp b10098's own `ggml_quantize_chunk`, called in-process from
  `ggml-base.dll`. That is the same code and build that wrote #113's arms.
- **Tensors:** `ffn_gate`, `ffn_up` and `ffn_down` of 10 blocks, 30 tensors in all.
  - Inside the fragile band: 51, 54, 57, 60, 63.
  - Outside it: 5, 15, 25, 35, 45.
- **Per tensor:**
  - ε for Q2_K, Q3_K, Q4_K, Q5_K and Q6_K with the imatrix.
  - ε for Q2_K without it.
  - The weighted-error eigen-spectrum for Q2_K (imatrix), Q2_K (no imatrix) and Q3_K (imatrix),
    at ranks 1–512.
- **Margin:** `M = 10·log10(ε_precision / ε_adapter)` at equal bits per weight.
  - The adapter side uses the best rank whose stored cost fits the budget.
  - The precision side is the hull.
  - The base is Q2_K. Q3_K is reported as well.
  - Budgets are +0.05, +0.10, +0.25 and +0.50 bits per weight. **+0.25** is primary: a tenth
    more bytes on a Q2_K tensor.
- **Tools, committed with this file:**
  - `weights/prereg114_spectrum.py` measures.
  - `weights/prereg114_score.py` scores. Its self-check gives 9 fixtures and 9 distinct verdicts.

**Validation gate, before any spectrum is computed.** The in-process quantizer must reproduce
llama-quantize byte for byte. The measurement script refuses to run otherwise, and the run is
VOID. It checks one Q2_K tensor against #113's SPREAD-I file (`blk.5.ffn_down`) and one Q4_K
tensor against OURS-I (`blk.51.ffn_down`), both quantized with the imatrix.

**Disclosed pilot.** The same check without an imatrix was run on Qwen3.5-4B before staking. Six
FFN tensors in Q2_K and Q3_K were byte-identical to llama-quantize's output, against
`Qwen3.5-4B-naive-q2k.gguf` from `Qwen3.5-4B-BF16.gguf`. No spectrum of any model was computed
before this file was committed.

## Predictions

- **P-1: where the error lives.** `ffn_down`'s weighted Q2_K error is the most concentrated
  tensor in its block. Its top-16 share must exceed both `ffn_gate`'s and `ffn_up`'s in at least
  **8 of 10** blocks. The reason: its input is the SwiGLU product, the heaviest-tailed activation
  in the block.
- **P-2: the imatrix eats the adapter's lunch.** Without the imatrix, the top-16 share of the
  weighted Q2_K error is higher than with it, in at least **25 of 30** tensors. An activation-aware
  quantizer suppresses error in exactly the columns an adapter would target.
- **P-3: the decision.** At +0.25 bits per weight on Q2_K, the median margin is **below 0 dB** for
  each of `ffn_gate`, `ffn_up` and `ffn_down`. Precision wins at equal bytes.
  - **If P-3 holds,** correction adapters are recorded as a dead end for imatrix K-quants on this
    model, and no adapter file is built.
  - **If the median margin reaches +0.5 dB for any kind,** Stage B is staked for that kind: a real
    LoRA GGUF, measured by KLD at equal total bytes.
  - **Between 0 and +0.5 dB,** the result is reported and not pursued. An adapter costs a second
    matmul per tensor at runtime, and a gain under 0.5 dB does not pay for it.

Band-versus-outside medians and the Q3_K-base margins are published without a staked direction.

## What would make this VOID rather than scored

- The validation gate fails: the precision points would not be llama-quantize's.
- The importance matrix lacks an entry for a sampled tensor.
- Any ε or spectral value is non-finite.
- For any spectrum, the Gram matrix's trace disagrees with the directly summed weighted error by
  more than 0.1%.

## What this stage cannot show

- **It sees only the diagonal metric.** The weighting uses each channel's mean square, not
  correlations between channels. A correction fitted to the full input covariance (EoRA, QERA) can
  do better than this one. If P-3 holds, the conclusion is scoped to diagonal-metric corrections,
  which is all llama.cpp's importance data supports. A full-covariance collector is the follow-up if
  the margin lands close to zero.
- **Per-tensor error is not end-to-end quality.** Stage B would measure KL divergence.

## When it runs

After #113 completes. It needs #113's importance matrix and arm files, and running alongside
#113's builds and evaluations would contend for CPU and RAM (C-14).
