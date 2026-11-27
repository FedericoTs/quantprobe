# Pre-registration #113 — does the depth-aware edge survive an importance matrix?

**Staked 2026-11-27, before the importance matrix was computed and before any arm was built.**

## Why this one

Every equal-bytes win this project has published was measured with **no importance matrix
(imatrix) in either arm**. Prereg #104 (MoE, 29.2%) and #112 (dense, 43.3%) both say so. Every
serious quant producer ships with one. In APEX's same-layout pairs an imatrix alone cut KL
divergence a further 9–29%, and Unsloth's layer selection *on top of* an imatrix gains only
~3–8%. The November 2026 opportunity report names this as the single experiment that decides
which half of the project deserves the remaining effort. It is also the cheapest one in the
backlog, because #112's source file is still on disk.

## Design

Everything is held fixed from #112 except the imatrix: the same `Qwen3.8-27B-Q8_0.gguf` source
(29,047,086,048 bytes, 8.51-bit), the same band (51–63), the same SPREAD placement (blocks
0,4,9…59), the same formats (FFN Q2_K / protected Q4_K, attention and SSM Q4_K, shared expert and
NextN Q8_0, token embedding Q4_K), llama.cpp b10098.

| arm | placement | imatrix |
|---|---|---|
| **OURS-I** | band 51–63 at Q4_K | yes |
| **SPREAD-I** | 13 blocks spread evenly at Q4_K | yes, same file |
| **UNIFORM-I** | llama-quantize's standard Q2_K mix, no overrides | yes, same file |

UNIFORM-I is what `llama-quantize --imatrix x Q2_K` gives a careful hobbyist. It has a different
byte budget, so it is **descriptive only**: reported with its size, never scored against the
equal-bytes pair.

**Importance matrix.** Computed here from the Q8_0 source with `llama-imatrix`, **128 chunks** of
512 tokens (65,536 tokens) of **WikiText-2 train**. A check of 400 random 200-character windows
of the test set found **0** of them verbatim in the train split, so calibration and evaluation
share no text. 128 chunks is a deliberate size. A thin imatrix would make this test easier to
pass, since it would rescue the badly placed arm less, so the matrix is not trimmed for speed.

**Size gate, before any quality number is read.** OURS-I and SPREAD-I must be within 0.5% in
bytes. An imatrix changes values, not formats, so they are expected byte-identical, as in #112.

**Evaluation.** `llama-perplexity --kl-divergence` against base logits saved from the Q8_0 source,
using the same file, chunk count (32) and `-ngl 12` as #112. It reports perplexity, mean and p99
KL divergence, and top-1 token agreement. The base run also re-measures the reference perplexity
and must reproduce #112's **5.5950** within 0.5%. If it does not, the protocol has drifted and the
run is VOID before any arm is scored.

## Predictions

The #112 twins are OURS (PPL 6.0590) and SPREAD (6.4136) with no imatrix, and reference 5.5950.

- **P-1 — the edge survives.** Mean KLD(OURS-I) < mean KLD(SPREAD-I) at identical bytes.
  **Kill rule:** if this fails, the depth-aware quantizer's edge does not survive standard
  practice, and the project pivots to prediction, as the report's decision gate says.
- **P-2 — but it shrinks.** The share of SPREAD-I's excess perplexity that OURS-I removes,
  (PPL_S − PPL_O) / (PPL_S − 5.5950), is **below #112's 43.3%**, staked **between 5% and 30%**.
  A share below 5% counts as collapsing to the ~3–8% that Unsloth's layer selection reaches, and
  triggers the pivot as well.
- **P-3 — the imatrix helps both arms.** PPL(OURS-I) < 6.0590 and PPL(SPREAD-I) < 6.4136.
- **P-4 — the mechanism behind P-2.** The imatrix rescues the badly placed arm more:
  (6.4136 − PPL_SPREAD-I) > (6.0590 − PPL_OURS-I).

Mean KLD is the primary quality metric. Perplexity carries the comparison with #112 because #112
measured nothing else. KLD p99 and top-1 agreement are published for every arm without a staked
threshold.

## What would make this VOID rather than scored

- The base run does not reproduce reference perplexity 5.5950 within 0.5%.
- The size gate fails.
- The imatrix computation, a build or an evaluation does not complete, or a disk fills mid-run.
  A truncated artifact is not a measurement.

## Stage 2, only if P-1 holds

Byte-matched head-to-heads of OURS-I against llama.cpp's `--target-bpw` allocator (merged 23
November 2026, so it needs a newer llama.cpp than b10098) and the nearest-size Unsloth Dynamic
quant. Both require downloads and will be staked separately.

## Publication

Scored either way, at full size. If P-1 fails, the README's quantizer claims get a scope line
saying the edge did not survive an importance matrix.
