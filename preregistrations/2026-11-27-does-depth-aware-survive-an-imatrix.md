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

## Verdict — scored 2026-11-27, 3 of 4

`python weights/prereg113_score.py --results weights/data/prereg113.json`, run unchanged after
every arm finished.

| | OURS-I | SPREAD-I | UNIFORM-I (descriptive) |
|---|---|---|---|
| bytes | 12,483,292,192 | 12,483,292,192 | 10,864,593,952 |
| perplexity | 5.8754 | 6.0810 | 6.1079 |
| mean KLD | **0.0828 ± 0.0024** | 0.1219 ± 0.0035 | 0.1473 ± 0.0037 |
| p99 KLD | 0.770 | 1.230 | 1.528 |
| top-1 agreement | 88.55% | 86.31% | 83.90% |

**Protocol checks.** The reference reproduced #112's perplexity exactly (5.5950 against 5.5950),
and the two scored arms are byte-identical, as in #112.

- **P-1 HIT: the edge survives.** At identical bytes, OURS-I's mean KLD is **32% lower** than
  SPREAD-I's (0.0828 against 0.1219). The gap is about nine times either arm's standard error.
  The kill rule does not fire, so the quantizer half of the project keeps its claim.
- **P-2 MISS: the edge did not erode.** I staked 5–30%. OURS-I removes **42.3%** of SPREAD-I's
  excess perplexity, against 43.3% in #112. The scorer's pre-committed branch reads "it shrank,
  but by less than staked". The honest reading is that it did not shrink: one point between two
  single-calibration runs is not evidence of erosion. My prediction drew on Unsloth's 3–8%, layer
  selection on top of an imatrix, and expected the imatrix to absorb most of the placement's
  value. It absorbed none of it.
- **P-3 HIT.** The imatrix improved both arms: OURS from 6.0590 to 5.8754 (−3.0%), SPREAD from
  6.4136 to 6.0810 (−5.2%).
- **P-4 HIT.** It helped the worse arm more in absolute terms: +0.333 PPL for SPREAD against
  +0.184 for OURS.

**The mechanism.** P-2 and P-4 look contradictory and are not. The imatrix removed about 40% of
each arm's excess perplexity: SPREAD's excess fell from 0.819 to 0.486 (−40.6%), OURS's from 0.464
to 0.280 (−39.6%). A proportional cut is larger in absolute terms for the arm with more excess,
which is P-4, and leaves the ratio between the arms where it was, which is P-2. **On this model
the importance matrix and the band placement compose; they do not substitute.** The imatrix
decides how each tensor is rounded, the placement decides which tensors get the bits, and each
removes about 40% of whatever loss the other leaves.

**Descriptive.** UNIFORM-I is what `llama-quantize --imatrix … Q2_K` gives a careful hobbyist. It
is 13% smaller and worse on every metric: mean KLD 0.147, which OURS-I beats by 44%. It is not
byte-matched, so this is scale, not a score.

**Consequences.**

- The README's quality claims stop needing their caveat ("measured with no imatrix in either
  arm"). They now cite the with-imatrix result alongside.
- The published 27B build is #112's, made without an imatrix. OURS-I is the same size and better
  on every metric measured. Republishing it is proposed, not done: it is a public upload.
- Stage 2 is unblocked: a byte-matched head-to-head against llama.cpp's `--target-bpw` allocator,
  which b12815 has. It will be staked separately.

**Scope.** One dense model, one calibration set (WikiText-2 train, 128 chunks), evaluated on
WikiText-2 test, with Q2_K/Q4_K formats. Composition with an imatrix is measured here only.
