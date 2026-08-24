# Pre-registration #112 — does depth-aware quantization pay on a DENSE model?

**Staked 2026-08-24, before any arm was built or scored.**

## Why this one

Every equal-bytes win this project has published is on a **mixture-of-experts** model
(Qwen3.6-35B, prereg #104: 29.2% less quality loss at byte-identical size). MoE models have a
structural feature dense models lack — most of each layer's weight is in routed experts that are
individually cheap to move between tiers. It is entirely possible that depth-aware placement works
*because* of that structure and does nothing on a dense stack, where every layer is fully read
every token.

If so, the technique is narrower than the README implies, and we should say so.

Qwen3.8-27B is the test: **dense**, 27.3B parameters, 64 decode blocks, and already in the atlas
(band 51-63 measured under prereg #101, ratio 2.35). It is also the class where 2-bit is not a
curiosity — Q4 is ~16 GB and does not fit an 8 GB card, while ~2.9-bit lands near 10 GB.

Related: **U-61** (does protecting the never-executed MTP block buy anything?) is answered by the
same three builds, at no extra cost.

## Arms

All three built from the same `Qwen3.8-27B-Q8_0.gguf` source (29.0 GB, `unsloth/Qwen3.8-27B-GGUF`),
same imatrix, same target bits, scored on the same held-out corpus.

| arm | what | why |
|---|---|---|
| **A — OURS** | depth-aware, band **51-63** protected at the higher tier | the recipe, with the MTP block excluded |
| **B — SPREAD** | the same number of blocks protected at the same tier, spread evenly across depth | the strong control: identical byte budget, wrong *placement* |
| **C — MTP** | depth-aware, band **51-64** (includes the MTP head) | U-61: does protecting a never-executed block cost anything |

**B is the control that matters.** Comparing against a naive uniform quant would confound
placement with budget; B holds the budget fixed and moves only *where* the protection sits.

## Size gate (applies before any quality number is read)

Prereg #104 rejected its first control for being 9% smaller — a smaller file losing on quality
proves nothing. Same gate here:

> **A and B must land within 0.5% of each other in bytes.** If they do not, the control is rebuilt
> before either is scored. Neither PPL is looked at until the gate passes.

C is expected to be *larger* than A (one more protected block); that difference is the measurement,
not a gate failure.

## Predictions

**P-1 — depth-aware beats spread at equal bytes on a dense model.**
PPL(A) < PPL(B). Staked at **>50% of the MoE effect**: prereg #104 removed 29.2% of the excess
loss over reference, so P-1 claims A removes **at least 14.6%** of B's excess.
*Refuted if* A removes less than 14.6%, and **inverted** if PPL(A) > PPL(B).

**P-2 — the effect is smaller on dense than on MoE.**
The share of excess loss removed is **below 29.2%**. Stated because the honest expectation is that
some of the MoE result came from expert structure, and if dense matches or beats MoE that is news
worth having staked against.

**P-3 — protecting the MTP block buys nothing (U-61).**
|PPL(C) − PPL(A)| < 0.01, i.e. indistinguishable, while C is **larger** than A by 0.2-2%.
*Kill rule:* if PPL(C) is measurably BETTER than A beyond that threshold, blk.64 is being executed
by some path the U-60 source analysis missed, and **U-60 P-1 must be re-opened.**

**P-4 — decode speed is unchanged across all three arms**, within 3%. Depth-aware moves *which*
blocks hold which format, not how many bytes are read per token.

## Scoring

- Corpus: **WikiText-2 test, held out**, 32 chunks — the same corpus and chunk count as #104, so
  the two results are comparable.
- Reference for "excess loss": the Q8_0 source's own PPL, measured once.
- The scorer is committed **before the arms are evaluated**, with `--self-check` proving every
  branch reachable, as with #104-#111.
- Usability gate: any arm whose PPL run does not complete is reported as such, not dropped.

## What would make this VOID rather than scored

- The size gate cannot be met after two control rebuilds.
- The source turns out not to be high-precision (a lossy-on-lossy build is not a quality claim).
- The box runs out of disk mid-run and an arm is truncated — a partial file is not a measurement.

## Publication

Scored either way, at full size, including an inversion. If P-1 is refuted the README's claim gets
a scope line saying depth-aware placement is demonstrated on MoE and does not transfer to dense —
which is exactly the kind of boundary this register exists to record.

---

# VERDICT — scored 1/2, two unscored (2026-08-24)

**P-1 HIT. P-2 MISS — and the miss is the result worth reading.**

| arm | placement | bytes | PPL (32 chunks, held-out WikiText-2) | excess over reference |
|---|---|---|---|---|
| reference | Q8_0 source, 8.51 bits | 29,047,086,048 | 5.5950 | — |
| **OURS** | measured band **51-63** | **12,483,292,128** | **6.0590** | **+0.4640** |
| SPREAD | 13 blocks at 0,4,9…59 | **12,483,292,128** | 6.4136 | +0.8186 |

**The size gate passed at 0.000% — the files are byte-identical, not merely close.** That is not
luck. In a dense stack every decode block has the same FFN dimensions, so moving *which* thirteen
get Q4_K changes placement and literally nothing else. This is the strongest form this control can
take, and it passed before any perplexity number was read.

## P-1 — HIT

Depth-aware placement removes **43.3%** of the control's excess loss (staked: ≥ 14.6%).

> **Depth-aware quantization is not a mixture-of-experts artefact.** It pays on a dense model, at
> byte-identical size, by a wide margin.

That was the real question. Every equal-bytes win this project had published was on MoE, where
most of a layer's weight sits in routed experts that move between tiers independently — a
structure a dense stack does not have. Had P-1 failed, every headline here would have needed a
scope line. It did not.

## P-2 — MISS. Dense did not merely match MoE; it beat it.

Staked: the dense share would come in **below** the MoE's 29.2% (prereg #104). Measured: **43.3%**,
about 1.5× the MoE effect.

I staked P-2 in that direction precisely so this outcome would count as news rather than a shrug,
and it does. The reasoning behind the prediction — that some of the MoE result came from expert
structure — is refuted. If anything the causation runs the other way: on an MoE only the routed
experts of a protected block get the better format while the always-active path is already
protected everywhere, so a "protected block" is a weaker intervention there than here, where the
whole FFN changes format. **Concentration of a fixed byte budget matters more when every protected
block is fully read every token.**

This does not license a general claim. One dense model is one dense model, and the same
architecture family (qwen35) supplied both arms. What it does establish is that the MoE-only
scope line is unnecessary, and that the dense case is worth more attention than it has had here.

## P-3, P-4 — UNSCORED, and stated as such

- **P-3 (U-61, the MTP block)** needs arm C, which is not built: the box was deliberately run
  without deleting the user's models, and three 12.5 GB arms do not fit beside a 29 GB source.
- **P-4 (decode unchanged)** needs a decode measurement on at least two arms. Not attempted here.
  It should be read against this box's record: preregs #110 and #111 both VOIDed on decode spread
  of 15-24%, and a 3% threshold cannot be resolved against that. Attempting it would produce a
  number, not an answer.

Neither is reported as a pass. An unscored prediction is unscored.

## Cost note, recorded because it shaped the design

The reference took **99.2 minutes**; each 12.5 GB arm took **4.9**. The reference is 29 GB against
16 GB of RAM on a disk that was 100% full, so every forward pass streamed the weights (L-29/D-29).
A 20× wall-clock penalty for one arm being over the RAM boundary is the same effect this project
documents for decode, showing up in evaluation. Had the arms themselves been that size, this
experiment would not have been practical on this hardware.

