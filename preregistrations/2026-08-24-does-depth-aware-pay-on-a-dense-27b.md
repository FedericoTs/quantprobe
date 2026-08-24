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
