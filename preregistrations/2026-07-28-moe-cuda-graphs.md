# Pre-registration #48 (scored inline): does the MoE flagship behave differently under CUDA graphs?

**Author:** Federico Sciuca · **Date:** 2026-07-28. **Diagnostic follow-up to #47**, run because
#47's null was measured on a DENSE model and the flagship is MoE — a different code path.

## Why the flagship might differ

`ggml-cuda.cu:3280` `[TAG_MUL_MAT_ID_CUDA_GRAPHS]` disables graphs when
`!ggml_is_quantized(src0) || ne[2] > mmvq_mmid_max`. Reading the table (`mmvq.cu:125`): **Q2_K on
Pascal has `mmvq_mmid_max = 4`**, and at decode `ne[2] = 1`. Q2_K is quantized. **So the condition
is FALSE and MUL_MAT_ID does not block graphs at batch 1** — my stated "next suspect" from #47 was
wrong on its own premise, which the source settles before any measurement.

But the MoE graph is structurally different regardless: router, top-k, gather, sum, clamp, div and
three `MUL_MAT_ID`s per layer, against a dense layer's handful of nodes. If a per-node cost exists,
a node-denser graph should expose it.

## Measured — split placement, tg128, r=3, position-controlled

| arm | run 1 | run 2 (position control) |
|---|---|---|
| graphs default (arch check disables) | 16.98 ± 0.32 | 17.12 ± 0.18 |
| **graphs FORCED ON** | **17.61 ± 0.24** | **17.57 ± 0.27** |

Capture verified firing (`[e5] cudaStreamBeginCapture fired #1/#2/#3`).
**Effect: +3.2%** (17.59 vs 17.05 averaged), consistent across positions.

## What this settles

- **The MoE path DOES benefit where the dense path does not** (+3.2% vs 0.0%), consistent with a
  per-node cost that a node-denser graph exposes — but the magnitude is small.
- **#47's kill stands.** The 15.4 ms fixed constant is not launch overhead: removing per-launch
  cost from a 48-layer MoE graph buys 3%, not the 25% staked, and not the ~35% the constant
  represents.
- **My "MUL_MAT_ID forces a sync" suspicion is refuted at source before measurement** — the
  condition cannot fire at decode for Q2_K on Pascal. Recorded because it was written down as the
  live suspect one turn earlier, and it was wrong.

## The one genuinely shippable finding

**llama.cpp's pre-Ampere graph disable is too conservative for MoE models.** On this Pascal card
graphs capture correctly and give +3.2% on a 30B MoE. That is small, free, and upstream-shaped —
the same class as the `GGML_OPENMP` build finding already filed as
[#26200](https://github.com/ggml-org/llama.cpp/issues/26200).

**Before it can be proposed upstream** it needs what this run did not do: an **output-identity
check** (llama-bench produces no text), and a **second Pascal-class card**, since a 3% claim from
one machine is inside the noise other people will measure.

**Wired into:** `findings/REGISTER.json:V-14` (the +3.2%, scoped and not yet upstream) ·
`D-15` (launch-overhead refutation strengthened, now on both dense and MoE paths).
