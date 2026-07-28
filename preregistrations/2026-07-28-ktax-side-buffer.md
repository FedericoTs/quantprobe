# Pre-registration #56: can a per-token activation-sum side buffer remove the K-quant min tax at llama.cpp's own geometry?

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurement. **STAKED.**

## What this tests, and why it exists

#55 closed the Q2_K question with "the two available fixes are mutually exclusive." The brainstorm
review (H-REOPEN) found the third fix that claim missed, and source reading upgraded the target:
`vec_dot_q4_K_q8_1_impl_vmmq` (`vecdotq.cuh:518`) spends **2 of its 4 dp4a per iteration**
computing `sum(u)` — a row-invariant activation sum recomputed for every output row. Same pattern
in Q5_K and Q2_K. This is the asymmetric K-quant family tax, and **Q4_K_M is the most-downloaded
format in the ecosystem.**

The candidate fix: precompute per-4-activation sums ONCE PER TOKEN into a 2 KB side buffer
(L2-resident — every row's block reads the same values), then the min term is a cached load +
FMA instead of a dp4a. No blocking change (#55 killed that), no `block_q8_1` struct change
(U-13's risk), CUDA-backend-local.

## Method — kernelprobe, at llama.cpp's ACTUAL decode geometry

The previous kernelprobe numbers (L1d/L1g) used 768-rows-per-block with smem-staged activations —
a structure #55 proved unavailable to llama.cpp. This oracle uses **mmvq geometry: one output row
per CUDA block, 128 threads, activations read from GLOBAL memory (L1/L2-cached)**, three arms on
the same 4.5-bit buffer:

| arm | per uint32 (8 weights) | mimics |
|---|---|---|
| (i) sum-via-dp4a | 2 weight-dp4a + **2 sum-dp4a** + 2 FMA | llama.cpp K-quant vec_dot |
| (ii) sum-via-side-buffer | 2 weight-dp4a + **2 cached loads** + 2 FMA | the proposed fix |
| (iii) symmetric control | 2 weight-dp4a + 2 FMA | Q4_0 (no min term) |

Arms (i) and (ii) compute mathematically identical results and are correctness-checked against
the already-verified L1d reference. Arm (iii) computes a different (offset-free) result and is a
throughput control only.

## Stakes

- **P-1 (THE CLAIM).** Arm (ii) recovers **≥ 60%** of the (i)→(iii) throughput gap. The sums move
  from the ALU port to the LD/ST port, which dual-issues on Pascal.
- **P-2 (the tax is real at this geometry).** Arm (i) is **≥ 15%** slower than arm (iii). If the
  gap is smaller, the sum-dp4a hides in the memory latency at 1-row geometry and the whole
  K-quant-tax story does not survive contact with the real structure.
- **P-3 (correctness).** Arms (i) and (ii) match the L1d double-checked reference bit-for-bit
  against each other (same math, same order within a row).
- **P-4 (context).** All mmvq-geometry arms land BELOW the 768-row L1d (228 GW/s) — quantifying
  for the first time what llama.cpp's 1-row structure itself costs vs block-level activation
  reuse. Recorded as a number, whatever it is.

## KILL RULE

**If P-2 fails, the min tax does not bind at real geometry** and the Q4_K_M/Q4_0 e2e gap needs a
different explanation — the in-tree patch is not written and U-13 is closed as refuted-by-oracle.
**If P-2 holds but P-1 fails**, the tax is real but the side buffer is not the cure (LD/ST does
not dual-issue as hoped); U-13 stays open with the oracle's numbers and no patch is proposed on
hope.

Only if BOTH hold does an in-tree `vec_dot_q4_K` patch get written and A/B'd on the real 7B
Q4_K_M — target: 22.72 toward Q4_0's 26.87 tok/s. e2e claims wait for that measurement.

**Wired into:** pending; `findings/REGISTER.json:U-13` cites this either way.
