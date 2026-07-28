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

---

## Scored (2026-07-28, kernelprobe run in session log)

**Verdict: P-2 FAILS, P-1 fails with it, P-3 passes, P-4 fails informatively. THE KILL RULE
FIRES on its first gate: the min tax does not bind at real geometry. U-13 closed as
refuted-by-oracle. Twelfth mechanism this project has named; twelfth moved by a control.**

| arm (mmvq geometry: 1 row/block, 128 threads, global cached x) | GB/s | GW/s |
|---|---|---|
| (i) sum-via-dp4a (llama.cpp K-quant pattern) | 130.7-132.6 | 232-236 |
| (ii) sum-via-side-buffer (the proposed fix) | 111.5 | 198 |
| (iii) symmetric control (no min term) | 131.4-133.1 | 234-237 |
| (iv) packed nibble scale+min decode | 125.4 | 223 |

- **P-2 MISS: the min-term dp4a costs 0.9-1.8% at real geometry** — it hides completely in
  memory latency at 1 row/block. The "8 dp4a where 4 suffice" instruction count is real in the
  source and IRRELEVANT to throughput. Counting instructions is not measuring them.
- **P-1 MISS (moot but recorded): the side buffer is 15-16% SLOWER.** Adding cached loads to a
  latency-bound loop hurts; recomputing on the ALU port is free. The "LD/ST dual-issue" story
  was wrong in the direction that matters.
- **P-3 PASS:** arms i and ii bitwise identical, so the null is a real null.
- **P-4 MISS, informatively:** mmvq geometry (132.6) BEAT the 768-row smem-staged L1d (128.7).
  llama.cpp's 1-row structure was never a penalty either; block-level activation reuse is worth
  nothing here. Global cached reads of a shared activation vector are as good as smem.
- **Extension (iv): packed metadata decode costs 4.1%** at real geometry — not the 12-19% the
  e2e format gaps need.

### What this refutes beyond the stake — including a mechanism shipped earlier TODAY

Every simplified cost model of the K-quant family is HEALTHY at real geometry (125-133 GB/s vs
real Q4_K_M 106.4, real Q2_K 65.4). Therefore:

1. **#52's mechanism sentence is overclaimed.** "K-quants decode a 6-bit scale AND min before
   any dot product" is true of the source and now measured to cost ~4-6%, not the 12-19% (Q4_K)
   or ~2x (Q2_K) observed e2e. The MEASURED FACTS of #52/#53 stand untouched (+19% Q4_0, Q2_K
   dominated); their attributed cause does not. `plan.py format_advice` text corrected in the
   same commit as this score.
2. **The brainstorm's H-REOPEN/N1 program is closed**, kill honoured: no in-tree patch, no
   upstream filing, on either the blocking route (#55) or the side-buffer route (this).
3. **What remains as suspects for the real K-quant deficit**, none yet tested: (a) register
   pressure/occupancy of the real vec_dot implementations (complex array indexing, more live
   state); (b) the real interleaved superblock LAYOUT WALK - partial coalescing across the
   144-byte blocks - versus these oracles' clean planar layouts; (c) something in mmvq's
   blocks_per_iter row walk. The next oracle is an arm that reproduces Q4_K's actual byte
   arrangement and load indexing at matched arithmetic.

**Wired into:** `findings/REGISTER.json:U-13` (closed, refuted-by-oracle) · `D-21` (min-term
and metadata-decode taxes acquitted at real geometry) · `L-15` (mechanism narrowed: the format
effect is real, its instruction-count explanation is dead; access pattern/occupancy now lead).
