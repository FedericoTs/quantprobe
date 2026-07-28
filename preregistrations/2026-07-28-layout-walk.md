# Pre-registration #57: is the real K-quant deficit the LAYOUT WALK?

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurement. **STAKED.**

## The last suspect standing

#56 acquitted the arithmetic: min-term dp4a 0.9-1.8%, packed metadata decode 4.1%, at real 1-row
geometry. Yet real Q4_K_M runs at 106.4 GB/s effective and real Q2_K at 65.4, where every clean
cost model runs 125-133. Two suspects remain: register pressure/occupancy, and the **layout walk**
— the real formats store weights in interleaved structs, not planar arrays:

| | struct | stride | qs region per block |
|---|---|---|---|
| block_q2_K | scales[16] + qs[64] + dm | **84 B** (odd, non-power-of-two) | 64 B, unaligned |
| block_q4_K | dm + scales[12] + qs[128] | **144 B** | 128 B, mostly aligned |
| planar (our oracles) | separate planes | — | fully coalesced |

At mmvq's thread mapping (16 lanes per block), a warp's Q2_K loads hit two disjoint unaligned 64 B
regions 84 B apart, plus scattered scale bytes at each block head — partial coalescing, wasted
sectors. Q4_K's 128 B qs region is far kinder. **This predicts the observed ordering**
(η 0.34 < 0.55 < 0.62) where the arithmetic could not.

## Method

kernelprobe, mmvq geometry (1 row/block, 128 threads, global cached x), two matched pairs where
the ONLY difference within a pair is byte placement — identical per-thread arithmetic, identical
logical weight assignment, bitwise-comparable outputs:

- **(a) planar 2.625-bit** (Q2_K cost model: 4 shifts+dp4a per u32, packed scale/min nibbles,
  fp16 d/dmin, min via dp4a) vs **(b) struct-interleaved 84 B blocks**, llama.cpp's lane→block map.
- **(c) planar 4.5-bit** (the #56 arm) vs **(d) struct-interleaved 144 B blocks**.

## Stakes

- **P-1 (THE CLAIM).** (b)/(a) ≤ **0.80** — the 84 B interleaved walk costs ≥ 20%.
- **P-2 (the ordering).** (d)/(c) > (b)/(a) — Q4_K's layout is measurably kinder than Q2_K's,
  matching the real e2e ordering.
- **P-3 (correctness).** (a)==(b) and (c)==(d) bitwise, proving the pairs differ only in bytes.
- **P-4 (sufficiency check, the honest one).** (b) lands within **±20%** of real Q2_K's 65.4 GB/s
  effective share of its kernel time. If (b) stays FAR above it, layout is a partial cause and
  occupancy/register pressure still hides the rest.

## KILL RULE

**If P-1 fails — the struct walk costs < 10% — layout is acquitted like the arithmetic before
it**, both named suspects for the K-quant deficit are dead, and the only remaining road is
profiling the REAL kernels (register counts via ptxas -v, occupancy calculators), not cost
models. I will stop building oracles and go read the real kernel's compilation artifacts.

## What a P-1 hit unlocks (why this matters for the "best kernel" goal)

If layout is the mechanism, the cure is a **load-time repack**: same GGUF file, same bits, same
quality — bytes reordered once into planar layout at model load, plus a vec_dot for the repacked
layout. llama.cpp has exact precedent (aarch64 `Q4_0_4_4` online repack). Our planar Q2_K-cost
kernel measured **356.6 GW/s vs llama.cpp Q2_K's 165.1** — if layout explains the gap, repack
recovers most of ~2×, in-tree, with output identity.

**Wired into:** pending; L-15 mechanism chain either narrows to layout or dies to profiling.
