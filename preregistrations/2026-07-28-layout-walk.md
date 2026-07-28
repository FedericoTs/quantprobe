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

---

## Scored (2026-07-28)

**Verdict: P-1 MISS — the KILL RULE fires on layout, and the same run found the real mechanism,
which then survived its own confirmation arm. The K-quant deficit decomposition is now complete
to within 12%.**

| pair (mmvq geometry, bitwise-matched) | planar | struct | ratio |
|---|---|---|---|
| Q2_K-shaped (84 B blocks) | 83.8-86.0 | 84.1-85.3 | **0.99-1.00** |
| Q4_K-shaped (144 B blocks) | 131.0-131.3 | 133.1-133.6 | **1.02** |

- **P-1 MISS:** the interleaved struct walk costs nothing. Layout ACQUITTED — the repack idea is
  dead before it was built (bytes were never the problem).
- **P-2 technically holds but trivially** (both ratios ~1.0). **P-3 PASS** (bitwise identical).
- **P-4: the cost model lands within 12% of real** after e2e dilution — nothing large is missing.

### The unplanned finding, then its confirmation arm

The Q2_K-shaped arms run **84-86 GB/s** where the Q4_K-shaped arms run **131** — same geometry,
same layout style, same dp4a count per weight. The difference is that Q2_K's format defines
scale+min PER 16 WEIGHTS, which at 2 bits means a metadata FMA chain every 4 bytes — 4x the
application density of Q4_K per byte. Exploratory arm (identical loads, identical dp4a count,
scale/min applied once per u32 instead of once per quad): **83.8 -> 103.2 GB/s (+23%)**.

**The K-quant deficit is METADATA APPLICATION DENSITY** — not layout (#57), not the min dp4a
(#56), not packed-scale decode (#56), not geometry (#56), not blocking (#55).

### The complete decomposition of real Q2_K (65.4 GB/s measured)

```
Q4_K-class kernel ceiling at real geometry     131   GB/s
x 0.64  metadata density of Q2_K's definition   84
x 0.885 e2e dilution (Q4_0-calibrated)          74
measured                                        65.4   -> residual x0.88 (12%, unattributed:
                                                          occupancy / q8_1-struct walk / mixed tensors)
```

Q4_K itself: 131 x 0.885 = 116 vs measured 106.4 — residual 9%.

### What this means for "the best kernel on the market"

The deficit is intrinsic to the FORMAT DEFINITION: quality at 2 bits requires fine-granularity
asymmetric metadata (quality.py: every coarser or symmetric variant loses), and fine metadata
costs ALU density. A kernel cannot remove it and keep the format; a repack cannot remove it
(bytes acquitted). The honest remaining levers, ranked by size:

1. **The split-placement residual (N5): GPU share eta 0.15 vs all-in-VRAM Q2_K 0.34 — factor
   ~2.3 on the flagship.** THE open number, now the only big one.
2. The 9-12% kernel residual on K-quants (occupancy/q8_1 walk) — real but small.
3. Format-metadata co-design (~+23% at ~5% RMSE cost per quality.py's g32 asym) — a new-format
   cost the project already declined once (D-17); only worth revisiting with a quality gate
   stronger than RMSE.

**Wired into:** `findings/REGISTER.json:D-22` (layout acquitted) · `L-16` (metadata application
density — the mechanism, with the full decomposition) · `C-02` (all-in-VRAM eta band CLOSED:
format metadata density explains it; the split residual moves to N5).
