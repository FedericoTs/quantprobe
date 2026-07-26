# The machine — exact specs, measured bandwidths, and what the next euro buys

All of this happened on one desktop I already owned. Exact specs, because reproducibility starts with honesty about hardware:

| component | spec | measured bandwidth / effect |
|---|---|---|
| CPU | Intel i5-7600K (4c/4t, 2017) | MoE decode saturates at 2 threads (memory-bound, measured) |
| GPU | GTX 1060 6 GB (Pascal, 2016) | 192 GB/s VRAM · η ≈ 0.35 for decode at **any** bit-width (see correction below) · **prefill** collapses ~6.8× on IQ-format quants |
| RAM | 16 GB DDR4 Corsair Vengeance | **2133 MT/s → 3000 (XMP): dense +52%, MoE +32% — pre-registered ×1.41, measured ×1.52** |
| SSD | Crucial MX500 (SATA) | 0.45 GB/s sequential (measured) — the 110B streaming tier |
| PCIe | 3.0 ×16 | 12.2 GB/s host→device (measured) |

The RAM line is the story in miniature: one free BIOS toggle, predicted in advance by the law, delivered within 8% — and it *moved the bottleneck* (the 30B went from bandwidth-bound to capacity-bound, exactly as a tiered system should behave).

### Correction (2026-07-26): there is no low-bit *decode* collapse

This table previously read "η ≈ 0.04 at 2-bit (decode-util collapse, measured)", and the planner
applied that below 4 bits. **It was wrong, and it was wrong in the direction that hurt users** — it
told anyone running a sub-4-bit quant that their GPU was useless for it. On `gemma4-12b` the tool
predicted 1.0 tok/s all-in-VRAM and recommended pure CPU at 3.9; the GPU placement it rejected
actually runs **9.56**.

[Pre-registration #16](../preregistrations/2026-07-26-gl-format-not-bitwidth.md) measured the same
7B in three quantizations, all in VRAM, changing nothing else:

| format | bits | decode tok/s | prefill pp2048 |
|---|---|---|---|
| Q4_K_M | 4.5 | 20.03 ± 0.04 | 27.49 |
| Q2_K | 2.8 | 19.17 ± 0.03 | 17.71 |
| IQ3_XS | 3.3 | 18.11 ± 0.05 | **4.04** |

**Decode does not care about bit-width** (a 10% band across 2.8–4.5 bits) and barely cares about
format (1.06×). **Prefill cares enormously about format** — IQ3_XS pays 6.8× — because
dequantization is compute, prefill is compute-bound, and decode is not. The old constant conflated
a real prefill effect with an imaginary decode one.

Two practical consequences, both counterintuitive:

- **If a model already fits in your VRAM, quantizing it further buys you almost no speed.** Going
  Q4_K_M → Q2_K here was 36% smaller and **4% slower**. Quantize to make a model *fit*; once it
  fits, stop — you are trading quality for nothing.
- **Avoid IQ-format quants on Pascal-class cards if you feed long prompts.** Decode is fine; it is
  prompt processing that falls off a cliff.

### Projections — what the law says the next euro buys

| upgrade | cost (mid-2026 market*) | predicted effect |
|---|---|---|
| +16 GB DDR4 | ~€35–50 used · €90–130 new | 30B hybrid leaves the RAM boundary → stable ~19–21 tok/s; caches half a 110B |
| NVMe SSD, 1 TB (Gen3 ×4 is enough — board caps there) | ~€150–190 new right now; worth waiting for <€100 deals | disk tier 0.45 → ~3.5 GB/s: the 110B goes 0.19 → **~1.5 tok/s** |
| Both | ~€200–320 at today's prices | a 2016 desktop serving a 30B at reading speed and a 110B at demo speed |

\* The 2026 AI-driven NAND/DRAM shortage has inflated component prices (~2× the 2024 floor) and they're volatile — the used DDR4 market is the value play, and NVMe deals reward patience. The *predictions* don't change with the prices; when the hardware arrives, measured numbers go in this table next to them.

