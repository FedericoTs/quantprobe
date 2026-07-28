# The machine — exact specs, measured bandwidths, and what the next euro buys

All of this happened on one desktop I already owned. Exact specs, because reproducibility starts with honesty about hardware:

| component | spec | measured bandwidth / effect |
|---|---|---|
| CPU | Intel i5-7600K (4c/4t, 2017) | MoE decode saturates at 2 threads (memory-bound, measured) |
| GPU | GTX 1060 6 GB (Pascal, 2016) | 192 GB/s VRAM · decode η 0.34–0.62 **by format** (Q4_0 0.619 · Q4_K_M 0.553 · Q2_K 0.340, same session on this card — the format, not the bits, sets decode η; L-15/L-16, see correction below) · **prefill** collapses ~6.8× on IQ-format quants |
| RAM | 16 GB DDR4 Corsair Vengeance | **2133 MT/s → 3000 (XMP): dense +52%, MoE +32% — pre-registered ×1.41, measured ×1.52** · delivers **26.1 GB/s** of its 48 GB/s spec (pre-registration #27) |
| SSD | Crucial MX500 (SATA) | 0.45 GB/s sequential (measured) — the 110B streaming tier |
| PCIe | 3.0 ×16 | 12.2 GB/s host→device (measured) |

The RAM line is the story in miniature: one free BIOS toggle, predicted in advance by the law, delivered within 8% — and it *moved the bottleneck* (the 30B went from bandwidth-bound to capacity-bound, exactly as a tiered system should behave). The other RAM lesson is that spec-sheet bandwidth is not delivered bandwidth: this box reads 26.1 GB/s of its 48 GB/s DDR4-3000 "peak" (pre-registration #27), and correcting for that moved the raw-decode wall we publish from 52.9 down to 41.1 tok/s.

One trap for anyone copying this table for their own box: **channel count is not stick count.** Consumer platforms are dual-channel no matter how many DIMMs are populated. The first external replication (register E-06, an RTX 3090 box) hit exactly this — 4 DIMMs counted as 4 channels quoted 173 GB/s where the platform peaks at ~86, a clean 2× input error. `detect` now defaults to dual-channel regardless of stick count (HEDT/server CPUs recognized by name go wider).

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

**Decode does not care about bit-width** (a 10% band across 2.8–4.5 bits). The "barely cares about
format" part (1.06×) held only for the three formats measured that day: a later same-card,
same-session pair of pre-registrations (#52/#53) measured Q4_0 at **26.87 tok/s vs Q4_K_M at
22.72** — +19% from the format alone, of which bytes explain 5.7%. Decode ignores the bits but
does care about the format, because the format sets the unpack instruction cost and the metadata
application density (L-15/L-16). **Prefill cares enormously about format** — IQ3_XS pays 6.8× —
because dequantization is compute, prefill is compute-bound, and decode is not. The old constant
conflated a real prefill effect with an imaginary decode one.

Two practical consequences, both counterintuitive:

- **If a model already fits in your VRAM, quantizing it further buys you almost no speed.** Going
  Q4_K_M → Q2_K here was 36% smaller and **4% slower**. Quantize to make a model *fit*; once it
  fits, stop — you are trading quality for nothing. One amendment since (v1.18): *within* a fit,
  the format is a real lever on pre-Ampere cards — prefer Q4_0 over Q4_K_M (+19% measured;
  speed-only, Q4_K_M is higher quality per byte; unverified on Ampere+), and never Q2_K when a
  4-bit file fits (Q2_K is 32% smaller and *slower* than Q4_0).
- **Avoid IQ-format quants on Pascal-class cards if you feed long prompts.** Decode is fine; it is
  prompt processing that falls off a cliff.

### Projections — what the law says the next euro buys

| upgrade | cost (mid-2026 market*) | predicted effect |
|---|---|---|
| +16 GB DDR4 | ~€35–50 used · €90–130 new | 30B hybrid leaves the RAM boundary → stable ~19–21 tok/s; caches half a 110B |
| NVMe SSD, 1 TB (Gen3 ×4 is enough — board caps there) | ~€150–190 new right now; worth waiting for <€100 deals | disk tier 0.45 → ~3.5 GB/s: the 110B goes 0.19 → **~1.5 tok/s** |
| Both | ~€200–320 at today's prices | a 2016 desktop serving a 30B at reading speed and a 110B at demo speed |

\* The 2026 AI-driven NAND/DRAM shortage has inflated component prices (~2× the 2024 floor) and they're volatile — the used DDR4 market is the value play, and NVMe deals reward patience. The *predictions* don't change with the prices; when the hardware arrives, measured numbers go in this table next to them.

### Measure your own box instead of trusting this table

Every number above was measured by hand; `quantprobe calibrate` (v1.19) does the same for your
box: RAM stream (a real read, not the spec sheet), disk on your own file, and GPU *sustained*
clocks — which catches the stuck-boost state that silently costs 25–30% and that only a reboot
clears on consumer cards (pre-registrations #60/#61). Results persist to
`~/.quantprobe/calibration.json` and `plan` consumes them automatically, tagged `[calibrated]`;
the optional anchor runs on your own GGUF make anchored predictions the default from there.

