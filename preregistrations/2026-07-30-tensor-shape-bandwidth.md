# Pre-registration #80: is decode bandwidth a property of the format, or of format x shape?

**Author:** Federico Sciuca · **Date staked:** 2026-07-30, BEFORE compiling the probe. **STAKED.**

Every `FORMAT_EBW` entry (Q4_0 119.1, Q4_K 106.4, the IQ ladder) was measured e2e on **dense 7B
files**, where a token is one long stream of LARGE FFN matrices. Prereg #79 priced a MoE's GPU
tier at attention's FORMAT number and over-promised in near-linear proportion to attention share
(**r = 0.87, n = 6**: 45–51% share → helped, 74–77% share → +27 points worse). U-30's reading:
attention-shaped tensors do not reach their format's headline bandwidth.

`tools/kernelprobe/shape.cu` holds format, bit-width, kernel, activation source and total bytes
constant and varies **only rows per tensor**, with one launch per tensor — the decode geometry
llama.cpp actually uses (#55, one output row per block). Real geometries at hidden 2048:
kv-proj ≈ 512 rows, q/o-proj ≈ 2048, FFN/expert ≈ 6144, large-FFN ≈ 8192+.

## Stakes

- **P-1 (the mechanism).** Bandwidth at **512 rows/tensor is ≥15% BELOW** bandwidth at
  8192 rows/tensor, same bytes, same format.
- **P-2 (monotone, not noise).** The sweep is monotone non-decreasing in rows/tensor across
  128 → 8192 (allowing ±3% for measurement noise). A U-shape or a flat line refutes the
  size-floor reading even if P-1's endpoints happen to differ.
- **P-3 (the magnitude matches the damage).** The attention-vs-FFN gap is **large enough to
  explain #79**: applying the measured attention-shape penalty to a 75%-attention token must
  move a Qwen3.6-class prediction by ≥15% — otherwise shape is real but not the dominant term
  and something else caused the +27 points.

## KILL RULE

**If P-1 fails, U-30 is refuted**: shape does not set decode bandwidth on this card, FORMAT_EBW
stays format-only, and #79's failure needs a different explanation (the `ne` weights, U-29,
become the sole suspect). If P-1 holds but P-3 fails, shape is real but secondary — recorded as
a scoped finding, NOT wired into the tool, because a term that cannot explain the error it was
invented for has not earned a place in the law.

**Wired into:** pending; `spec.FORMAT_EBW` would become shape-classed only if all three hold.

---

## SCORED — 2026-07-30. **ALL THREE STAKES HIT.**

Raw log: `weights/data/prereg80_shape.log`. GTX 1060, 384 MB swept per point, 4.5-bit layout,
same kernel and activation source throughout; **only rows-per-tensor varies**.

| rows/tensor | GB/s | of spec peak | shape class | penalty vs best |
|---|---|---|---|---|
| 128 | 31.0 | 16% | attention (kv proj) | **−69.2%** |
| 256 | 46.7 | 24% | attention (kv proj) | −53.6% |
| 512 | 64.5 | 34% | attention (kv proj) | **−35.8%** |
| 1024 | 80.1 | 42% | attention (q/o proj) | −20.2% |
| 2048 | 88.8 | 46% | attention (q/o proj) | −11.6% |
| 4096 | 95.2 | 50% | FFN / expert | −5.2% |
| 8192 | 99.6 | 52% | FFN / expert | −0.9% |
| 16384 | 100.5 | 52% | large FFN | 0.0% |

- **P-1 HIT, by more than double the bar.** 512 rows/tensor runs **−35.8%** against 8192 —
  the stake was ≥15%.
- **P-2 HIT.** Strictly monotone across all eight points, no U-shape, no plateau until ~8192
  where it saturates at the format's true ceiling.
- **P-3 HIT.** Attention-shaped work (mean of kv-proj and q/o-proj: 76.7 GB/s) runs **23% below**
  FFN-shaped (99.6). On a 75%-attention token that is a **21% over-promise** — the right size to
  explain #79's damage, which was worst exactly on the highest-attention models.

**The finding: `FORMAT_EBW` is not a format constant. It is a format × shape constant, and every
entry we ship was measured at the FFN end of the range** — i.e. each is an upper bound that only
large homogeneous matrices reach. A 4.5-bit weight decodes at 100 GB/s in an FFN matrix and
31 GB/s in a 128-row projection **on the same card, in the same format, in the same kernel**.

This is #59's small-model size floor and L-19's CPU-attention term seen at a third granularity:
work that is small per launch does not reach the bandwidth its bytes suggest. Three independent
regimes, one shape.
