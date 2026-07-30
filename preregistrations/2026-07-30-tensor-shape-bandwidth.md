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
