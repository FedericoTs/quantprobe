# Pre-registration #27: decompose the 2.4× decode gap — which share of the wall is capturable?

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## The question

L-11 computes the decode wall for the flagship (Qwen3-30B-A3B Q2_K, split placement, reference
box) at **69.5 tok/s** (η=1, theoretical DDR4) / **52.9 tok/s** (realistic stream). We measure
**21.58**. The whole 2.4× gap currently lives inside one fitted constant, η — a description, not
an explanation. The goal from here on is explicit: **get as close as possible to 52.9, or past it.**
Nothing can be captured until it is attributed, so this measures where the 46.3 ms/token actually
goes, by ablation.

Three mechanisms could each own a share, with OPPOSITE implications:

1. **Kernel/compute share** — llama.cpp's CPU path not saturating DRAM. Capturable in principle
   (ktransformers' claim), hard on 4 Kaby Lake cores without AMX.
2. **Memory-level parallelism share** — 4 cores physically cannot keep DDR4 saturated during
   GEMV's access pattern. NOT capturable by any software.
3. **GPU↔CPU synchronisation share** — 32 host layers × per-layer round trips. Capturable by a
   runtime that batches/overlaps transfers; if large, D-05's no-fork verdict reopens.

## Protocol (one session, GPU state logged, every arithmetic step shown)

1. **Stream ceiling.** Multithreaded memcpy + read-only benchmark (numpy, 1/2/4 threads, ≥1 GB
   arrays, r=3). Gives the box's real attainable DRAM bandwidth, replacing the 48 GB/s spec sheet.
2. **Kernel arm.** `llama-bench -ngl 0 -t 4` flagship `tg32`. Pure CPU: no GPU, no sync — the
   1.217 GB/token transits host DRAM only. Effective GB/s = 1.217 ÷ t. The ratio to (1) is the
   kernel+MLP efficiency, mechanisms 1+2 combined, uncontaminated by 3.
3. **Sync arm.** Split placement `tg128` same session. Expected time if sync were free:
   `t_vram_share + t_cpu_pure × (0.516/1.217)`. The measured excess over that is mechanism 3.

## Stakes

- **P-1 (spec sheet vs reality).** Measured stream is **34–42 GB/s**, i.e. the 48 is not
  attainable and the "realistic wall" of 52.9 was computed on the right basis.
- **P-2 (the kernel arm is the bulk).** Pure-CPU effective bandwidth lands at **55–70% of measured
  stream** — the gap is mostly mechanisms 1+2, living in the CPU path itself.
- **P-3 (sync is minor).** The sync share explains **<20%** of the hybrid's total token time. If
  it explains more, D-05's no-fork verdict is wrong for this box and REOPENS — a patched runtime
  that batches transfers would be worth real money here, and I will say so.
- **P-4 (no law changes).** Measurement only; anchors bit-identical.

## What "capturable" will mean, quantitatively

After this, the road to 52.9 has a budget: `capturable ≈ (stream − effective) × kernel-headroom +
sync share`, and each open lever (U-07 top-k, speculation #28, batching) multiplies from whatever
base this establishes. If P-2 shows the CPU path already runs at ≥70% of *measured* stream, then
the honest conclusion is that raw decode on this box is within ~1.4× of its true wall and **the
only route to 52.9+ is speculation** — which is measured next, in #28, regardless.
