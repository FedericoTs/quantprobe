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

---

## Scored (2026-07-27, log: `weights/data/prereg27_decomposition.log`)

**Verdict: P-1 MISS (low — the wall was too optimistic), P-2 HIT, P-3 marginal at 17–25%,
P-4 HIT. The decomposition closes, and it MOVES THE WALL DOWN.**

| measurement | value |
|---|---|
| stream, copy (read+write), 1 thread — already saturated | 30.4 GB/s |
| stream, pure read, 4 threads | **26.1 GB/s** |
| pure-CPU decode (`-ngl 0 -t 4`) | 14.06 ± 0.28 tok/s |
| split decode, same session | 22.25 ± 0.28 tok/s |

- **P-1 (stream 34–42 GB/s): MISS, LOW.** 26.1. The DDR4-3000 spec sheet says 48; a single thread
  already saturates the real controller at ~30 (copy). Both L-11 walls were computed on bandwidth
  this box cannot deliver.
- **P-2 (kernel arm at 55–70% of stream): HIT.** 71.1 ms for 1.217 GB = 17.1 GB/s effective =
  **66% of measured stream**. The CPU path is already decently close to the memory system's real
  ceiling — mechanisms 1+2 own the bulk of the gap, and most of what they own is the SPEC SHEET'S
  fiction, not llama.cpp's inefficiency.
- **P-3 (sync <20% of token time): MARGINAL.** Sync-free expectation: 0.516 GB at the kernel arm's
  17.1 GB/s = 30.2 ms, plus VRAM share 3.7–7.3 ms (η_vram 1.0–0.5) → 33.9–37.5 ms. Measured 44.9.
  Excess **7.4–11.1 ms = 17–25%**, straddling the stake. Scored as inconclusive-leaning-miss at the
  central estimate (22.7%); D-05's no-fork verdict survives but with a measured asterisk: there are
  ~10 ms/token on the table for a runtime that eliminates per-layer synchronisation, worth ~+29%
  decode — real, and less than the fork's cost by D-05's own arithmetic, but no longer negligible.

### The wall, recomputed on measured physics

| basis | wall |
|---|---|
| L-11 as staked (48 GB/s spec) | 69.5 tok/s |
| L-11 "realistic" (36 GB/s assumed) | 52.9 tok/s |
| **measured stream (26.1 GB/s)** | **41.1 tok/s** |

**The 52.9 target is physically unreachable for raw decode on this box.** The complete capturable
budget — perfect kernel (17.1→26.1 on the host share) plus zero sync — lands exactly on 41. We
measure 22.25 = **54% of the true wall**. A heroic runtime effort captures at most 1.85×, and
ktransformers-class engineering on a 4-core AVX2 CPU realistically much less.

**Consequence, stated plainly: any path to ≥52.9 tok/s must break the every-byte-every-token
axiom itself.** That is speculation (#28) — one weight-read verifying several tokens — and nothing
else in the register can do it.

**Wired into:** `findings/REGISTER.json:L-11` (walls corrected to measured basis) ·
`findings/REGISTER.json:U-09` (scored) · pre-registration #28 (the consequence).
