# Pre-registration #31: the capturable half — zero-patch kernel and sync levers

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## The budget this attacks

#27 decomposed the 44.9 ms token: **~10.4 ms kernel shortfall** (the CPU path delivers 17.1 GB/s
against a measured 26.1 GB/s stream — 66%) and **~7.4–11.1 ms sync excess** (GPU↔CPU round trips
beyond the sync-free expectation). Capturing all of both would land on the 41.1 wall exactly.
Nobody captures all of anything, but llama.cpp exposes four relevant knobs that cost NO code:

- **`-t 3` vs `-t 4`** — the classic: the CUDA driver needs a core for its sync thread; on a
  4-core CPU, 4 compute threads mean the driver preempts a GEMV worker every layer boundary.
- **`--poll 0/50/100`** — spin-wait policy. Spinning burns a core but reacts instantly; yielding
  frees the core but adds wakeup latency ~32× per token (once per host layer group).
- **`--prio 2`** — Windows scheduler priority; the desktop steals timeslices at priority 0.
- **`--cpu-strict 1`** — pin threads to physical cores; stops the scheduler migrating a worker
  mid-GEMV and cold-restarting its cache state.

Plus one deeper lever that OUR OWN LAW predicts and #27's design missed: **C-05 says a quantized
byte is not a byte** — dequant cost depends on format. The kernel arm measured 17.1 GB/s *on
Q2_K*, the most dequant-expensive K-format. If the effective CPU bandwidth is format-dependent,
part of the "kernel shortfall" is not llama.cpp's fault at all — it is the price of 2.95 bits, and
the flagship might decode FASTER from a fatter file.

## Arms (split placement, tg128, r=3, one session, GPU state logged)

| arm | change |
|---|---|
| K0 | baseline `-t 4` (reference 22.25) |
| K1 | `-t 3` |
| K2 | `--poll 0` and `--poll 100` |
| K3 | `--prio 2` |
| K4 | `--cpu-strict 1` |
| K5 | best combination of K1–K4 |
| F | pure-CPU decode (`-ngl 0`), Qwen2.5-7B at Q2_K vs Q4_K_M vs IQ3_XS: effective GB/s per format |

## Stakes

- **P-1 (something is free).** At least one single lever K1–K4 gains **≥5%** over K0.
- **P-2 (the combination is material).** K5 gains **≥12%** over K0 (≥24.9 tok/s). The sync share
  is 17–25% of the token; capturing half of it via scheduling alone is the bet.
- **P-3 (C-05 strikes a fourth time — the format is part of the "kernel gap").** On pure-CPU
  decode, Q4_K_M delivers **≥20% more effective GB/s** than Q2_K at identical architecture. If it
  does, the 66%-of-stream figure is partly a FORMAT property, the flagship's Q2_K choice is
  costing raw decode, and "requantize fatter for the CPU tier" becomes a measurable
  recommendation candidate.
- **P-4 (no law changes).** Anchors bit-identical.

## Refuted if

P-1 and P-2 both miss: scheduling is not where the sync share lives, and capturing it needs actual
code (async transfer batching) — upstream-PR territory, priced by #27 at ≤29% and left there.

## What ships

Any lever that HITS goes into the emitted `run it:` command with its measured number — these are
free flags, exactly like `--no-mmap` was. P-3 hitting opens a follow-up (fatter-file decode test
on the flagship) but ships nothing by itself.
