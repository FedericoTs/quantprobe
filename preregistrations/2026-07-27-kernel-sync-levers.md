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

---

## Scored (2026-07-27, log: `weights/data/prereg31_kernel_sync.log`)

**Verdict: P-1 MISS, P-2 MISS (the refutation clause fires), P-3 MISS as staked — and arm F
caught two unstaked findings that matter more than every staked one.**

### The scheduling grid: dead, cleanly

| arm | tg128 |
|---|---|
| K0 baseline `-t 4` | 21.42 ± 0.20 |
| K1 `-t 3` | 20.92 ± 0.28 (−2.3%) |
| K2a `--poll 0` | 21.40 ± 0.36 |
| K2b `--poll 100` | 21.03 ± 0.48 |
| K3 `--prio 2` | 21.13 ± 0.09 |
| K4 `--cpu-strict 1` | 21.42 ± 0.35 |

Nothing clears +5%; two levers lose. K5 (the combination) is vacuous — there is nothing positive
to combine. **The sync share is not scheduler-shaped.** Capturing it needs actual code (async
transfer batching), which is upstream-PR territory priced at ≤29% by #27, and llama.cpp's defaults
are, on this box, already right.

### Arm F, finding 1: the "kernel gap" is not the kernel — it is MoE scatter

Pure-CPU decode, dense 7B, effective GB/s (= bytes × tok/s):

| format | tok/s | effective GB/s |
|---|---|---|
| Q2_K | 10.11 ± 0.06 | **28.4** |
| Q4_K_M | 6.81 ± 0.12 | **29.7** |
| IQ3_XS | 3.39 ± 0.05 | **10.6** |

**Dense K-quant CPU decode already runs AT the stream wall** (28–30 GB/s vs 26.1 measured
pure-read, 30.4 copy). llama.cpp's dense kernel has nothing left to give. The flagship's 17.1 GB/s
(#27) is therefore not kernel inefficiency — it is the **MoE expert-scatter penalty, ~40%**:
8 scattered expert GEMVs per token defeat the prefetcher where one dense GEMV streams. That is the
slab-hopping property already recorded in Law 4's scatter note, now with its own number. P-3 as
staked (Q4_K_M ≥ +20% over Q2_K) is a MISS — the K-quants tie at the wall, which is a *better*
result than the stake: within K-formats there is no fat-file bonus to chase, and the flagship's
Q2_K choice is vindicated for the CPU tier.

### Arm F, finding 2: I-quants are catastrophic on CPU tiers — SHIPPED as a warning

IQ3_XS delivers **10.6 GB/s where K-quants deliver ~29** — a 2.7× decode penalty for any
host-resident placement, invisible to a user who picked the IQ file because it was smaller. The
planner now warns when a >30% I-quant file lands on a host tier, and stays silent in VRAM where
IQ formats measured mid-pack (the η study). This is C-05's fourth instance: dequant cost is a
format property, and on the CPU tier the IQ codebook lookup is compute-shaped where K-format
dequant is bandwidth-shaped.

### Where this leaves the ceiling chase

22.25 → 41.1 needs ~1.85×. Now attributed: **~40% MoE scatter** (needs expert-gather kernels —
real code, upstream scale), **~17–25% sync** (needs async transfer batching — real code), **0%
scheduling** (measured), **0% K-format choice** (measured at the wall). On this box, with stock
llama.cpp, raw decode is done: every remaining percent costs upstream engineering, and the
measured payoff for all of it combined is bounded by 1.85×. The register's conclusion stands —
the wall is passed by not reading bytes (speculation, 50–59 tok/s), not by reading them better.

**Wired into:** `quantprobe/spec.py:from_gguf` (iq_share) · `quantprobe/plan.py` (the warning) ·
`tests/smoke.py:t_iq_quants_warned_on_cpu_tiers` · `findings/REGISTER.json:D-11, L-11 (scatter
attribution), C-05 (fourth instance)`.
