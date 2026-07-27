# Pre-registration #34: the barrier is a RUNTIME CHOICE — OpenMP kernel sleeps vs ggml's spin

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## What the research found

The E3 profiler measured 30.8 ms/token of barrier time (33%) on `build-cpu`. The desk research
then found the confound: **that build is OpenMP (`GGML_USE_OPENMP`), and mingw libgomp's barrier
has NO spin phase** — symbol-level evidence: `bar.o` references `sem_wait`/`ReleaseSemaphore`/
`WaitForSingleObject`, no `gomp_spin_count_var`. Every barrier is a kernel transition, ~2,070
times per token. ggml's own non-OpenMP fallback (`ggml-cpu.c:578-603`) is a pure user-space spin
with `_mm_pause` — order 10-50× cheaper. And the SHIPPED MSVC binary imports `libomp140`, whose
barriers spin for `KMP_BLOCKTIME` (default 200 ms) before sleeping — a third behavior.

So before any fusion prototype: how much of the 33% is the barrier MECHANISM rather than the
barrier COUNT?

## Arms

- **A (gcc, spin).** Rebuild `build-cpu` with `-DGGML_OPENMP=OFF` → ggml's spin barrier. Same
  compiler, same flags otherwise. Measure tg32 (t=4, k=8) and the E3 profile.
- **B (shipped, zero rebuild).** `KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active` on the b10098
  MSVC binary — libomp never sleeps. Measure tg32.

## Stakes

- **P-1.** Arm A gains **≥15%** over the OpenMP gcc build (11.92 → ≥13.7 tok/s).
- **P-2.** Arm A's E3 barrier total drops to **≤10 ms/token** (from 30.8).
- **P-3.** Arm B gains **≥5%** on the shipped binary (13.15 → ≥13.8) — if libomp already spins
  effectively at default BLOCKTIME, this is ~0 and the shipped numbers were never taxed.

## Why this matters beyond one box

If P-1/P-2 hit, the fusion prototype's baseline changes and part of the "+18-35% upstream prize"
is actually a BUILD-CONFIGURATION finding (Windows OpenMP builds pay kernel semaphores per node)
worth reporting upstream as such - cheaper than code, reaches everyone who builds with gcc on
Windows.
