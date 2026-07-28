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

---

## Scored (2026-07-27, log: `weights/data/prereg34_barrier_runtime.log`)

**Verdict: P-1 HIT at +40%, P-2 marginal miss (10.8 vs staked ≤10), P-3 MISS (+4.6%, bars
overlap). The barrier tax was largely the MECHANISM, and it is a build property.**

| build / env | tg32 (t=4, k=8) | barrier ms/token |
|---|---|---|
| gcc + OpenMP (mingw libgomp, kernel semaphores) | 11.92 | 30.8 |
| **gcc + `GGML_OPENMP=OFF` (ggml spin barrier)** | **16.64 ± 1.97** | **10.8** |
| shipped MSVC + libomp, default | 13.28 ± 0.83 | — |
| shipped MSVC + libomp, `KMP_BLOCKTIME=infinite` | 13.89 ± 0.55 | — |

- **P-1 (≥15%): HIT at +40%.** One cmake flag. The no-OpenMP build also beats the SHIPPED MSVC
  binary by ~25% on pure-CPU decode — new best: **16.64 tok/s**.
- **P-2 (≤10 ms/token): miss by 8%** — 10.8. The residual is what the fusion prototype now
  targets, and the honest fusion prize shrinks accordingly: ~7–8 ms max, ~+12%.
- **P-3 (≥5% on shipped): MISS at +4.6%**, error bars overlapping. LLVM libomp's default
  BLOCKTIME (200 ms) already spins across a 60 ms token — the shipped binary never paid the
  kernel-semaphore tax. The tax is SPECIFIC to mingw-libgomp builds, whose port has no spin
  phase at all (symbol-level: `sem_wait`/`ReleaseSemaphore`, no `gomp_spin_count_var`).

### What this is, precisely

Not a llama.cpp bug — a **build-configuration trap on Windows/mingw**: `GGML_OPENMP=ON` (the
default) routes every one of ~2,070 per-node barriers through a Windows kernel semaphore, costing
~40% of CPU decode on 4 threads. ggml's own fallback barrier (pure spin, `_mm_pause`) is the right
mechanism for this graph shape and is one flag away. Upstream deliverable #1 is therefore
DOCUMENTATION/BUILD GUIDANCE with this data attached - cheaper than code and reaches everyone who
builds with gcc on Windows. Deliverable #2 (elementwise-chain fusion under the existing
`ggml_cpu_try_fuse_ops` hook, whose TODO already points at planning-time fusion) now targets the
residual 10.8 ms.

### Scope notes, stated rather than discovered later

- The split-placement decode (the shipped configuration) runs on the CUDA build with libomp,
  which mostly spins already - the +40% does NOT transfer as-is; a CUDA+no-OpenMP build is the
  follow-up measurement.
- quantprobe's CPU-tier eta was fitted on libomp-family binaries. A user running a mingw-libgomp
  build sits ~40% BELOW our CPU predictions - a build-dependence the tool cannot see from
  hardware specs. Recorded as C-07.

**Wired into:** `findings/REGISTER.json:C-07` · `FUTURE.md` (upstream deliverables re-ordered) ·
task #28.
