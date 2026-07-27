# DRAFT — upstream issue for ggml-org/llama.cpp (NOT POSTED; for Federico's review)

**Proposed title:** CPU decode: `GGML_OPENMP=ON` with mingw-w64 libgomp costs ~40% on
small-op-heavy graphs — every graph barrier is a kernel semaphore, and the port cannot spin

## Summary

On Windows/mingw-w64 builds with the default `GGML_OPENMP=ON`, `ggml_barrier` resolves to
`#pragma omp barrier` in libgomp — and the mingw-w64 libgomp port has **no spin phase**: every
barrier takes pthread mutex traffic plus a Windows kernel semaphore sleep/wake per waiter. A
MoE decode graph executes ~2,000 barriers per token, so the barrier *mechanism* alone costs ~40%
of decode throughput on a 4-core CPU. Building with `-DGGML_OPENMP=OFF` (ggml's own spin barrier,
`ggml-cpu.c` fallback path) recovers all of it.

We measured this while profiling per-node costs; the numbers, evidence, and two supporting
experiments are below. The likely useful outcomes: a note in the build docs, and/or defaulting
`GGML_OPENMP=OFF` for mingw toolchains, and/or a startup warning when the OpenMP runtime is
detected to be sleep-only.

## Measurements

Hardware: i5-7600K (4C/4T), DDR4-3000 (measured stream: 26.1 GB/s pure-read).
Model: Qwen3-30B-A3B, Q2_K (`llama-bench -ngl 0 -t 4 -mmp 0 -n 32 -p 0 -r 3..5`).

| build | tg32 tok/s |
|---|---|
| gcc 16.1 (w64devkit) + libgomp, `GGML_OPENMP=ON` (default) | 11.92 |
| MSVC + LLVM libomp (spins `KMP_BLOCKTIME` before sleeping) | 13.28 |
| gcc 16.1, `-DGGML_OPENMP=OFF` (ggml spin barrier) | **16.64** (+40% vs gomp) |

A per-node profiler (timing each node's compute and its trailing barrier separately from
thread 0; ledger balances against wall-clock to 0.3%) attributes the difference:

| | barrier ms/token |
|---|---|
| libgomp build | **30.8** (33% of the token; ~19 µs × ~1,640 inter-node barriers) |
| `GGML_OPENMP=OFF` | **10.8** |

The extreme case is instructive: at batch size 1, `GGML_OP_ADD` runs 432 times/token on this
graph (per-expert aggregation), doing **0.35 ms of compute behind 7.7 ms of barriers** on the
gomp build — a 22× sync-to-work ratio for ops where three of four threads have no rows to
process and arrive at the barrier instantly.

## Why mingw libgomp cannot be tuned around

Symbol-level: in the shipped `libgomp.a` (w64devkit gcc 16.1.0), `bar.o` references
`pthread_mutex_lock/unlock`, `sem_init/sem_post`, `gomp_sem_wait`, and contains **no reference to
`gomp_spin_count_var`** — the spin-count knob (`GOMP_SPINCOUNT`, `OMP_WAIT_POLICY=active`) is
compiled out of this port's barrier path. `sem.o` maps `gomp_sem_wait → sem_wait`, and
winpthreads' semaphore imports are `CreateSemaphoreA` / `ReleaseSemaphore` /
`WaitForSingleObject`. Every barrier is kernel transitions; no environment variable changes that.
(LLVM libomp is fine: default `KMP_BLOCKTIME=200ms` spins across an entire token; we measured
`KMP_BLOCKTIME=infinite` worth only ~+4.6% on the MSVC build.)

## Two supporting datapoints

1. **The barrier COUNT is not the cheap lever once the mechanism is right.** We prototyped two
   CPU fusions to remove ~570 of the ~1,640 inter-node barriers/token (a CPU port of the
   CUDA/Vulkan `topk-moe` pattern, and an n-ary fuse of the dependent expert-sum ADD chain).
   Output byte-identical; throughput on the spin-barrier build: **no measurable gain**
   (19.42 ± 0.86 baseline vs 19.55 ± 1.08 fused, same session, position-controlled) — because
   the barriers removed are precisely the ones a spin barrier already makes nearly free. The
   node-count reduction only pays where barriers are kernel-priced, and there the right fix is
   the build flag. (Happy to share the prototype; its three correctness requirements — order-
   identical arithmetic, tie bailout around unstable `std::sort` argsort, and a memory-range
   overlap check analogous to `ggml_cuda_check_fusion_memory_ranges` — may be useful to anyone
   porting fusions to the CPU backend.)
2. **This is independent of chunk distribution.** Expert-major dispatch (whole experts per
   thread instead of row-splitting each expert) measured exactly null on this CPU, consistent
   with PR #25048's work-stealing being a separate axis (barrier WAIT on heterogeneous cores)
   from the mechanism cost reported here.

## Repro

```
cmake -B build-omp   -DGGML_CUDA=OFF                    # default GGML_OPENMP=ON
cmake -B build-noomp -DGGML_CUDA=OFF -DGGML_OPENMP=OFF
cmake --build build-omp --target llama-bench -j && cmake --build build-noomp --target llama-bench -j
build-omp/bin/llama-bench   -m <moe-model.gguf> -ngl 0 -t <cores> -n 32 -p 0 -r 5
build-noomp/bin/llama-bench -m <moe-model.gguf> -ngl 0 -t <cores> -n 32 -p 0 -r 5
```

The effect scales with graph node count per token — MoE models with per-expert aggregation show
it most; dense models less so. Windows + mingw-w64 gcc required for the pathological case.
