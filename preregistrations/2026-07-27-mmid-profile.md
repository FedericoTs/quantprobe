# Pre-registration #33: profiling the expert path — dispatch geometry vs marginal expert cost

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## What this measures

#32 proved the ~40% MoE CPU penalty is code, not memory. This attributes it WITHIN the code, with
two instruments:

1. **The E2c A/B.** A prior session left an experiment in `tools/llama.cpp-src`
   (`ggml-cpu.c:1641`, "expert-major dispatch"): at batch-1 decode, give each thread WHOLE experts
   (one long sequential stream each, zero atomic contention) instead of the stock path where all
   threads cooperate on each expert's GEMV in strided chunks. Runtime toggle `GGML_MMID_EM=0/1`,
   already built in `build-cpu/bin/llama-bench.exe` (gcc, CPU-only). The A/B is within one binary.
2. **The k-sweep.** `expert_used_count` baked into a scratch copy (the #18 technique), k ∈
   {1,2,4,8}: fit `t(k) = a + b·k`. Slope b = marginal per-expert cost (per-expert bytes at what
   GB/s?); intercept a = fixed per-token cost (attention + router + graph dispatch).

## Stakes

- **P-1 (compiler parity).** The gcc build at `GGML_MMID_EM=0` (stock semantics) lands within
  **±15%** of the shipped MSVC binary's 13.15–14.06 tok/s. Outside that, the two binaries are not
  comparable and every cross-binary conclusion is void.
- **P-2 (dispatch geometry is a first-order cost).** Expert-major (`EM=1`) beats stock (`EM=0`)
  by **≥10%** at `-t 4`. The stock path runs 4 threads in strided chunks over ONE ~2 MB slab at a
  time — 4 interleaved streams into the same slab, ~1,150 times per token; expert-major runs 4
  long streams into 4 different slabs. #32 showed long streams are what the memory system wants.
- **P-3 (the marginal expert is efficient).** The k-sweep slope implies **≥20 GB/s** on marginal
  expert bytes — i.e. the per-expert GEMV itself is near-wall once running, and the deficit lives
  in the intercept (fixed dispatch) plus stream geometry, not in the inner loop.

## Refuted if

P-2 misses AND P-3 shows the marginal expert itself is slow (<15 GB/s): then the inner GEMV is the
problem, expert-major dispatch is a dead end, and the PR target moves to the vec_dot kernel itself.

## What ships

Numbers and attribution only — the patch, if E2c wins, ships as an UPSTREAM PR (task #28), never a
fork. Every measured number lands in D-05's evidence chain.
