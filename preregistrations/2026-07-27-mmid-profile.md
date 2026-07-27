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

---

## Scored (2026-07-27, log: `weights/data/prereg33_mmid_profile.log`)

**Verdict: P-1 HIT, P-2 MISS (the dispatch hypothesis dies with proof of mode), P-3 HIT — and the
k-sweep relocates the entire problem.**

- **P-1 (compiler parity ±15%): HIT.** gcc `EM=0` 11.92 ± 0.24 vs the shipped MSVC binary's
  13.15 — −9.4%. Cross-binary comparisons are valid; within-binary A/Bs are clean.
- **P-2 (expert-major dispatch ≥10%): MISS — null.** 11.75 (ON, banner captured) vs 11.92 (OFF).
  Giving each thread whole experts — long sequential streams, zero atomic contention — changes
  NOTHING. Second independent refutation of every geometry/scatter story (#32 was the first).
- **P-3 (marginal expert ≥20 GB/s): HIT — 23.1 GB/s.** The k-sweep (k=1/2/4/8 baked into a
  scratch copy: 18.06 / 16.81 / 15.02 / 11.81 tok/s) is linear with slope **4.19 ms/expert** =
  96.8 MB of expert weights at 23.1 GB/s, essentially the stream wall. **The per-expert GEMV
  inner loop is already at physics.**

### The relocation: the cost is the FIXED per-token machinery

| component of the 84.7 ms token (k=8) | ms | share | attribution |
|---|---|---|---|
| 8 experts × 4.19 | 33.5 | 40% | **AT PHYSICS** (23.1 GB/s) |
| always-active bytes (0.44 GB) at the wall | ~19 | 22% | AT PHYSICS |
| **unattributed fixed machinery** | **~32** | **38%** | **CODE** |

The intercept (51.2 ms) dwarfs what always-active bytes can explain. The remaining ~32 ms/token is
per-token, per-op machinery — the leading suspect is the CPU graph executor's **per-op thread
barrier**: a 48-layer MoE graph carries roughly 3–4× the op count of a dense model (router, topk,
gather, three expert matmuls per layer), and ~700+ barriers × tens of µs on 4 threads lands
exactly in this range. This is also consistent with everything previously measured: it explains
the dense-vs-MoE gap (op count, not access pattern), the E2c null (the experts were never the
problem), the #31 scheduling nulls (OS-level knobs cannot remove ggml-internal barriers), and the
#32 microbench (memory is fine).

### Consequence for task #28

The upstream target SHIFTS: not `MUL_MAT_ID` dispatch — the graph executor's per-op
synchronization on small-op-heavy graphs. Next discriminator, stated now: the fixed cost must
scale with GRAPH OP COUNT, not with bytes — measurable by fitting the intercept across models of
different layer counts, and by a barrier-fusion prototype in `build-cpu`. Ceiling if fully
captured: 84.7 → ~53 ms ≈ **19 tok/s pure-CPU** (from 11.8), and proportionally on the split.

**Wired into:** `findings/REGISTER.json:D-05` evidence chain · task #28 (retargeted) ·
`weights/data/prereg33_mmid_profile.log`. The 11 GB `_k_sweep_scratch.gguf` is retained on D: for
future sweeps (its `expert_used_count` is restored to 8).
