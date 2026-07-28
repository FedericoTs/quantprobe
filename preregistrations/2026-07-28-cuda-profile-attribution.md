# Pre-registration #46: attribute the split token on a CUDA build — the two open numbers, one run

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurement. **Status: STAKED.**

## What this closes

Two numbers have been open and both are blocked on the same missing instrument:

1. **C-09**: ~20 ms of the split placement's 50.6 ms token is unattributed. Five candidate causes
   are each refuted by their own control (CPU expert read at physics #33; per-layer round trips
   #43; scheduling #31; memory scatter #32; graph barriers #34).
2. **C-02 / L-14**: llama.cpp's quantized VRAM decode reaches η 0.513 where the card independently
   delivers η 0.84 (#44) — a **1.64×** gap that is not hardware and not arithmetic (`sm_61` has
   native `__dp4a` at ~4× fp32), leaving the kernel's block-unpacking work as the only remaining
   explanation, **unmeasured**.

The E3 per-op profiler (`GGML_OP_PROFILE=1`, written in this project, `ggml-cpu.c`) already times
every graph node's compute and its trailing barrier and balances to 0.3% of wall clock. It has
only ever run on **CPU-only** builds, because this box had no `nvcc` able to target Pascal. CUDA
12.9 supplies it.

## Method

Build llama.cpp with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61`, E3 profiler compiled in.
Run the shipped split recipe (`-ngl 99 -ot "blk\.(16..47)\.ffn_.*_exps\.=CPU" -ub 1024`), tg128,
`GGML_OP_PROFILE=1`. Compare the per-op ledger against the byte model.

## Stakes

- **P-1 (the ledger balances).** Profiler total is within **±5%** of measured wall-clock time per
  token. If it does not balance, the instrument is wrong on this build and nothing below is
  interpretable — the same check that validated it at 0.3% on CPU-only.
- **P-2 (the 20 ms gets a name).** A single op class accounts for **≥ 40%** of the unattributed
  ~20 ms. My expectation, stated before looking: `MUL_MAT_ID` on the CUDA side (the 16 VRAM expert
  layers) is the largest single contributor, because that is the op the 1.64× kernel gap lives in.
- **P-3 (the two open numbers are the same number).** The CUDA-side `MUL_MAT_ID` + `MUL_MAT`
  compute time exceeds the byte model's prediction for the VRAM share (3.6 ms at η=1) by a factor
  within **±30% of 1.64** — i.e. C-09's remainder and C-02's kernel gap are one mechanism seen
  from two directions, not two independent problems.
- **P-4 (no anchors move).** Measurement only; all four published anchors bit-identical.

## Refuted if

**P-3 fails** — the CUDA-side ops are close to their byte model while ~20 ms still hides elsewhere.
Then C-09 and C-02 are genuinely independent, the unattributed time is somewhere neither profiler
nor byte model has looked (driver, WDDM scheduling, PCIe descriptor overhead), and the next
instrument is a driver-level trace, not a graph-level one.

## What ships

Attribution only, into the register. **No constant in `plan.py` moves on the strength of one
profiled run** — the tool's η values describe what llama.cpp delivers to users, and #44 already
established that raising them to a ceiling no shipped runtime reaches would make the tool lie.

If P-2 and P-3 both hold, the follow-up is a CUDA kernel experiment against `MUL_MAT_ID` with a
staked prize of the measured 1.64×, and that is the first time this project would have a
GPU-side software target with a number attached.
