# Pre-registration #32: the fork gate — is the 40% scatter penalty memory or code?

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## The decision this gates

The user asks: if llama.cpp limits us, why not fork it or build our own runtime? The measured
record says llama.cpp's dense kernel is AT the memory wall and scheduling is optimal — but the
**~40% MoE scatter share** (#31: flagship CPU path 17.1 GB/s vs dense 28.4 at the same format) is
attributed to "the prefetcher" on an argument, not a measurement. That attribution decides
everything:

- If the penalty is the MEMORY SYSTEM (scattered access is inherently slower), no code recovers
  it, the fork buys nothing, and D-05 closes permanently for this tier.
- If scattered slab reads actually stream at full bandwidth, the 40% is **llama.cpp's expert-path
  code** (per-expert dispatch, unfused loops, router overhead) — capturable, and the fork/PR
  question REOPENS with a measured prize.

## The experiment

The flagship's expert access pattern, isolated from all code: the active experts are 8 slabs of
~2.0 MB per layer (0.774 GB ÷ 48 layers ÷ 8 experts). Read a 1 GB array in 2 MB chunks,
**identical chunk mechanics, only the ORDER differs**: sequential vs shuffled. Same python-call
overhead in both arms, r=3, 4 threads (matching `-t 4`).

## Stakes

- **P-1 (the attribution).** Shuffled 2 MB chunk reads lose **≥25%** versus sequential — the
  memory system explains the scatter share, and the fork stays closed. Staked at 25% because the
  measured penalty is ~40%: if the access pattern explains most of it, it must show at least this.
- **P-2 (the control).** Sequential chunked reads reproduce the stream benchmark (26.1 GB/s ± 15%)
  — otherwise the harness, not the memory system, is being measured.

## Refuted if

**P-1 misses badly (shuffled loses <10%).** Then ~2 MB slabs re-engage the prefetcher just fine,
the 40% is llama.cpp's expert-path CODE, and the capturable prize for an expert-gather kernel is
real: up to ~+65% raw decode (17.1 → 28.4 GB/s on the host share). That flips my own advice: the
upstream contribution becomes the highest-value engineering target this project has, and I will
say so and stake the follow-up.

## What ships

Nothing either way — this is a decision gate. The result lands in D-05 (the fork verdict) as its
final evidence, whichever direction it points.
