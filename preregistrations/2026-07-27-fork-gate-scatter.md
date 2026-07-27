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

---

## Scored (2026-07-27, log: `weights/data/prereg32_fork_gate.log`)

**Verdict: P-1 REFUTED — decisively, and in the direction that reopens the question. P-2 HIT
(sequential chunks at 23.88 GB/s = 91% of the stream benchmark; harness valid).**

| access pattern, 1 GB in 2 MB slabs, 4 threads | GB/s |
|---|---|
| sequential | 23.88 |
| **SHUFFLED** | **24.56 (+2.9% — noise)** |

The memory system is INDIFFERENT to expert-slab-sized scatter. The staked "≥25% loss" needed for
the memory-system attribution did not appear at all. Therefore **the ~40% MoE penalty (17.1 GB/s
vs dense 28.4 at identical format and threads) is llama.cpp's expert-path CODE** — and my earlier
attribution of it to prefetch defeat, inherited from Law 4's scatter note, is refuted by direct
measurement. An argument had been standing in for a measurement since 2026-07-26.

(The 128 KB rows in the log are python-overhead-dominated and not interpretable; disclosed, unused.)

### The localization diagnostic

MoE pure-CPU thread scaling: 6.05 → 9.92 → 13.15 tok/s (1×/1.64×/2.17× at 1/2/4 threads),
saturating at **16.0 GB/s where dense reaches 28.4 with the same 4 threads**. Neither
bandwidth-saturated nor serial-overhead-flat: consistent with per-expert dispatch cost plus
partial-bandwidth small-GEMV work — ~1,150 small matmuls per token against dense's ~140 large
ones. Precise attribution within that path is the first task of the engineering itself.

### The decision, re-made on the corrected evidence

**D-05 REOPENS, scoped.** The prize for fixing ggml's CPU `MUL_MAT_ID` path is real and bounded:
host share at dense-kernel efficiency ≈ **+65%**, taking raw decode from 22.25 toward **~30
realistic, 41 at perfection**. The correct vehicle is an **upstream PR**, not a fork and not a
from-scratch runtime, for measured reasons:

1. Every OTHER component measured at the wall (dense kernel, scheduling, K-format choice) — a
   from-scratch runtime re-implements months of parity to chase the same bounded gap.
2. A fork pays a permanent maintenance tax and kills the property that quantprobe's advice runs
   on stock llama.cpp — the distribution channel every user already has.
3. Even PERFECT capture (41 tok/s) lands below the free speculation number (50–59) — this prize
   matters specifically for NOVEL generation, where speculation is closed (D-10).

**Wired into:** `findings/REGISTER.json:D-05` (reopened, scoped to the CPU expert path) ·
`findings/REGISTER.json:L-11` (scatter attribution corrected) · `FUTURE.md` (the PR target).
