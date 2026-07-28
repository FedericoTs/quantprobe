# Pre-registration #49 (scored inline): the first per-op ledger of the split token on a CUDA build

**Author:** Federico Sciuca · **Date:** 2026-07-28. **Diagnostic.** The E3 per-op profiler
(`GGML_OP_PROFILE=1`) has existed since #33 but could only run on CPU-only builds — this box had no
`nvcc` able to target Pascal until CUDA 12.9 was installed this session. This is its first run
against the shipped MoE split placement with CUDA active.

## Measured — MoE flagship, split placement, tg128, CUDA 12.9 `sm_61` build

```
[e3] op            calls   compute_ms   barrier_ms
[e3] MUL_MAT_ID    12384       2379.6        783.3
[e3] GLU            4128          5.3         17.1
[e3] GET_ROWS        129          1.4          0.0
```

129 tokens. `12384 / 129 = 96` = 32 CPU-resident layers × 3 matmuls (gate/up/down) — **the
instrument is measuring exactly what it should**, which is the first thing to check.

| component | ms/token | share |
|---|---|---|
| CPU expert compute (`MUL_MAT_ID`) | **18.45** | 32% |
| CPU barriers | 6.20 | 11% |
| CPU other ops | 0.05 | 0% |
| **CPU executor total** | **24.70** | **42%** |
| **NOT VISIBLE TO E3** | **33.44** | **58%** |
| token wall-clock | 58.14 | 100% |

## The two results

**1. The CPU side is finished, and now measured rather than inferred.**
0.516 GB of expert weights in 18.45 ms = **28.0 GB/s**, *above* the 26.1 GB/s numpy pure-read
stream and matching the 30.4 GB/s copy figure. The CPU expert path is at the memory system's
limit. Everything #27/#31/#32/#33 concluded about the CPU tier is confirmed directly. **There is
nothing left there.**

**2. 58% of the token is outside the CPU executor, and the byte model explains almost none of it.**
The GPU holds 0.700 GB, which at the card's independently measured 161.3 GB/s (#44) should take
**4.34 ms**. The non-CPU budget is **33.44 ms**. That is an effective **20.9 GB/s — η 0.11**, an
eighth of what the same card delivers to CuPy, with **29.1 ms unexplained**.

## What this rules out, and what it leaves

C-09's "~20 ms unattributed" is now located: it is **~29 ms and it is on the GPU/scheduler side**,
not in the CPU path. Already excluded as causes, each by its own control:

| candidate | status |
|---|---|
| CPU expert read | at physics — 28.0 GB/s, measured here directly |
| CPU graph barriers | 6.20 ms, fully accounted in the ledger above |
| kernel-launch overhead | refuted — CUDA graphs give +3.2% on MoE, 0.0% on dense (#47, #48) |
| `MUL_MAT_ID` stream sync | refuted at source — cannot fire at `ne[2]=1` for Q2_K on Pascal (#48) |
| per-layer GPU↔CPU round trips | refuted — 1 crossing performs identically to 32 (#43) |
| memory scatter | refuted (#32) |
| scheduling knobs | refuted, six arms (#31) |

**No mechanism is named for the 29 ms.** That is deliberate: this project has killed seven
mechanism hypotheses with controls, and every attribution made without one has been wrong.

## The instrument this now demands

E3 profiles `ggml-cpu.c`'s executor. The 29 ms is by definition *outside* it — GPU kernel time,
`ggml-backend-sched` orchestration, driver/WDDM submission, or the GPU idling while the CPU works.
Distinguishing those needs **GPU-side timing**: CUDA events around each backend subgraph, or Nsight
Systems. That is the next instrument, and unlike every previous blocker it is buildable today —
the CUDA toolchain now exists on this box.

One arithmetic note that should shape the expectation: the CPU does 24.70 ms of work and the GPU
should do 4.34 ms. If the two were perfectly overlapped the token would be ~25 ms (40 tok/s). It is
58 ms. **The system behaves as if nothing overlaps and something additionally waits** — which is
consistent with the GPU idling ~22 ms per token that the red-team independently computed, and is
the strongest remaining lead in the project.

**Wired into:** `findings/REGISTER.json:C-09` (relocated from "unattributed ~20 ms, unknown side"
to "~29 ms, GPU/scheduler side, CPU side closed") · `L-11` (CPU tier confirmed at physics).
