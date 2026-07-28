# Pre-registration #43 (scored inline): where are the missing 20 tok/s in the SPLIT token?

**Author:** Federico Sciuca · **Date:** 2026-07-28. **Diagnostic, not a hypothesis test** — run to
answer a direct question: the split placement measures ~21 tok/s (47–50 ms/token) against a
41.1 tok/s wall (24.3 ms). Where does the excess go?

## Instrument 1 — sweep the CPU-resident layer count

`t(K) = fixed + K × per-layer cost`, tg128, r=3, one session:

| K (layers of experts on CPU) | tg128 | ms/token | marginal per layer |
|---|---|---|---|
| 32 (what we ship) | 19.76 | 50.61 | — |
| 36 | 16.07 | 62.23 | 2.90 ms |
| 40 | 15.28 | 65.45 | 0.81 ms |
| 44 | 15.01 | 66.62 | 0.29 ms |
| 48 | 15.01 | 66.62 | 0.00 ms |

**The linear model FAILS** — fitting K=32..40 gives a negative intercept (−7.4 ms), which is
physically impossible and is reported rather than massaged. The marginal cost per CPU layer is not
constant; it decays from 2.90 ms to zero. Physics says one layer's experts are 16 MB = **0.62 ms**
at the measured 26.1 GB/s, so the first layers moved cost ~**4.7× their bytes** and the last cost
nothing.

## Instrument 2 — the hypothesis that decay suggested, and its refutation

Decaying marginal cost with a large first step looked like **per-layer GPU↔CPU round trips**: the
`-ot` split leaves attention on the GPU and experts on the CPU for each of 32 layers, so a token
crosses the boundary ~32 times. A **pure layer split** (`-ngl N`, no `-ot`) puts whole layers on
one side — **one** crossing per token.

| configuration | crossings/token | tg128 | ms/token |
|---|---|---|---|
| `-ngl 99 -ot` 32 expert layers on CPU | ~32 | **19.76** | 50.61 |
| **`-ngl 20` pure layer split** | **1** | **19.70** | 50.76 |
| `-ngl 16` | 1 | 15.96 | 62.66 |
| `-ngl 12` | 1 | 12.88 | 77.64 |

**Eliminating 31 of 32 boundary crossings saves nothing (19.70 vs 19.76).** The round-trip
hypothesis is refuted. Per-layer GPU↔CPU synchronisation is not where the time goes — it is either
free or already overlapped.

Useful side result: at matched work division the two placement styles are **equivalent**, so the
simpler `-ngl 20` is as good as the `-ot` regex on this model for decode. Measured, not assumed.

## What this means for "where are the other 20 tok/s"

Byte accounting for the shipped split (K=32): CPU holds 0.516 GB → 19.8 ms at measured stream;
GPU holds 0.70 GB → 3.6 ms at η=1. Total **23.4 ms** against **50.6 ms** measured. **~27 ms is
unexplained**, and it is now known NOT to be:

| candidate | status |
|---|---|
| CPU expert read | at physics — 23.1 GB/s marginal, 88% of stream (#33) |
| per-layer GPU↔CPU round trips | **refuted here** — 31 fewer crossings, zero gain |
| thread scheduling, affinity, poll, priority | null, six arms (#31) |
| memory access pattern / scatter | refuted (#32) |
| ggml graph barriers | 10.8 ms/token on a spin build, and the shipped CUDA build already spins (#34) |
| GPU-side inefficiency | **partly** — C-02 has the GPU at η 0.32–0.56, which accounts for only ~5 ms here |

**Honest conclusion: ~20 ms of the split token is unattributed, and I will not name a mechanism
for it.** The instrument that would close it is the E3 per-op profiler running on a **CUDA** build,
which this box cannot produce — there is no `nvcc` installed. That is the specific blocker, and it
is a tooling gap, not a conceptual one: install the CUDA toolkit and the decomposition is one run.

This is the fifth mechanism hypothesis in this project killed by its own control, and the pattern
is worth stating: **every time the excess has been attributed without a control, the attribution
was wrong** (fixed overhead, clock state, bytes-per-token, memory scatter, and now round trips).

**Wired into:** `findings/REGISTER.json:C-09` (the unattributed remainder, with the named blocker)
· `findings/REGISTER.json:V-13` (pure layer split ≡ `-ot` split at matched division).
