# Prereg #85 — the launch cost, measured without the ruler that broke #83

**Staked:** 2026-07-30, in `tools/kernelprobe/launch.cu` source comments, before the
binary was compiled. **Scored:** same day, same session, GPU idle at 795 MiB / 139 MHz,
no other CUDA process.

## Why

Prereg #83 measured per-op GPU time from inside llama.cpp using an instrumented build
that records **a CUDA event pair around every op**. U-33 then measured that harness at
**3.80–8.32 µs per op call** by diffing its `tg32` against the clean ladder — the same
order as the 5–16 µs op times it was reporting. You cannot measure a 6 µs cost with a
6 µs ruler. L-21's headline (non-matmul overhead 8.4 %–27.0 %) rests on that ruler.

This probe inverts the geometry: **one event pair around N launches**. The harness
contributes exactly two events per measurement regardless of N.

## Staked predictions and kill rules

| # | prediction | kill rule | result |
|---|---|---|---|
| **P1** | marginal launch cost < **6.0 µs**, i.e. materially below #83's 7.72–15.51 µs/call for RMS_NORM | ≥ 6.0 µs ⇒ #83's magnitude was not inflation, L-21 stands | **HELD — 3.742 µs** |
| **P2** | per-launch cost independent of work across 896→28672 floats, spread ≤ 1.5× | > 2.0× ⇒ work-bound, D-24's premise wrong | **MISS — 2.31×** |
| **P3** | a dependent chain costs ≥ **1.3×** an independent burst | ≤ 1.0× ⇒ serialization is free, D-25's per-layer framing wrong | **MISS — 1.00×** |

## Measured

```
arm A  empty kernel, pure dispatch floor          2.10 µs/launch (flat, N=100..10000)
arm B  rmsnorm, independent, vs vector length     896f 3.245 → 28672f 7.035  (2.31×)
arm C  rmsnorm n=3584, burst 1→1024               4.27 → 3.77, marginal 3.742 µs
arm D  dependent chain (reads previous output)    3.740 µs — identical to independent
```

## What this settles

**1. #83's per-op magnitudes were roughly 2–4× inflated.** A real norm-shaped kernel at
Qwen2.5-7B's hidden size costs **3.74 µs**, not the 15.51 µs #83 reported. The
instrumentation was most of what #83 measured. L-21's 8.4 %–27.0 % must come down.

Re-pricing the launch-bound bucket at the measured 3.7 µs:

| model | launch-bound calls/token | true launch cost | share of token | #83 claimed |
|---|---|---|---|---|
| Qwen2.5-0.5B | 124 | 0.46 ms | **7.0 %** | 27.0 % |
| Qwen2.5-7B | 144 | 0.53 ms | **1.2 %** | 8.4 % |
| gemma4-12B | 581 | 2.15 ms | **2.7 %** | 14.2 % |

**2. This kills the overhead framing outright, and that is the useful part.** D-24 fitted
the op-count term at **72.9 µs per op**. The physical launch cost is **3.7 µs** — a
**20× discrepancy**. A term fitted at twenty times the cost of the thing it claims to
price is not pricing that thing; it is absorbing something else. The residual Law 4
leaves on the table is **not launch overhead**. It is bandwidth.

**3. P3's miss retracts D-25's stated mechanism.** D-25 explained #83's anti-correlation
(µs/call falling as ops-per-layer rises) as back-to-back kernels *pipelining* and sharing
launch latency. Arm C shows burst length does not reduce per-launch cost (3.64–4.42 µs
flat from N=1 to N=1024) and arm D shows a data dependency costs nothing extra — so there
is no overlap to share. The observation stands; my explanation was wrong. The likeliest
remaining cause is **gap attribution inside the profiler**: an isolated op absorbs more of
the inter-kernel gap than a dense run of ops does. That is a property of the ruler, not
the hardware — which is the same conclusion P1 reached from the other direction.

**4. P2's miss is partly my own spec error.** I wrote "896 → 28672 floats, the hidden
sizes on our ladder". 28672 is not a hidden size, it is an **FFN width**; no RMS_NORM on
this ladder is that wide. Over the range norms actually run at (896–3840) the spread is
**1.24×**, inside P2's 1.5 % band. Scored as staked it is a miss, and the arm is still
informative: there is a real work component (~2.1 µs dispatch floor + work on top), it is
just small over the widths that matter.

## Where this leaves the next step

Op count is dead as a pricing axis (D-24, now with a physical reason rather than only a
cross-validation result). **U-32 — tensor shape via the L-20 knee curve — is the only
surviving candidate**, and it now carries the whole burden: r = +0.769 with zero fitted
parameters, kill rule already staked at LOO median < 8.7 % and max < 18.6 %.

## Reproduce

```
nvcc -O3 -arch=sm_61 tools/kernelprobe/launch.cu -o launch.exe && ./launch.exe
```
