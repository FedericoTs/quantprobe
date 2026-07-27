# Pre-registration #23: the evict-KV advice sits on a VRAM cliff and must be gated

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## The defect

v1.14.0/v1.14.1 recommend `-nkvo 1` with `-ub 2048` for long-prompt workloads on MoE, quoting
**391.72 tok/s** prefill. Pre-registration #22 found that figure is conditional on the card being
otherwise clear:

| desktop VRAM held | pp2048 |
|---|---|
| 462–472 MiB | **437–438** (four runs, ±0.5%) |
| 713–714 MiB | **193–195** |

~250 MiB of occupancy flips it **2.3×**, to a value *worse than every other frontier point*. We
are telling long-prompt users to pick the configuration that is fastest on an idle card and worst
on a working one, with no warning.

## Mechanism, and why `-ub` maps the same boundary

Evicting KV frees VRAM, which lets the `-ub 2048` compute buffer grow to a size that only just
fits. Whether it fits depends on `model_VRAM + compute_buffer + desktop ≤ capacity`. Sweeping the
**ubatch** varies the compute buffer directly, so it finds the same edge from the inside — no
external VRAM allocator needed (none is installed on this box).

llama.cpp prints its own `compute buffer size` per backend, which converts the edge into the
number a gate can use.

## Stakes

`Qwen3-30B-A3B-Q2_K`, split placement, `-nkvo 1`, `llama-bench -p 2048 -n 0 -r 3`, `-ub` swept
512 → 4096. GPU state logged before and after.

- **P-1 (a cliff exists, not a slope).** Prefill rises with `-ub` up to some `ub*`, then **falls by
  ≥30% in a single step**. A smooth roll-off instead would mean #22's 2.3× was not a capacity
  boundary and the gate should be built on something else.
- **P-2 (the compute buffer explains it).** llama.cpp's reported CUDA compute buffer grows roughly
  linearly with `-ub`, and the step where prefill collapses is the step where
  `model_VRAM + compute_buffer` first exceeds **~5.4 GB** (6144 MiB minus the 0.8–1.5 GB desktop
  reserve measured in #13).
- **P-3 (the gate is computable).** From the sweep, the compute buffer at `ub 2048` is
  **≥ 1.0 GB** — large enough that `UBATCH_HEADROOM_GB = 1.5`, the constant already shipped in
  `ubatch_flags`, is the right order of magnitude rather than a guess.

## Refuted if

P-1 fails — no single-step collapse. Then #22's bimodality has another cause (driver fallback,
allocator fragmentation) and gating on free VRAM would be treating a symptom.

## What ships

`ubatch_flags` and the `-nkvo` advice gated on **measured free VRAM** rather than nominal
capacity, in the same shape as `DESKTOP_VRAM_RESERVE` for the expert split. If the headroom is not
there, the advice is withheld — never emitted with a caveat, since a user who follows advice they
were warned about is still worse off than one who was given the right advice.
