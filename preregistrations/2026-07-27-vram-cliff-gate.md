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

---

## Scored (2026-07-27, log: `weights/data/prereg23_vram_cliff.log`)

**Verdict: P-1 HIT, P-2 HIT, P-3 HIT. And the shipped command was on the wrong side of the cliff.**

Rather than occupy VRAM externally (no torch, no cupy on this box), the same boundary was mapped
**from the inside** — sweeping `-ub`, which moves the compute buffer directly, on the split
placement with `-nkvo 1`: the exact configuration v1.14.x shipped as the prefill champion.

| `-ub` | CUDA compute buffer | pp2048 |
|---|---|---|
| 512 | 300.75 MiB | 303.32 ± 1.57 |
| 1024 | 601.50 MiB | **387.37 ± 1.68** |
| 1536 | 902.25 MiB | 381.21 ± 1.86 |
| 2048 | 1203.00 MiB | **209.64 ± 0.26** |
| 3072 | — | 209.46 ± 0.83 |
| 4096 | — | 210.84 ± 0.33 |

- **P-1 (≥30% collapse in one step): HIT.** −45.0% from `ub 1536` to `ub 2048`. Not a taper — a
  step, and the three points past it are flat within 0.7% of each other. The runtime does not fail
  or warn; it spills and holds the degraded speed forever after.
- **P-2 (the buffer explains it): HIT, and more exactly than staked.** The buffer is not "roughly
  linear" — it is linear to four figures: 0.5874 MiB per ubatch token at 1024, 1536 and 2048 alike.
  The cliff sits between 902 and 1203 MiB of buffer, i.e. the card had **~1 GB** of usable headroom
  behind a 4.4 GB split. Smooth demand, hard supply, discontinuous speed.
- **P-3 (buffer at ub 2048 ≥ 1.0 GB): HIT.** 1203.00 MiB. `UBATCH_HEADROOM_GB = 1.5` is the right
  order of magnitude, though the sweep shows it should have been a *sizing* input, not a yes/no
  threshold.

### The defect this found in shipped code

`MOE_FRONTIER` row 3 quoted **391.72 tok/s** for `-b 2048 -ub 2048 -nkvo 1`. That same command
measures **209.64** on the same machine. Both numbers are real; the difference is ~250 MiB of
desktop occupancy — one browser window — and #22 caught the flip (437 vs 193) without identifying
the axis. The re-measured safe point:

| | pp2048 | tg128 |
|---|---|---|
| shipped `-ub 2048` | 209.64 (or 391.72 on a clear card) | 16.54 |
| **corrected `-ub 1024`** | **386.14 ± 2.08** | **18.06 ± 0.09** |

The correction gives up 1.4% of the best-case prefill figure and **gains 9.2% of decode**, while
removing a 1.85× dependency on whether the user has a browser open.

### Why this generalises past one row

The error was not the number. It was **quoting a peak measured at the edge of a resource as if it
were a property of the configuration.** Every frontier figure in this project is measured on a
6 GB card, and any of them can sit on a cliff we did not sweep across. The tool's own defence is
the one the sweep demonstrates: *a value that is flat in its neighbourhood is a result; a value
with a 45% step next to it is a coincidence.* Neighbourhood-checking now belongs in the protocol,
not in the judgement of whoever ran the benchmark.

**Wired into:** `quantprobe/plan.py:safe_ubatch` · `quantprobe/plan.py:COMPUTE_BUFFER_MIB_PER_UB_TOKEN`
· `quantprobe/plan.py:MOE_FRONTIER` (row 3) · `tests/smoke.py:t_ubatch_is_sized_not_pinned`
· `tests/smoke.py:t_frontier_rows_are_off_the_cliff`
