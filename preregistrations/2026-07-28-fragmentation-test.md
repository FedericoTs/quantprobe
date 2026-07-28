# Pre-registration #51: is the split's GPU inefficiency caused by graph FRAGMENTATION?

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurement. **Status: STAKED.**

## The hypothesis and why it is not yet a conclusion

#50 measured the split placement's GPU share at **η 0.15** (23.88 ms device-busy for 4.34 ms of
byte-model work), against the same runtime's all-in-VRAM **η 0.51** and cuBLAS's **η 0.84**. The
same measurement showed **33 `ggml_backend_cuda_graph_compute` calls per token** — the graph is cut
into 33 subgraphs at CPU/GPU boundaries, each carrying ~21 MB and ~0.72 ms. A GPU needs large
contiguous streams to reach 161 GB/s; 21 MB segments may not qualify.

**The tension that stops this being a conclusion:** #43 measured the pure layer split (`-ngl 20`,
ONE boundary) at the same *end-to-end* speed as the 32-boundary `-ot` split. If fragmentation cost
what this hypothesis claims, one boundary should have been faster.

**The resolution the hypothesis predicts:** `-ngl 20` also moves 28 layers of *attention* onto the
CPU, so it does strictly more CPU work. Better GPU η and worse CPU load can cancel end-to-end.
E6 can now measure η directly rather than inferring it from wall-clock — which is exactly the
measurement #43 could not make.

## Arms — same model, tg128, E3+E6 both on, one session

| arm | placement | GPU-resident active bytes | expected calls/token |
|---|---|---|---|
| A | `-ngl 99 -ot` experts 16..47 → CPU | 0.700 GB | 33 (measured) |
| B | `-ngl 20` pure layer split | 20/48 × 1.217 = 0.507 GB | few |
| C | `-ngl 12` pure layer split | 12/48 × 1.217 = 0.304 GB | few |

η is computed per arm as `GPU_bytes / device_busy_time`, using E6's device-busy — not wall-clock,
and not end-to-end tok/s.

## Stakes

- **P-1 (fragmentation is real and structural).** Arm B has **≤ 8** `graph_compute` calls/token
  versus arm A's 33. This is bookkeeping, not physics, but if it fails the whole framing is wrong
  because the placements are not doing what I think they are.
- **P-2 (THE CLAIM).** Arm B's GPU **η is ≥ 1.8×** arm A's 0.15 — i.e. ≥ 0.27. Fewer, larger
  subgraphs must convert into better device efficiency, or fragmentation is not the mechanism.
- **P-3 (the #43 tension resolves as predicted).** Arm B's end-to-end tok/s stays within **±10%**
  of arm A's despite the better η, because its CPU-side work grows. If B is instead much *faster*
  end-to-end, #43's null was wrong and something else changed.
- **P-4 (monotone).** Arm C's η ≥ arm B's η. Fewer GPU bytes in the same small number of subgraphs
  should not get *less* efficient; if it does, per-call fixed cost dominates and the story is
  "cost per call", not "bytes per call".

## KILL RULE

**If P-2 fails — fewer subgraphs do not improve η — fragmentation is refuted** and the split's
η 0.15 must be explained by something intrinsic to the kernels themselves (dequant cost at small
occupancy, MMVQ tile shapes on Pascal, MUL_MAT_ID's gather). That would make the target a kernel
rewrite rather than a scheduling change, and I will say so rather than keep the hypothesis alive.

## What ships

Nothing to `plan.py`. This is diagnosis: it decides whether the ~24 ms of GPU device-busy is
attackable by **restructuring the graph** (a scheduling change, plausibly upstream-shaped) or only
by **rewriting kernels** (expensive, and previously priced as not worth it).

Honest expectation recorded before measuring: I expect P-1 and P-3 to hit and **P-2 to be the coin
flip**. If η is flat across wildly different subgraph counts, that is the more informative outcome.
