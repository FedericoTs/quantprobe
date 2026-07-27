# The upstream plan: ggml CPU executor — what we measured, what upstream knows, what we build

*2026-07-27. Sources: pre-registrations #27–#34, the E3 per-op profiler, and a three-agent
adversarially-verified survey of upstream (all claims checked against primary sources, 3/3
refuters returned refuted=false).*

## The measured state (this box, 4-core i5-7600K, Qwen3-30B-A3B Q2_K, pure CPU)

| build | tg32 | barrier ms/token |
|---|---|---|
| gcc + mingw libgomp (kernel-semaphore barriers) | 11.92 | 30.8 |
| shipped MSVC + libomp (spins 200 ms by default) | 13.28 | — |
| **gcc + `GGML_OPENMP=OFF` (ggml spin barrier)** | **16.64** | **10.8** |

~2,070 barriers/token (~1,640 inter-node + ~430 intra-op), counted from the graph-building code
and confirmed by the E3 profiler to 0.3% ledger balance. GEMV compute (57.7 ms) is at the memory
wall. Small-op compute is 1.6 ms. Everything else is synchronization and dispatch.

## What upstream already knows (verified survey)

- **Sync-scheme replacements are a graveyard.** slaren, PR #7993 (merged, the current design):
  "several PRs to change the synchronization mechanism in the past... always resulted in worse
  performance." Do NOT propose a new sync scheme.
- **The barrier primitive is already optimized** (#9598 fixed false sharing, +21–47% on ARM at
  high thread counts). **The barrier COUNT is unattacked.** Nobody upstream has a per-op
  compute/barrier ledger — our E3 data is novel motivation.
- **The threadpool rework (#8672) explicitly left per-node sync untouched**; a "Threadpool-V3"
  was deferred.
- **Adjacent in-flight work:** PR #25048 (atomic work-stealing for mul_mat chunks, +14% on a
  hybrid i3) reduces barrier WAIT, not barrier count — cite, don't duplicate. Our E2c null is
  evidence the two mechanisms are independent on non-hybrid cores.
- **The maintainers' own stated direction is fusion**: the `ggml_cpu_try_fuse_ops` hook exists
  (currently only RMS_NORM+MUL on CPU), its TODO invites planning-time fusion, and CUDA/Vulkan
  already carry merged `topk-moe` and norm-chain fusions that stacked to +27% on gpt-oss-20b
  (discussion #17621). Test coverage for these patterns exists in `test-backend-ops`.
- **Industry consensus** (ORT/TVM/IREE/ExecuTorch/PyTorch): nobody runs lockstep
  all-threads-per-graph with per-node team barriers; tiny ops run inline on one thread; chains
  get fused. A data-dependent serial chain is irreducible by scheduling — fusion is the only
  lever on our ~19 ms serial line.

## The deliverables, in order

1. **Build guidance (DONE measuring — write the upstream issue).** `GGML_OPENMP=ON` on
   Windows/mingw routes every barrier through a kernel semaphore (symbol-level: `sem_wait` /
   `ReleaseSemaphore`, no spin phase in this libgomp port). One flag recovers +40%. Attach the
   #34 A/B and the symbol evidence. Cheapest deliverable, reaches every gcc-on-Windows builder.
2. **CPU `topk-moe` fusion** — port the CUDA/Vulkan-validated pattern
   (soft_max → argsort_top_k → get_rows → sum_rows → clamp → div, single-threaded over 128
   floats) into `ggml_cpu_try_fuse_ops` via `ggml_can_fuse_subgraph`, exactly as `topk-moe.cu`
   does. Removes ~5 nodes + barriers per layer. Merged precedent on two backends, tests exist.
3. **N-ary weighted expert-sum** — replace MUL + 7 chained 1-row ADDs (the 22×-overhead
   emblem) with one node. Removes ~7 nodes + barriers per layer.
4. **Plan-time `needs_barrier` bitmap** — skip the inter-node barrier when the next node is
   provably dependency-free and neither touches shared wdata; all matmuls stay barrier points
   in v1. Only after 2–3 land; reviewers treat this code as correctness-fragile (three merged
   race/ordering fixes in its history) and will review hard.

Staked expectation for 2+3 combined, before building: they touch ~400 of the ~1,640 inter-node
barriers plus the serial dispatch around them — on the spin-barrier build that is worth
**+5–12%**, on top of #34's +40%. If measurement lands under +3%, fusion on CPU at batch-1 is
not worth upstream's review bandwidth and we stop at deliverable #1.

## What we will NOT do, and why (measured, not opinion)

- No sync-scheme redesign (upstream graveyard, #7993).
- No fork (D-05: distribution + maintenance; every other component at the wall).
- No expert-major dispatch (E2c: measured null).
- No scheduling flags (six arms, all flat, #31).
