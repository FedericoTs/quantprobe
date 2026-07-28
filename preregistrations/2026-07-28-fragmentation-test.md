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

---

## Scored (2026-07-28, logs: `weights/data/prereg51_fragmentation.log`, `prereg51_e8_orphans.log`)

**Verdict: P-2 MISS. THE KILL RULE FIRES — fragmentation is refuted, and it is refuted at the
strongest possible contrast rather than the weak one this design originally produced.**

### P-1 failed first, and its failure was the discovery

`-ngl 20` does **not** produce one boundary. Measured dispatches per token (divisor 129 confirmed
three ways: 4257/33, 3870/30, 4902/38 are all exact integers):

| arm | GPU layers | dispatches/token |
|---|---|---|
| A `-ot` | 48 attn + 16 expert | 33.0 |
| B `-ngl 20` | 20 | 30.0 |
| C `-ngl 12` | 12 | **38.0** |

Fewer GPU layers produced *more* dispatches. So I instrumented the split map (E7,
`GGML_SPLIT_MAP=1`) and found the cause:

```
29 x CUDA split of exactly 2 nodes   (MUL Qcur_normed-0 .. -28)   <- one per CPU-resident layer
 1 x CUDA split of 1219 nodes        (the actual 20 offloaded layers)
30 x CPU  split of ~61 nodes
```

**llama.cpp bounces a 2-node fragment onto the GPU inside every CPU-resident layer**, each paying a
host->device copy of two inputs and a device->host copy of the result for a few KB of work. Cause is
in `ggml-backend.cpp` pass 3: unassigned nodes pick `n_supported > n_supported_best` scanning
backends from highest priority, so a node whose inputs all live on the CPU still lands on CUDA0 when
the counts tie.

### The contrast P-2 actually needed

E8 (`GGML_SCHED_MIN_GPU_RUN=N`, `ggml-backend.cpp` pass 3.5) flips runs of fewer than N consecutive
GPU nodes back to the CPU. At N=8 it moved 58 nodes and collapsed the graph **from 60 splits to 2**.

| arm | dispatches/token | device-busy ms/tok | submission ms/tok | tok/s |
|---|---|---|---|---|
| B `-ngl 20` | 30.0 | 20.42 | 1.44 | 15.07, 14.93 |
| **B + E8** | **1.0** | **19.14** | **0.213** | **15.80, 16.16** |
| A `-ot` | 33.0 | 24.84 | 2.90 | 16.82 |
| A + E8 | 33.0 (0 flips) | 22.58 | 2.87 | 16.85 |

- **P-1 MISS** — 30, not <=8.
- **P-2 MISS, decisive.** A **30x** reduction in dispatch count moved device efficiency by **6.3%**
  (20.42 -> 19.14 ms). eta 0.154 -> 0.164 (bytes 0.507 GB; the conclusion is unchanged at any
  plausible byte assumption, e.g. 0.402 GB excluding the output head gives 0.122 -> 0.130).
- **P-3 HIT** — B stayed within 6% of A end-to-end.
- **P-4 MISS** — C's eta fell below B's; per-dispatch cost does not dominate.

**Submission/wait collapsed 85%** (1.44 -> 0.213 ms/token) exactly as fragmentation predicts, and
that term was only ever ~2 ms. Device-busy, which is 23.9 of the token's 61 ms, barely moved. The
mechanism is real but an order of magnitude too small to be the answer.

### What E8 is worth on its own, honestly bounded

**+6.5% end-to-end on `-ngl N` partial offload** (paired adjacent runs: +4.8%, +8.2%), free, and it
does not fire at all on the `-ot` placement (0 flips) — so **it does not change this project's
shipped advice**, where `-ot` (16.85) still beats the fixed layer split (15.98).

**Output is NOT byte-identical.** Moving a node from CUDA to CPU changes float rounding order and
at temp 0 flips a near-tie argmax; the text diverges from the third sentence. The necessary control
says this is llama.cpp's own behaviour class, not a defect in the pass: stock llama.cpp with E8 OFF
also produces different text for `-ngl 20` vs `-ngl 19`, while being exactly reproducible run to
run. Disclosed rather than claimed as free.

### Where this leaves the gap — and why the next move is not in llama.cpp

Three consecutive llama.cpp-level interventions have now been measured end to end:

| intervention | effect |
|---|---|
| CUDA graphs forced on (#48) | +3.2% |
| `MUL_MAT_ID` stream sync (#48) | refuted at source, 0% |
| **maximal defragmentation, 30 dispatches -> 1 (this)** | **+6.5%** |

E6 says **23.88 ms of the 61 ms token is device-busy** — the GPU actually executing. No scheduling
change can touch device-busy time. The remaining 5.5x between the byte model (4.34 ms) and measured
device execution lives **inside the quantized CUDA kernels**, and that is now the only place left to
look. Ninth mechanism refuted.

**Wired into:** `findings/REGISTER.json:D-16` (fragmentation refuted at 30x contrast) ·
`V-15` (E8 scheduler orphan fix, +6.5% on `-ngl` offload, numerics-changing, not upstreamed) ·
`C-02` (narrowed to kernel efficiency — the next test must bypass llama.cpp entirely).
