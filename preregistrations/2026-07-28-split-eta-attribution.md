# Pre-registration #58: WHERE does the split placement's GPU device-busy time actually go?

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurement. **STAKED.**

## The number this attacks — the last big one

#50 (E6, CUDA events): the MoE flagship split placement keeps the GPU **device-busy 23.88 ms/token**
for work the byte model prices at 4.34 ms — effective 29.3 GB/s, **η 0.15**. Yet the same runtime
does all-in-VRAM Q2_K at η 0.34 (#53). The factor **~2.3** between those is the largest unexplained
number left on this box, now that C-02's all-in-VRAM band is closed (L-16). Everything else this
session was kernel-scale; this is the flagship's actual wall.

## Hypotheses, stated with their arithmetic

**H1 (staked as the claim): the small-op latency floor.** The split graph runs ~2000+ GPU nodes
per token (48 layers of attention + router/gather/norm/rope/softmax chains; the E7 map showed
~60 nodes/layer). At batch 1 on a 10-SM WDDM card, a tiny kernel has a ~4–10 µs floor regardless
of its bytes. 1800 small nodes × ~7 µs ≈ 13 ms — most of the 23.88 ms, without any kernel being
"slow". All-in-VRAM 7B is healthier because it has fewer layers, fewer boundary ops, and
matmul-dominated time.

**H2 (alternative): the matmuls themselves are sick in the split config** (expert-shaped GEMVs at
768×2048 with poor occupancy, or MUL_MAT_ID overhead) — then the time concentrates IN matmul ops
and their implied bandwidth is far below the format's ceiling.

**H3 (minor): boundary transfers** — cudaMemcpy device time for the 33 CPU↔GPU crossings.
Predicted small (~1 ms) since the tensors are KB-scale.

## Method — E9: per-op CUDA event attribution

Extend the E6 instrument: `GGML_GPU_PROFILE_OPS=1` records an event pair around EVERY node the
CUDA backend executes, bucketed by (ggml op, src0 quant type, batch-1 vs not), accumulated and
printed at exit. Overhead is measured (E6 totals with the flag on vs off) and disclosed; shares
remain valid even if totals inflate.

Runs, same session: (A) MoE flagship, split `-ot` placement — the target; (B) Qwen2.5-7B Q2_K
all-in-VRAM — the healthy-comparison arm.

## Stakes

- **P-0 (validity).** E9's per-op sum reconciles with E6's device-busy total within **±15%** on
  arm A. If the instrument cannot reconcile with the instrument it extends, nothing is read.
- **P-1 (THE CLAIM, H1).** Non-matmul ops (everything except MUL_MAT / MUL_MAT_ID) account for
  **≥ 50%** of arm A's device-busy time.
- **P-2 (the floor is latency, not bytes).** The mean device time of non-matmul nodes is
  **≥ 4 µs** — a launch/latency floor, an order of magnitude above their byte cost.
- **P-3 (the matmuls are healthy).** Arm A's MUL_MAT(+_ID) device time alone implies an effective
  bandwidth **≥ 55 GB/s** on the GPU-resident weight bytes (η ≥ 0.29 vs spec) — i.e. within the
  all-in-VRAM Q2_K band, meaning no separate "split matmul sickness" exists.
- **P-4 (it explains the contrast).** Arm B's non-matmul share is **at least 15 points lower**
  than arm A's, accounting for the η 0.34 vs 0.15 ordering.

## KILL RULE

**If P-1 fails and the time concentrates in matmuls (H2), the small-op story dies** and the split's
deficit is a matmul-shape/occupancy problem — a much harder fix (kernel work at expert shapes),
and I will say the 2.3× is kernel-bound after all, reversing my current lean in public.

**If P-1 holds**, the fix direction is NOT faster kernels but FEWER/FUSED small ops or bigger
effective batches (speculative verify rounds amortize the floor across draft tokens — connecting
directly to the existing Law 6 speculation machinery), and the tool's split-placement η constant
gains a mechanistic justification instead of being a fitted number.

**Wired into:** pending; the N5 entry in `KERNEL_BREAKTHROUGH_BRAINSTORM.md` scores either way.
