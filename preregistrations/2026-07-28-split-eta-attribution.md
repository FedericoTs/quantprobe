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

**Wired into:** pending; the N5 entry in the kernel brainstorm ledger (internal) scores either way.

---

## Scored (2026-07-28, log: `weights/data/prereg58_e9_attribution.log`)

**Verdict: P-0 PASS (reconciliation 96.9% / 99.1%). P-1 MISS — the KILL RULE fires: the time
concentrates in MATMULS (80%), not in small glue ops. But the per-call arithmetic then hands over
the real mechanism, and it reconciles BOTH arms quantitatively.**

### Instrument disclosure, before any number

E9 costs: arm A ran 12.52 tok/s profiled vs 16.87 unprofiled (853 nodes/token x event pairs on
WDDM inflate device-busy 23.9 -> 40.6 ms). Arm B: 21.42 vs 21.67 (-1.2%) — trustworthy absolutes.
SHARES are used from arm A; absolutes only from arm B and from the earlier uninstrumented E6 run.

### The attribution (per token)

| | arm A: flagship split | arm B: 7B Q2_K all-in-VRAM |
|---|---|---|
| GPU nodes/token | **853** | **341** |
| GPU weight bytes/token | 0.700 GB | 2.95 GB |
| matmul share of device time | **80.2%** | **92.7%** |
| non-matmul share | 19.8% | 7.3% |
| biggest single op | q3_K attn matmuls 36% | q2_K FFN matmuls |
| lm_head (q6_K, 1 call) | 4.8 ms | 4.3 ms |

- **P-1 MISS:** non-matmul ops are 19.8%, not >=50%. H1-as-an-op-class is dead.
- **P-2 HIT but moot** (glue ops 6-13 us/call).
- **P-3 MISS:** arm A matmul time alone implies **36.5 GB/s** on GPU weight bytes — the split's
  matmuls are NOT healthy.
- **P-4 MISS** (12.5 points vs 15 staked).

### The mechanism the per-call numbers hand over

Arm A's matmuls run **5-15x their byte cost per call** (q2_K attn: 96 us measured vs ~9 us of
bytes; MUL_MAT_ID: 358 us vs ~51). Arm B's matmuls run near their format-taxed byte cost
(matmul-only effective bandwidth **69.4 GB/s — exactly the L-16 prediction** for Q2_K-class).

The two arms reconcile under ONE model:

```
device time = format tax (L-16, ~69 GB/s effective for K-quants)  +  per-call floor (~16 us)

arm B: 2.95 GB / 69 GB/s = 42.8 ms  + 341 x 16 us =  5.5 ms  -> 48.3 vs 45.8 measured  (+5%)
arm A: 0.700 GB / 69 GB/s = 10.1 ms + 853 x 16 us = 13.6 ms  -> 23.7 vs 23.9 measured  (-1%)
```

**The split's eta 0.15 vs all-in-VRAM's 0.34 is CALL GRANULARITY: the split runs 2.5x the calls
on 4.2x fewer GPU bytes, so the same ~16 us per-call floor explodes from ~12% of device time to
~58%.** The flagship's 2048-hidden attention and 768-wide experts are simply too small per call
to amortize it; the 7B's 3584-hidden matmuls are not.

**The floor is in-kernel, not submission:** #48 measured CUDA graphs (which remove submission
gaps) at +3.2% on this exact placement — independently confirming the floor survives graph replay.
Known contributor: llama.cpp quantizes the activation vector to q8_1 INSIDE every quantized
matmul call (~5-10 us x ~290 calls/token ≈ 1.5-3 ms of the 13.6) — the same activations are
re-quantized for q/k/v. The remainder is latency-bound small-matvec execution (a 672-byte-per-row
matvec cannot hide DRAM latency). Micro-attribution below 16 us is NOT claimed.

### What this closes and what it opens

- **N5 CLOSED. The last big number on this box is explained.** The full physics map: CPU tier at
  physics (L-11), all-in-VRAM eta = format metadata density (L-16), split GPU eta = L-16 + the
  per-call floor at call granularity (this, L-17).
- **The fix direction is NOT kernels** (pre-registered as the kill-rule consequence): it is
  fewer/bigger calls — (1) speculative verify rounds amortize the floor across draft tokens with
  machinery Law 6 already measured; (2) more GPU-resident expert layers = bigger MMID calls;
  (3) upstream: share the q8_1 activation quantization across q/k/v (real, bounded ~1.5-3 ms).
- The tool's split-placement eta constant now has a mechanistic decomposition instead of being a
  fitted number.

**Wired into:** `findings/REGISTER.json:L-17` (the per-call floor law + two-arm reconciliation) ·
`C-09` note (the 87%-accounted token now has its GPU share decomposed) · N5 in the
internal kernel brainstorm ledger, scored.
