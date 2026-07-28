# Kernel-level brief for external red-teaming (Grok / any reviewer)

**Date:** 2026-07-28 · **Author:** Federico Sciuca (quantprobe) · **Status:** all numbers measured
this session on one machine, same session, position-controlled where it matters.

**What changed since the last brief:** the project stopped measuring *through* llama.cpp. There is
now a standalone CUDA benchmark (`tools/kernelprobe/bench.cu`) with **zero llama.cpp and zero ggml**
— our own kernels, our own buffers, our own timing, every kernel correctness-checked against a
double-precision host reference. That finally separates "this card cannot" from "this runtime does
not", which every previous measurement in this project was structurally unable to do.

**Read this document adversarially.** Five things I asserted during this session were refuted by
controls I ran afterwards, and they are all recorded below as retractions rather than deleted.
The pattern is the point: assume the remaining claims have the same error rate.

---

## 1. Hardware, measured not spec

| | |
|---|---|
| GPU | NVIDIA GTX 1060 6GB, compute capability **6.1** (Pascal), 10 SMs |
| VRAM bandwidth, spec | 192.2 GB/s (4004 MHz × 192-bit) |
| VRAM bandwidth, **measured by our own streaming kernel** | **161.0 GB/s (0.84 of spec)** |
| VRAM bandwidth, measured by cuBLAS fp32 GEMV | 161.3 GB/s |
| `__dp4a` (4-way INT8 dot product) | available, sm_61 qualifies exactly |
| OS | Windows 10, WDDM |
| CUDA | 12.9 (13.x dropped Pascal; `nvcc --list-gpu-arch` starts at compute_75) |

**0.84, not 1.00, is the real ceiling.** Every efficiency below is quoted against spec 192.2 so it
can be compared to published numbers, but 0.84 is the number a kernel can actually reach.

---

## 2. The central result: the decode wall is UNPACK INSTRUCTION COST, not bandwidth

All rows below are the same harness, same 512 MB buffer, same access pattern, same reduction.
Only the weight format and the unpack instruction sequence change.

| level | bits/weight | GB/s | vs spec | vs stream | **GWeights/s** |
|---|---|---|---|---|---|
| L0 pure stream read | — | 161.0 | 0.84 | 1.00 | — |
| L1b fp16 matvec, **no unpack** | 16.0 | 152.5 | **0.79** | **0.95** | 76.2 |
| L1c int8 → `__dp4a` | 8.5 | 126.2 | 0.66 | 0.78 | 118.8 |
| L1 4.5-bit → float (naive shift/mask/convert) | 4.5 | 67.8 | 0.35 | 0.42 | 120.5 |
| L1d 4.5-bit → `__dp4a` | 4.5 | 128.7 | 0.67 | 0.80 | 228.8 |
| L1h 4.5-bit `__dp4a`, **warp-per-row** | 4.5 | **132.1** | **0.69** | 0.82 | **234.9** |
| L1e 2.5-bit sym → `__dp4a` | 2.5 | 119.6 | 0.62 | 0.74 | **382.7** |
| L1g Q2_K's exact cost model → `__dp4a` | 2.625 | 117.0 | 0.61 | 0.73 | **356.6** |

Two facts, both internally controlled:

1. **A matvec with no unpacking reaches 95% of the streaming ceiling.** Memory layout, shared
   memory staging, and the reduction are all essentially free. Everything below 0.79 is the unpack.
2. **Same buffer, same bytes, only the instruction changed: 67.8 → 128.7 GB/s = 1.90×.** Widening
   nibbles to float one at a time is the wall; Pascal's INT8 dot product is not.

**GWeights/s is the decision-relevant metric**, not GB/s — decode time for a given model is
weights/second. Note that GB/s *falls* as bits drop (progressively more ALU-bound) while GW/s
*rises*. Those are not in conflict; they are the same fact seen from two sides.

---

## 3. It transfers to real models, twice

Qwen2.5-7B, all-in-VRAM (`-ngl 99`), tg128, arms interleaved, position-controlled, same session.

| format | file | tok/s | effective GB/s | η | GW/s |
|---|---|---|---|---|---|
| Q4_K_M | 4.36 GiB | 22.72 | 106.4 | 0.553 | 173.1 |
| **Q4_0** | 4.12 GiB | **26.87** | **119.1** | **0.619** | **204.7** |
| Q2_K | 2.80 GiB | 21.67 | 65.4 | 0.340 | 165.1 |

- **Q4_0 beats Q4_K_M by +19.0%** where the byte difference explains only **+5.7%**. Effective
  bandwidth *rises* 106.4 → 119.1 GB/s, which a pure byte model forbids.
- **Q2_K is SLOWER in absolute tok/s than Q4_0 (21.67 vs 26.87) while being 32% smaller** — and
  lower quality. On this card Q2_K is strictly dominated whenever Q4_0 also fits.
- The byte model predicts Q2_K at 39.5 tok/s. It measures 21.67 — **45% short.**

### Consequence for the project's own physics model

Law 4 was `tok/s = η(tier) × BW / active-bytes-per-token`. **η is not a function of the tier alone;
it is a function of the FORMAT's unpack cost.** Same tier, same card, same model, same session:
η 0.553 vs 0.619 purely from how the weights are packed. This is the mechanism behind a pattern the
project had logged six times without being able to name ("a quantized byte is not a byte").

---

## 4. Where llama.cpp actually stands — better than expected

**llama.cpp is not the limitation.** It uses `__dp4a` in 61 call sites, MoE decode routes through
MMVQ, and its Pascal-specific tuning survived independent testing (§6).

| | GW/s | fraction of our best hand-written kernel |
|---|---|---|
| llama.cpp Q4_0, real 7B, **end-to-end** | 204.7 | 204.7 / 234.9 = **87%** |
| llama.cpp Q2_K, real 7B, **end-to-end** | 165.1 | 165.1 / 356.6 = **46%** |

Q4_0 end-to-end — including attention, KV, norms and sampling — reaches 87% of a pure matvec
microbenchmark. **There is essentially nothing to win at 4 bits.**

Q2_K reaches 46%, and the source explains it exactly: `vec_dot_q2_K_q8_1_impl_mmvq` issues
**8 `dp4a` per 16 weights where 4 suffice**, because its min term is computed as
`dp4a(m_broadcast, u[i])` = `m × sum(activations)` — a value that does not depend on the output row
— plus 4 lane-splat ops per group. Q4_0, being symmetric, has no min term and issues 4.

**But this is diagnosed and NOT claimable** — see §6. Do not file it upstream.

---

## 5. The bounded headroom, stated as a budget

At 4 bits, end-to-end, on this card:

```
llama.cpp Q4_0                119.1 GB/s   0.62
our best kernel (warp/row)    132.1 GB/s   0.69     <- +11%, and only +2.6% of that came from
                                                        removing the per-row block barrier
matvec with NO unpack         152.5 GB/s   0.79     <- +15% more, and NO known instruction
                                                        sequence recovers it
pure stream read              161.0 GB/s   0.84     <- +6% more, unreachable by any matvec
```

**Total remaining at 4 bits: 1.28× end-to-end, of which 1.11× is kernel engineering and 1.15× is
unpack cost nobody has shown how to remove.** Anyone claiming more than this on this hardware at
4 bits is claiming something these measurements contradict.

**The larger lever is bits, not the kernel:** 4.5-bit = 234.9 GW/s, 2.625-bit = 356.6 GW/s
(**1.52×**), 2.5-bit = 382.7 GW/s (**1.63×**). Weights/second is bought with bit-width, provided
the unpack stays dp4a-native.

---

## 6. Retractions — five claims I made this session and then refuted myself

Listed because the error rate is information about the remaining claims.

| # | claim I made | the control that killed it |
|---|---|---|
| 1 | Graph **fragmentation** explains the split placement's η 0.15 | Cut dispatches/token 30 → 1 (60 splits → 2). Gained **+6.5%**, η 0.154 → 0.164. Submission time collapsed 85% as predicted, but that term was only ~2 ms; device-busy barely moved. |
| 2 | "My hand-written kernel is 40% slower than llama.cpp's, so theirs is already good" | Compared a pure matvec against a whole-model decode. Not apples to apples; withdrawn. |
| 3 | **Q2_A**, a new dp4a-native asymmetric 2-bit format at measured Q2_K quality parity, is 1.70× faster | Built Q2_K's *exact cost model* in the same harness: **352.7 vs Q2_A's 282.0 GW/s — Q2_K's format is 25% FASTER than mine.** The nibble unpack I designed around costs ~2 ALU ops per 16 weights; my extra 0.5 bits/weight costs real bytes. **New-format project dropped.** |
| 4 | llama.cpp forfeits row reuse; giving asymmetric K-quants **multi-row blocking** would recover ~1.9× | Forced it on in-tree (validity-gated on the blocking actually changing, `rows/block 1→4`): **22.25 → 17.33 tok/s, 22% SLOWER.** And decisively: **Q4_0, at 87% of ceiling, also runs 1 row per block** — so blocking never distinguished the healthy format from the sick one. llama.cpp's `slow_pascal` carve-out is correct; upstream measured it right. |
| 5 | MoE **expert gather** costs a further ~1.8–1.9× | Swept the gather ratio: 1-in-2 → **1.00× penalty**, 1-in-4 → 1.04×, 1-in-8 → 1.18×, 1-in-16 → 1.97×. The penalty tracks **block count**, not scatter — my benchmark was starving a 10-SM card. A real MoE expert matmul launches thousands of blocks. **Scattered expert reads cost nothing.** |

Retraction #4 closes a loop worth stating plainly: the only way to remove Q2_K's extra `dp4a` is to
amortise a row-invariant term across rows, which requires multi-row blocking, which is measurably
worse on this hardware. **The two available fixes are mutually exclusive and llama.cpp already
chose the better one.** The 2.07× Q2_K headroom is a real *ceiling* and an *unclaimable* one.

---

## 7. What we want from you — open questions, in priority order

Constraints: consumer hardware, batch size 1 (single user, interactive), no quality loss beyond
what is stated, and answers must be **falsifiable on one GPU**.

**Q1. Where does GWeights/s peak as bits → 0?** Measured: 4.5-bit = 234.9, 2.625-bit = 356.6,
2.5-bit = 382.7 GW/s. The trend has not been pushed below 2.5 bits, and the ALU cost per weight is
roughly constant across these (4 dp4a per 16 weights) while bytes keep falling. Is there a 1.5-bit
or 1-bit dp4a-native layout where GW/s is still rising? Where does it turn over, and why?

**Q2. Is there an unpack sequence cheaper than shift/mask/`dp4a`?** We use
`(v >> 2j) & 0x03030303` per group. Pascal has `LOP3.LUT` (three-input logic in one instruction),
`__byte_perm`, and `PRMT`. Vectorised `uint4` loads (16 B/thread) are untried. The gap to close is
0.69 → 0.79 of spec, i.e. **1.15×** — is any of it reachable, or is 0.69 the instruction-count floor
for 4-bit?

**Q3. Why is the no-unpack matvec at 0.79 and not 0.84?** L1b does nothing but load fp16, convert,
FMA, reduce — and loses 5% against a pure streaming read. Is that the fp16→fp32 conversion, the
shared-memory activation reads, or the reduction? This 6% is the least-understood number here.

**Q4. Does any of this change on Ampere+?** The entire effect is that this card's ALU is weak
relative to its bandwidth. On a GPU with more ALU headroom the unpack should hide behind the memory
transfer and the format ranking may **invert** — Q4_K_M could beat Q4_0 again. We cannot test this;
one `llama-bench` run of Q4_0 vs Q4_K_M on a modern card would settle it and we would publish the
result either way, including if it contradicts us.

**Q5. Is there a fundamentally different structure we have not considered?** Everything above
assumes: read packed weights → unpack → dot with activations. Candidates we have *not* evaluated —
warp-level `__shfl` codebook lookup for non-uniform quantisation; activation-conditional weight
skipping; storing weights pre-permuted into dp4a lane order at quantisation time (we do this and it
helps, but only for the layouts we tried); persistent kernels; fusing the three MoE projections.

---

## 8. Reproduce

```bash
nvcc -O3 -arch=sm_61 -o kernelprobe bench.cu && ./kernelprobe 512 16   # the full ladder
python quality.py model-Q8_0.gguf                                      # format reconstruction RMSE
```

`tools/kernelprobe/bench.cu` — every kernel checked against a double-precision host reference
before its timing is reported; a mismatch prints `*** MISMATCH ***` and the number is void.
`tools/kernelprobe/quality.py` — reconstruction error of candidate formats vs Q2_K on real
dequantized weights.

**Run-to-run variance is ±10%** on this box (thermal). Only same-run comparisons are used for any
conclusion above; cross-run numbers are quoted for context only.

**Known limits of everything here:** one GPU, one OS, batch size 1, reconstruction RMSE used as a
proxy for quality (**no perplexity has been run**), and the Q4_0 test file was requantized *from*
Q4_K_M so its quality is strictly worse than a Q4_0 built from source weights — the +19% is a
**speed** result and nothing more.
