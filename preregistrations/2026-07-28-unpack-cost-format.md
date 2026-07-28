# Pre-registration #52: the decode wall on a weak GPU is UNPACK COST, not bytes — so a cheaper format should win

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurement. **Status: STAKED.**

## What the bare-metal probe established (no llama.cpp in the loop)

`tools/kernelprobe/bench.cu` — our own CUDA kernels, our own buffers, correctness-checked:

| level | GB/s | vs spec | GW/s |
|---|---|---|---|
| L0 pure stream read | 160.0 | 0.83 | — |
| L1b fp16 matvec, **no unpack** | 151.3 | 0.79 | 75.7 |
| L1 4.5-bit → float (naive) | 68.2 | 0.35 | 121.2 |
| **L1d 4.5-bit → `__dp4a`** | **128.3** | **0.67** | **228.1** |
| L1c int8 → `__dp4a` | 124.1 | 0.65 | 116.8 |

Two facts, both internally controlled:

1. **A matvec with no unpacking runs at 95% of the streaming ceiling.** Memory layout, shared
   memory, and the reduction are all free. Everything below 0.79 is the unpack.
2. **Same bytes, same format, same buffer — only the instruction changed — is 1.88×**
   (68.2 → 128.3, rel.err 1.3e-08). Widening nibbles to float one at a time is the wall;
   Pascal's INT8 dot-product is not.

llama.cpp already uses `__dp4a` (61 call sites, and MoE decode routes through MMVQ), so it is *not*
missing the instruction — my earlier reading of its ~98 GB/s as "leaving 1.3× on the table" was
wrong and is corrected here. The remaining difference is **what else the format makes you unpack**:
my probe format carries plain fp16 scales per 32 weights; K-quants pack **6-bit scales AND 6-bit
mins** that must be decoded per block before any dot product happens.

## The claim this makes testable on a real model

llama.cpp already ships a format shaped like the probe's: **Q4_0** — one fp16 scale per 32 weights,
symmetric, no min. Against **Q4_K_M** — packed 6-bit scales and mins.

| | bits/weight | per-block unpack |
|---|---|---|
| Q4_0 | 4.50 | 1 fp16 scale |
| Q4_K_M | ~4.83 | 6-bit scale + 6-bit min, both bit-packed |

Q4_K_M reads **7% more bytes**. If Law 4's byte model were the whole story, Q4_0 should be ~7%
faster and no more. If unpack cost is a first-class term, the gap should be much larger.

## Stakes — Qwen2.5-7B, all-in-VRAM (the regime with no CPU tier to confound), tg, position-controlled

- **P-1 (THE CLAIM).** Q4_0 decodes **≥ 15%** faster than Q4_K_M — more than double what the byte
  difference alone can explain.
- **P-2 (it is unpack, not bytes).** The speedup exceeds the byte ratio (1.07×) by at least 2×,
  i.e. effective GB/s *rises* for Q4_0 rather than staying flat.
- **P-3 (scope honesty).** The effect is a property of this ALU-weak card, so I record it as
  hardware-conditional and do **not** generalize it to Ampere+ without a second card.

## KILL RULE

**If P-1 fails — Q4_0 is within 7% of Q4_K_M, i.e. explained by bytes alone — then unpack cost does
not survive contact with a real model**, the microbenchmark's 1.88× does not transfer through a
whole decode (where attention, KV, norms and sampling dilute it), and Law 4's pure byte model stands
unamended. I will say that plainly rather than keep the microbenchmark result as if it were a
system-level one.

## Why this matters beyond one flag

Universal community advice is "K-quants are strictly better than legacy quants at equal size."
If P-1 holds, that advice is **wrong on ALU-weak hardware for speed**, and the tool gains a real
lever: on pre-Ampere GPUs, prefer the cheap-unpack format. It also predicts the reverse of the usual
intuition about going *smaller*: below 4 bits the unpack gets more expensive while the bytes saved
shrink, so Q2_K should be **worse per weight** than Q4_0 — the opposite of what bytes predict.

**Wired into:** pending P-1.
