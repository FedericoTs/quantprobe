# Pre-registration #53: does going to 2 bits in llama.cpp actually buy weights-per-second?

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurement. **Status: STAKED.**

## Why ask this before writing a new quant type

The bare-metal probe (`tools/kernelprobe/bench.cu`) now measures a full unpack ladder on this card,
all correctness-checked:

| format | bits/w | GB/s | GWeights/s |
|---|---|---|---|
| fp16, no unpack | 16.0 | 150.6 | 75.3 |
| int8 `__dp4a` | 8.5 | 127.2 | 119.7 |
| 4.5-bit naive→float | 4.5 | 69.8 | 124.1 |
| 4.5-bit `__dp4a` | 4.5 | 130.1 | 231.2 |
| **2.5-bit `__dp4a`, pre-permuted** | **2.5** | 107.0 | **342.4** |

GB/s *falls* going 4.5→2.5 bit (more ALU-bound per byte) but **weights/second rises 1.48×**.
So a dp4a-native 2-bit format is worth building — *if* llama.cpp's existing 2-bit format is not
already getting that. Q2_K is 2.625 bits/weight, close to our 2.5, so this is nearly a like-for-like
comparison and it costs one bench run instead of a new quant type.

Q2_K's unpack is *not* like our format's: per 16 weights it decodes a packed 4-bit scale **and** a
4-bit min from a shared byte, on top of the 2-bit extraction. Ours reads one fp16 scale per 32 and
has its offset term hoisted out of the row loop entirely.

## The arithmetic that makes this falsifiable

Qwen2.5-7B, all-in-VRAM, same session. Already measured in #52: **Q4_0 = 27.03 tok/s at 4.424 GB.**
Q2_K of the same model is **3.016 GB**.

- **Pure byte model (Law 4 unamended)** predicts Q2_K at 27.03 × 4.424/3.016 = **39.6 tok/s**.
- **Unpack-cost model (#52's amendment)** predicts substantially less, because Q2_K spends more ALU
  per weight than Q4_0 does.

In weights/second (params × tok/s, 7.62 B params) the byte model predicts Q2_K at 302 GW/s against
Q4_0's 206 GW/s.

## Stakes

- **P-1 (THE CLAIM).** Q2_K lands **below 34 tok/s** — i.e. at least 15% short of what its byte
  count alone predicts. The unpack tax is real and it grows as bits shrink.
- **P-2 (the ordering that justifies a new format).** Q2_K's **effective GB/s is LOWER than Q4_0's**
  119.6 GB/s. Fewer bits bought at higher ALU cost must show up as slower bytes, exactly as the
  probe's 130.1 → 107.0 does.
- **P-3 (the prize is quantified, not assumed).** If P-1 and P-2 hold, the gap between Q2_K's
  measured GW/s and the probe's 342.4 GW/s ceiling is the headroom a dp4a-native 2-bit format
  would target. I record that number rather than claiming it is all recoverable.

## KILL RULE

**If P-1 fails — Q2_K reaches ~39.6 tok/s, i.e. the byte model is right at 2 bits — then the unpack
tax does NOT grow as bits shrink**, #52's amendment is bounded to the 4-bit K-quant case, and
building a custom 2-bit format is not worth it because llama.cpp's Q2_K is already at the byte
ceiling. I will drop the new-format project rather than build it on a microbenchmark alone.

## What this does NOT decide

Quality. Our 2.5-bit probe format is **symmetric with one fp16 scale per 32 weights**; Q2_K is
asymmetric with a scale *and* a min per 16. On quality Q2_K should win clearly, and a speed victory
for a format nobody can use is worth nothing. Quality is a separate, later measurement and no claim
is made here.

**Wired into:** pending P-1.
