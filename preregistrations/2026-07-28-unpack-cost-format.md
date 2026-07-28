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

---

## Scored (2026-07-28, log: `weights/data/prereg52_format_unpack.log`)

**Verdict: P-1 HIT, P-2 HIT. Unpack cost survives contact with a real model, and it is a
first-class term — not a rounding correction on the byte model.**

Qwen2.5-7B, all-in-VRAM (`-ngl 99`), tg128, r=2, arms interleaved for position control:

| format | file | tg128 run 1 | run 2 | mean | bytes/token | effective GB/s | eta |
|---|---|---|---|---|---|---|---|
| Q4_K_M | 4.36 GiB | 22.78 ± 0.01 | 22.66 ± 0.03 | 22.72 | 4.681 GB | 106.4 | 0.553 |
| **Q4_0** | 4.12 GiB | 26.97 ± 0.18 | 27.09 ± 0.00 | **27.03** | 4.424 GB | **119.6** | **0.622** |

- **P-1 HIT.** +19.0%, against a 15% stake.
- **P-2 HIT.** Bytes alone predict +5.7% (the file-size ratio). Measured is **3.3x that**, and the
  decisive form of the statement is that **effective bandwidth rises 106.4 -> 119.6 GB/s (+12.5%)**.
  A pure byte model requires effective bandwidth to stay flat. It does not.
- **P-3 honoured.** Scoped to this ALU-weak pre-Ampere card. Not generalized to Ampere+, where the
  unpack has more headroom to hide, and explicitly flagged as needing a second card.

### The amendment this forces on Law 4

Law 4 is `tok/s = eta(tier) x BW / active-bytes-per-token`. The measurement says **eta is not a
function of the tier alone — it is a function of the FORMAT's unpack cost.** Same tier
(all-in-VRAM), same card, same model, same session: eta 0.553 vs 0.622 purely from how the weights
are packed. This is the mechanism behind C-05 ("a quantized byte is not a byte"), which this project
has logged six times as a pattern without ever being able to name the cause. The cause is that
K-quants decode a 6-bit scale **and** a 6-bit min per block before any dot product runs, while Q4_0
decodes one fp16 scale.

The bare-metal probe predicted exactly this and bounds it: with unpacking removed entirely, the same
kernel reaches 0.79 of spec; naive nibble-to-float unpacking drops it to 0.35; `__dp4a` recovers it
to 0.67. The real-model gap (0.553 -> 0.622) sits inside that band.

### The honest limits of this result

- **This is a SPEED measurement, not a quality one.** Q4_0 is worse per bit than Q4_K_M, and the
  test file was requantized *from* the Q4_K_M rather than from the original weights, so its quality
  is worse still. Nothing here says Q4_0 is a better model — only that it decodes faster. A quality
  comparison needs a Q4_0 built from source weights and is a separate measurement.
- **One card.** The whole effect is that this GPU's ALU is weak relative to its bandwidth. On a card
  with more ALU headroom the unpack can hide behind the memory transfer and the gap should shrink.
- It does **not** rescue the MoE flagship directly: that model is Q2_K, whose unpack is *more*
  complex still, but there is no simple-format equivalent at 2 bits to swap to. The prediction that
  falls out — Q2_K should be worse per weight than Q4_0 — is untested and logged as such.

**Wired into:** `findings/REGISTER.json:L-14` (Law 4 amended: eta is format-dependent, mechanism =
unpack instruction cost, measured) · `C-05` (pattern explained, cause named) ·
`V-16` (format lever: +19% on pre-Ampere at 5.7% fewer bytes, speed-only claim) ·
`U-11` (untested: does the effect shrink on Ampere+? needs a second card).
