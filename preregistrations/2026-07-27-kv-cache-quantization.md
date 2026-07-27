# Pre-registration #25: KV cache quantization — the one lever our own laws say should pay on both axes

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## Why this is the top untried lever

Three established findings point at it and none of them was followed to this conclusion:

- **L-02** prices the KV cache as a first-class bandwidth consumer — decode falls 20.35 → 6.30 tok/s
  from `d0` to `d16384`. Fewer KV bytes per token is therefore *directly* a decode lever.
- **L-06** proved the VRAM claimants are **fungible**: evicting KV recovered 2.42× of prefill on the
  split placement, because the compute buffer was short of exactly what KV was holding.
- **L-08** gives the compute buffer's exact slope (0.5874 MiB per ubatch token), so freed VRAM
  converts into a known number of extra ubatch tokens.

Put together: halving the KV cache should buy decode **and** free the headroom that buys prefill.
Every other lever in the register trades one against the other — the workload frontier exists
precisely because prefill and decode pull opposite ways. **This is the only candidate that might
pay on both axes at once, and we have never run it.**

It is also the cheapest possible test of **C-05**, the pattern that has now bitten twice: a constant
assumed to hold across quantization formats turning out to be a property *of* the format. `ETA_KV =
0.70` was fitted **without ever varying the KV cache type**. This varies it.

## Configurations

`Qwen3-30B-A3B-Q2_K`, split placement (`-ot "blk\.(16..47)\.ffn_.*_exps\.=CPU" -mmp 0`) — the
placement where L-06 showed VRAM actually binds. Flash attention **on for every arm**, because
quantized V requires it and an unmatched `-fa` would confound the comparison.

| arm | `-ctk` / `-ctv` | nominal KV bytes |
|---|---|---|
| K1 | `f16` / `f16` | 1.00× (baseline) |
| K2 | `q8_0` / `q8_0` | ~0.53× |
| K3 | `q4_0` / `q4_0` | ~0.28× |

Measured at `-d 0` and `-d 16384`, `r=3`, one session, GPU state logged before and after.

## Stakes

- **P-1 (it pays at depth).** At `d16384`, K2 improves decode by **≥20%** over K1. L-02 says KV
  bandwidth dominates at depth and K2 removes ~47% of those bytes; if this does not show, the KV
  term is not describing what I think it describes.
- **P-2 (it is free at zero depth).** At `d0`, K2 is neutral within **±5%**. *Stated in full
  awareness that the identical reasoning MISSED for `-nkvo` in pre-registration #21* — I predicted
  an empty cache could not matter and was wrong, because that flag moved where attention was
  computed. `-ctk`/`-ctv` change the cache's *element type*, not its location, so the reasoning
  should hold here. If P-2 misses again, the lesson is that I still do not understand what these
  flags do, and P-1 becomes uninterpretable for the same reason #21's P-3 nearly was.
- **P-3 (the fungibility payoff — the reason this is worth doing).** At `d16384` on the split,
  **K2 with KV in VRAM beats the currently-shipped `-nkvo 1` row on BOTH prefill and decode.** If it
  does, the frontier's prefill champion is obsolete: we would be recommending a configuration that
  gives up 47% of decode to buy prompt speed that KV quantization delivers without the trade.
- **P-4 (C-05, the third instance).** The measured decode ratio K1:K2 at `d16384` differs from the
  nominal byte ratio (1.00 : 0.53) by **more than 10%**. That would make `ETA_KV` format-dependent —
  the same failure mode as D-06 and C-02, found a third time, at which point it stops being a
  coincidence and becomes something the five-layer gate should test for directly.

## Refuted if

**P-1 fails.** Then quantized KV does not buy decode, the lever is purely a VRAM-freeing trick, and
it collapses into the existing ubatch sizing story rather than being a finding of its own.

P-2 failing does not refute the lever but **invalidates my model of the flag**, and P-1 would then
need re-measuring against a control that isolates location from element type.

## What ships, and what does not

**Not the quality cost — that is not measured here and nothing ships without it.** Quantized KV
degrades output, and this project's own D-01 killed a 1.335× speed lever for costing 1.206×
perplexity. A speed win here means nothing until the perplexity cost is measured against the bit
ladder the same way, and I am recording that *before* seeing an attractive number.

If P-1 and P-3 hold **and** the quality cost is subsequently measured and acceptable, the shipped
change is to the frontier: KV quantization would replace `-nkvo 1` as the long-prompt
recommendation. If only P-1 holds, it becomes a disclosed option, in the same shape as v1.13.1 —
state the trade, name the command, let the user choose.

**Explicitly NOT claimed:** that this generalises off a GTX 1060, or off this model. `ETA_KV` was
fitted on one architecture and this varies one flag on that same architecture.

---

## Scored (2026-07-27, log: `weights/data/prereg25_kv_quant.log`)

**Verdict: P-1 HIT, P-2 HIT, P-3 MISS by the letter, P-4 HIT decisively. And the shipped
"evict KV to RAM" row turns out to be dominated.**

All arms, one session, identical flags (`-fa 1`, split placement, `r=3`):

| arm | tg32 @ d0 | tg32 @ d16384 |
|---|---|---|
| K1 `f16` KV in VRAM | 21.81 ± 0.32 | 7.73 ± 0.13 |
| **K2 `q8_0` KV in VRAM** | 21.10 ± 0.18 | **10.59 ± 0.18** |
| K3 `q4_0` KV in VRAM | 20.76 ± 0.62 | 10.80 ± 0.09 |
| **`-nkvo 1`, `f16` — the SHIPPED row** | 17.06 ± 0.87 | **3.48 ± 0.05** |

- **P-1 (≥20% at depth): HIT.** K2 gives **+37.0%** over K1 at 16k.
- **P-2 (neutral at d0, ±5%): HIT.** −3.3%. Stated in advance that the identical reasoning missed
  for `-nkvo` in #21; this time the flag behaved as modelled, and the difference is exactly the one
  predicted — `-ctk`/`-ctv` change the cache's element type, `-nkvo` changes where attention is
  computed.
- **P-3 (beats the shipped row on BOTH axes): MISS.** Prefill 382.17 ± 3.11 against the shipped
  386.14 — **−1.0%**, inside the error bar, so it does not *beat* it. Recording this as a miss
  rather than rounding a tie into a win.
- **P-4 (>10% departure from the byte model): HIT, decisively, and only below 8 bits.**

### The saturation, which is the real finding

Decomposing each arm's decode time into weight and KV components (`t_KV = 1/tg(16k) − 1/tg(d0)`,
using each arm's own `d0` so no cross-arm assumption is smuggled in):

| arm | t_KV, normalised to f16 | nominal byte ratio | departure |
|---|---|---|---|
| `q8_0` | 0.563 | 0.53 | **+6.2%** — the byte model works |
| `q4_0` | 0.532 | 0.28 | **+90%** — the byte model fails completely |

`q4_0` halves the KV bytes again and returns **essentially nothing** (10.80 vs 10.59, +2.0%). The
saving stops being delivered somewhere between 8 and 4 bits: dequantisation cost replaces the
bandwidth cost it was supposed to remove.

**This is the third instance of C-05**, and the three now share one shape:

| finding | quantity assumed constant | what it actually depends on |
|---|---|---|
| D-06 | the sub-4-bit decode collapse | weight FORMAT, not bit-width |
| C-02 | η, the VRAM-tier efficiency | weight FORMAT CLASS, not one constant |
| **#25** | `ETA_KV = 0.70` | **KV format — flat to 8 bits, collapses below** |

`ETA_KV` was fitted without ever varying the cache type, and it holds only above 8 bits. Three
times is not a coincidence: **a quantized byte is not a byte.** Below a format-specific threshold,
bytes removed from the bandwidth bill reappear as compute, and every constant in this project that
was fitted at one format is suspect until varied.

### The shipped row is dominated — and it is dominated worst where it is recommended

`MOE_FRONTIER` row 3 evicts KV to host RAM to buy prompt speed. Measured against `q8_0` KV kept in
VRAM, under identical flags:

| | pp2048 | tg @ d0 | tg @ d16384 |
|---|---|---|---|
| shipped: `-nkvo 1`, f16 KV | 386.14 | 17.06 | 3.48 |
| **`q8_0` KV in VRAM** | 382.17 (−1.0%, tied) | **20.18 (+18%)** | **10.59 (+204%)** |

Same prefill, **3.04× the decode at depth.** And the row exists specifically to serve long-prompt
workloads — RAG at 50:1, document QA at 200:1 — which are precisely the workloads that run at
depth, where it is three times worse. We are recommending it in the one regime it loses hardest.

The `-ub 2048` cliff (L-08) reproduces here unchanged at 205.54 vs 382.17, confirming that what
overflows is the compute buffer and not KV: quantizing the cache does not buy a bigger safe ubatch.
That part of the fungibility argument does **not** hold.

### What ships — and the condition stated before any of this was measured

**Nothing yet, and the reason was written into the stake before the numbers existed:** the quality
cost is not measured here. D-01 killed a 1.335× decode lever for costing 1.206× perplexity, and a
3.04× number is exactly when it is most tempting to skip that step. `q8_0` KV degrades output by an
amount nobody in this project has measured.

The next measurement is therefore perplexity at `-ctk q8_0 -ctv q8_0` against `f16`, on the same
model, judged against the bit ladder the same way D-01 was. If the cost is small, row 3 of the
frontier is **replaced**, not amended. If it is not, `q8_0` KV becomes a disclosed option and the
`-nkvo` row is withdrawn anyway — because it loses to *something*, and a row that is dominated at
depth should not be recommended for deep workloads regardless of what replaces it.

**Explicitly NOT claimed:** that this generalises off a GTX 1060 or off Qwen3-30B-A3B. `ETA_KV` was
fitted on one architecture and this varied one flag on that same architecture.

**Wired into:** `findings/REGISTER.json:C-05` (third instance, pattern promoted) ·
`findings/REGISTER.json:U-01` (scored) · the frontier is NOT changed until perplexity is measured.
