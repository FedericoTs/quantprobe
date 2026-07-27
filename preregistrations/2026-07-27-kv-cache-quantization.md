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
