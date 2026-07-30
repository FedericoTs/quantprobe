# Pre-registration #78: does better accounting make the machine actually FASTER?

**Author:** Federico Sciuca · **Date staked:** 2026-07-30, BEFORE any measurement. **STAKED.**

#76 (gather-only embeddings) and #77 (conservative codebook pricing) were accuracy fixes. But
freeing phantom bytes from the always-active budget frees real VRAM, and the emitted commands
moved: **7 of 14** now keep MORE expert layers resident (30B 34→33 CPU layers, Coder 34→33,
APEX-Mini 32→31, Qwen3.6 32→31, APEX-MTP 31→30). #62 measured that resident-expert count is a
real lever (+6% tg / +15% pp per its sweep).

**The question this settles: is the new advice FASTER on the machine, or just better-numbered?**

## Stakes

- **P-1 (the headline).** The flagship Qwen3-30B-A3B, on its NEW emitted `-ot` (33 CPU layers vs
  34), beats its own previous measurement of **21.33 tok/s** — same file, same session, same
  binary, only the tool's advice changed.
- **P-2 (the champion).** APEX-Mini on its new emit (31 vs 32 CPU layers) beats **22.20**.
- **P-3 (no cliff).** No re-emitted config runs SLOWER than its predecessor. One extra resident
  layer is a small step, but #13 measured a hard cliff one step past the ceiling; the whole point
  of a conservative budget is not to cross it, and if any arm regresses the freed budget is being
  spent too aggressively.

## KILL RULE

If P-3 fails — any arm slower than before — the freed budget is over-spent: the emit reverts to
the prior conservatism for that class and the regression publishes. If P-1 and P-2 both fail
while P-3 holds, the accounting fix is accuracy-only: honest, worth shipping, and explicitly NOT
a speed claim. Whatever happens, the MACHINE_LADDER numbers are updated to whatever is measured.

**Wired into:** pending; MACHINE_LADDER + the v1.24 release note either way.

---

## SCORED — 2026-07-30, same session (clocks 139 MHz idle -> loaded, healthy)

| model | previous emit | new emit | delta |
|---|---|---|---|
| Qwen3-30B-A3B Q2_K | 21.33 | **21.54** | +1.0% |
| Qwen3-Coder-30B Q2_K_L | 21.49 | **21.77** | +1.3% |
| **Qwen3.5-35B APEX-Mini** | 22.20 | **22.55** | **+1.6%** |
| Qwen3.6-35B Q2_K_XL | 16.68 | **16.87** | +1.1% |
| Qwen3.6-35B APEX-MTP-Nano | 17.26 | **17.53** | +1.6% |

- **P-1 HIT** (30B 21.33 → 21.54) and **P-2 HIT** (APEX-Mini 22.20 → **22.55, a new measured
  best for this box**).
- **P-3 HIT.** 5 of 5 improved, **zero regressions** — the freed budget was spent one layer at a
  time and no arm crossed the #13 cliff.
- **Honest size of the win: +1.3% mean.** Real, reproducible, in the predicted direction, and
  SMALL. Freeing phantom bytes buys one more resident expert layer, and #62 already measured
  that one layer is worth about this much. Anyone reading "better accounting made it faster"
  should read the second decimal, not the headline: the accuracy fix was worth ~8 points of
  median prediction error; the speed it unlocked is worth ~1.3%.

**The honest summary of the day's chain:** the accounting fixes were worth far more as *truth*
than as *throughput* — and both are now measured rather than assumed.
