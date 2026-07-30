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
