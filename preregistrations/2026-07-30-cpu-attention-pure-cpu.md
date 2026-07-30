# Pre-registration #75: the CPU-attention term's limiting case — pure CPU at depth

**Author:** Federico Sciuca · **Date staked:** 2026-07-30, prediction captured BEFORE the run. **STAKED.**

#74 validated the CPU-attention term (1.55 µs/pos/CPU-layer) on dense SPLITS out-of-sample.
Extending it to the pure-CPU row was forced by consistency: with every layer on the CPU, that row
pays the term over ALL layers, and leaving it un-termed ranked pure CPU above configs with
strictly fewer CPU layers — a ranking inversion, not a prediction. But **no pure-CPU-at-depth
measurement exists anywhere in this project** (all prior depth work was MoE with attention in
VRAM), so the extension is untested and must not stand on assumption.

**Arm:** Qwen2.5-7B-Instruct Q4_K_M, `-ngl 0`, tg @ d16384, 4 threads, one session.

**Staked (the shipped tool with the term applied over all 28 layers): 1.1 tok/s.**

- **P-1.** Measured lands within ±25% of 1.1.
- **P-2 (the ranking is right).** Measured pure CPU is SLOWER than the same model's measured
  dense split at the same depth (3.44 tok/s, #73) — the inversion the extension exists to fix.

**KILL RULE:** if P-1 fails, the term does not apply at full strength to pure CPU: the extension
is reverted and the pure-CPU-at-depth row gets an explicit unvalidated-regime note instead of a
number. If P-2 fails (pure CPU genuinely beats the split at depth), the inversion was real
physics and BOTH the extension and #74's term need re-derivation on this regime.

**Wired into:** pending; `plan.evaluate` pure-CPU row.
