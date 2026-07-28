# Pre-registration #60: the original case, retested — the shipped tool vs naive llama.cpp, fresh

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the runs. **STAKED.**

## What this is

The project's original case is Qwen3-30B-A3B Q2_K on this 6 GB card. After a session that closed
the mechanism ledger (#51–#59), this retests the ORIGINAL claims end to end, in one clean session,
against what `quantprobe plan --model qwen3-30b --machine 2016-xmp` prints TODAY:

| arm | command | shipped prediction |
|---|---|---|
| (a) pure CPU | `-ngl 0` | **11.9 tok/s** ±25% → **[8.9, 14.9]** |
| (b) naive llama.cpp | `-ngl 20` (most layers that fit, no -ot) | no tool prediction — this is the baseline the tool exists to beat; today's earlier stock measurements: 14.9–15.8 |
| (c) tool-advised | `-ngl 99 -ot "blk.(11..47).ffn_.*_exps.=CPU" --no-mmap -b 1024 -ub 1024` | **22.0 tok/s** ±25% → **[16.5, 27.5]** |

## Also on record before the runs (informational, no kill attached)

The session's v2 floor model (LOO-validated poorly: 10–30% errors, DS-Lite −28% structural miss)
predicts arm (c) at **25.2 tok/s**, band widened to [17.6, 25.2] by the DS-bias caveat. Logged to
score the model's usefulness, not as a claim.

## Disclosures

- The binary is this project's instrumented build (b1-f113e02): E5/E6/E8/E9 toggles all OFF by
  default; the E2c expert-major mmid dispatch is compiled ON (measured ~neutral when introduced).
  Not a stock release binary — stated because a baseline claim must name its binary.
- Thermal: this box drifts up to +25% cold→warm; arms interleaved (c,b,a then repeat), r=2 each.

## Stakes

- **P-1.** Arm (c) lands in the tool's printed band **[16.5, 27.5]**.
- **P-2.** Arm (a) lands in **[8.9, 14.9]**.
- **P-3 (the headline the project rests on).** Arm (c) ≥ **1.25×** arm (b) — the tool's advice
  must beat naive llama..cpp usage by at least 25% on the original case, fresh, same session.

## KILL RULE

If **P-3** fails, the tool's headline value proposition on its own flagship case is stale and the
README/plan copy must be corrected to whatever the fresh ratio is before anything else ships.
If **P-1** fails, the frontier row constants are re-measured and updated (the ±25% band is the
tool's own printed promise — missing it is a shipped-claim failure, handled as such).

**Wired into:** pending scoring; this is the project's recurring honesty checkpoint.
