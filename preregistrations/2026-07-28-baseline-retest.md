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

---

## Scored (2026-07-28, log: `weights/data/prereg60_baseline_retest.log`)

**Verdict: P-2 HIT, P-1 MISS, P-3 FAILS — both kill rules fire, and both shipped-copy
corrections were made in the same commit as this score.**

### First, an instrument catch that would have corrupted the verdict

The first six-run block measured everything 25-30% low with a 34% intra-run error bar on the CPU
arm. Cause found before scoring: a **runaway `find` orphan with 16,285 CPU-seconds** (4.5
CPU-hours, from an earlier shell pipeline) holding idle load at 47%. Killed; load fell to 9%;
the block is recorded as INVALIDATED in the log and excluded. Fourth harness artifact this
project has caught by refusing to read a suspicious number.

### The clean block (interleaved x2, r=2 each)

| arm | tg128 | staked band | verdict |
|---|---|---|---|
| (c) tool-advised `-ot` split + `-b/-ub 1024` | 15.04, 15.53 → **15.29** | [16.5, 27.5] | **P-1 MISS** (7% under the floor) |
| (b) plain `-ngl 20` | 15.08, 15.40 → **15.24** | — | c/b = **1.003** → **P-3 FAILS** |
| (a) pure CPU `-ngl 0` | 13.18, 13.19 → **13.19** | [8.9, 14.9] | **P-2 HIT** (tool under-predicted by 11% — CPU is the arm the box's thermal state hurts least) |

### What the two failures actually mean, and the corrections shipped

**P-3 (the -ot split does not beat a plain -ngl split on generation).** This is not new physics —
prereg #43 measured the same equality weeks ago (19.70 vs 19.76) and today reconfirms it
(15.29 vs 15.24). What failed is the COPY: the plan output's ranked list reads as if the -ot row
wins generation. Correction shipped: the plan now states the tg equality explicitly and grounds
the -ot advice where it is actually earned — prompt processing (2.2x measured), KV-in-VRAM
safety, and enabling the speculation numbers. The v2 floor model, for the record, predicted this
equality (its GPU-call term barely distinguishes the two placements); the old byte-only intuition
did not.

**P-1 (the printed 22.0 ±25% missed at 15.29).** The frontier constant was measured on a fresh
box; today's measurement came after ~9 hours of sustained benchmarking. Same-day earlier runs of
a near-identical arm gave 16.87 — the box degrades under sustained load beyond the printed band.
Correction shipped: the plan output now labels reference numbers as COLD-BOX ceilings and quotes
the measured -29% loaded-state figure with the prereg reference. The constants were NOT rewritten
from tonight's degraded state — overwriting cold-box calibration with end-of-marathon numbers
would replace good data with bad; instead the band's meaning is now stated honestly.

### The fresh original-case scoreboard, as it stands tonight

```
pure CPU                 13.19 tok/s
plain -ngl 20            15.24
tool -ot split           15.29     (equal tg; wins pp 2.2x and enables speculation)
copy-regime + ngram      ~4.7x on top of split decode (measured #28, cold-box)
```

**Wired into:** `findings/REGISTER.json:C-10` (new: printed band vs loaded-state drift) ·
`quantprobe/plan.py` workload copy (both corrections, tests green) · V-01 scope note.
