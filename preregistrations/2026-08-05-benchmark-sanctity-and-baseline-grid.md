# BENCHMARK SANCTITY (program law) + Phase A baseline grid

**Date staked:** 2026-08-05, before the grid harness existed and before any full-bench number
was measured on this box. This pre-registration does two things: it enacts the decontamination
rule as a PERMANENT law of the specialist program (Phases A-D and any successor), and it stakes
Phase A's gates.

## Part 1 — the decontamination rule (permanent, program-wide)

**Standard benchmarks are eval-only, forever.** The protected set, pinned by version hash in
the harness: MBPP+ and HumanEval+ (EvalPlus releases), plus any bench later added to the grid.

1. **No training sample may derive from a protected bench** - not its prompts, not its
   solutions, not paraphrases generated FROM its tasks. Consequences applied today: P0's 86
   plus-verified lane winners are **radioactive for training** (they solve eval tasks) and are
   so marked in the register; the earlier "generate training data on the untouched 265 MBPP+
   tasks" design is dead.
2. **Mechanical screen on every training batch, no exceptions:** (a) exact prompt-hash match
   against all protected prompts AND canonical solutions; (b) sliding **8-gram token overlap**
   (whitespace-normalized, case-folded) between any training sample and any protected prompt or
   canonical solution. Any hit -> the training sample is EXCLUDED and counted; counts are
   published with every training run. 8-gram is deliberately strict: the failure direction is
   losing training data, never contaminating eval.
3. **The screen is a kill rule:** a training run whose batch log cannot show the screen ran is
   void - not "assumed clean". A post-hoc discovered hit voids the affected checkpoint's
   bench claims entirely.
4. **Scope clause:** any model this project publishes carries the claim "trained on zero
   protected-bench derivatives, screen-verified" or it carries no bench numbers at all.

## Part 2 — Phase A: the baseline grid

**What:** full MBPP+ (378 tasks) and full HumanEval+ (164 tasks), on the local ladder:
{0.6B, 4B, 7B} x {pass@1 greedy, verified best-of-16 lanes} + 30B pass@1.
30B lanes is excluded by prior evidence, not laziness: U-39 measured the MoE expert-offload
batching cap at ~2.0x aggregate - 16 lanes on the split placement is economically pointless
and ~10 GPU-hours; the exclusion is cited here and in the results.

**Protocol:** P0's harness generalized - same sandbox, same selection honesty (selector reads
base tests only, all arms scored on plus tests), same truncation quarantine, one session per
row, shared mkdir-or-refuse lock, GPU state logged, resumable per-row JSON so a crash never
re-measures a completed row. Qwen3-family rows run with the thinking soft-switch off and
npredict 1024 (the P0 0.6B-lanes truncation lesson, fixed rather than re-suffered).

**Staked predictions (falsifiable, scored when the grid lands):**
- **P-A1 (subset representativeness):** every P0-measured cell (0.6B/7B/30B single, 0.6B/7B
  lanes) reproduces on full MBPP+ within **+/-6 pts** of its 113-task subset number. A miss
  means the seeded subset was unrepresentative and the P0 verdict gains a caveat - recorded,
  not hidden.
- **P-A2 (mechanism transfers across benches):** 7B lanes beats 7B single by **>= +5 pts on
  HumanEval+** - a bench the lane machinery has never touched. A miss bounds the
  verified-lanes claim to MBPP-style tasks and the P0b design must react.
- **P-A3 (ladder sanity):** on every bench, single-shot order is 0.6B < 4B < 7B <= 30B. Any
  inversion is a finding (and a warning about the 4B's placement or the 30B's Q2 damage).

**Kill rules:**
- **KR-A1 (harness integrity, per bench):** every reference solution must pass its own plus
  tests in our sandbox; exclusions counted and published; >10% on any bench =
  precondition-block for that bench, not a result.
- **KR-A2 (degraded-row rule):** any row where >20% of tasks have >= half their lanes
  truncated is marked DEGRADED and its number is a floor, stated wherever cited (the 0.6B
  P0 census caveat, promoted to a first-class rule).
- **KR-A3 (state):** one machine state per row; overlapping measurement = the overlap window's
  data is deleted, never explained.
- **KR-A4 (cross-bench integrity):** zero prompt-hash overlap between MBPP+ and HumanEval+
  task sets, asserted before any row runs (they should be disjoint; if not, the overlap list
  is published and those tasks excluded from both).

**What Phase A feeds:** the grid is the baseline table for Phases C/D (every training claim is
scored against these cells), the public comparison surface for the README, and the
representativeness check that decides how much the P0 numbers can be trusted as proxies.

Raw under `weights/data/grid_*`. Verdict appended here. Phases B/C/D get their own stakes.
