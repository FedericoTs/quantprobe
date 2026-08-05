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

---

## VERDICT (Phase A, scored 2026-08-05): P-A1 PASS, P-A2 PASS, P-A3 FAIL - and the fail is a promotion

Full grid, 14 cells (12 measured + 30B lanes excluded by stake, 4B lanes deferred-then-run),
371 MBPP+ / 164 HumanEval+ tasks, harness v2, one protocol for every surviving row.

| pass@1 / lanes16 | MBPP+ | HumanEval+ |
|---|---|---|
| 0.6B | 36.7 / 56.9 (+20.2) | 24.4 / 56.1 (+31.7) |
| 4B | 66.0 / **75.2** (+9.2) | 76.8 / **88.4** (+11.6) |
| 7B | 68.5 / 75.7 (+7.2) | 72.0 / 84.8 (+12.8) |
| 30B Coder | 75.5 / - | 87.8 / - |

- **P-A1 PASS 5/5:** every P0-measured cell reproduced on the full bench within the +/-6 band
  (+3.9, +2.0, +3.1, -0.4, +3.8) - across a harness upgrade and a protocol change, which is
  what the band was staked to absorb. The P0 subset was representative.
- **P-A2 PASS:** 7B lanes transferred to a bench the machinery never touched at **+12.8**
  (bar: +5). Verified-lanes is a mechanism, not an MBPP artifact.
- **P-A3 FAIL, mechanism identified:** staked order 0.6B < 4B < 7B <= 30B holds on MBPP+ and
  breaks on HumanEval+ - the 4B beats the 7B by 4.8 pts (76.8 vs 72.0). Clean rows, no
  truncation degradation: **model generation (Qwen3.5) beats parameter count (Qwen2.5) on
  this bench.** Published as a miss against the stake and acted on: the 4B is promoted to
  primary lanes-engine candidate for Phases C/D.
- **KR-A1 honored the hard way:** v1 of the executor was precondition-blocked by its own gate
  on HumanEval+ (69% reference exclusions - JSON transport mangling Python types). Harness v2
  (pickle transport): 0/164 and 7/378 (1.9%) exclusions. **KR-A2:** zero degraded rows after
  the thinking-template fix. **KR-A3:** one overlap incident (release-verify bench fired
  during the grid) - the affected row was voided and re-measured; verify.py now refuses to
  bench under any runner lock. **KR-A4:** zero cross-bench overlap.

**The headline, stated with its caveats:** on a 2016 GTX 1060, a 4B model with 16
execution-verified lanes matches the strongest local 30B coder single-shot on MBPP+ (75.2 vs
75.5) and beats it on HumanEval+ (88.4 vs 87.8) - at 5-8x the wall-clock (16.6s vs 3.4s
median), with test-availability assumed, and with the 30B denied its own lanes by the staked
U-39 exclusion (its batching caps ~2x; a lanes-30B arm would be slow, not impossible).
Same-cost comparisons live in P0b.

**Feeds forward:** the grid is the Phase C/D baseline table; the 4B promotion is Phase C's
first casting decision; lanes-vs-size scaling (+20/+9/+7 MBPP) says verification compounds
hardest exactly where training is cheapest. Raw: weights/data/grid_*.json, grid_run.log.
