# Pre-registration #96: width or depth — how to spend a fixed token budget on a small model

**Author:** Federico Sciuca · **Date staked:** 2026-08-08, **BEFORE any arm ran.** **STAKED.**

## The question nobody has answered

Two people spent extra compute on the same problem and got the same headline from opposite
directions.

- **Ours (P0):** 16 candidates in *parallel*, winner picked by visible tests. 4B went
  76.8 → **88.4** on HumanEval+ and passed a 30B single-shot at 87.8.
- **[@danpacary](https://x.com/danpacary/status/2085794035197960418) (2026-08-07, M4 Max):**
  *one sequential repair* after showing each model its own test failures. A 16 GB model went
  53% → 100%; a 93 GB model went 96% → 100%. His conclusion — *"the 93 GB model was never
  smarter, it tolerated a worse harness"* — is our thesis stated better than we have stated it.

**Neither of us controlled for budget.** He spent his on depth, we spent ours on width, and the
comparison has never been made on one axis. That is the experiment.

Credit where the framing came from: the sequential-repair arm and the "same exam, only the
harness changed" design are his. The equal-budget constraint and the predictions below are ours,
and the predictions are what is being staked.

## Design: one budget, spent three ways

Fix a total generation budget **B** per task (measured in generated tokens, not calls — a
repair round that emits 3x the tokens is not one unit of anything).

- **Arm W (pure width):** k independent candidates at temperature 1.0, winner picked by the
  problem's visible tests. This is our shipped lanes path.
- **Arm D (pure depth):** 1 candidate, then d sequential repair rounds, each shown the *actual
  failing visible-test output* from the previous round. Execution-grounded, not introspection.
- **Arm H (hybrid):** sqrt-ish split — 4 candidates, best-by-visible-tests, then 4 repair rounds
  on that one. Same B.

Scoring: **hidden plus-tests, one submission per task**, exactly as P0 (`weights/p0_lanes.py`
selects on `base_input` and scores on `plus_input`). 164 HumanEval+ tasks, the full set. Budget
is matched by measured generated tokens per arm and **reported**, not assumed — if the arms
land more than 10% apart in tokens the comparison is void and re-run.

## Staked expectations

- **P-1 (width beats depth at equal budget).** Arm W > Arm D by **≥3 points** on hidden tests.
  Mechanism: a wrong first attempt is usually wrong *structurally*, and repair inherits its
  frame. Sampling gets a fresh frame; repair rarely escapes one. Our Phase B is consistent —
  introspective repair recovered only **7.3%** (20/273) — though that arm repaired *tests*
  without execution feedback, so it is supporting evidence, not the same measurement.
- **P-2 (the hybrid wins).** Arm H ≥ max(W, D). Width buys diverse frames, depth exploits the
  near-misses; a budget spent entirely on either leaves one of those on the table.
- **P-3 (THE DECISION RULE — the useful result).** Repair's benefit is **conditional on how
  close the first attempt was**. Among tasks whose first candidate fails **≤2** visible tests,
  Arm D's per-task gain exceeds Arm W's; among tasks failing **>2**, Arm W's exceeds Arm D's.
  If this holds it ships as advice: *few failures → repair, many failures → resample.*
- **P-4 (the small model is the one that moves).** The width-over-depth gap is larger on the
  0.6B than on the 4B. Weak models produce more structurally-wrong first frames.

## KILL RULES

- **If P-1 fails and depth wins**, our lanes framing is wrong about *why* it works, the P0
  posts get an amendment at full prominence, and `-np` lanes stop being the headline
  recommendation until re-derived. This is the outcome that costs us the most, which is why it
  is stated first.
- **If P-3 shows no conditional split**, no decision rule ships. "It depends" without a
  measured boundary is not advice, and we do not publish it as one.
- **If token budgets diverge >10%** between arms, every number is void — a width-vs-depth
  result at unequal budget is the exact confound this prereg exists to remove.

## Why it is cheap

The lanes harness, the visible/hidden split, the per-candidate outcome logs and the sandboxed
executor all exist and are committed. Arm D is the only new code: feed the failing-test output
back into the prompt. One evening of GPU behind the current queue.

**Wired into:** pending — P0's public claims, the lanes advice in `plan`, E-17/U-43's
difficulty work (near-misses vs structural failures is the same axis as difficulty), and a
chart: the width-vs-depth frontier at fixed budget, which nobody currently has.
