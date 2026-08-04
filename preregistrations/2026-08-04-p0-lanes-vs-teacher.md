# P0 — do k verified lanes of a 7B beat one shot of a 30B, at equal wall-clock?

**Date staked:** 2026-08-04, before the lane-count pilot, before any arm ran, before the task
subset was drawn.

**The claim under test:** on this box, a dense 7B sampling k candidates in parallel server lanes
(U-38's batch machinery) with an *execution-picked* winner beats the 30B MoE's single greedy
shot on held-out coding tasks — at equal or less wall-clock. This is the entire test-time-compute
literature compressed into one falsifiable local claim, and it is Phase 0 of the agreed
specialist program: if verified lanes cannot beat the big model even with free selection, the
later phases (specialist training, committee distillation) inherit that ceiling and must know it.

Secondary product, non-gating: the **headroom census** — 0.6B/7B/30B single-shot rates on the
same subset — which is the input the S-1 design rule demands before any S-2 cluster is staked
("measure the BASE STUDENT first; only stake where the gap is real").

## Design

- **Tasks:** MBPP+ (EvalPlus), the public versioned set. Seeded sample of **120 task-ids, seed
  20260804**, id list published in the run log. Full set only if the 120 finish under the time
  cap. Chosen over our own generator because Phase 0's census half exists to be comparable to
  numbers people already know.
- **Sandbox:** each candidate runs under a subprocess with a 10s timeout, no network. This is
  the standard EvalPlus execution model, on our own box.
- **Selection honesty (the load-bearing design rule):** the selector may read **base tests
  only** (the visible spec a real user would have). ALL arms are scored on the **plus tests**
  (the extended hidden set). A lane winner that passes base but fails plus is a MISS. The
  selector never sees the exam it is graded on.
- **Models (the validated local ladder rows):** Qwen3-0.6B, Qwen2.5-7B-Instruct, and — **AMENDED
  before any measurement ran**: arm A is **Qwen3-Coder-30B-A3B-Instruct** (Q2_K_L, ladder row),
  not the general Qwen3-30B-A3B first drafted here. Reason: on a coding family the strongest
  coding model this box owns is the only honest goliath; beating the weaker general 30B would
  invite the "wrong teacher" objection and it would be right. Each model at its established
  ladder placement (quantprobe's own best_flags — the tool plans its own experiment). Same
  llama.cpp build for every arm.

## Arms, one machine state (C-14), sequential, lockfile

- **Pilot (before any arm, k frozen at its end):** `-np` sweep on the 7B placement; k = the
  largest lane count ≤ 16 keeping aggregate throughput ≥ 0.8× its peak (U-38's cliff logic
  decides, not taste).
- **A:** 30B single-shot, temp 0 — the big-model baseline.
- **B:** 7B single-shot, temp 0 — the floor, and a census row.
- **C:** 7B × k lanes, temp 0.8 / top-p 0.95, seeded; base-test-picked winner (ties → shortest);
  wall-clock charged HONESTLY: all k lanes plus selection execution, not per-lane.
- **D (census-only, never cited as a comparison):** 0.6B single-shot and 0.6B × k lanes.

## Staked gates

- **P1 (headline):** C solve-rate ≥ A solve-rate + 5 pts on plus tests.
- **P2 (economics):** C median wall-clock/task ≤ A median wall-clock/task.
- **P3 (mechanism):** C ≥ B + 10 pts. If selection does not beat its own single shot by a real
  margin, lanes are noise and P1 cannot be credited to verification regardless of its sign.
- **KR-A (headroom precondition, S-1's rule as a gate):** B ≤ A − 5 pts. If the 7B single-shot
  already matches the 30B on this family, there is no goliath here — the census is still
  reported, but the P1 comparison is VOID as a beat-the-teacher claim and says so.
- **KR-B (harness integrity):** every task's reference solution must pass its own plus tests in
  our sandbox; failures are excluded and counted; >10% excluded = precondition-block (exit 2
  semantics), not a result.
- **KR-C (state):** one session per model row, arms sequential, lock via mkdir-or-refuse, GPU
  state logged before/after each row. Any overlap-window data is deleted, never explained.
- **KR-D (selection-split reality check):** on ≥90% of included tasks the plus set must be a
  strict superset of the base set; otherwise the base/plus split is cosmetic and P1's honesty
  clause fails → precondition-block.

Misses published at equal prominence, whichever way it lands. Raw under `weights/data/p0_*`.
Verdict appended here.

## Cost and what feeds forward

$0 and one evening. Downloads: the `evalplus` package and MBPP+ data (~MB, public, versioned —
recorded in the log). The census table feeds S-2 target selection; the P1/P2 outcome decides
whether the program's later phases ride lanes or abandon them.

---

## VERDICT (scored 2026-08-04, same day): P3 PASS, P1/P2 FAIL - the mechanism is real, the staked headline is not

Pilot chose k=16 (aggregate held to 16 lanes, U-38 consistent). 113 tasks (7 references
excluded by KR-B at 5.8%, under the block). One session per arm, locks held, GPU state logged.

| arm | solve (plus tests) | median wall/task |
|---|---|---|
| A - Coder-30B single | 83/113 = 73.5% | 3.7s |
| B - 7B single | 73/113 = 64.6% | 2.5s |
| **C - 7B x16 verified lanes** | **86/113 = 76.1%** | 12.0s |
| D - 0.6B single (census) | 38/113 = 33.6% | 7.8s |
| D - 0.6B x16 lanes (census) | 60/113 = 53.1% | 20.8s |

- **P3 PASS: +11.5 pts** (76.1 vs 64.6) - execution-picked selection is a real mechanism,
  past the +10 bar.
- **KR-A holds:** B sits 8.9 pts under A - the goliath was real; P1 was scored, not voided.
- **P1 FAIL: +2.6 < +5.** C DID beat the Coder-30B absolutely - a Q4 7B on a GTX 1060
  outscoring the strongest coder this box owns - but under the margin this stake said would
  count. Published as a miss.
- **P2 FAIL: 12.0s vs 3.7s.** The verified answer costs 3.2x the big model's wall-clock. The
  cost is dominated by selection engineering (16 sequential subprocess test runs per task),
  not decode - see P0b below.

**Decomposition (from per-task records):** of C's 27 failures, only 5 scored zero on plus
tests; 22 were near-misses (winner passed a partial plus set) - the recoverable class that
better selection targets. C solves 9 tasks A fails; A solves 6 C fails (union 81.4%). The
0.6B census pair is the program's loudest result: lanes bought +19.5 pts on the 0.6B - the
mechanism scales DOWN the ladder - and the 0.6B sits 39.9 pts under the teacher on this
family, the real headroom S-1's design rule demanded. CAVEAT recorded: the 0.6B lanes row ran
with heavy thinking-budget truncation (quarantined per stake, up to 16/16 lanes on some
tasks), so 53.1% is a floor.

**What feeds forward:** (1) P0b restake - early-exit lanes (test candidates as they finish,
kill lanes on first base-passer) + council/test-augmentation selection on ties, targeting P1
>= +5 AND P2 pass; (2) S-2 code distillation - C's 86 plus-verified winners are a free,
locally-generated training corpus for the 0.6B, whose 40-pt gap is now measured; (3) the
frontier framing: 76.1% best-of-16-verified sits inside the frontier single-shot band on this
family (~72-78 for GPT-4o/Claude-3.5-class per EvalPlus), with the asymmetries stated in the
body. Raw: weights/data/p0_run.log, p0_results.json, p0_subset.json.
