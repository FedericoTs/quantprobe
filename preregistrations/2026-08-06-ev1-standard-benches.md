# EV-1 — the most-adopted benchmarks, scored by the most-adopted harness, on this box

**Date staked:** 2026-08-06, before any EV-1 item was measured. Extends the capability table
(prereg 2026-08-05 Part 2) with standard suites; the sanctity law applies - every bench here
is eval-only forever and enters the decontamination protected set.

## The harness decision, recorded

Singles run through **lm-evaluation-harness v0.4.12** (pinned) - the community-standard
scorer - pointed at OUR llama-server placements (quantprobe-planned, same models as the
grid). Their canon, our provenance: locks, one state per row, logs committed. Rationale: for
adopted benchmarks, comparability IS the product; a home-grown scorer would cost trust
exactly where trust is the point. Strategy columns (majority-vote, constraint-verified) come
later via our harness on the same items (EV-1b), where lm-eval has no equivalent.

## Tasks (lm-eval names, pinned) and models

- `gsm8k_cot_zeroshot` (1,319) - `ifeval` (541) - `hendrycks_math500` (500) -
  `aime24` + `aime25` (~60) - `gpqa_main_zeroshot` (448; GATED dataset - if access is not
  granted the row is a published precondition-block, not a quiet skip) - `mmlu` subset
  deferred to night 2+.
- Models: 0.6B, 4B, 7B, 30B at their established placements. Night 1: 0.6B + 4B on
  math/ifeval/aime. Night 2: 7B + 30B + gsm8k completions + gpqa attempt.

## Protocol notes, stated before results exist

- **Thinking-as-served:** lm-eval drives the chat endpoint; per-request template kwargs are
  not available, so Qwen3-family rows run with thinking ON - the opposite of the grid's
  protocol. Stated here, printed in results; scores are NOT comparable to grid cells, only
  to public numbers for the same models (which also run as-served).
- `max_gen_toks` 2048 on generative tasks (256 default would truncate CoT); temperature 0;
  seeds and full lm-eval configs committed with results.
- Full sets, no sampling - public comparability is the entire point.

## Staked expectations and kill rules

- **P-E1 (sanity vs public numbers):** each model's GSM8K/MATH-500/IFEval single lands within
  **+/-10 pts** of its publicly reported number for the same task where one exists (quant +
  harness variance). A larger gap = investigate placement/truncation BEFORE believing either.
- **P-E2 (the ladder holds):** 0.6B < 4B and 7B <= 30B on every completed task (the P-A3
  4B-vs-7B question gets its second bench family).
- **KR-E1:** scorer is lm-eval as pinned - no post-hoc metric edits; result JSONs committed
  raw. **KR-E2:** one model row per server session, shared locks, GPU state logged.
  **KR-E3:** any row with >10% empty/truncated responses is DEGRADED and marked.

Raw under `weights/data/ev1_*`. Verdicts + the table/media updates as rows land.

---

## AMENDMENT (protocol v2, 2026-08-06, before any v2 row ran)

Night-1 v1 exposed the thinking-as-served protocol's failure mode within five rows:
llama-server routes thought into `reasoning_content`, which lm-eval never reads - so
thinking-family models burned their token budgets INVISIBLY and the scored answers were
truncated stumps (MATH-500 samples end mid-sentence at ~200 visible tokens). KR-E3 applies:
every v1 row is DEGRADED, archived under `weights/data/ev1_protocol_v1_degraded/`, and
published only as floors with this mechanism attached. IFEval's separate failure was a
missing `langdetect` dependency - installed.

**Protocol v2, staked before re-running:** thinking-family servers run `--reasoning off`
(b10098's server-side switch) - budgets stay 2048, and EV-1 becomes protocol-COHERENT with
the grid (both thinking-off), restoring cross-table comparability that v1 had explicitly
given up. All night-1 rows re-run under v2; P-E1/P-E2 score against v2 numbers only.
