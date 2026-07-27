# Pre-registration #39: does speculation COMPOSE with batching? (the free-AI question)

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurement. **Status: STAKED.**

## Why this is the highest-value untested thing left

Both multipliers are measured, and they attack the SAME denominator by different means:

- **Batching** amortises one weight read across N slots (#26: ~2.0–2.25×, saturating by 4 slots).
- **Speculation** amortises one weight read across k tokens within a slot (#36/#37: 5.0×,
  108–115 tok/s).

Nobody has run them together. If they compose, a 2016 desktop serves **~200+ tok/s aggregate** —
and "how many people can one cheap machine serve" is the question that decides whether local AI
is a hobby or an alternative to renting someone else's GPU.

There is a real mechanism for INTERFERENCE, which is why this needs measuring rather than
multiplying on paper: with continuous batching, a slot's speculative verify batch (up to 384
drafted tokens) occupies the same forward pass that other slots want to put their tokens in. If
one slot's long draft monopolises the batch, the other slots stall and batching's gain evaporates.
The MoE expert-union tax (Law 6) also grows with batch diversity: 4 slots × 8 experts can touch
far more distinct experts than 1 slot × 8.

## Arms — 2×2, one session, llama-server, split placement, `-cb`, edit task, temp 0

| arm | slots | speculation |
|---|---|---|
| A | `-np 1` | none |
| B | `-np 1` | `ngram-simple m 384 n 4` |
| C | `-np 4` | none |
| D | `-np 4` | `ngram-simple m 384 n 4` |

Concurrency is driven by 4 genuinely simultaneous HTTP requests (threads), not sequential ones.
**Aggregate** tok/s (sum over slots) AND **per-request** tok/s are both reported — quoting only the
aggregate is the half-number this project has already had to correct twice (#26).

## Stakes

- **P-1 (batching reproduces).** C ≥ 1.7 × A. Below that the harness is not actually running
  slots concurrently and everything else is void.
- **P-2 (THE HEADLINE).** D ≥ **150 tok/s aggregate**.
- **P-3 (composition, the actual question).** D/B ≥ 0.8 × (C/A) — batching's multiplier survives
  with speculation on. If D/B collapses toward 1.0, the two levers are fighting for the same
  forward pass and a server must choose one.
- **P-4 (identity).** Each slot's output in D is byte-identical to B's, at matched request index
  (per #38: compare like with like — prompt-cache state, not speculation, moves the sha).

## KILL RULE

**If D ≤ B, speculation already saturates the machine** and concurrency adds nothing on top —
the honest recommendation becomes "one speculating stream is the best this box can do", and the
free-AI ceiling for a 2016 desktop is ~110 tok/s total, not per user.

## What ships

The measured aggregate and per-request numbers, as the tool's serving guidance. If they compose,
that is the headline result of this entire line of work: the cost per user-token on a six-year-old
GPU, measured rather than extrapolated.

---

## WITHDRAWN, incomplete (2026-07-28, log: `weights/data/prereg39_spec_x_batch.log`)

**Status: not scored. Two of four arms ran before the project's priority changed from multi-user
serving to single-user quality; arms C and D (`-np 4`) were never measured.**

Recorded rather than deleted, because a staked prediction that quietly disappears is how a
register stops being trustworthy — and because the release gate caught the omission (layer 5:
"pre-registration #39 is not cited by any register entry"), which is precisely the job it was
built for.

What did run, single slot, edit task, 300 tokens:

| arm | aggregate tok/s | per-request tok/s |
|---|---|---|
| A `-np 1`, no speculation | 16.16 | 20.62 |
| B `-np 1`, speculation `m 384 n 4` | 42.79 | 99.28 |

One methodological note worth keeping: **aggregate and per-request diverge sharply** (42.79 vs
99.28 for the same run) because the aggregate divides total tokens by wall time, which includes
prompt processing, while `predicted_per_second` times generation only. Both are honest; they
answer different questions, and quoting either alone would mislead. Any future concurrency work
must report both — the same discipline #26 imposed.

**If resumed**, the open question is unchanged and still valuable: batching amortises one weight
read across slots, speculation amortises it across tokens, and whether they compose or contend for
the same forward pass is unmeasured.
