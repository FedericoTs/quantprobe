# Pre-registration #37: raising draft AVAILABILITY — the constraint #36 left standing

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurement. **Status: STAKED.**

## The constraint, precisely

#36 took speculation from 49.80 to **90.33 tok/s** by raising the draft budget, and identified
where it stops: at `size-m` ≥ 192 the drafted/accepted counts FREEZE (1100/735) — the n-gram
store has no longer matching spans. Meanwhile the batched verify forward sustains ~405–445 tok/s
(#26). So we sit at ~1/4.5 of speculation's own ceiling, bounded by **draft availability**.

Two untried levers attack availability directly:

- **`--spec-ngram-simple-size-n`** (lookup length, default 12): the drafter must match n tokens of
  recent context before it will propose anything. A *shorter* lookup matches more often, so more
  decode steps get a draft at all. It should also produce *worse* drafts — and #36 established
  that acceptance rate is the wrong target, so worse-but-more may still win.
- **Drafter stacking**: `--spec-type` takes a comma-separated list. If one drafter running dry is
  the constraint, a second may fire on the steps the first misses.

## Arms (all at the #36 winner `size-m 384`; edit task, split, f16 KV, temp 0, r=2)

| arm | setting |
|---|---|
| ref | `ngram-simple`, size-n 12 (the #36 winner: 90.33) |
| N6 | size-n 6 |
| N4 | size-n 4 |
| N20 | size-n 20 (the control — availability should FALL) |
| ST | `--spec-type ngram-simple,ngram-cache` |

## Stakes

- **P-1 (availability is the constraint, and n controls it).** N6 drafts **≥30% more tokens**
  than ref (from 1100). Rejected drafts being cheap, more drafting should mean fewer verify
  rounds for the same output.
- **P-2 (the headline).** Some arm beats **104 tok/s** (≥15% over 90.33).
- **P-3 (the control fires the right way).** N20 drafts **fewer** tokens than ref. If a *longer*
  required match somehow increases availability, I do not understand this drafter and P-1/P-2 are
  uninterpretable.
- **P-4 (identity holds).** Every arm byte-identical to ref. Speculation is output-preserving by
  construction; if changing the drafter changes the output, that is a bug in llama.cpp worth
  reporting, and every speculation number in this project needs re-examination.

## KILL RULE — stated before measuring

**If nothing beats 90.33 by more than 5%, drafter tuning is closed.** The n-gram store is then at
its practical ceiling for this workload, the remaining 4.5× to the verify ceiling belongs to a
fundamentally better drafter (a model, an indexed corpus), and #28's finding that a 0.6B draft
model is net-negative here says that road is expensive. We would stop and ship the m=384 result.

## What ships

Any winning combination goes into `speculation_advice` with its measured numbers and its
copy-regime scope, exactly like the m=384 flag. No law changes; these are flags.
