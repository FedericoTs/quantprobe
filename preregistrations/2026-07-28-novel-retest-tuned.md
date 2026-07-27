# Pre-registration #41: does the TUNED drafter reach novel content? (the untested combination)

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurement. **Status: STAKED.**

## The gap this closes

#28 and #30 measured novel generation at **0% draft acceptance** and closed that line by kill
rule. Both were run at llama.cpp's **defaults: `size-n 12`, `size-m 48`**. #37 then found that
`size-n 4` — a three-times shorter required match — is worth +21% on copy-regime work.

**The combination "tuned drafter × novel content" has never been measured.** A 4-token lookup
matches vastly more often than a 12-token one, and novel prose/code still contains repeated
short spans (`    return `, `self.`, ` the `, indentation runs). D-10 may be an artifact of the
defaults rather than a property of novel generation, and if so this project closed a line early.

## Arms (llama-server, split, Q2_K, temp 0, fresh server, request 1 only per #38)

Two novel tasks — nothing to copy from context:
- **NC** novel code: "write a Python function `schedule(jobs)` … maximum-weight non-overlapping
  jobs by dynamic programming, with a docstring."
- **NP** novel prose: "explain in plain English why reading a file from disk is slower than from
  memory, about 200 words."

| arm | flags |
|---|---|
| base | no speculation |
| def | `ngram-simple` at defaults (m 48, n 12) — reproduces the #28/#30 null |
| tuned | `ngram-simple m 384 n 4` — the untested cell |

## Stakes

- **P-1 (the tuned drafter fires at all on novel content).** `tuned` drafts **> 50 tokens** on at
  least one novel task, against `def`'s ~0. This is the claim that D-10 was defaults-scoped.
- **P-2 (but it does not pay).** `tuned` is within **±10%** of `base` on both novel tasks. I
  expect firing without profit: short matches produce short runs, and #37 showed short runs are
  what makes n=2 lose. **If this is exceeded upward, D-10 must be reopened and novel generation
  is not closed after all.**
- **P-3 (identity).** All arms byte-identical at matched request index.

## Refuted / reopened if

**P-2 exceeded upward (tuned > base by >10% on a novel task).** Then novel-generation speculation
is real at tuned settings, D-10 was closed on defaults-scoped evidence, and the headline extends
from "copy-regime only" to "all content, with a smaller multiplier on novel" — a materially
different claim for users, and one this project would have to publish as a correction of its own
prior conclusion.

## What ships

Either the confirmation that D-10 holds at tuned settings (tightening a claim we already make), or
its reopening with measured numbers. The plan output's "novel generation gains nothing" sentence
is downstream of this either way.
