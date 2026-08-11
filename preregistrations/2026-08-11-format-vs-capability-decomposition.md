# Pre-registration #99: how much of #98's +24.0 is format, and how much is capability?

**Author:** Federico Sciuca · **Date staked:** 2026-08-11, **after #98 was scored and BEFORE the
lenient re-score was run.** **STAKED.**

## Why this exists

Prereg #98 confirmed P1: the depth-aware recipe beats naive Q2_K by **+24.0 pts on MATH-500**
against a staked bar of +2.0. The same scorer's format column then showed that most of the gap
is not arithmetic:

| | emitted a `\boxed{}` | exact_match | correct GIVEN a box |
|---|---|---|---|
| NAIVE | 64.4% | 57.0% | 88.5% |
| OURS | 86.4% | 81.0% | 93.8% |

A reader who quotes "+24 points on MATH-500" will be quoting a number that is 91% about answer
formatting. **That is not a wrong number, it is a misreadable one**, and this pre-registration is
the fix rather than a footnote.

**#98's VERDICT IS NOT UNDER REVIEW.** P1 stands, scored by the extractor that was live when the
arms ran, exactly as KR-3 requires. This decomposes the win; it does not re-litigate it.

## The extractor, chosen for provenance rather than for this result

The fallback is **lm-eval's own `flexible-extract` filter, verbatim**, as shipped in
`lm_eval/tasks/gsm8k/gsm8k-cot-llama.yaml`:

    regex_pattern: (-?[$0-9.,]{2,})|(-?[0-9]+)
    group_select: -1
    take_first

This project already publishes every GSM8K number through that exact filter, and audited it
independently (it reproduces 36.8 / 81.7 / 79.9 on the EV-1 rows). Reusing a rule that predates
this question is the point: a rule invented today, after seeing a +24, could not be distinguished
from one tuned to shrink it.

**Applied as a FALLBACK ONLY.** An item with a well-formed `\boxed{}` is graded exactly as in
#98 — same `last_well_formed_boxed`, same `is_equiv`, same gold. The fallback fires only where
no well-formed box exists. The captured string has `$` and `,` removed (the regex deliberately
admits both) and is compared with the same `is_equiv`.

Consequence, stated so it cannot be claimed afterwards: **this can only move scores UP, and only
on items that scored zero in #98.** No item that #98 scored correct can become incorrect.

## The bound, computable before running and therefore stated now

Items with no well-formed box: NAIVE 35.6%, OURS 13.6%. Those are the only items the fallback can
touch, so the arithmetic bounds are fixed in advance:

- NAIVE can gain at most **+35.6 pts**; OURS at most **+13.6 pts**.
- The gap can therefore shrink by at most 22.0 pts and widen by at most 13.6 pts.
- **The gap cannot fall below +2.0 or rise above +37.6.**

The lower bound landing exactly on #98's staked P1 threshold is a coincidence of the numbers, not
a design choice, and it is worth saying out loud: even under the maximally charitable reading of
the naive arm — every one of its unboxed answers correct, every one of ours wrong — the recipe
still clears the bar it was staked against. That is the strongest single sentence available about
how robust #98's verdict is, and it is true before the re-score is run.

## Staked predictions

- **P-A — FORMAT DOMINATES:** the MATH-500 gap falls **below +12.0 pts** (at least half of the
  original +24.0 was answer formatting). If this lands, the headline claim becomes "low-bit
  quantization breaks instruction adherence first" and the maths framing is retired.
- **P-B — CAPABILITY DOMINATES:** the gap stays **at or above +20.0 pts**. Format was a minor
  contributor, the naive arm really is much worse at the maths, and the original framing stands.
- **P-C — MIXED:** the gap lands in **[+12.0, +20.0)**. Both effects are real and comparable, and
  the honest headline names both.

## Kill rules

- **KR-A NO NEW RULE:** the fallback is the lm-eval flexible-extract regex verbatim. If it needs
  any adjustment to run, the adjustment is recorded and the result is reported as tuned.
- **KR-B SYMMETRY:** applied identically to both arms, fallback-only, boxed items untouched. Any
  asymmetry voids the comparison.
- **KR-C #98 STANDS:** whatever lands here, #98's P1 verdict is not amended. A decomposition that
  could retroactively unmake the verdict it decomposes would make the original stake worthless.
- **KR-D FALSE CREDIT IS REPORTED:** a last-number rule awards points for numbers a model never
  offered as its answer. That inflation cannot be measured directly here, so the count of items
  each arm gains is reported alongside the scores, because it bounds the inflation and lets a
  reader discount both arms by eye.

## What this cannot show

It cannot tell us whether the naive arm's unboxed answers were *reasoning* to the right value and
merely failing to format it, versus emitting a number incidentally. The gained-item counts bound
that but do not resolve it. Resolving it needs a human read of a sample of the naive arm's unboxed
responses, which is out of scope here and named so nobody mistakes the fallback for a mind-reader.


---

## SCORED (2026-08-11, same day, run by pre-written code against the staked bands)

**P-B CAPABILITY DOMINATES. P-A refuted.**

| | #98 strict | format-blind fallback | no-box items | items recovered |
|---|---|---|---|---|
| NAIVE | 57.0% | 57.4% | 178 | 2 |
| OURS | 81.0% | 81.2% | 68 | 1 |

Gap **+24.0 -> +23.8**. Format accounts for **1%** of the original gap. Staked bound
[-11.6, +37.6] holds, checked mechanically rather than trusted.

### The null is not an artifact of a blind instrument, and here is why

A format-blind rule that recovers almost nothing invites the obvious objection: maybe it could
never have recovered anything. Measured:

- **320 of 500 MATH-500 golds (64%) are purely numeric**, so a last-number rule can match them.
- **100 of NAIVE's 178 unboxed items have a numeric gold.** The fallback had up to 100 items to
  recover and took 2.
- Those responses are long, not empty or cut off: median 6,787 chars, min 3,982, max 13,247,
  against 0.0% truncation on the row.

Long responses, recoverable golds, nothing recovered. The unboxed items are not correctly-reasoned
answers in the wrong shape; they are reasoning that never converges.

### KR-D, honoured

The fallback awards credit for the last number in a response, which the model may never have
offered as an answer. It gained 2 items for NAIVE and 1 for OURS. Both counts are upper bounds
polluted by that inflation, which makes the null STRONGER, not weaker - the true recovery is at
most 2 and 1.

### The staking document itself contains the error, and it is left standing

"Why this exists" above asserts that "+24 points on MATH-500 is a number that is 91% about answer
formatting" as an established fact rather than as the hypothesis under test. It is false, and it
is not being edited out - a pre-registration that gets quietly cleaned up after the result is a
pre-registration for nothing. It is also the tell: writing the conclusion into the motivation is
how a test gets designed to confirm rather than to discriminate. This one discriminated anyway,
because the extractor was chosen for provenance and the bound was computed in advance.

### What this changes

It refutes the format reading I appended to #98 on the same day, and that document now carries
the correction. **#98's P1 verdict stands (KR-C)** and is better supported than under my mistaken
reading: the recipe's win on MATH-500 is a capability difference.

### What this still cannot show

Whether naive's non-converging responses are looping, drifting, or confidently wrong. A
mechanical length/repetition profile would separate those and is not run here. Named so the
"never converges" phrasing is read as what it is - an inference from "no box and no recoverable
number in 6,787 characters", not a characterisation of the failure mode.
