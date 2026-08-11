"""Score pre-registration #98 - depth-aware quantization recipe vs the naive default command.

WRITTEN BEFORE ANY SCORE WAS READ. At the time this file was committed exactly one arm-row
existed on disk (Q35-NAIVE/math500_boxed) and its number had not been opened. That ordering is
the whole point: a verdict computed by code written in ignorance of the data cannot be tuned to
the data. If you are reading this after the fact, `git log --diff-filter=A` on this file against
the mtimes under data/ev1/Q35-* is the check.

    python weights/prereg98_score.py

The prereg staked three outcomes on the POWERED THREE only - MATH-500, GSM8K, IFEval. AIME is
n=30 with ~7pp standard error and is excluded from the verdict by the prereg itself; it is
printed here for completeness and never enters P1/P2/P3.

    P1 RECIPE PAYS      ours - naive >= +2.0 pts on MATH-500, AND not worse than -1.0 pt on
                        either GSM8K or IFEval.
    P2 RECIPE HURTS     naive beats ours by >= 2.0 pts on MATH-500, OR ours loses by more than
                        1.0 pt on two or more of the powered three.
    P3 NULL             every powered benchmark differs by < 2.0 pts in absolute value.

These three are NOT exhaustive and NOT mutually exclusive, and this scorer says so instead of
forcing the data into one of them:

  * UNCLASSIFIED is possible. MATH +2.5 with GSM8K -1.5 satisfies none of the three - P1 fails
    on the guard, P2 fails both clauses, P3 fails on MATH. If that happens the honest report is
    that the prereg's outcome space had a hole, not that the nearest prediction "basically" hit.
  * P2 and P3 can BOTH hold. MATH -1.5 / GSM8K -1.5 / IFEval 0.0 is inside P3's null band while
    tripping P2's two-benchmark clause. Both are printed; neither is silently dropped.

Kill rules enforced mechanically here:

  KR-3 SAME SCORER   both arms come through ev1_report.load_rows with re-grading ON, so the
                     C-26 extractor fix applies identically to both or to neither.
  KR-4 DEGRADED      if EITHER arm trips KR-E3 (>10% empty or truncated) on a benchmark, that
                     benchmark leaves the verdict for BOTH arms - not just for the arm that
                     tripped it, which would silently select the friendlier half of the pair.
  KR-5 NO PEEKING    the verdict block refuses to print until both arms hold all three powered
                     benchmarks. Partial pairs print as "pending", never as a partial verdict.

KR-1 already FAILED and is reported as a standing banner rather than a footnote: the arms are
not byte-matched (ours is +2.51%), so any win ours posts is "the recipe plus 2.5% more bytes".
The direction is unfavourable to us, which is why the comparison is still worth reporting.
"""
from __future__ import annotations
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ev1_report as R                                        # noqa: E402
import ev1_score as S                                         # noqa: E402

NAIVE, OURS = "Q35-NAIVE", "Q35-OURS"
POWERED = ["math500_boxed", "gsm8k_cot_zeroshot", "ifeval"]    # verdict-bearing, per the prereg
REPORTED_ONLY = ["aime24_boxed", "aime25_boxed"]               # n=30, excluded by power argument
PRIMARY = "math500_boxed"

WIN = 2.0        # P1/P2 threshold on the primary, in percentage points
GUARD = 1.0      # "not worse by more than" on the secondaries
NULL = 2.0       # P3 band, absolute

BYTES = {OURS: 13_272_701_568, NAIVE: 12_939_594_368}          # measured at build, pre-eval


def _pct(rows, model, task):
    """The reported metric for a row, in percentage points, or None if the row is absent."""
    d = rows.get((model, task))
    if not d:
        return None
    metric = R.REPORTED.get(task, (None, None))[0]
    v = d.get(metric)
    return None if v is None else 100.0 * v


def degraded_benchmarks(tasks, health=None):
    """{task: [reasons]} for benchmarks KR-4 removes from the verdict.

    Checked for BOTH arms and pooled deliberately: KR-4 drops the benchmark for the pair, so
    one degraded arm is enough to remove it. A row whose samples are missing is not treated as
    healthy - it is reported as unknown, because assuming health is how a degraded row sneaks
    into a headline.

    `health` is injectable so the scoring logic can be exercised without reading the live run's
    sample files - the tests must not do disk work on a box that is mid-measurement (C-14).
    """
    health = health or S.response_health
    out = {}
    for task in tasks:
        for arm in (NAIVE, OURS):
            h = health(arm, task)
            if h is None:
                out.setdefault(task, []).append(f"{arm}: samples unavailable, health unknown")
                continue
            n, empty, trunc = h
            if n and (empty + trunc) / n > S.DEGRADED_AT:
                out.setdefault(task, []).append(
                    f"{arm}: {100 * (empty + trunc) / n:.1f}% unusable ({empty + trunc} of {n})")
    return out


def score(rows=None, health=None):
    rows = rows if rows is not None else R.load_rows()

    prem = 100.0 * (BYTES[OURS] - BYTES[NAIVE]) / BYTES[NAIVE]
    print("prereg #98 - depth-aware recipe vs naive Q2_K, same source, same box, same harness")
    print(f"  KR-1 FAILED and stands: ours is {prem:+.2f}% bytes vs naive "
          f"({BYTES[OURS]:,} vs {BYTES[NAIVE]:,}).")
    print("  Read every ours-win as 'the recipe PLUS 2.5% more bytes'. An ours-loss is "
          "correspondingly stronger.\n")

    # KR-3 has a failure mode the kill rule's wording does not spell out: if one arm's samples
    # are missing, load_rows leaves that row on the RAW lm-eval verdict while the other arm gets
    # the C-26 re-grade. Comparing those two numbers is comparing two different scorers, which is
    # exactly what KR-3 forbids - so it is an abort, not a caveat.
    unregraded = [f"{a}/{t}" for t in POWERED + REPORTED_ONLY for a in (NAIVE, OURS)
                  if rows.get((a, t), {}).get("_rescore_unavailable")]
    if unregraded:
        print("  KR-3 ABORT: re-grading unavailable for " + ", ".join(unregraded)
              + " - that arm carries a\n  different scorer than its partner. Restore the samples "
              "or re-run the row; do not compare.\n")

    missing = [(a, t) for t in POWERED for a in (NAIVE, OURS) if _pct(rows, a, t) is None]
    dropped = degraded_benchmarks(POWERED, health) if not missing else {}
    live = [t for t in POWERED if t not in dropped]

    print(f"  {'benchmark':<22} {'naive':>8} {'ours':>8} {'delta pp':>10}  status")
    deltas = {}
    for task in POWERED + REPORTED_ONLY:
        a, b = _pct(rows, NAIVE, task), _pct(rows, OURS, task)
        if a is None or b is None:
            have = [n for n, v in ((NAIVE, a), (OURS, b)) if v is not None]
            print(f"  {task:<22} {'-':>8} {'-':>8} {'-':>10}  PENDING"
                  + (f" (have {'/'.join(have)})" if have else ""))
            continue
        d = b - a
        if task in REPORTED_ONLY:
            status = "reported only - EXCLUDED from verdict (n=30, ~7pp stderr)"
        elif task in dropped:
            status = "KR-4 DROPPED for both arms: " + "; ".join(dropped[task])
        else:
            status = "counts"
            deltas[task] = d
        print(f"  {task:<22} {a:>7.1f}% {b:>7.1f}% {d:>+9.1f}   {status}")

    # A boxed-task delta mixes "got it wrong" with "never produced a gradeable answer", and the
    # two are different claims. This column separates them and prints WITH the number, always.
    #
    # It is here because I read it wrong the first time. Seeing NAIVE at 64.4% emitted-box against
    # OURS at 86.4%, I concluded most of the +24.0 was answer FORMATTING and said so in the #98
    # scored section and in d7910e6. Prereg #99 staked the test and refuted it: re-graded with a
    # format-blind last-number rule the gap moves +24.0 -> +23.8, and only 2 of NAIVE's 178
    # unboxed responses held the right answer anywhere - out of 100 whose gold a number rule
    # COULD have matched. The missing box is not a lost format, it is a lost answer.
    #
    # So the column stays, with the opposite reading: a low emitted-box rate is evidence of
    # capability collapse, not of a scorer being fussy. See prereg #99.
    boxed = [t for t in POWERED + REPORTED_ONLY if t in R.BOXED_TASKS
             and _pct(rows, NAIVE, t) is not None and _pct(rows, OURS, t) is not None]
    if boxed:
        print("\n  FORMAT DECOMPOSITION - a boxed delta measures formatting as well as accuracy")
        print(f"  {'row':<28} {'emitted box':>12} {'exact':>8} {'correct GIVEN a box':>21}")
        for task in boxed:
            for arm in (NAIVE, OURS):
                d = rows[(arm, task)]
                em = d.get("emitted_boxed,none")
                ex = d.get(R.REPORTED[task][0])
                cond = (100 * ex / em) if em else float("nan")
                print(f"  {arm + '/' + task:<28} {100*em:>11.1f}% {100*ex:>7.1f}% {cond:>20.1f}%")

    if missing or unregraded:
        print("\n=== NO VERDICT ===")
        if missing:
            print("  KR-5: the arms have not both run to completion on the powered three. "
                  "Missing: " + ", ".join(f"{a}/{t}" for a, t in missing))
            print("  Partial results are not consulted, and a partial verdict is not printed.")
        if unregraded:
            print("  KR-3: " + ", ".join(unregraded) + " could not be re-graded.")
        return {"verdict": None, "missing": missing, "unregraded": unregraded,
                "deltas": deltas, "dropped": dropped}

    if not live:
        print("\n=== NO VERDICT - KR-4 ===")
        print("  Every powered benchmark was dropped as degraded. The experiment measured "
              "nothing about the recipe.")
        return {"verdict": None, "missing": [], "deltas": deltas, "dropped": dropped}

    primary = deltas.get(PRIMARY)
    losses = [t for t, d in deltas.items() if d < -GUARD]

    if primary is None:
        p1 = p2_a = False
        p2_b = len(losses) >= 2
        p3 = all(abs(d) < NULL for d in deltas.values())
        print(f"\n  NOTE: {PRIMARY} was dropped, so the primary discriminator is gone. P1 cannot "
              "be met\n  at all and P2 survives only through its two-benchmark clause.")
    else:
        p1 = primary >= WIN and not losses
        p2_a = primary <= -WIN
        p2_b = len(losses) >= 2
        p3 = all(abs(d) < NULL for d in deltas.values())
    p2 = p2_a or p2_b

    print("\n=== verdict ===")
    print(f"  benchmarks in the verdict: {', '.join(live)}"
          + (f"  (dropped: {', '.join(dropped)})" if dropped else ""))
    if primary is None:
        why = "primary benchmark dropped"
    else:
        why = (f"primary {primary:+.1f} vs +{WIN}, "
               + ("no benchmark worse than -1.0" if not losses
                  else "worse than -1.0 on " + ", ".join(losses)))
    print(f"  P1 RECIPE PAYS   {'CONFIRMED' if p1 else 'not met':<10} ({why})")
    print(f"  P2 RECIPE HURTS  {'CONFIRMED' if p2 else 'not met':<10} "
          f"(primary <= -{WIN}: {p2_a}; two or more losses > {GUARD} pt: {p2_b} {losses or ''})")
    print(f"  P3 NULL          {'CONFIRMED' if p3 else 'not met':<10} "
          f"(all |delta| < {NULL} pts across {len(deltas)} scored benchmark(s))")

    hits = [n for n, ok in (("P1", p1), ("P2", p2), ("P3", p3)) if ok]
    if not hits:
        print("\n  UNCLASSIFIED - the data satisfies none of the three staked outcomes. That is a "
              "hole in\n  the prereg's outcome space, and it is reported as one. Do NOT round it "
              "to the nearest\n  prediction; amend the prereg with the outcome that actually "
              "occurred and say it was\n  unanticipated.")
    elif len(hits) > 1:
        print(f"\n  OVERLAP - {' and '.join(hits)} both hold. The prereg's outcomes were not "
              "written to be\n  mutually exclusive; both are reported rather than one being "
              "chosen after the fact.")
    return {"verdict": hits, "missing": [], "deltas": deltas, "dropped": dropped,
            "p1": p1, "p2": p2, "p3": p3}


if __name__ == "__main__":
    score()
