"""Score pre-registration #100 - the 4B ceiling chain: BF16 original vs OURS-Q2K vs NAIVE-Q2K.

WRITTEN AND COMMITTED BEFORE ANY CHAIN ARM HAD RUN A SINGLE BENCHMARK ITEM. The fragility probe
had run (band 24-31, KR-C6) and the children were building; no results_*.json existed for any of
the three new arms. Same discipline as prereg98_score.py, same reason: a verdict computed by
code written in ignorance of the data cannot be tuned to the data.

    python weights/prereg100_score.py

Staked outcomes (prereg #100, verbatim thresholds):

    P-C1 EXPECTED       BF16 - OURS in (0, 12] on MATH-500, AND OURS - NAIVE >= 5 on MATH-500.
    P-C2 NEAR-LOSSLESS  |BF16 - OURS| < 2 on every powered benchmark (not expected at 2 bits).
    P-C3 SIZE BINDS     BF16 - OURS > 25 on MATH-500 (recipe helps but the class is the wall).
    P-C4 VIOLATION      NAIVE >= OURS on MATH-500, or OURS > BF16 + 2 anywhere - front page.

These are NOT exhaustive: BF16-OURS in (12, 25] with OURS-NAIVE >= 5 satisfies none of the
four. That region prints as BETWEEN STAKED BANDS and is never rounded to a neighbour - the #99
lesson, now structural. P-C2 implies P-C1's first clause can overlap (a deficit in (0, 2) is
inside both); overlaps print together.

Kill rules enforced here: KR-C3 one scorer (ev1_report, re-grade on; a row that cannot be
re-graded aborts, same as #98). KR-C4 a KR-E3-degraded benchmark (>10% empty or truncated in
ANY arm) leaves the verdict for ALL arms. KR-C5 no partial verdicts - every powered row of all
three new arms, or nothing. KR-C1 byte premium is read from the files at score time and printed
in the header of every table; >5% flips the OURS/NAIVE comparison to SIZE-CONFOUNDED.

The Q4_K_M reference rows (EV-1, older conversion with the inert MTP head) print as context,
clearly labeled, and enter no staked outcome.
"""
from __future__ import annotations
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ev1_report as R                                        # noqa: E402
import ev1_score as S                                         # noqa: E402

BF16, OURS, NAIVE = "4B-BF16", "4B-OURS", "4B-NAIVE"
REF_Q4 = "4B"                                                  # EV-1 rows, context only
POWERED = ["math500_boxed", "gsm8k_cot_zeroshot", "ifeval"]
PRIMARY = "math500_boxed"

FILES = {BF16:  "D:/evo-compress-data/gguf/Qwen3.5-4B-BF16.gguf",
         OURS:  "D:/evo-compress-data/gguf/Qwen3.5-4B-ours-depthaware.gguf",
         NAIVE: "D:/evo-compress-data/gguf/Qwen3.5-4B-naive-q2k.gguf"}

EXPECTED_MAX = 12.0     # P-C1 upper edge of the BF16-OURS deficit
GUARD = 5.0             # P-C1 minimum OURS-NAIVE margin
LOSSLESS = 2.0          # P-C2 band, absolute
WALL = 25.0             # P-C3 threshold
NOISE = 2.0             # P-C4 "beats BF16 by more than noise"
CONFOUND = 5.0          # KR-C1 byte-premium ceiling, percent


def _pct(rows, model, task):
    d = rows.get((model, task))
    if not d:
        return None
    v = d.get(R.REPORTED.get(task, (None, None))[0])
    return None if v is None else 100.0 * v


def byte_premium():
    """OURS vs NAIVE size premium in percent, or None while a file is missing."""
    try:
        a, b = os.path.getsize(FILES[OURS]), os.path.getsize(FILES[NAIVE])
    except OSError:
        return None
    return 100.0 * (a - b) / b


def degraded(tasks, health=None):
    """KR-C4: {task: [reasons]} pooled across ALL THREE arms; unknown health is not healthy."""
    health = health or S.response_health
    out = {}
    for task in tasks:
        for arm in (BF16, OURS, NAIVE):
            h = health(arm, task)
            if h is None:
                out.setdefault(task, []).append(f"{arm}: samples unavailable, health unknown")
                continue
            n, empty, trunc = h
            if n and (empty + trunc) / n > S.DEGRADED_AT:
                out.setdefault(task, []).append(
                    f"{arm}: {100 * (empty + trunc) / n:.1f}% unusable")
    return out


def score(rows=None, health=None):
    rows = rows if rows is not None else R.load_rows()

    prem = byte_premium()
    print("prereg #100 - the ceiling chain: BF16 original vs recipe vs naive, one parent file")
    if prem is None:
        print("  KR-C1: byte premium unavailable (a child file is missing)")
    else:
        tag = "SIZE-CONFOUNDED - exceeds the 5% ceiling" if abs(prem) > CONFOUND else "disclosed"
        print(f"  KR-C1: OURS carries {prem:+.2f}% bytes vs NAIVE ({tag})")

    unregraded = [f"{a}/{t}" for t in POWERED for a in (BF16, OURS, NAIVE)
                  if rows.get((a, t), {}).get("_rescore_unavailable")]
    if unregraded:
        print("  KR-C3 ABORT: " + ", ".join(unregraded) + " carry a different scorer than "
              "their partners.\n")

    missing = [(a, t) for t in POWERED for a in (BF16, OURS, NAIVE) if _pct(rows, a, t) is None]
    dropped = degraded(POWERED, health) if not missing else {}

    print(f"\n  {'benchmark':<22} {'BF16':>7} {'OURS':>7} {'NAIVE':>7} {'Q4ref*':>7}  status")
    d_bo, d_on = {}, {}
    for task in POWERED:
        b, o, n = (_pct(rows, a, task) for a in (BF16, OURS, NAIVE))
        q = _pct(rows, REF_Q4, task)
        cells = [f"{v:>6.1f}%" if v is not None else f"{'-':>7}" for v in (b, o, n, q)]
        if any(v is None for v in (b, o, n)):
            status = "PENDING"
        elif task in dropped:
            status = "KR-C4 DROPPED: " + "; ".join(dropped[task])
        else:
            status = "counts"
            d_bo[task] = b - o
            d_on[task] = o - n
        print(f"  {task:<22} " + " ".join(cells) + f"  {status}")
    print("  * Q4ref: EV-1 rows on the older conversion (inert MTP head) - context, no verdict")

    if missing or unregraded:
        print("\n=== NO VERDICT ===")
        if missing:
            print("  KR-C5: not every arm holds every powered row. Missing: "
                  + ", ".join(f"{a}/{t}" for a, t in missing))
        if unregraded:
            print("  KR-C3: " + ", ".join(unregraded))
        return {"verdict": None, "missing": missing, "unregraded": unregraded,
                "dropped": dropped, "premium": prem}

    if PRIMARY in dropped or not d_bo:
        print("\n=== NO VERDICT - KR-C4 ===")
        print("  The primary benchmark (or every benchmark) was dropped as degraded.")
        return {"verdict": None, "missing": [], "unregraded": [], "dropped": dropped,
                "premium": prem}

    bo, on = d_bo[PRIMARY], d_on[PRIMARY]
    p_c1 = (0 < bo <= EXPECTED_MAX) and (on >= GUARD)
    p_c2 = all(abs(v) < LOSSLESS for v in d_bo.values())
    p_c3 = bo > WALL
    p_c4 = (on <= 0) or any(-v > NOISE for v in d_bo.values())

    print("\n=== verdict ===")
    print(f"  BF16 - OURS on {PRIMARY}: {bo:+.1f} pts   OURS - NAIVE: {on:+.1f} pts")
    print(f"  P-C1 EXPECTED       {'CONFIRMED' if p_c1 else 'not met'}"
          f"  (deficit in (0, {EXPECTED_MAX}]: {0 < bo <= EXPECTED_MAX}; margin >= {GUARD}: {on >= GUARD})")
    print(f"  P-C2 NEAR-LOSSLESS  {'CONFIRMED' if p_c2 else 'not met'}"
          f"  (all |BF16-OURS| < {LOSSLESS}: {p_c2})")
    print(f"  P-C3 SIZE BINDS     {'CONFIRMED' if p_c3 else 'not met'}  (deficit > {WALL}: {p_c3})")
    print(f"  P-C4 VIOLATION      {'CONFIRMED' if p_c4 else 'not met'}"
          f"  (NAIVE >= OURS: {on <= 0}; OURS > BF16 + {NOISE} anywhere: "
          f"{any(-v > NOISE for v in d_bo.values())})")

    hits = [n for n, ok in (("P-C1", p_c1), ("P-C2", p_c2), ("P-C3", p_c3), ("P-C4", p_c4)) if ok]
    if not hits:
        print("\n  BETWEEN STAKED BANDS - the outcome satisfies none of the four staked "
              "predictions.\n  Report the numbers as they are and amend the prereg to say the "
              "region was unanticipated.\n  Do not round to the nearest band.")
    elif len(hits) > 1:
        print(f"\n  OVERLAP - {' and '.join(hits)} hold together; all are reported.")
    return {"verdict": hits, "d_bf16_ours": d_bo, "d_ours_naive": d_on,
            "dropped": dropped, "premium": prem}


if __name__ == "__main__":
    score()
