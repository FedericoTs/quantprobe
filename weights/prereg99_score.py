"""Score pre-registration #99 - how much of #98's +24.0 on MATH-500 is format, not capability?

Written and committed BEFORE being run, against a prereg staked the same way. #98's verdict is
not under review here (KR-C); this decomposes the win it measured.

The fallback is lm-eval's own gsm8k `flexible-extract` filter, verbatim:

    regex_pattern: (-?[$0-9.,]{2,})|(-?[0-9]+)
    group_select: -1
    take_first

chosen because it PREDATES this question - this project already publishes every GSM8K number
through it. A rule invented after seeing a +24 could not be told apart from one tuned to shrink
it, which is the only reason provenance matters more than elegance here.

It fires ONLY where no well-formed \\boxed{} exists. Boxed items are graded by the identical
code path #98 used, so the fallback can only move an item from wrong to right, never the
reverse - a property worth stating because it makes the direction of any change unambiguous.

    python weights/prereg99_score.py
"""
from __future__ import annotations
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm_eval_tasks"))
import ev1_report as R                                        # noqa: E402
import math500_utils as M                                     # noqa: E402

# lm_eval/tasks/gsm8k/gsm8k-cot-llama.yaml, flexible-extract. Copied, not adapted (KR-A).
FLEX = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")

NAIVE, OURS = "Q35-NAIVE", "Q35-OURS"
TASK = "math500_boxed"

# Staked bands, from the prereg. Written here so the verdict is a lookup, not a judgement call.
FORMAT_DOMINATES = 12.0        # P-A: gap falls below this
CAPABILITY_DOMINATES = 20.0    # P-B: gap stays at or above this


def _flexible(response):
    """lm-eval's flexible-extract: last regex match, first non-empty group, $ and , removed."""
    hits = FLEX.findall(response)
    if not hits:
        return None
    last = hits[-1]                                   # group_select: -1
    for g in (last if isinstance(last, tuple) else (last,)):
        if g:                                         # take_first over the groups
            return g.replace("$", "").replace(",", "")
    return None


def grade(arm, task=TASK, root=None):
    """{strict, lenient, gained, n} for one arm - #98's grading plus the fallback."""
    root = root or os.path.join(R.DATA, "ev1")
    files = sorted(glob.glob(os.path.join(root, arm, task, "**", "samples_*.jsonl"),
                             recursive=True))
    if not files:
        return None
    n = strict = lenient = gained = no_box = 0
    seen = set()
    with open(files[-1], encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("doc_id") in seen:
                continue
            seen.add(d.get("doc_id"))
            resp = d.get("resps") or [[""]]
            txt = resp[0][0] if isinstance(resp[0], list) else str(resp[0])
            r = M.process_results(d["doc"], [txt])    # identical to #98
            n += 1
            strict += r["exact_match"]
            if r["emitted_boxed"]:
                lenient += r["exact_match"]           # boxed items untouched (KR-B)
                continue
            no_box += 1
            cand = _flexible(txt)
            ok = 0
            if cand is not None:
                try:
                    ok = int(bool(M.is_equiv(cand, M._gold(d["doc"]))))
                except Exception:
                    ok = 0
            lenient += ok
            gained += ok
    if not n:
        return None
    return {"n": n, "strict": strict / n, "lenient": lenient / n,
            "gained": gained, "no_box": no_box}


def score():
    a, b = grade(NAIVE), grade(OURS)
    if not a or not b:
        print("NO VERDICT - samples missing for "
              + ", ".join(x for x, v in ((NAIVE, a), (OURS, b)) if not v))
        return None

    print("prereg #99 - format vs capability in #98's MATH-500 result")
    print("  fallback: lm-eval flexible-extract, verbatim, applied ONLY where no well-formed box\n")
    print(f"  {'arm':<12} {'n':>4} {'#98 strict':>11} {'lenient':>9} {'no box':>8} {'gained':>7}")
    for name, r in ((NAIVE, a), (OURS, b)):
        print(f"  {name:<12} {r['n']:>4} {100*r['strict']:>10.1f}% {100*r['lenient']:>8.1f}% "
              f"{r['no_box']:>8} {r['gained']:>7}")

    strict_gap = 100 * (b["strict"] - a["strict"])
    lenient_gap = 100 * (b["lenient"] - a["lenient"])
    print(f"\n  gap under #98's scorer : {strict_gap:+.1f} pts")
    print(f"  gap under the fallback : {lenient_gap:+.1f} pts")
    print(f"  format's share of the original gap: "
          f"{100*(strict_gap-lenient_gap)/strict_gap:.0f}%" if strict_gap else "")

    # The bound was computable before running and is re-checked here rather than trusted.
    lo = strict_gap - 100 * a["no_box"] / a["n"]
    hi = strict_gap + 100 * b["no_box"] / b["n"]
    inside = lo - 0.05 <= lenient_gap <= hi + 0.05
    print(f"  staked bound [{lo:+.1f}, {hi:+.1f}]: "
          + ("holds" if inside else "VIOLATED - the fallback did something it cannot do"))
    if not inside:
        raise SystemExit("bound violated; the fallback is not behaving as pre-registered")

    verdict = ("P-A FORMAT DOMINATES" if lenient_gap < FORMAT_DOMINATES else
               "P-B CAPABILITY DOMINATES" if lenient_gap >= CAPABILITY_DOMINATES else
               "P-C MIXED")
    print(f"\n=== verdict === {verdict}")
    print(f"  P-A gap < {FORMAT_DOMINATES}: {lenient_gap < FORMAT_DOMINATES}")
    print(f"  P-B gap >= {CAPABILITY_DOMINATES}: {lenient_gap >= CAPABILITY_DOMINATES}")
    print(f"\n  KR-D false credit: the fallback awards points for the LAST NUMBER in a response, "
          f"which a\n  model may never have offered as its answer. It gained {a['gained']} items "
          f"for {NAIVE} and\n  {b['gained']} for {OURS}; both figures are inflated by an unknown "
          "amount and neither is a\n  clean capability measurement. #98's verdict stands (KR-C).")
    return {"verdict": verdict, "strict_gap": strict_gap, "lenient_gap": lenient_gap,
            "naive": a, "ours": b}


if __name__ == "__main__":
    score()
