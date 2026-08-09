"""Score the EV-1 pre-registration against the rows on disk - mechanically, not by eye.

prereg 2026-08-06-ev1-standard-benches staked three things. Two of them are checkable from
the data alone and are implemented here; the third needs numbers from outside this repo and
says so rather than being quietly skipped.

    P-E2   the ladder holds: 0.6B < 4B, and 7B <= 30B, on every COMPLETED task.
    KR-E3  any row with >10% empty or truncated responses is DEGRADED and marked.
    P-E1   each model lands within +/-10 pts of its publicly reported number for the same
           task. NOT scoreable here - it needs published figures, which do not live in this
           repo and must not be typed in from memory. Left explicit.

KR-E3 had never been evaluated. It is a kill rule with teeth: a row that generated nothing
useful can still produce a plausible-looking percentage, and "the model scored 20%" reads very
differently from "the model returned an empty string 40% of the time and scored 20% on the
rest". Truncation is detected as generation stopped at the budget without an end-of-answer,
which is the same shape that hid C-26.

    python weights/ev1_score.py
"""
from __future__ import annotations
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ev1_report as R                                       # noqa: E402

DEGRADED_AT = 0.10               # KR-E3 threshold, as staked
LADDER = [("0.6B", "4B", "<"), ("7B", "30B", "<=")]          # P-E2, as staked


def response_health(model, task, root=None):
    """(n, empty, truncated) for a row, read from the logged responses."""
    root = root or os.path.join(R.DATA, "ev1")
    fs = sorted(glob.glob(os.path.join(root, model, task, "**", "samples_*.jsonl"),
                          recursive=True))
    if not fs:
        return None
    n = empty = trunc = 0
    seen = set()
    with open(fs[-1], encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("doc_id") in seen:
                continue
            seen.add(d.get("doc_id"))
            n += 1
            r = d.get("resps") or [[""]]
            txt = r[0][0] if isinstance(r[0], list) else str(r[0])
            if not txt.strip():
                empty += 1
                continue
            # Truncated = ran to the cap with no terminal punctuation and no closing brace.
            # Deliberately conservative: a long answer that ends in a full stop is not
            # truncated no matter how long it is.
            tail = txt.rstrip()[-1:] if txt.strip() else ""
            if len(txt) > 20000 and tail not in ".!?}$)":
                trunc += 1
    return n, empty, trunc


def score(rows=None):
    rows = rows if rows is not None else R.load_rows()
    got = {(m, t): v for (m, t), d in rows.items()
           for k, v in d.items() if k == R.REPORTED.get(t, ("", ""))[0]}

    print("KR-E3 - response health per row (>10% empty or truncated = DEGRADED)")
    print(f"  {'row':<26} {'n':>5} {'empty':>7} {'truncated':>11}  verdict")
    degraded, borderline = [], []
    for (model, task) in sorted(rows):
        h = response_health(model, task)
        if not h:
            continue
        n, empty, trunc = h
        bad = (empty + trunc) / n
        # A row that clears a threshold by zero margin is not the same as one that clears it
        # comfortably, and printing the same word for both hides which is which. The 4B's AIME
        # rows sit at exactly 10.0% - 3 items of 30 - against a rule of ">10%". They pass by one
        # item. Say so, or the next reader takes "ok" at face value.
        margin_items = (DEGRADED_AT * n) - (empty + trunc)
        if bad > DEGRADED_AT:
            verdict = "DEGRADED"
            degraded.append(f"{model}/{task} ({100*bad:.1f}% unusable)")
        elif margin_items < 1:
            verdict = "ok - AT THRESHOLD, one item from degraded"
            borderline.append(f"{model}/{task} ({100*bad:.1f}%, {empty+trunc} of {n})")
        else:
            verdict = "ok"
        print(f"  {model + '/' + task:<26} {n:>5} {100*empty/n:>6.1f}% {100*trunc/n:>10.1f}%  "
              f"{verdict}")

    print("\nP-E2 - the ladder holds on every COMPLETED task")
    fails, pending = [], []
    for task in sorted({t for _, t in rows}):
        for small, big, op in LADDER:
            a, b = got.get((small, task)), got.get((big, task))
            if a is None or b is None:
                pending.append(f"{small} vs {big} on {task}")
                continue
            ok = (a < b) if op == "<" else (a <= b)
            mark = "HOLDS" if ok else "FAILS"
            if not ok:
                fails.append(f"{small} {100*a:.1f}% vs {big} {100*b:.1f}% on {task}")
            print(f"  {task:<22} {small:>5} {100*a:>5.1f}% {op:>2} {big:>5} {100*b:>5.1f}%  {mark}")

    print("\nP-E1 - NOT SCORED: needs publicly reported figures for these exact models and "
          "tasks.\n  Those do not live in this repo and are not typed in from memory. Score it "
          "when the\n  numbers are collected with citations, or withdraw it.")

    print("\n=== verdict so far ===")
    print(f"  KR-E3: {len(degraded)} degraded row(s)" + (": " + "; ".join(degraded) if degraded
                                                         else " - every row usable"))
    if borderline:
        print(f"  KR-E3: {len(borderline)} row(s) AT THRESHOLD, one item from degraded: "
              + "; ".join(borderline))
    print(f"  P-E2 : {len(fails)} failure(s)" + (": " + "; ".join(fails) if fails
                                                 else " - holds on every completed task"))
    if pending:
        print(f"  P-E2 : {len(pending)} comparison(s) still pending a row")
    return {"degraded": degraded, "borderline": borderline,
            "p_e2_fails": fails, "p_e2_pending": pending}


if __name__ == "__main__":
    score()
