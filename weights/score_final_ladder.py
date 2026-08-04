"""Score the final-ladder run against the stake in
weights/data/final_ladder_20260731_1910_STAKE.md.

Exits 0 only if P-1..P-4 all hold (P-5 downgrades the verdict to UNINFORMATIVE, exit 4).
Usage: python weights/score_final_ladder.py RESULT.json [REFERENCE.json]
"""
import json
import statistics
import sys

REF_DEFAULT = "weights/data/ladder_PRE_v124_2dc97d41_backup.json"
BAND = (6.8, 10.8)
EXPECT_CAL = "2dc97d41"
CTRL_NAME = "Qwen2.5-0.5B Q8_0"
CTRL_REF = 151.76
CTRL_TOL = 0.03


def main():
    res = json.load(open(sys.argv[1], encoding="utf-8"))
    ref = json.load(open(sys.argv[2] if len(sys.argv) > 2 else REF_DEFAULT, encoding="utf-8"))
    fails, notes = [], []

    # P-2 completeness first: the median is only meaningful over 14 scored rows.
    scored = [r for r in res if r.get("measured") and r.get("err_pct") is not None]
    print(f"P-2 rows scored: {len(scored)}/14 (rows present: {len(res)})")
    if len(scored) != 14 or len(res) != 14:
        fails.append(f"P-2 completeness: {len(scored)} scored rows of 14 (present {len(res)})")

    # P-3 one machine state.
    cals = sorted({r.get("cal_id") for r in res})
    print(f"P-3 cal_ids: {cals}")
    if cals != [EXPECT_CAL]:
        fails.append(f"P-3 cal_id: expected exactly ['{EXPECT_CAL}'], got {cals}")

    # P-1 median band.
    med = statistics.median([abs(r["err_pct"]) for r in scored]) if scored else None
    print(f"P-1 median |err|: {med}%  band {BAND[0]}-{BAND[1]}")
    if med is None or not (BAND[0] <= med <= BAND[1]):
        fails.append(f"P-1 median {med}% outside staked band {BAND[0]}-{BAND[1]}%")

    # P-4 prediction determinism vs the reference ladder (42 fields).
    refmap = {r["name"]: r for r in ref}
    diffs = []
    for r in res:
        b = refmap.get(r["name"])
        if not b:
            diffs.append(f"{r['name']}: absent from reference")
            continue
        for k in ("predicted", "placement", "emit"):
            if r.get(k) != b.get(k):
                diffs.append(f"{r['name']}.{k}: ref={b.get(k)!r} run={r.get(k)!r}")
    print(f"P-4 prediction-field diffs: {len(diffs)}/42")
    for d in diffs:
        print("   ", d)
    if diffs:
        fails.append(f"P-4 determinism: {len(diffs)} of 42 prediction fields moved")

    # P-5 control row -> gates whether the verdict is informative at all.
    ctrl = next((r for r in res if r["name"] == CTRL_NAME), None)
    uninformative = False
    if ctrl and ctrl.get("measured"):
        lo, hi = CTRL_REF * (1 - CTRL_TOL), CTRL_REF * (1 + CTRL_TOL)
        print(f"P-5 control {CTRL_NAME}: {ctrl['measured']} tok/s  band {lo:.2f}-{hi:.2f}")
        if not (lo <= ctrl["measured"] <= hi):
            uninformative = True
            notes.append(f"P-5 control row {ctrl['measured']} outside {lo:.2f}-{hi:.2f}: "
                         "machine state differs, P-1 verdict is UNINFORMATIVE")
    else:
        uninformative = True
        notes.append("P-5 control row missing or unmeasured")

    print()
    for n in notes:
        print("NOTE:", n)
    if fails:
        for f in fails:
            print("FAIL:", f)
        print("VERDICT: FAIL")
        return 1
    if uninformative:
        print("VERDICT: UNINFORMATIVE (kill rules on P-1..P-4 held, but P-5 says the box moved)")
        return 4
    print("VERDICT: PASS - median within 2 points of the 8.8% baseline (C-18: report as "
          "UNCHANGED, never as an improvement)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
