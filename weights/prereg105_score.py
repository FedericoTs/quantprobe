"""Scorer for prereg #105 - published speed vs experienced speed.

Written and committed BEFORE the arms were run. It reads the raw arm log and prints a verdict
per prediction with no judgement calls left to the person reading the output.

Usage:  python weights/prereg105_score.py weights/data/prereg105_ctx_sweep.json
"""
from __future__ import annotations
import json
import os
import sys

REFERENCE = 14.86          # llama-bench tg128, N=5, as published on the model card
P1_FLOOR = 13.5            # P-1: -c 512 must reach this
P3_CENTER, P3_HALFWIDTH = 11.4, 1.0   # P-3: default context reproduces the sanity log


def _mean_hr(vals):
    """Mean and half-range. Two reps make a standard deviation meaningless; half-range is the
    honest spread statistic for n=2 and it is what the prereg said would be reported."""
    return sum(vals) / len(vals), (max(vals) - min(vals)) / 2.0


def score(data):
    arms = {int(k): _mean_hr(v) for k, v in data["ctx_arms"].items() if v}
    if not arms:
        return ["VOID - no arm produced a rate; nothing to score"], {}
    lines, verdicts = [], {}

    lines.append(f"reference (llama-bench tg128, this session): "
                 f"{data.get('bench_tg128', 'not re-run')}")
    lines.append(f"card headline: {REFERENCE} tok/s")
    lines.append("")
    for c in sorted(arms):
        m, hr = arms[c]
        lines.append(f"  -c {c:<6d} {m:6.2f} +/- {hr:.2f} tok/s   "
                     f"({m / REFERENCE * 100:5.1f}% of the published figure)")
    lines.append("")

    # P-1: the smallest context recovers most of the gap
    if 512 in arms:
        m, hr = arms[512]
        ok = m >= P1_FLOOR
        verdicts["P-1"] = ok
        lines.append(f"P-1  -c 512 >= {P1_FLOOR} tok/s ....... {'HIT' if ok else 'MISS'} "
                     f"({m:.2f})")
    else:
        verdicts["P-1"] = None
        lines.append("P-1  VOID - the -c 512 arm did not produce a rate")

    # P-2: monotone non-increasing in context, allowing one error bar of overlap
    ordered = sorted(arms)
    breaks = []
    for a, b in zip(ordered, ordered[1:]):
        ma, ha = arms[a]
        mb, hb = arms[b]
        if mb > ma + (ha + hb):        # a rise larger than the combined error bars
            breaks.append(f"-c {a}={ma:.2f} -> -c {b}={mb:.2f}")
    ok = not breaks
    verdicts["P-2"] = ok
    lines.append(f"P-2  monotone non-increasing in -c ... {'HIT' if ok else 'MISS'}"
                 + (f"  (reversals beyond error bars: {'; '.join(breaks)})" if breaks else ""))

    # P-3: the default context reproduces the sanity-log rate
    if 4096 in arms:
        m, hr = arms[4096]
        ok = abs(m - P3_CENTER) <= P3_HALFWIDTH
        verdicts["P-3"] = ok
        lines.append(f"P-3  -c 4096 within {P3_CENTER} +/- {P3_HALFWIDTH} ... "
                     f"{'HIT' if ok else 'MISS'} ({m:.2f})")
    else:
        verdicts["P-3"] = None
        lines.append("P-3  VOID - the -c 4096 arm did not produce a rate")

    # P-4: token count is not the cause (added in the pre-data amendment)
    b128, b512 = data.get("bench_tg128"), data.get("bench_tg512")
    if b128 and b512:
        ok = abs(b512 - b128) <= 1.0
        verdicts["P-4"] = ok
        lines.append(f"P-4  llama-bench n=512 within 1.0 of n=128 ... {'HIT' if ok else 'MISS'} "
                     f"({b128:.2f} vs {b512:.2f})")
    else:
        verdicts["P-4"] = None
        lines.append("P-4  VOID - both llama-bench arms are required")

    # The kill rule, applied mechanically. This is the part that must not be re-litigated
    # after seeing the numbers, which is why it lives in code staked before the run.
    lines.append("")
    if verdicts.get("P-1") is True and verdicts.get("P-2") is True:
        lines.append("KILL RULE -> VRAM-DISPLACEMENT branch.")
        lines.append("  The card KEEPS 14.86 but must state the context it was measured at and")
        lines.append("  carry the -c curve: the default costs the user ~23% and they deserve why.")
    elif verdicts.get("P-1") is False:
        lines.append("KILL RULE -> HEADLINE-CHANGES branch.")
        lines.append("  14.86 is not reachable from the documented command on this hardware.")
        lines.append("  The card headline becomes the llama-cli figure; 14.86 is demoted to a")
        lines.append("  labelled harness measurement. Edited the same day.")
    else:
        lines.append("KILL RULE -> UNRESOLVED (P-1 void). Re-run before editing the card.")
    return lines, verdicts


def main():
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "prereg105_ctx_sweep.json")
    data = json.load(open(p, encoding="utf-8"))
    lines, _ = score(data)
    print("\n".join(lines))


def _self_check():
    """The scorer must be able to fail. Two synthetic datasets, one per branch, prove the
    kill rule actually branches instead of always printing the comfortable answer."""
    displaced = {"ctx_arms": {"512": [14.2, 14.0], "2048": [13.1, 13.0],
                              "4096": [11.5, 11.3], "8192": [9.9, 9.7]}}
    _, v = score(displaced)
    assert v["P-1"] is True and v["P-2"] is True and v["P-3"] is True, v
    flat = {"ctx_arms": {"512": [11.4, 11.3], "2048": [11.4, 11.2],
                         "4096": [11.4, 11.4], "8192": [11.3, 11.2]}}
    _, v = score(flat)
    assert v["P-1"] is False, v
    rising = {"ctx_arms": {"512": [11.0, 11.0], "4096": [14.0, 14.0]}}
    _, v = score(rising)
    assert v["P-2"] is False, v
    print("self-check OK: both branches reachable, P-2 can fail")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
