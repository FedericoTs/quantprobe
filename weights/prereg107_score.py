"""Scorer for prereg #107 - is the expert-count knob a real speed lever?

Written and committed BEFORE the arms ran. Reads the raw JSON and prints a verdict per
prediction plus the branch of the kill rule that fires.

Usage:  python weights/prereg107_score.py weights/data/prereg107_kcurve.json
"""
from __future__ import annotations
import json
import os
import sys

# Law 4 predictions, computed from the FILE's byte split before any measurement (see the prereg
# table). Hard-coded here so the scorer cannot quietly re-derive them from the outcome.
PREDICTED = {8: 1.000, 6: 1.059, 4: 1.125, 2: 1.200, 1: 1.242}
P1_CEILING = 1.50            # k=1 speedup must stay below this for "weak lever" to hold
P3_K, P3_TOL = 2, 0.15       # Law 4 point test at k=2, +/-15%
P4_K, P4_COST = 4, 0.50      # halving the experts must cost at least this much PPL
PPL_K8 = 5.7796              # prereg #104's committed figure, same chunks


def _mean_hr(vals):
    return sum(vals) / len(vals), (max(vals) - min(vals)) / 2.0


def score(d):
    speed = {int(k): _mean_hr(v) for k, v in (d.get("speed") or {}).items() if v}
    ppl = {int(k): v for k, v in (d.get("ppl") or {}).items() if v}
    lines, verd = [], {}
    if 8 not in speed:
        return ["VOID - no k=8 baseline in this session; every speedup would be uncomparable"], {}

    base = speed[8][0]
    lines.append(f"baseline k=8: {base:.2f} tok/s (warmed, same session)")
    lines.append(f"{'k':>3s} {'tok/s':>14s} {'measured':>10s} {'Law 4':>8s} {'error':>8s}")
    for k in sorted(speed, reverse=True):
        m, hr = speed[k]
        got = m / base
        pred = PREDICTED.get(k)
        err = f"{(got/pred - 1)*100:+7.1f}%" if pred else "      -"
        lines.append(f"{k:3d} {m:8.2f} +/-{hr:4.2f} {got:9.3f}x "
                     f"{(f'{pred:.3f}x' if pred else '-'):>8s} {err:>8s}")
    if ppl:
        lines.append("")
        # Deltas are against THIS SESSION's k=8, never against PPL_K8. That constant came from
        # prereg #104, which used a different eval corpus (D:/evo-compress-data/eval/
        # wiki.test.raw, 1,290,590 bytes) than these arms (weights/data/wikitext2_test.raw,
        # 1,307,975 bytes, different hash). Mixing the two printed a k=4 delta of +1.69 where
        # the real in-session cost is +1.51. P-4 was always computed correctly; only this line
        # was wrong, and a wrong line in a verdict is still a wrong verdict to whoever reads it.
        base_ppl = ppl.get(8)
        for k in sorted(ppl, reverse=True):
            lines.append(f"  k={k:<2d} PPL {ppl[k]:.4f}"
                         + (f"   {ppl[k]-base_ppl:+.4f} vs k=8" if k != 8 and base_ppl else ""))
        lines.append(f"  (corpus: this session's own k=8 is the baseline; prereg #104's 5.7796 "
                     f"is NOT comparable - different eval file)")
    lines.append("")

    # P-1: is the lever weak?
    if 1 in speed:
        got = speed[1][0] / base
        verd["P-1"] = got < P1_CEILING
        lines.append(f"P-1  k=1 speedup < {P1_CEILING} (weak lever) ... "
                     f"{'HIT' if verd['P-1'] else 'MISS'} ({got:.3f}x)")
    else:
        verd["P-1"] = None
        lines.append("P-1  VOID - no k=1 arm")

    # P-2: monotone as k falls
    ks = sorted(speed, reverse=True)          # 8, 4, 2, 1
    breaks = []
    for a, b in zip(ks, ks[1:]):
        ma, ha = speed[a]
        mb, hb = speed[b]
        if mb < ma - (ha + hb):               # got SLOWER with fewer experts, beyond error bars
            breaks.append(f"k={a}:{ma:.2f} -> k={b}:{mb:.2f}")
    verd["P-2"] = not breaks
    lines.append(f"P-2  monotone as k falls ............... {'HIT' if verd['P-2'] else 'MISS'}"
                 + (f"  (reversals: {'; '.join(breaks)})" if breaks else ""))

    # P-3: Law 4's point prediction at k=2
    if P3_K in speed:
        got = speed[P3_K][0] / base
        pred = PREDICTED[P3_K]
        verd["P-3"] = abs(got / pred - 1) <= P3_TOL
        lines.append(f"P-3  k={P3_K} within +/-{P3_TOL*100:.0f}% of {pred:.3f}x ..... "
                     f"{'HIT' if verd['P-3'] else 'MISS'} ({got:.3f}x, "
                     f"{(got/pred-1)*100:+.1f}%)")
    else:
        verd["P-3"] = None
        lines.append(f"P-3  VOID - no k={P3_K} arm")

    # P-4: the quality cost of halving the experts
    if P4_K in ppl and 8 in ppl:
        cost = ppl[P4_K] - ppl[8]
        verd["P-4"] = cost >= P4_COST
        lines.append(f"P-4  k={P4_K} costs >= {P4_COST} PPL .......... "
                     f"{'HIT' if verd['P-4'] else 'MISS'} ({cost:+.4f})")
    else:
        verd["P-4"] = None
        lines.append(f"P-4  VOID - need PPL at k={P4_K} and k=8")

    lines.append("")
    if verd.get("P-1") is False:
        lines.append("KILL RULE -> LAW 4 IS INCOMPLETE HERE.")
        lines.append("  k=1 beat the bandwidth-only ceiling, so on a model larger than free RAM")
        lines.append("  the residency term belongs in the PREDICTION, not only the disclosure.")
        lines.append("  That is a Law 4 amendment and it gets its own stake.")
    elif verd.get("P-4") is False:
        lines.append("KILL RULE -> k IS A CHEAP LEVER. Expose it in the tool with the measured")
        lines.append("  quality curve attached. Weak but cheap still beats weak and expensive.")
    elif verd.get("P-1") and verd.get("P-4"):
        lines.append("KILL RULE -> BOUNDED, EXPENSIVE LEVER.")
        lines.append("  Register it as such. quantprobe states the CEILING computed from the")
        lines.append("  file's byte split rather than offering the dial as a win - the useful")
        lines.append("  output is the formula, not the knob.")
    else:
        lines.append("KILL RULE -> UNRESOLVED (a required arm is void). Re-run before publishing.")
    if verd.get("P-3") is False and verd.get("P-1"):
        lines.append("  NOTE: P-3 refuted with P-1 holding - the lever is weak as claimed but")
        lines.append("  Law 4 mis-sizes it. Publish the miss beside the ceiling claim.")
    return lines, verd


def main():
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "prereg107_kcurve.json")
    print("\n".join(score(json.load(open(p, encoding="utf-8")))[0]))


def _self_check():
    """Every branch reachable, every prediction able to fail."""
    weak = {"speed": {"8": [14.0, 14.1], "4": [15.7, 15.8], "2": [16.8, 16.9],
                      "1": [17.4, 17.4]},
            "ppl": {"8": 5.7796, "4": 6.9}}
    _, v = score(weak)
    assert v == {"P-1": True, "P-2": True, "P-3": True, "P-4": True}, v

    cheap = dict(weak, ppl={"8": 5.7796, "4": 5.9})
    _, v = score(cheap)
    assert v["P-4"] is False, v

    residency = {"speed": {"8": [14.0, 14.0], "4": [18.0, 18.0], "2": [21.0, 21.0],
                           "1": [23.0, 23.0]}, "ppl": {"8": 5.7796, "4": 6.9}}
    _, v = score(residency)
    assert v["P-1"] is False and v["P-3"] is False, v

    # every list here is REPS, not (mean, spread) - getting that wrong once is why this fixture
    # is spelled out with two near-identical reps per arm
    reversal = {"speed": {"8": [14.0, 14.0], "4": [12.0, 12.0]}, "ppl": {}}
    _, v = score(reversal)
    assert v["P-2"] is False, v
    print("self-check OK: all kill-rule branches reachable, every P can fail")


if __name__ == "__main__":
    _self_check() if "--self-check" in sys.argv else main()
