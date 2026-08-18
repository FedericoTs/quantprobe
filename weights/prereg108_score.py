"""Scorer for prereg #108 - is the expert dial a prefill lever?

Written and committed BEFORE the arms ran.

Usage:  python weights/prereg108_score.py weights/data/prereg108_prefill.json
"""
from __future__ import annotations
import json
import os
import sys

# From the file's PARAM split (compute scales with params, not bytes), staked in the prereg.
PREDICTED = {8: 1.000, 4: 1.206, 2: 1.345, 1: 1.426}
# prereg #107's measured DECODE gains on the same file and placement.
DECODE = {8: 1.000, 4: 1.146, 2: 1.175, 1: 1.451}
P1_K, P1_TOL = 4, 0.15
P3_K, P3_FLOOR = 2, 1.25


def _mean_hr(v):
    return sum(v) / len(v), (max(v) - min(v)) / 2.0


def score(d):
    pf = {int(k): _mean_hr(v) for k, v in (d.get("prefill") or {}).items() if v}
    lines, verd = [], {}
    if 8 not in pf:
        return ["VOID - no k=8 prefill baseline; every speedup would be uncomparable"], {}
    base = pf[8][0]
    lines.append(f"baseline k=8 prefill: {base:.2f} tok/s (warmed, same session)")
    lines.append(f"{'k':>3s} {'prefill tok/s':>16s} {'gain':>8s} {'predicted':>10s} "
                 f"{'error':>8s} {'decode #107':>12s}")
    for k in sorted(pf, reverse=True):
        m, hr = pf[k]
        got = m / base
        pr = PREDICTED.get(k)
        err = f"{(got/pr-1)*100:+7.1f}%" if pr else "      -"
        lines.append(f"{k:3d} {m:10.2f} +/-{hr:4.2f} {got:7.3f}x "
                     f"{(f'{pr:.3f}x' if pr else '-'):>10s} {err:>8s} "
                     f"{DECODE.get(k, float('nan')):11.3f}x")
    lines.append("")

    if P1_K in pf:
        got, pr = pf[P1_K][0] / base, PREDICTED[P1_K]
        verd["P-1"] = abs(got / pr - 1) <= P1_TOL
        lines.append(f"P-1  k={P1_K} within +/-{P1_TOL*100:.0f}% of {pr:.3f}x ... "
                     f"{'HIT' if verd['P-1'] else 'MISS'} ({got:.3f}x, {(got/pr-1)*100:+.1f}%)")
    else:
        verd["P-1"] = None
        lines.append(f"P-1  VOID - no k={P1_K} arm")

    beats = {k: (pf[k][0] / base) > DECODE[k] for k in (4, 2) if k in pf}
    if len(beats) == 2:
        verd["P-2"] = all(beats.values())
        detail = ", ".join(f"k={k}: {'beats' if v else 'LOSES TO'} decode" for k, v in beats.items())
        lines.append(f"P-2  prefill beats decode at k=4 and k=2 ... "
                     f"{'HIT' if verd['P-2'] else 'MISS'}  ({detail})")
    else:
        verd["P-2"] = None
        lines.append("P-2  VOID - needs both the k=4 and k=2 arms")

    if P3_K in pf:
        got = pf[P3_K][0] / base
        verd["P-3"] = got >= P3_FLOOR
        lines.append(f"P-3  k={P3_K} reaches {P3_FLOOR}x ............... "
                     f"{'HIT' if verd['P-3'] else 'MISS'} ({got:.3f}x)")
    else:
        verd["P-3"] = None
        lines.append(f"P-3  VOID - no k={P3_K} arm")

    lines.append("")
    if verd.get("P-2") and verd.get("P-3"):
        lines.append("KILL RULE -> IT IS A PREFILL LEVER.")
        lines.append("  quantprobe's ceiling line gains a second number computed from the PARAM")
        lines.append("  share, stating the gain lands on time-to-first-token - and never free of")
        lines.append("  prereg #107's quality bill (k=4 costs +1.51 PPL, k=2 costs +15.5).")
    elif verd.get("P-2") is False:
        lines.append("KILL RULE -> BAD EVERYWHERE.")
        lines.append("  V-22 goes from 'not recommended' to 'no known workload', and the tool")
        lines.append("  keeps exactly one ceiling line.")
    elif verd.get("P-3") is False and verd.get("P-2"):
        lines.append("KILL RULE -> BETTER THAN DECODE, STILL NOT WORTH IT.")
        lines.append("  Prefill is the right place for this knob and the knob is still too small")
        lines.append("  to pay for its quality bill. Record the ordering, recommend nothing.")
    else:
        lines.append("KILL RULE -> UNRESOLVED (a required arm is void). Re-run before publishing.")
    if verd.get("P-1") is False and verd.get("P-2"):
        lines.append("  NOTE: P-1 refuted with P-2 holding - the FLOP-share model mis-prices")
        lines.append("  prefill. The excess is a new term (per-expert overhead), and it gets its")
        lines.append("  own stake before anything is claimed about it.")
    return lines, verd


def main():
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "prereg108_prefill.json")
    print("\n".join(score(json.load(open(p, encoding="utf-8")))[0]))


def _self_check():
    def arms(d):
        return {"prefill": {str(k): [v, v] for k, v in d.items()}}

    # prefill tracks the FLOP share and beats decode everywhere
    _, v = score(arms({8: 100.0, 4: 120.6, 2: 134.5, 1: 142.6}))
    assert v == {"P-1": True, "P-2": True, "P-3": True}, v
    # flat prefill: loses to decode, and never reaches 1.25
    _, v = score(arms({8: 100.0, 4: 101.0, 2: 102.0, 1: 103.0}))
    assert v["P-2"] is False and v["P-3"] is False, v
    # beats decode but stays under the usefulness floor
    _, v = score(arms({8: 100.0, 4: 118.0, 2: 121.0, 1: 130.0}))
    assert v["P-2"] is True and v["P-3"] is False, v
    # far above the FLOP model - the confounded wall-clock hint coming true
    _, v = score(arms({8: 100.0, 4: 180.0, 2: 220.0, 1: 260.0}))
    assert v["P-1"] is False and v["P-2"] is True, v
    print("self-check OK: every kill-rule branch reachable, every P can fail")


if __name__ == "__main__":
    _self_check() if "--self-check" in sys.argv else main()
