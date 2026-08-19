"""Scorer for prereg #111 - is the pure-CPU overshoot real, or was it baseline noise?

Written and committed BEFORE the arms ran. Uses MEDIAN (robust to CPU-scheduler jitter, which is
why the #110 probe's mean was untrustworthy at 47% spread).

Usage:  python weights/prereg111_score.py weights/data/prereg111_ngl0.json
"""
from __future__ import annotations
import json
import os
import statistics as st
import sys

CEILING = {6: 1.000, 4: 1.224, 2: 1.577, 1: 1.843}   # DeepSeek-Lite byte ceilings, from the file
P1_MIN = 1.90            # k=1 gain must clear this to call the overshoot real
P2_SPREAD = 0.15         # #108 usability gate on the baseline
KS = [6, 4, 2, 1]


def _spread(v):
    return (max(v) - min(v)) / st.median(v) if v and st.median(v) else float("inf")


def score(d):
    arm = {int(k): v for k, v in (d.get("decode") or {}).items() if v}
    lines, verd = [], {}
    if 6 not in arm or not arm[6]:
        return ["VOID - no k=6 baseline; gains uncomparable"], {}

    base = st.median(arm[6])
    bspread = _spread(arm[6])
    gain = {k: st.median(arm[k]) / base for k in arm}
    excess = {k: gain[k] / CEILING[k] for k in arm}

    lines.append(f"k=6 baseline: median {base:.2f} tok/s, spread {bspread*100:.1f}% "
                 f"(gate {P2_SPREAD*100:.0f}%){'  UNUSABLE' if bspread > P2_SPREAD else ''}")
    lines.append(f"{'k':>3s} {'median':>8s} {'n':>3s} {'gain':>8s} {'ceiling':>8s} {'excess':>8s}")
    for k in KS:
        if k in arm:
            lines.append(f"{k:3d} {st.median(arm[k]):8.2f} {len(arm[k]):3d} {gain[k]:7.3f}x "
                         f"{CEILING[k]:7.3f}x {excess[k]:7.3f}")
    lines.append("")

    # P-2 first: the method check that can void everything.
    verd["P-2"] = bspread <= P2_SPREAD
    lines.append(f"P-2  baseline spread <= {P2_SPREAD*100:.0f}% .......... "
                 f"{'HIT' if verd['P-2'] else 'MISS'} ({bspread*100:.1f}%)")

    if not verd["P-2"]:
        lines.append("")
        lines.append("KILL RULE -> VOID. The baseline is still unusable, so gains built on it are")
        lines.append("  not trustworthy. This box cannot measure the overshoot cleanly; U-59 needs")
        lines.append("  different hardware. A hardware limit, not a result.")
        verd["P-1"] = verd["P-3"] = None
        return lines, verd

    # P-1: the overshoot is real
    if 1 in arm:
        verd["P-1"] = gain[1] > P1_MIN
        lines.append(f"P-1  k=1 gain > {P1_MIN} (overshoot real) . "
                     f"{'HIT' if verd['P-1'] else 'MISS'} ({gain[1]:.3f}x, "
                     f"excess {excess[1]:.3f} over ceiling)")
    else:
        verd["P-1"] = None
        lines.append("P-1  VOID - no k=1 arm")

    # P-3: excess grows as experts are removed
    if all(k in excess for k in (4, 2, 1)):
        steps_ok = excess[1] > excess[2] > excess[4]
        verd["P-3"] = steps_ok
        lines.append(f"P-3  excess grows as k falls ......... "
                     f"{'HIT' if steps_ok else 'MISS'} "
                     f"(k=4 {excess[4]:.3f} -> k=2 {excess[2]:.3f} -> k=1 {excess[1]:.3f})")
    else:
        verd["P-3"] = None
        lines.append("P-3  VOID - need k=4,2,1")

    lines.append("")
    if verd.get("P-1") and verd.get("P-3"):
        lines.append("KILL RULE -> REAL, AND PER-EXPERT.")
        lines.append("  The overshoot survives a usable baseline and GROWS as experts are removed,")
        lines.append("  in a clean bandwidth-bound fitting regime - residency and capacity both")
        lines.append("  excluded. The excess is a per-expert term. quantprobe's ceiling gains a")
        lines.append("  caveat: on CPU-resident-expert placements the byte number is a FLOOR, not a")
        lines.append("  cap. U-59 advances to measured-on-one-model.")
    elif verd.get("P-1") and verd.get("P-3") is False:
        lines.append("KILL RULE -> REAL BUT FLAT.")
        lines.append("  Overshoots, but the excess does not grow with k - a fixed per-call offset,")
        lines.append("  not per-expert. Recorded; weaker; its own follow-up.")
    elif verd.get("P-1") is False:
        lines.append("KILL RULE -> THE PROBE'S 2x WAS NOISE.")
        lines.append("  With a usable baseline the overshoot does not hold. The per-expert-term")
        lines.append("  thread weakens and L-32's undershoot reading stands. Said plainly.")
    else:
        lines.append("KILL RULE -> UNRESOLVED (a required arm void). Re-run.")
    return lines, verd


def main():
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "prereg111_ngl0.json")
    print("\n".join(score(json.load(open(p, encoding="utf-8")))[0]))


def _self_check():
    def arms(dec):
        return {"decode": {str(k): v for k, v in dec.items()}}

    # real + per-expert: tight baseline, k=1 clears 1.90, excess grows
    good = arms({6: [5.0, 5.1, 5.0, 5.05, 5.0], 4: [6.3, 6.4, 6.3, 6.35, 6.3],
                 2: [8.4, 8.5, 8.4, 8.45, 8.4], 1: [10.3, 10.4, 10.3, 10.35, 10.3]})
    _, v = score(good)   # gain1=10.33/5.02=2.06>1.90; excess grows 1.03,1.06,1.12
    assert v == {"P-2": True, "P-1": True, "P-3": True}, v

    # unusable baseline -> void everything
    noisy = arms({6: [3.0, 7.0, 4.0, 6.0, 5.0], 4: [6.3]*5, 2: [8.4]*5, 1: [10.3]*5})
    _, v = score(noisy)
    assert v["P-2"] is False and v["P-1"] is None, v

    # overshoot real but flat excess -> per-call not per-expert
    flat = arms({6: [5.0]*5, 4: [6.7]*5, 2: [8.6]*5, 1: [10.1]*5})   # excess ~1.09,1.09,1.09
    _, v = score(flat)
    assert v["P-1"] is True and v["P-3"] is False, v

    # probe was noise: k=1 undershoots
    under = arms({6: [5.0]*5, 4: [5.6]*5, 2: [6.8]*5, 1: [7.4]*5})   # gain1=1.48<1.90
    _, v = score(under)
    assert v["P-1"] is False, v
    print("self-check OK: void / real+per-expert / real+flat / noise-probe all reachable")


if __name__ == "__main__":
    _self_check() if "--self-check" in sys.argv else main()
