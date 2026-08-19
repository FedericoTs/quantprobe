"""Scorer for prereg #110 - does starving RAM enlarge the expert lever?

Written and committed BEFORE the balloon arms ran.

Usage:  python weights/prereg110_score.py weights/data/prereg110_balloon.json
"""
from __future__ import annotations
import json
import os
import sys

CEILING_K1 = 1.843            # byte-share bandwidth ceiling at k=1 (file property)
P1_ENLARGE = 1.15             # starved lever must be this much bigger than fits
P2_FITS_MAX = 1.90            # fits must not overshoot the ceiling beyond noise
P3_STARVED_MIN = 2.03         # starved must overshoot: ceiling x 1.10
USABLE_SPREAD = 0.15          # #108's baseline-usability bar, per condition


def _mean_hr(v):
    return sum(v) / len(v), (max(v) - min(v)) / 2.0


def _cond(d, key):
    arm = {int(k): _mean_hr(v) for k, v in (d.get(key) or {}).items() if v}
    if 6 not in arm:
        return None, None, None
    base, bhr = arm[6]
    spread = bhr / base if base else float("inf")
    gains = {k: arm[k][0] / base for k in arm}
    return arm, gains, spread


def score(d):
    lines, verd = [], {}
    fa, fg, fs = _cond(d, "fits")
    sa, sg, ss = _cond(d, "starved")
    if fg is None or sg is None:
        return ["VOID - a condition is missing its k=6 baseline; gains uncomparable"], {}

    for name, arm, g, spr in (("FITS", fa, fg, fs), ("STARVED", sa, sg, ss)):
        flag = "" if spr <= USABLE_SPREAD else "  - UNUSABLE (baseline too noisy)"
        lines.append(f"{name}  k=6 baseline {arm[6][0]:.2f} tok/s +/-{arm[6][1]:.2f} "
                     f"({spr*100:.1f}%{flag})")
        for k in sorted(arm, reverse=True):
            lines.append(f"   k={k}: {arm[k][0]:7.2f} tok/s   gain {g[k]:.3f}x"
                         + (f"   (ceiling {CEILING_K1:.3f}x)" if k == 1 else ""))
        lines.append("")

    fits_bad = fs > USABLE_SPREAD
    starv_bad = ss > USABLE_SPREAD
    if fits_bad or starv_bad:
        who = " and ".join(n for n, b in (("FITS", fits_bad), ("STARVED", starv_bad)) if b)
        lines.append(f"!! {who} baseline exceeds the {USABLE_SPREAD*100:.0f}% usability bar - "
                     f"any prediction that leans on it is VOID.")
        lines.append("")

    f1, s1 = fg.get(1), sg.get(1)

    # P-1: starving enlarges the lever
    if f1 and s1 and not fits_bad and not starv_bad:
        ratio = s1 / f1
        verd["P-1"] = ratio >= P1_ENLARGE
        lines.append(f"P-1  starved lever >= {P1_ENLARGE}x the fitting lever ... "
                     f"{'HIT' if verd['P-1'] else 'MISS'} "
                     f"(k=1: starved {s1:.3f}x vs fits {f1:.3f}x = {ratio:.2f}x)")
    else:
        verd["P-1"] = None
        lines.append("P-1  VOID - needs a usable k=1 gain in both conditions")

    # P-2: fits does not overshoot
    if f1 and not fits_bad:
        verd["P-2"] = f1 <= P2_FITS_MAX
        lines.append(f"P-2  fits k=1 <= {P2_FITS_MAX}x (no overshoot) ... "
                     f"{'HIT' if verd['P-2'] else 'MISS'} ({f1:.3f}x)")
    else:
        verd["P-2"] = None
        lines.append("P-2  VOID - needs a usable fits k=1 gain")

    # P-3: starved overshoots the ceiling
    if s1 and not starv_bad:
        verd["P-3"] = s1 > P3_STARVED_MIN
        lines.append(f"P-3  starved k=1 > {P3_STARVED_MIN}x (overshoots ceiling) ... "
                     f"{'HIT' if verd['P-3'] else 'MISS'} ({s1:.3f}x)")
    else:
        verd["P-3"] = None
        lines.append("P-3  VOID - needs a usable starved k=1 gain")

    lines.append("")
    if verd.get("P-1") and verd.get("P-3"):
        lines.append("KILL RULE -> RESIDENCY IS THE MECHANISM, AND IT IS LEVERAGE.")
        lines.append("  The #107/#108 overshoot is explained and reproduced on demand by memory")
        lines.append("  pressure alone. quantprobe gains a real recommendation: on a box where the")
        lines.append("  model does not fit RAM, the expert dial buys MORE than the file predicts.")
        lines.append("  L-32 is corrected from 'an Amdahl floor' to 'a residency ceiling that")
        lines.append("  memory pressure lifts'.")
    elif verd.get("P-1") and verd.get("P-3") is False:
        lines.append("KILL RULE -> RESIDENCY ENLARGES THE LEVER BUT NOT PAST THE CEILING.")
        lines.append("  Real but modest; the recommendation is quantitative, not categorical.")
    elif verd.get("P-1") is False:
        lines.append("KILL RULE -> NOT RESIDENCY.")
        lines.append("  Starving a fitting model did not enlarge the lever, so the #107/#108")
        lines.append("  overshoot is something else. The per-expert-term hypothesis returns with")
        lines.append("  its own stake.")
    else:
        lines.append("KILL RULE -> UNRESOLVED (a required arm is void). Re-run before publishing.")
    return lines, verd


def main():
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "prereg110_balloon.json")
    print("\n".join(score(json.load(open(p, encoding="utf-8")))[0]))


def _self_check():
    def cond(vals):
        return {str(k): [v, v] for k, v in vals.items()}

    # residency-as-leverage: fits undershoots, starved overshoots and the lever grows
    win = {"fits": cond({6: 100, 4: 118, 2: 150, 1: 170}),
           "starved": cond({6: 40, 4: 55, 2: 80, 1: 95})}   # fits 1.70x, starved 2.375x
    _, v = score(win)
    assert v == {"P-1": True, "P-2": True, "P-3": True}, v

    # residency real but modest: lever grows, starved does not clear 2.03
    mod = {"fits": cond({6: 100, 4: 115, 2: 140, 1: 155}),
           "starved": cond({6: 50, 4: 62, 2: 82, 1: 95})}   # fits 1.55, starved 1.90, ratio 1.23
    _, v = score(mod)
    assert v["P-1"] is True and v["P-3"] is False, v

    # not residency: starving does not enlarge the lever
    no = {"fits": cond({6: 100, 4: 118, 2: 150, 1: 170}),
          "starved": cond({6: 60, 4: 71, 2: 90, 1: 102})}   # both ~1.70x
    _, v = score(no)
    assert v["P-1"] is False, v

    # noisy starved baseline voids the predictions that lean on it
    noisy = {"fits": cond({6: 100, 4: 118, 2: 150, 1: 170}),
             "starved": {"6": [20, 80], "4": [55, 56], "2": [80, 81], "1": [95, 96]}}
    _, v = score(noisy)
    assert v["P-1"] is None and v["P-3"] is None, v
    print("self-check OK: every kill-rule branch reachable, every P can fail or void")


if __name__ == "__main__":
    _self_check() if "--self-check" in sys.argv else main()
