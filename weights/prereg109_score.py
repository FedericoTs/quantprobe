"""Scorer for prereg #109 - is the excess over Law 4 a missing term, or residency?

Written and committed BEFORE the control model's arms ran.

Usage:  python weights/prereg109_score.py weights/data/prereg109_control.json
"""
from __future__ import annotations
import json
import os
import sys

# Ceilings computed from DeepSeek-Coder-V2-Lite-Base-IQ2_XS's own byte/param split, staked in
# the prereg. 64 experts, default k=6, routed share 54.9% of bytes and 55.1% of params.
DECODE_PRED = {6: 1.000, 4: 1.224, 2: 1.577, 1: 1.843}
PREFILL_PRED = {6: 1.000, 4: 1.225, 2: 1.580, 1: 1.848}
P1_K, P1_TOL = 1, 0.15
P2_K, P2_TOL = 2, 0.25
P3_MAX_EXCESS = 0.60          # the fitting model's prefill excess must stay under this
REFERENCE_EXCESS = 1.80       # what the NON-fitting model overshot by at k=2 (prereg #108)
USABLE_SPREAD = 0.15          # same baseline bar as #108


def _mean_hr(v):
    return sum(v) / len(v), (max(v) - min(v)) / 2.0


def _arm(d, key):
    return {int(k): _mean_hr(v) for k, v in (d.get(key) or {}).items() if v}


def score(d):
    dec, pre = _arm(d, "decode"), _arm(d, "prefill")
    lines, verd = [], {}
    if 6 not in dec or 6 not in pre:
        return ["VOID - the k=6 baseline is missing from an arm; gains would be uncomparable"], {}

    for name, arm, pred in (("DECODE", dec, DECODE_PRED), ("PREFILL", pre, PREFILL_PRED)):
        base, bhr = arm[6]
        spread = bhr / base if base else float("inf")
        lines.append(f"{name}  baseline k=6: {base:.2f} tok/s +/-{bhr:.2f} "
                     f"({spread*100:.1f}% spread{'' if spread <= USABLE_SPREAD else ' - UNUSABLE'})")
        for k in sorted(arm, reverse=True):
            m, hr = arm[k]
            g = m / base
            lines.append(f"   k={k}: {m:8.2f} +/-{hr:5.2f}   {g:6.3f}x   "
                         f"ceiling {pred[k]:.3f}x   {(g/pred[k]-1)*100:+7.1f}%")
        lines.append("")

    # P-1: decode lands on its ceiling
    if P1_K in dec:
        g = dec[P1_K][0] / dec[6][0]
        verd["P-1"] = abs(g / DECODE_PRED[P1_K] - 1) <= P1_TOL
        lines.append(f"P-1  decode k={P1_K} within +/-{P1_TOL*100:.0f}% of "
                     f"{DECODE_PRED[P1_K]:.3f}x ... {'HIT' if verd['P-1'] else 'MISS'} "
                     f"({g:.3f}x, {(g/DECODE_PRED[P1_K]-1)*100:+.1f}%)")
    else:
        verd["P-1"] = None
        lines.append(f"P-1  VOID - no decode k={P1_K} arm")

    # P-2: prefill lands on its ceiling
    excess = None
    if P2_K in pre:
        g = pre[P2_K][0] / pre[6][0]
        excess = g / PREFILL_PRED[P2_K] - 1
        verd["P-2"] = abs(excess) <= P2_TOL
        lines.append(f"P-2  prefill k={P2_K} within +/-{P2_TOL*100:.0f}% of "
                     f"{PREFILL_PRED[P2_K]:.3f}x ... {'HIT' if verd['P-2'] else 'MISS'} "
                     f"({g:.3f}x, {excess*100:+.1f}%)")
    else:
        verd["P-2"] = None
        lines.append(f"P-2  VOID - no prefill k={P2_K} arm")

    # P-3: and it is far smaller than the non-fitting model's excess
    if excess is None:
        verd["P-3"] = None
        lines.append("P-3  VOID - needs the prefill k=2 arm")
    else:
        verd["P-3"] = excess < P3_MAX_EXCESS
        lines.append(f"P-3  prefill excess < {P3_MAX_EXCESS*100:.0f}% ......... "
                     f"{'HIT' if verd['P-3'] else 'MISS'} ({excess*100:+.1f}%, against "
                     f"{REFERENCE_EXCESS*100:.0f}% on the model that does NOT fit)")

    lines.append("")
    if verd.get("P-1") and verd.get("P-2") and verd.get("P-3"):
        lines.append("KILL RULE -> THE EXCESS IS RESIDENCY, NOT A MISSING TERM.")
        lines.append("  On a model that fits, Law 4's ceiling holds. No new physics is owed -")
        lines.append("  the law needs its REGIME stated, which v1.29-v1.31 already ship.")
        lines.append("  The 'open edge of Law 4' in #107 and #108 closes as EXPLAINED by L-29/L-31.")
    elif verd.get("P-3") is False:
        lines.append("KILL RULE -> A REAL PER-EXPERT TERM, ARCHITECTURE-INDEPENDENT.")
        lines.append("  The excess survives on a model with 6.6 GiB of headroom, so residency")
        lines.append("  cannot explain it. Law 4 is missing a term that scales with k and not")
        lines.append("  with bytes. That is a law-amendment track with its own programme.")
    elif verd.get("P-1") is False or verd.get("P-2") is False:
        lines.append("KILL RULE -> RESIDENCY IS NOT THE WHOLE STORY.")
        lines.append("  Part of the excess survives where nothing can be evicted, so a per-expert")
        lines.append("  term exists even if residency also contributes. Size it before claiming it.")
    else:
        lines.append("KILL RULE -> UNRESOLVED (a required arm is void). Re-run before publishing.")
    return lines, verd


def main():
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "prereg109_control.json")
    print("\n".join(score(json.load(open(p, encoding="utf-8")))[0]))


def _self_check():
    def mk(dec, pre):
        return {"decode": {str(k): [v, v] for k, v in dec.items()},
                "prefill": {str(k): [v, v] for k, v in pre.items()}}

    # lands on the ceiling in both -> residency explains everything
    on = mk({6: 100, 4: 122.4, 2: 157.7, 1: 184.3}, {6: 100, 4: 122.5, 2: 158.0, 1: 184.8})
    _, v = score(on)
    assert v == {"P-1": True, "P-2": True, "P-3": True}, v

    # overshoots as badly as the non-fitting model -> a real per-expert term
    big = mk({6: 100, 4: 160, 2: 300, 1: 400}, {6: 100, 4: 200, 2: 440, 1: 520})
    _, v = score(big)
    assert v["P-1"] is False and v["P-2"] is False and v["P-3"] is False, v

    # decode clean, prefill mildly over -> residency not the whole story
    mixed = mk({6: 100, 4: 122.4, 2: 157.7, 1: 184.3}, {6: 100, 4: 140, 2: 220, 1: 250})
    _, v = score(mixed)
    assert v["P-1"] is True and v["P-2"] is False and v["P-3"] is True, v
    print("self-check OK: every kill-rule branch reachable, every P can fail")


if __name__ == "__main__":
    _self_check() if "--self-check" in sys.argv else main()
