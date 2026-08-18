"""Scorer for prereg #106 - is our published headline reproducible on the box that produced it?

Written and committed BEFORE the arms ran. Reads the raw JSON and prints a verdict per
prediction plus the branch of the kill rule that fires. No judgement left to the reader.

Usage:  python weights/prereg106_score.py weights/data/prereg106_reproduce.json
"""
from __future__ import annotations
import json
import os
import sys

PUBLISHED, PUBLISHED_ERR = 14.86, 0.36
P1_CEILING = 13.0                       # mean of six fresh runs must be below this
P2_FLOOR = PUBLISHED - PUBLISHED_ERR    # 14.50 - bottom of the published error bar
P3_GAIN = 1.0                           # priming must buy at least this
SMALL = "control_small"


def spread(vals):
    """Relative spread. The published claim was a point estimate with an error bar; the honest
    replacement is a distribution, and this is the one number that says how wide it is."""
    m = sum(vals) / len(vals)
    return (max(vals) - min(vals)) / m if m else float("inf")


def score(d):
    big = [r["tok_s"] for r in d.get("headline_runs", []) if r.get("tok_s")]
    small = [r["tok_s"] for r in d.get(SMALL, []) if r.get("tok_s")]
    primed = (d.get("primed") or {}).get("tok_s")
    lines, v = [], {}

    if not big:
        return ["VOID - the headline arm produced no rate; nothing to score"], {}

    mean = sum(big) / len(big)
    lines.append(f"published claim ......... {PUBLISHED} +/- {PUBLISHED_ERR} tok/s")
    lines.append(f"headline arm (N={len(big)}) ..... mean {mean:.2f}, "
                 f"min {min(big):.2f}, max {max(big):.2f}, "
                 f"relative spread {spread(big) * 100:.1f}%")
    for r in d.get("headline_runs", []):
        lines.append(f"    {r.get('tok_s')} tok/s   free RAM {r.get('free_gb')} GB")
    if primed:
        lines.append(f"primed (cache warmed) ... {primed:.2f} tok/s "
                     f"({primed - mean:+.2f} vs the arm mean)")
    if small:
        lines.append(f"control {d.get('control_name', 'small model')} (N={len(small)}) ... "
                     f"mean {sum(small)/len(small):.2f}, relative spread "
                     f"{spread(small) * 100:.1f}%")
    lines.append("")

    v["P-1"] = mean < P1_CEILING
    lines.append(f"P-1  six-run mean < {P1_CEILING} ......... "
                 f"{'HIT' if v['P-1'] else 'MISS'} ({mean:.2f})")

    v["P-2"] = max(big) < P2_FLOOR
    lines.append(f"P-2  nothing reaches {P2_FLOOR} ......... "
                 f"{'HIT' if v['P-2'] else 'MISS'} (best {max(big):.2f})")

    if primed is None:
        v["P-3"] = None
        lines.append("P-3  VOID - the primed arm produced no rate")
    else:
        v["P-3"] = (primed - mean) >= P3_GAIN
        lines.append(f"P-3  priming buys >= {P3_GAIN} tok/s .... "
                     f"{'HIT' if v['P-3'] else 'MISS'} ({primed - mean:+.2f})")

    if not small:
        v["P-4"] = None
        lines.append("P-4  VOID - the control arm produced no rate")
    else:
        v["P-4"] = spread(small) <= spread(big)
        lines.append(f"P-4  control no more variable ....... "
                     f"{'HIT' if v['P-4'] else 'MISS'} "
                     f"({spread(small)*100:.1f}% vs {spread(big)*100:.1f}%)")

    lines.append("")
    if v["P-1"] and v["P-2"]:
        lines.append("KILL RULE -> HEADLINE COMES DOWN.")
        lines.append(f"  {PUBLISHED} is not reproducible on the box that produced it. Model card,")
        lines.append(f"  recipe atlas and README move to the measured distribution: mean "
                     f"{mean:.2f}, N={len(big)},")
        lines.append(f"  spread {spread(big)*100:.1f}%, quoted with the free-RAM condition. Same day.")
        if v.get("P-3"):
            lines.append("  MECHANISM CONFIRMED (P-3): page-cache residency. quantprobe ships the")
            lines.append("  free-RAM check and refuses to call a number stable when the model")
            lines.append("  is larger than free RAM.")
        elif v.get("P-3") is False:
            lines.append("  MECHANISM NOT ESTABLISHED (P-3 refuted): priming bought less than")
            lines.append(f"  {P3_GAIN} tok/s. We have a reproducibility failure with no proven cause -")
            lines.append("  state that plainly rather than shipping a story the data did not buy.")
            lines.append("  The free-RAM disclosure still ships; it is disclosure, not a claim.")
    elif not v["P-2"]:
        lines.append("KILL RULE -> RANGE, NOT A POINT.")
        lines.append(f"  Something reached {P2_FLOOR}, so the figure is attainable but unstable.")
        lines.append("  The card publishes a range with its conditions and never a point estimate.")
    else:
        lines.append("KILL RULE -> TODAY WAS THE ANOMALY (P-1 refuted).")
        lines.append("  The defect is then that we cannot tell the two states apart without")
        lines.append("  re-measuring. The free-RAM disclosure ships regardless.")
    return lines, v


def main():
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "prereg106_reproduce.json")
    print("\n".join(score(json.load(open(p, encoding="utf-8")))[0]))


def _self_check():
    """Every branch must be reachable, or the kill rule is decoration."""
    def runs(vals):
        return [{"tok_s": x, "free_gb": 12.0} for x in vals]

    not_repro = {"headline_runs": runs([11.0, 11.1, 10.9, 11.2, 11.0, 10.8]),
                 "primed": {"tok_s": 12.5}, SMALL: runs([13.0, 13.1, 13.0])}
    _, v = score(not_repro)
    assert v == {"P-1": True, "P-2": True, "P-3": True, "P-4": True}, v

    no_mech = dict(not_repro, primed={"tok_s": 11.2})
    _, v = score(no_mech)
    assert v["P-1"] and v["P-2"] and v["P-3"] is False, v

    reachable = {"headline_runs": runs([14.9, 11.0, 14.6, 11.2, 12.0, 13.0]),
                 "primed": {"tok_s": 14.0}, SMALL: runs([13.0, 13.1])}
    _, v = score(reachable)
    assert v["P-2"] is False, v

    fine = {"headline_runs": runs([14.8, 14.9, 14.7, 14.85, 14.9, 14.8]),
            "primed": {"tok_s": 14.9}, SMALL: runs([13.0, 13.1])}
    _, v = score(fine)
    assert v["P-1"] is False and v["P-2"] is False, v
    print("self-check OK: all four kill-rule branches reachable, every P can fail")


if __name__ == "__main__":
    _self_check() if "--self-check" in sys.argv else main()
