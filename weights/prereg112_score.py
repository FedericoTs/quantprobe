"""Score pre-registration #112 - does depth-aware quantization pay on a DENSE model?

Committed BEFORE the arms are evaluated. Reads a JSON results file and prints a verdict that
nobody gets to adjust after seeing the numbers.

    python weights/prereg112_score.py --results weights/data/prereg112.json
    python weights/prereg112_score.py --self-check     # every branch reachable, no data needed

Results schema (bytes are exact file sizes, ppl is 32-chunk WikiText-2 held out):

    {"ref_ppl": 5.1234,
     "arms": {"OURS":   {"ppl": 5.40, "bytes": 9900000000, "tps": 4.1},
              "SPREAD": {"ppl": 5.55, "bytes": 9910000000, "tps": 4.0},
              "MTP":    {"ppl": 5.40, "bytes": 10050000000, "tps": 4.1}}}
"""

import argparse
import json
import sys

SIZE_GATE = 0.005  # A vs B must be within 0.5% in bytes, checked BEFORE any PPL is read
P1_FLOOR = 0.146  # at least half the MoE effect (prereg #104 removed 29.2%)
MOE_EFFECT = 0.292
P3_PPL_EPS = 0.01  # |PPL(MTP) - PPL(OURS)| below this is "indistinguishable"
P3_SIZE_LO, P3_SIZE_HI = 0.002, 0.02  # MTP arm larger by 0.2-2%
P4_TPS = 0.03  # decode unchanged within 3%


def score(d):
    """-> (lines, verdict_counts). Pure: no I/O, so --self-check can drive every branch."""
    out, hits, misses, unscored = [], [], [], []
    ref = d.get("ref_ppl")
    arms = d.get("arms") or {}
    a, b, c = arms.get("OURS"), arms.get("SPREAD"), arms.get("MTP")

    # --- size gate first, exactly as staked: no quality number is read until it passes ---
    if not (a and b):
        out.append("VOID: OURS and SPREAD are both required; one is missing.")
        return out, ("VOID", [], [], [])
    dev = abs(a["bytes"] - b["bytes"]) / max(a["bytes"], b["bytes"])
    out.append(f"size gate: OURS {a['bytes']:,} B vs SPREAD {b['bytes']:,} B -> {dev * 100:.3f}%")
    if dev > SIZE_GATE:
        out.append(
            f"  FAILED (> {SIZE_GATE * 100:.1f}%). Control must be rebuilt. "
            "No PPL is scored - a size-mismatched pair cannot separate placement from budget."
        )
        return out, ("VOID", [], [], [])
    out.append(f"  passed (<= {SIZE_GATE * 100:.1f}%)")

    if ref is None:
        out.append("VOID: no reference PPL, so 'excess loss' has no denominator.")
        return out, ("VOID", [], [], [])

    # --- P-1: depth-aware beats the equal-bytes control, by at least half the MoE effect ---
    ex_a, ex_b = a["ppl"] - ref, b["ppl"] - ref
    out.append(f"\nreference PPL {ref:.4f} | OURS {a['ppl']:.4f} (+{ex_a:.4f}) "
               f"| SPREAD {b['ppl']:.4f} (+{ex_b:.4f})")
    if ex_b <= 0:
        out.append("P-1 UNSCORED: control is at or below the reference; 'excess loss' is undefined.")
        unscored.append("P-1")
        share = None
    else:
        share = (ex_b - ex_a) / ex_b
        out.append(f"P-1: depth-aware removes {share * 100:.1f}% of the control's excess loss "
                   f"(staked: >= {P1_FLOOR * 100:.1f}%)")
        if a["ppl"] > b["ppl"]:
            out.append("  INVERTED - the control WON. Depth-aware placement did not transfer to "
                       "a dense model; the README needs a scope line.")
            misses.append("P-1 (inverted)")
        elif share >= P1_FLOOR:
            out.append("  HIT")
            hits.append("P-1")
        else:
            out.append("  MISS - it won, but by less than half the MoE effect.")
            misses.append("P-1")

    # --- P-2: the dense effect is SMALLER than the MoE one ---
    if share is None:
        out.append("P-2 UNSCORED: depends on P-1's share.")
        unscored.append("P-2")
    else:
        out.append(f"P-2: share {share * 100:.1f}% vs MoE {MOE_EFFECT * 100:.1f}% "
                   f"(staked: below)")
        if share < MOE_EFFECT:
            out.append("  HIT")
            hits.append("P-2")
        else:
            out.append("  MISS - dense matched or beat MoE. That is the more interesting answer: "
                       "the effect is not an expert-structure artefact.")
            misses.append("P-2")

    # --- P-3 / U-61: protecting the never-executed MTP block ---
    if not c:
        out.append("\nP-3 UNSCORED: no MTP arm.")
        unscored.append("P-3")
    else:
        dppl = c["ppl"] - a["ppl"]
        dsz = (c["bytes"] - a["bytes"]) / a["bytes"]
        out.append(f"\nP-3 (U-61): MTP arm PPL {c['ppl']:.4f} vs OURS {a['ppl']:.4f} "
                   f"(delta {dppl:+.4f}) | bytes {dsz * 100:+.2f}%")
        if dppl < -P3_PPL_EPS:
            out.append(f"  KILL RULE FIRED: protecting blk.64 measurably HELPED ({dppl:+.4f} < "
                       f"{-P3_PPL_EPS}). The block is being executed by some path the U-60 source "
                       "analysis missed. U-60 P-1 must be RE-OPENED.")
            misses.append("P-3 (kill rule)")
        elif abs(dppl) <= P3_PPL_EPS and P3_SIZE_LO <= dsz <= P3_SIZE_HI:
            out.append("  HIT - indistinguishable quality, larger file: the block is dead weight.")
            hits.append("P-3")
        elif abs(dppl) <= P3_PPL_EPS:
            out.append(f"  PARTIAL->MISS - quality indistinguishable as staked, but the size "
                       f"delta {dsz * 100:+.2f}% is outside the staked "
                       f"{P3_SIZE_LO * 100:.1f}-{P3_SIZE_HI * 100:.1f}% band.")
            misses.append("P-3")
        else:
            out.append("  MISS - protecting blk.64 measurably HURT, which no mechanism predicts.")
            misses.append("P-3")

    # --- P-4: decode unchanged ---
    tps = [x["tps"] for x in (a, b, c) if x and x.get("tps")]
    if len(tps) < 2:
        out.append("\nP-4 UNSCORED: fewer than two arms carry a decode measurement.")
        unscored.append("P-4")
    else:
        spread = (max(tps) - min(tps)) / max(tps)
        out.append(f"\nP-4: decode spread across arms {spread * 100:.1f}% "
                   f"(staked: <= {P4_TPS * 100:.0f}%)")
        if spread <= P4_TPS:
            out.append("  HIT")
            hits.append("P-4")
        else:
            out.append("  MISS - placement moved decode more than the byte count explains.")
            misses.append("P-4")

    n = len(hits) + len(misses)
    out.append(f"\nVERDICT: {len(hits)}/{n} scored" + (f", {len(unscored)} unscored" if unscored else ""))
    return out, ("SCORED", hits, misses, unscored)


SELF = [
    ("size gate fails", {"ref_ppl": 5.0, "arms": {
        "OURS": {"ppl": 5.4, "bytes": 9_000_000_000}, "SPREAD": {"ppl": 5.5, "bytes": 9_900_000_000}}}),
    ("missing arm", {"ref_ppl": 5.0, "arms": {"OURS": {"ppl": 5.4, "bytes": 9_000_000_000}}}),
    ("no reference", {"arms": {
        "OURS": {"ppl": 5.4, "bytes": 9_000_000_000}, "SPREAD": {"ppl": 5.5, "bytes": 9_000_000_000}}}),
    ("control below reference", {"ref_ppl": 6.0, "arms": {
        "OURS": {"ppl": 5.4, "bytes": 9_000_000_000}, "SPREAD": {"ppl": 5.5, "bytes": 9_000_000_000}}}),
    ("P-1 hit, P-2 hit", {"ref_ppl": 5.0, "arms": {
        "OURS": {"ppl": 5.40, "bytes": 9_000_000_000, "tps": 4.0},
        "SPREAD": {"ppl": 5.50, "bytes": 9_000_000_000, "tps": 4.0},
        "MTP": {"ppl": 5.405, "bytes": 9_090_000_000, "tps": 4.0}}}),
    ("P-1 miss (small win)", {"ref_ppl": 5.0, "arms": {
        "OURS": {"ppl": 5.495, "bytes": 9_000_000_000}, "SPREAD": {"ppl": 5.50, "bytes": 9_000_000_000}}}),
    ("P-1 inverted", {"ref_ppl": 5.0, "arms": {
        "OURS": {"ppl": 5.60, "bytes": 9_000_000_000}, "SPREAD": {"ppl": 5.50, "bytes": 9_000_000_000}}}),
    ("P-2 miss (dense beats MoE)", {"ref_ppl": 5.0, "arms": {
        "OURS": {"ppl": 5.20, "bytes": 9_000_000_000}, "SPREAD": {"ppl": 5.50, "bytes": 9_000_000_000}}}),
    ("P-3 kill rule", {"ref_ppl": 5.0, "arms": {
        "OURS": {"ppl": 5.40, "bytes": 9_000_000_000}, "SPREAD": {"ppl": 5.50, "bytes": 9_000_000_000},
        "MTP": {"ppl": 5.30, "bytes": 9_090_000_000}}}),
    ("P-3 size outside band", {"ref_ppl": 5.0, "arms": {
        "OURS": {"ppl": 5.40, "bytes": 9_000_000_000}, "SPREAD": {"ppl": 5.50, "bytes": 9_000_000_000},
        "MTP": {"ppl": 5.402, "bytes": 9_000_100_000}}}),
    ("P-3 hurt", {"ref_ppl": 5.0, "arms": {
        "OURS": {"ppl": 5.40, "bytes": 9_000_000_000}, "SPREAD": {"ppl": 5.50, "bytes": 9_000_000_000},
        "MTP": {"ppl": 5.60, "bytes": 9_090_000_000}}}),
    ("P-4 miss", {"ref_ppl": 5.0, "arms": {
        "OURS": {"ppl": 5.40, "bytes": 9_000_000_000, "tps": 4.0},
        "SPREAD": {"ppl": 5.50, "bytes": 9_000_000_000, "tps": 3.0}}}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        seen = set()
        for name, data in SELF:
            lines, (kind, hits, misses, unscored) = score(data)
            tag = f"{kind}:{','.join(hits)}|{','.join(misses)}|{','.join(unscored)}"
            seen.add(tag)
            print(f"  {name:28s} -> {tag}")
        print(f"\n{len(SELF)} fixtures, {len(seen)} distinct verdicts - "
              "every branch above is reachable.")
        assert len(seen) >= 9, f"only {len(seen)} distinct outcomes: a branch is unreachable"
        return 0
    if not a.results:
        sys.exit("pass --results <file.json> or --self-check")
    lines, _ = score(json.load(open(a.results, encoding="utf-8")))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
