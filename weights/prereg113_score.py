"""Score pre-registration #113 - does the depth-aware edge survive an importance matrix?

Committed BEFORE the importance matrix was computed or any arm built. Reads the results JSON
written by weights/prereg113_run.py and prints a verdict nobody adjusts after seeing the numbers.

    python weights/prereg113_score.py --results weights/data/prereg113.json
    python weights/prereg113_score.py --self-check   # every branch reachable, no data needed
"""

import argparse
import json
import sys

REF_112 = 5.5950  # #112's Q8_0 reference perplexity; the base run must reproduce it
REF_TOL = 0.005  # within 0.5%, or the protocol has drifted and the run is VOID
TWIN_OURS, TWIN_SPREAD = 6.0590, 6.4136  # #112's arms, no imatrix
SHARE_112 = 0.433
P2_LO, P2_HI = 0.05, 0.30
SIZE_GATE = 0.005


def score(d):
    """-> (lines, (kind, hits, misses, unscored)). Pure, so --self-check can drive every branch."""
    out, hits, misses, unscored = [], [], [], []
    ref = d.get("ref_ppl_measured")
    if ref is None:
        return ["VOID: the base run did not report a reference perplexity."], ("VOID-no-reference", [], [], [])
    drift = abs(ref - REF_112) / REF_112
    out.append(f"reference reproduced: {ref:.4f} vs #112 {REF_112:.4f} ({drift * 100:.2f}% apart)")
    if drift > REF_TOL:
        out.append(f"  VOID - the protocol drifted more than {REF_TOL * 100:.1f}% from #112.")
        return out, ("VOID-reference-drift", [], [], [])

    arms = d.get("arms") or {}
    o, s = arms.get("OURS-I"), arms.get("SPREAD-I")
    if not (o and s) or any(x.get("kld_mean") is None or x.get("ppl") is None for x in (o, s)):
        out.append("VOID: OURS-I and SPREAD-I both need a completed evaluation.")
        return out, ("VOID-arm-incomplete", [], [], [])
    gap = abs(o["bytes"] - s["bytes"]) / max(o["bytes"], s["bytes"])
    out.append(f"size gate: {o['bytes']:,} vs {s['bytes']:,} bytes -> {gap * 100:.3f}%")
    if gap > SIZE_GATE:
        out.append("  VOID - a size-mismatched pair cannot separate placement from budget.")
        return out, ("VOID-size-gate", [], [], [])

    out.append(f"\nmean KLD: OURS-I {o['kld_mean']:.5f} | SPREAD-I {s['kld_mean']:.5f}")
    if o["kld_mean"] < s["kld_mean"]:
        out.append("P-1 HIT - the depth-aware edge survives an importance matrix.")
        hits.append("P-1")
    else:
        out.append("P-1 MISS - KILL RULE: the edge does not survive standard practice. Pivot to "
                   "prediction; the README's quantizer claims need a scope line.")
        misses.append("P-1 (kill rule)")

    ex_s = s["ppl"] - REF_112
    if ex_s <= 0:
        out.append("P-2 UNSCORED: SPREAD-I is at or below the reference, so no excess exists.")
        unscored.append("P-2")
        share = None
    else:
        share = (s["ppl"] - o["ppl"]) / ex_s
        out.append(f"P-2: OURS-I removes {share * 100:.1f}% of SPREAD-I's excess perplexity "
                   f"(#112: {SHARE_112 * 100:.1f}%; staked {P2_LO * 100:.0f}-{P2_HI * 100:.0f}%)")
        if share < P2_LO:
            out.append("  MISS - collapsed to the Unsloth-level band. Decision gate: pivot.")
            misses.append("P-2 (collapsed)")
        elif share <= P2_HI:
            out.append("  HIT - it survives, and it shrank as predicted.")
            hits.append("P-2")
        elif share < SHARE_112:
            out.append("  MISS - it shrank, but by less than staked.")
            misses.append("P-2")
        else:
            out.append("  MISS - it did not shrink at all. The imatrix did not erode the edge.")
            misses.append("P-2 (no shrink)")

    both = o["ppl"] < TWIN_OURS and s["ppl"] < TWIN_SPREAD
    out.append(f"\nP-3: OURS {TWIN_OURS} -> {o['ppl']:.4f} | SPREAD {TWIN_SPREAD} -> {s['ppl']:.4f}")
    out.append("  HIT" if both else "  MISS - the imatrix did not help both arms.")
    (hits if both else misses).append("P-3")

    gain_s, gain_o = TWIN_SPREAD - s["ppl"], TWIN_OURS - o["ppl"]
    out.append(f"P-4: imatrix gain SPREAD {gain_s:+.4f} vs OURS {gain_o:+.4f} PPL")
    out.append("  HIT" if gain_s > gain_o else "  MISS - the imatrix did not favour the worse arm.")
    (hits if gain_s > gain_o else misses).append("P-4")

    for name in ("OURS-I", "SPREAD-I", "UNIFORM-I"):
        x = arms.get(name)
        if x and x.get("kld_mean") is not None:
            out.append(f"\n  {name:9s} {x['bytes']:>15,} B  PPL {x['ppl']:.4f}  KLD {x['kld_mean']:.5f}"
                       f"  p99 {x.get('kld_p99', float('nan')):.4f}"
                       f"  top-1 {x.get('top1', float('nan')):.2f}%")
    n = len(hits) + len(misses)
    out.append(f"\nVERDICT: {len(hits)}/{n} scored" + (f", {len(unscored)} unscored" if unscored else ""))
    return out, ("SCORED", hits, misses, unscored)


def _arms(o_ppl, o_kld, s_ppl, s_kld, s_bytes=12_000_000_000):
    return {"OURS-I": {"bytes": 12_000_000_000, "ppl": o_ppl, "kld_mean": o_kld},
            "SPREAD-I": {"bytes": s_bytes, "ppl": s_ppl, "kld_mean": s_kld}}


SELF = [
    ("no reference", {"arms": _arms(5.9, 0.1, 6.1, 0.2)}),
    ("reference drifted", {"ref_ppl_measured": 5.70, "arms": _arms(5.9, 0.1, 6.1, 0.2)}),
    ("arm missing", {"ref_ppl_measured": 5.595, "arms": {"OURS-I": {"bytes": 1, "ppl": 5.9,
                                                                    "kld_mean": 0.1}}}),
    ("size gate", {"ref_ppl_measured": 5.595, "arms": _arms(5.9, 0.1, 6.1, 0.2, 13_000_000_000)}),
    ("all hit", {"ref_ppl_measured": 5.595, "arms": _arms(5.95, 0.10, 6.05, 0.12)}),
    ("P-1 kill rule", {"ref_ppl_measured": 5.595, "arms": _arms(6.05, 0.13, 6.00, 0.12)}),
    ("P-2 collapsed", {"ref_ppl_measured": 5.595, "arms": _arms(5.99, 0.10, 6.00, 0.11)}),
    ("P-2 shrank too little", {"ref_ppl_measured": 5.595, "arms": _arms(5.84, 0.10, 6.00, 0.12)}),
    ("P-2 no shrink", {"ref_ppl_measured": 5.595, "arms": _arms(5.75, 0.08, 6.00, 0.12)}),
    ("P-2 unscored", {"ref_ppl_measured": 5.595, "arms": _arms(5.50, 0.05, 5.55, 0.06)}),
    ("P-3 miss", {"ref_ppl_measured": 5.595, "arms": _arms(6.10, 0.10, 6.20, 0.12)}),
    ("P-4 miss", {"ref_ppl_measured": 5.595, "arms": _arms(5.70, 0.10, 6.40, 0.12)}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        seen = set()
        for name, data in SELF:
            _, (kind, h, m, u) = score(data)
            tag = f"{kind}:{','.join(h)}|{','.join(m)}|{','.join(u)}"
            seen.add(tag)
            print(f"  {name:24s} -> {tag}")
        print(f"\n{len(SELF)} fixtures, {len(seen)} distinct verdicts")
        assert len(seen) == len(SELF), f"{len(seen)} distinct outcomes for {len(SELF)} fixtures"
        return 0
    if not a.results:
        sys.exit("pass --results <file.json> or --self-check")
    lines, _ = score(json.load(open(a.results, encoding="utf-8")))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
