"""Score pre-registration #114 - does a low-rank correction buy more quality per byte than precision?

Committed BEFORE any spectrum of the 27B was measured. Reads the JSON written by
weights/prereg114_spectrum.py and prints a verdict nobody adjusts after seeing the numbers.

    python weights/prereg114_score.py --results weights/data/prereg114.json
    python weights/prereg114_score.py --self-check   # every branch reachable, no data needed

Margin M = 10*log10(eps_precision / eps_adapter) at equal bits per weight. eps_precision is the
lower convex hull of the K-quant ladder (mixing whole tensors interpolates linearly in eps);
eps_adapter is the base format's error minus the energy of the best rank whose stored cost fits
the budget. Positive M: the adapter removes more error than the same bytes of precision.
"""

import argparse
import json
import math
import statistics
import sys

LADDER = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]
KINDS = ["ffn_gate", "ffn_up", "ffn_down"]
N_TENSORS = 30
PRIMARY = 0.25  # bpw above the base: a tenth more bytes on a Q2_K tensor
BUDGETS = [0.05, 0.10, 0.25, 0.50]
STAGE_B_DB = 0.5  # median margin that justifies building a real adapter and paying a KLD run
P1_MIN, P2_MIN = 8, 25


def hull_eps(points, b):
    """eps on the lower convex hull of (bpw, eps) points, at bits-per-weight b."""
    pts = sorted(points)
    hull = []
    for p in pts:  # monotone chain, lower hull
        while len(hull) >= 2 and ((hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1])
                                  - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0])) <= 0:
            hull.pop()
        hull.append(p)
    if b <= hull[0][0]:
        return hull[0][1]
    for (x0, y0), (x1, y1) in zip(hull, hull[1:], strict=False):  # adjacent pairs
        if x0 <= b <= x1:
            return y0 + (y1 - y0) * (b - x0) / (x1 - x0)
    return hull[-1][1]


def share(rec, fmt, rank):
    sp = rec["spectra"][fmt]
    return sp["top"][sp["ranks"].index(rank)] / rec["formats"][fmt]["eps"]


def margin(rec, base, budget):
    """(dB, rank) for the best adapter on `base` within `budget` bpw, against the precision hull."""
    f = rec["formats"]
    pts = [(f[k]["bpw"], f[k]["eps"]) for k in LADDER]
    sp = rec["spectra"][base]
    best_r, removed = 0, 0.0
    for r, top in zip(sp["ranks"], sp["top"], strict=True):
        if rec["adapter_bpw"][str(r)] <= budget:
            best_r, removed = r, top
    eps_ad = f[base]["eps"] - removed
    eps_pr = hull_eps(pts, f[base]["bpw"] + budget)
    return 10 * math.log10(eps_pr / eps_ad), best_r


def score(d):
    """-> (lines, (kind, hits, misses, decision)). Pure, so --self-check can drive every branch."""
    out, hits, misses = [], [], []
    if not (d.get("protocol") or {}).get("validated"):
        return ["VOID: in-process quantization was never shown to match llama-quantize."], \
            ("VOID-unvalidated", [], [], "")
    ts = d.get("tensors") or {}
    if d.get("status") != "complete" or len(ts) != N_TENSORS:
        return [f"VOID: {len(ts)}/{N_TENSORS} tensors measured, status {d.get('status')}."], \
            ("VOID-incomplete", [], [], "")
    for rec in ts.values():
        nums = [v["eps"] for v in rec["formats"].values()] + [x for s in rec["spectra"].values() for x in s["top"]]
        if not all(math.isfinite(x) and x >= 0 for x in nums):
            return [f"VOID: non-finite or negative value in {rec['name']}."], ("VOID-nonfinite", [], [], "")

    by = {}
    for rec in ts.values():
        by.setdefault(rec["block"], {})[rec["kind"]] = rec

    # P-1: ffn_down's weighted error is the most concentrated in its block
    wins = sum(1 for b in by.values()
               if share(b["ffn_down"], "Q2_K", 16) > max(share(b["ffn_gate"], "Q2_K", 16),
                                                         share(b["ffn_up"], "Q2_K", 16)))
    out.append(f"P-1: ffn_down most concentrated (top-16 share) in {wins}/{len(by)} blocks "
               f"(staked >= {P1_MIN})")
    (hits if wins >= P1_MIN else misses).append("P-1")

    # P-2: the importance matrix flattens the spectrum an adapter would exploit
    flat = sum(1 for r in ts.values() if share(r, "Q2_K-noimat", 16) > share(r, "Q2_K", 16))
    out.append(f"P-2: top-16 share higher WITHOUT the imatrix in {flat}/{len(ts)} tensors "
               f"(staked >= {P2_MIN})")
    (hits if flat >= P2_MIN else misses).append("P-2")

    # P-3: at the primary budget the adapter does not beat precision, for any tensor kind
    med = {}
    for k in KINDS:
        ms = [margin(r, "Q2_K", PRIMARY)[0] for r in ts.values() if r["kind"] == k]
        med[k] = statistics.median(ms)
        out.append(f"P-3: {k:8s} median margin at +{PRIMARY} bpw on Q2_K: {med[k]:+.2f} dB "
                   f"(range {min(ms):+.2f}..{max(ms):+.2f})")
    if all(v < 0 for v in med.values()):
        out.append("  HIT - precision wins at equal bytes for every kind.")
        hits.append("P-3")
    else:
        misses.append("P-3")
    win = [k for k, v in med.items() if v >= STAGE_B_DB]
    if win:
        decision = "STAGE-B:" + ",".join(win)
        out.append(f"  MISS - the adapter wins by >= {STAGE_B_DB} dB on {win}: Stage B (a real LoRA "
                   "file, KLD at equal bytes) is staked for those kinds.")
    elif "P-3" in misses:
        decision = "MARGINAL"
        out.append(f"  MISS, but below {STAGE_B_DB} dB: reported, not pursued - a runtime adapter "
                   "costs a second matmul per tensor.")
    else:
        decision = "DEAD-END"
        out.append("  Correction adapters are a dead end for imatrix K-quants on this model "
                   "(scoped to diagonal-metric corrections).")

    out.append("\ndescriptive: median margin (dB) by budget, base Q2_K / Q3_K, band | outside")
    for k in KINDS:
        row = []
        for base in ("Q2_K", "Q3_K"):
            for band in (True, False):
                ms = [margin(r, base, PRIMARY)[0] for r in ts.values() if r["kind"] == k and r["band"] == band]
                row.append(f"{statistics.median(ms):+.2f}")
        out.append(f"  {k:8s} Q2_K {row[0]} | {row[1]}   Q3_K {row[2]} | {row[3]}")
    for b in BUDGETS:
        out.append(f"  +{b:.2f} bpw on Q2_K: " + "  ".join(
            f"{k} {statistics.median([margin(r, 'Q2_K', b)[0] for r in ts.values() if r['kind'] == k]):+.2f}"
            for k in KINDS))
    n = len(hits) + len(misses)
    out.append(f"\nVERDICT: {len(hits)}/{n} predictions hit; decision {decision}")
    return out, ("SCORED", hits, misses, decision)


# ---------------------------------------------------------------------------- self-check
RANKS = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512]
SHAPES = {"ffn_gate": (17408, 5120), "ffn_up": (17408, 5120), "ffn_down": (5120, 17408)}


def _adapter_bpw(r, m, n):
    return (r * n * (8.5 if n % 32 == 0 else 16.0) + m * r * (8.5 if r % 32 == 0 else 16.0)) / (m * n)


def _rec(kind, block, s16, noimat_ratio):
    m, n = SHAPES[kind]
    eps = {"Q2_K": 1.0, "Q3_K": 0.32, "Q4_K": 0.09, "Q5_K": 0.025, "Q6_K": 0.007, "Q2_K-noimat": 1.3}
    bpw = {"Q2_K": 2.625, "Q3_K": 3.4375, "Q4_K": 4.5, "Q5_K": 5.5, "Q6_K": 6.5625, "Q2_K-noimat": 2.625}

    def sp(fmt, s):
        s = min(s, 0.99)
        return {"ranks": RANKS, "top": [eps[fmt] * (1 - (1 - s) ** (r / 16)) for r in RANKS], "min_dim": min(m, n)}

    return {"name": f"blk.{block}.{kind}.weight", "block": block, "kind": kind, "band": block >= 51,
            "formats": {f: {"bpw": bpw[f], "eps": eps[f]} for f in eps},
            "spectra": {"Q2_K": sp("Q2_K", s16), "Q2_K-noimat": sp("Q2_K-noimat", s16 * noimat_ratio),
                        "Q3_K": sp("Q3_K", s16)},
            "adapter_bpw": {str(r): _adapter_bpw(r, m, n) for r in RANKS}}


def _data(gate, up, down, noimat_ratio=1.5, validated=True, status="complete", drop=0, poison=False):
    ts = {}
    for blk in [51, 54, 57, 60, 63, 5, 15, 25, 35, 45]:
        for kind, s in (("ffn_gate", gate), ("ffn_up", up), ("ffn_down", down)):
            r = _rec(kind, blk, s, noimat_ratio)
            ts[r["name"]] = r
    for name in list(ts)[:drop]:
        del ts[name]
    if poison:
        next(iter(ts.values()))["formats"]["Q2_K"]["eps"] = float("nan")
    return {"protocol": {"validated": validated}, "status": status, "tensors": ts}


WHITE = 0.0074  # ~ the Marchenko-Pastur top-16 share of a 5120 x 17408 white matrix
SELF = [
    ("unvalidated", _data(WHITE, WHITE, 0.02, validated=False)),
    ("incomplete", _data(WHITE, WHITE, 0.02, drop=3)),
    ("non-finite", _data(WHITE, WHITE, 0.02, poison=True)),
    ("dead end, all hit", _data(WHITE, WHITE, 0.02)),
    ("P-1 miss", _data(0.02, WHITE, 0.01)),
    ("P-2 miss", _data(WHITE, WHITE, 0.02, noimat_ratio=0.8)),
    ("marginal (down)", _data(WHITE, WHITE, 0.05)),
    ("stage B (down)", _data(WHITE, WHITE, 0.10)),
    ("stage B (all)", _data(0.10, 0.10, 0.12)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        seen = set()
        for name, data in SELF:
            _, (kind, h, m, dec) = score(data)
            tag = f"{kind}:{','.join(h)}|{','.join(m)}|{dec}"
            seen.add(tag)
            print(f"  {name:20s} -> {tag}")
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
