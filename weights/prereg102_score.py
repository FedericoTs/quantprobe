"""Score pre-registration #102 - does per-layer effective rank predict the fragile band?

WRITTEN AND COMMITTED BEFORE THE FIRST SINGULAR VALUE WAS COMPUTED (KR-3). No
weights/data/rank_bands.json existed when this file was staked; the ground truth below is
copied from probe artifacts that predate the question. Same discipline as prereg100_score.py,
same reason: a verdict computed by code written in ignorance of the data cannot be tuned to
the data.

    python weights/prereg102_score.py

The prereg (preregistrations/2026-08-16-rank-predicts-fragility.md) staked five models; the
amendment committed alongside this scorer adds qwen3-30b as a sixth - its band deltas were
already sitting in quantprobe/recipes/qwen3-30b.json with the same 4-band probe method, and
leaving a measured row out of a 5-row experiment needs a better reason than "the table was
drawn first". The bar scales with it: ">= 4 of 5" becomes ">= 5 of 6". Mistral stays
load-bearing either way (KR-4): it is the only front-fragile model, so any signal that only
ever says "the back" must turn around on it or the verdict is confirmation theater.

Contract for weights/data/rank_bands.json, produced later by the rank computation:

    {"<model key>": [b0, b1, b2, b3], ...}                      four band scores, front->back
    {"<model key>": {"bands": [...], "source": "<file>"}, ...}  also accepted; source printed
                                                                (the prereg requires the per-
                                                                model source file in the log)

Band boundaries are frozen in BANDS below, copied from the probe artifacts, so the rank
computation and this scorer cannot disagree about which layers "band 2" means (the prereg's
"identical band boundaries" clause). A staked model missing from the json, or a staked row
without exactly 4 finite band scores, refuses the whole verdict (SystemExit): no partial
scoring, all six models or nothing.

Staked outcomes (thresholds are the module constants below):

    P-A RANK PREDICTS            rho <= -0.8 in >= 5 of 6 models, mistral-7b among them.
                                 Staked direction: LOWER rank score <-> HIGHER delta_ppl.
    P-B PREDICTIVE, WRONG SIGN   rho >= +0.8 in >= 5 of 6 models, mistral-7b among them.
    P-C RANK FAILS               anything else - weak, inconsistent, or Mistral refuses
                                 to turn around.

n = 4 points per model, so with distinct ranks Spearman rho is quantized to multiples of
0.2: |rho| >= 0.8 means the band ordering is at most one adjacent swap from perfect. That
coarseness is why the prereg's bar is "models consistent", not a p-value ritual.
"""
from __future__ import annotations
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RANK_BANDS = os.path.join(HERE, "data", "rank_bands.json")

RHO_PREDICTS = -0.8      # P-A per-model bar: staked direction, low rank <-> high delta
RHO_WRONG_SIGN = 0.8     # P-B per-model bar: same strength, opposite sign
NEEDED = 5               # of the 6 staked models (amendment: the prereg's five + qwen3-30b)
ANCHOR = "mistral-7b"    # KR-4: the front-fragile control must be in the passing set
EPS = 1e-9               # float guard at the exact +-0.8 boundary (see spearman())

# Ground truth: measured band delta_ppl per model, probe band order (front -> back).
# KR-2 READ-ONLY: copied verbatim from the committed probe artifacts named per row; no
# probe is re-run to make a row agree.
GROUND_TRUTH = {
    # quantprobe/recipes/mistral-7b.json band_deltas (the front-fragile control, 27x median)
    "mistral-7b":  [6.525, 0.148, 0.2248, 0.2587],
    # quantprobe/recipes/qwen2.5-7b.json band_deltas
    "qwen2.5-7b":  [0.8829, 0.5406, 0.5814, 1.8545],
    # quantprobe/recipes/qwen3-30b.json band_deltas (the amendment's sixth model)
    "qwen3-30b":   [0.5907, 0.0165, 0.1352, 1.3374],
    # quantprobe/recipes/qwen3.5-35b.json band_deltas
    "qwen3.5-35b": [0.0909, 0.1531, 0.1579, 0.4492],
    # weights/data/prereg100_probe_4b.log, "layers L-L: PPL p (delta d)" lines, 4dp
    "qwen3.5-4b":  [0.1778, 0.3727, 0.6275, 1.0677],
    # weights/data/prereg101_probe_qwen38_q4.log, same line format, 4dp
    "qwen3.8-27b": [0.0452, 0.2122, 0.2925, 0.5926],
}

# The probe's own band boundaries (inclusive layer ranges), frozen here so the rank
# computation scores exactly the layers the probe quantized - the prereg's "identical band
# boundaries" clause made checkable.
BANDS = {
    "mistral-7b":  [(0, 7), (8, 15), (16, 23), (24, 31)],    # quantprobe/recipes/mistral-7b.json
    "qwen2.5-7b":  [(0, 6), (7, 13), (14, 20), (21, 27)],    # quantprobe/recipes/qwen2.5-7b.json
    "qwen3-30b":   [(0, 11), (12, 23), (24, 35), (36, 47)],  # quantprobe/recipes/qwen3-30b.json
    "qwen3.5-35b": [(0, 9), (10, 19), (20, 29), (30, 39)],   # quantprobe/recipes/qwen3.5-35b.json
    "qwen3.5-4b":  [(0, 7), (8, 15), (16, 23), (24, 31)],    # weights/data/prereg100_probe_4b.log
    "qwen3.8-27b": [(0, 16), (17, 33), (34, 50), (51, 64)],  # weights/data/prereg101_probe_qwen38_q4.log
}


def _ranks(xs):
    """1-based average ranks; ties share the mean of their positions."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return out


def spearman(xs, ys):
    """Spearman rho as Pearson over average ranks - tie-safe, unlike the 6*d^2 shortcut.

    Returns None when either side has constant ranks: an undefined rho counts toward
    neither staked direction. The single sqrt over the PRODUCT of squared deviations keeps
    the distinct-rank case exact at the +-0.8 threshold: sqrt(5)*sqrt(5) != 5.0 in floats,
    sqrt(5*5) is, and a verdict must not hinge on which way a sqrt rounded (EPS guards the
    remaining tie cases).
    """
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    d2 = sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    if d2 == 0:
        return None
    return num / math.sqrt(d2)


def _load_bands(path):
    """{model: (4 band scores, source or None)} for the staked models, or SystemExit.

    Refusal, not partial scoring: a verdict over a subset of the staked models is exactly
    the confirmation theater KR-4 exists to prevent (score 5 back-fragile rows, skip the
    control, declare victory).
    """
    if not os.path.exists(path):
        raise SystemExit(
            f"prereg #102: {path} not found - the rank computation has not run yet. "
            "That is the expected state at commit time (KR-3: scorer first). No verdict.")
    with open(path, encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except ValueError as e:
            raise SystemExit(f"prereg #102: {path} is not valid JSON ({e}). No verdict.")
    if not isinstance(raw, dict):
        raise SystemExit(
            f"prereg #102: {path} must be an object keyed by model, per the contract in "
            "prereg102_score.py. No verdict.")

    missing = sorted(m for m in GROUND_TRUTH if m not in raw)
    if missing:
        raise SystemExit(
            "prereg #102: rank_bands.json is missing staked models: " + ", ".join(missing)
            + ". All six or nothing (KR-4: Mistral is not optional, and neither is the "
            "rest of the stake). No verdict.")

    out = {}
    for model in GROUND_TRUTH:
        row = raw[model]
        source = None
        if isinstance(row, dict):
            source = row.get("source")
            row = row.get("bands")
        ok = (isinstance(row, list) and len(row) == 4
              and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      and math.isfinite(v) for v in row))
        if not ok:
            raise SystemExit(
                f"prereg #102: {model} does not carry exactly 4 finite band scores "
                f"(got {row!r}). The probe measured 4 bands per model; anything else is "
                "a bug in the rank computation, not a scoreable row. No verdict.")
        out[model] = ([float(v) for v in row], source)
    return out


def score(path=None):
    """Print the per-model table and the mechanical verdict; return {verdict, per_model}."""
    path = path or RANK_BANDS
    bands = _load_bands(path)

    print("prereg #102 - does effective rank predict the fragile band? (U-46, the fork decider)")
    print(f"  rank scores: {path}")
    print("  ground truth: 6 measured band-delta rows, embedded read-only (KR-2)")
    print(f"  staked: rho <= {RHO_PREDICTS} (LOWER rank <-> HIGHER delta) in >= {NEEDED} of "
          f"{len(GROUND_TRUTH)} models, {ANCHOR} required (KR-4)")

    extras = sorted(set(json.load(open(path, encoding="utf-8"))) - set(GROUND_TRUTH))
    if extras:
        print("  note: rows outside the stake, ignored: " + ", ".join(extras))

    rhos = {}
    for model in GROUND_TRUTH:
        scores, source = bands[model]
        deltas = GROUND_TRUTH[model]
        rho = spearman(scores, deltas)
        rhos[model] = rho

        if rho is None:
            tag = "rho undefined (constant band scores) - counts toward neither direction"
        elif rho <= RHO_PREDICTS + EPS:
            tag = f"rho {rho:+.2f}  staked direction, past threshold - counts toward P-A"
        elif rho >= RHO_WRONG_SIGN - EPS:
            tag = f"rho {rho:+.2f}  wrong sign, past threshold - counts toward P-B"
        else:
            lean = "leans staked" if rho < 0 else ("leans wrong-sign" if rho > 0 else "flat")
            tag = f"rho {rho:+.2f}  {lean}, below threshold - counts toward neither"

        label = " / ".join(f"{lo}-{hi}" for lo, hi in BANDS[model])
        src = f"  [source: {source}]" if source else ""
        print(f"\n  {model}  (bands {label}){src}")
        print("    rank score " + " ".join(f"{v:>8.4f}" for v in scores))
        print("    delta_ppl  " + " ".join(f"{v:>8.4f}" for v in deltas))
        print(f"    {tag}")

    neg = sorted(m for m, r in rhos.items() if r is not None and r <= RHO_PREDICTS + EPS)
    pos = sorted(m for m, r in rhos.items() if r is not None and r >= RHO_WRONG_SIGN - EPS)

    print("\n=== verdict ===")
    print(f"  staked direction (rho <= {RHO_PREDICTS}): {len(neg)} of {len(GROUND_TRUTH)}"
          + (f"  [{', '.join(neg)}]" if neg else "")
          + f"; {ANCHOR} {'IN' if ANCHOR in neg else 'NOT in'} the set")
    print(f"  wrong sign      (rho >= +{RHO_WRONG_SIGN}): {len(pos)} of {len(GROUND_TRUTH)}"
          + (f"  [{', '.join(pos)}]" if pos else "")
          + f"; {ANCHOR} {'IN' if ANCHOR in pos else 'NOT in'} the set")

    if len(neg) >= NEEDED and ANCHOR in neg:
        verdict = "P-A"
        print("\n  P-A RANK PREDICTS - the breakthrough. The staked consequence: build "
              "probe --fast\n  (rank-only band pick), validate it against these measured "
              "bands, and the probe\n  becomes its confirmation tool. A screen is "
              "validated, not a probe replacement.")
    elif len(pos) >= NEEDED and ANCHOR in pos:
        verdict = "P-B"
        print("\n  P-B PREDICTIVE, WRONG SIGN - a predictor is a predictor. Report the "
              "staked\n  direction (low rank -> fragile) as wrong; the follow-up recipe "
              "work inverts the sign.")
    else:
        verdict = "P-C"
        print("\n  P-C RANK FAILS - Law 3 strengthened: effective rank joins kurtosis and "
              "architecture\n  family on the refuted-predictor list. The probe remains the "
              "only honest instrument,\n  and the fork decision input is: no cheap "
              "breakthrough here.")
        if ANCHOR not in neg and len(neg) >= NEEDED - 1:
            print(f"  ({ANCHOR} refused to turn around - the failure mode the prereg "
                  "named as decisive.)")

    return {"verdict": verdict, "per_model": rhos}


if __name__ == "__main__":
    score()
