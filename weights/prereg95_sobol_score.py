"""Score pre-registration #95 stage 2 - Sobol variance attribution on the Morris survivors.

WRITTEN AND COMMITTED BEFORE THE FIRST STAGE-2 CSV ROW EXISTED. The design doc
(docs/DESIGN_DOE_SOBOL.md, section 5) precommits this scorer the same way stage 1
precommitted weights/prereg95_score.py, and for the same reason: a verdict computed by
code written in ignorance of the data cannot be tuned to the data. Neither
weights/data/doe_sobol_stage2.csv nor the --stage2 harness mode that will write it
existed when this file was staked.

    python weights/prereg95_sobol_score.py               # score the real CSV (refuses pre-data)
    python weights/prereg95_sobol_score.py --self-check  # hand-derived fixture + drills, exit 0

Reads  weights/data/doe_sobol_stage2.csv (append-only, written by weights/doe_morris.py
--stage2), writes weights/data/prereg95_sobol_verdict.json and prints the table. It
scores ONLY P-3 (plus the Morris-vs-Sobol adjudication state); it refuses everything
else, including any request to re-score the stage-1 stakes. It edits no prereg, no
README, no plan.py and no chart - it PRINTS the exact scope-label text and the OPERATOR
ships it.

This module carries its OWN copy of the seeded Saltelli design generator
(build_stage2_design), and the harness will carry another - two copies ON PURPOSE, same
rule as stage 1: this copy is the FROZEN precommit expectation, and importing the design
from the harness would let a post-stake harness edit move the expectation with it. At
score time the harness's live build_stage2_design is imported and compared over the full
N_cap stream, REFUSING on drift (the frozen copy wins). Today the harness does not carry
--stage2 yet - this scorer is committed first, per house rule - so an absent function is
a loud warning, not a refusal: the frozen copy joined against the CSV's run_ids is the
load-bearing gate either way.

Estimators, named in advance (design section 5.3): first-order S_i from Saltelli et al.
2010 Table 2, estimator (b); total-order ST_i from Jansen 1999. P-3 keys on the ST
argmax, as staked ("highest Sobol TOTAL-ORDER index"). Bootstrap: 1000 seeded resamples
of Saltelli blocks; the argmax is DECIDED iff the modal top factor holds rank 1 in
>= 950/1000 resamples (section 5.4).

The five-case P-3 decision table (design section 1.3), with s = Sobol ST argmax after
the decidability gate, m = the frozen Morris argmax, MAP = the frozen mapping set:

    s == m and s in MAP        PASS               publishable
    s == m and s not in MAP    FAIL               publishable (methods agree, mapping refuted)
    s != m and s not in MAP    FAIL               publishable (mapping fails under EITHER
                                                  argmax; only the top-factor identity is
                                                  held for Taguchi)
    s != m and s in MAP        HELD_FOR_TAGUCHI   prereg kill rule 3: neither method
                                                  publishes until a Taguchi arm adjudicates
    gate undecided at N_cap    UNDECIDABLE        the scope label ships anyway (amendment
                                                  item 7: stricter than staked, declared)

A theorem of the frozen constants, restated so no reader expects a PASS: both Morris
argmaxes (ngl, t) sit OUTSIDE their mapping sets, so the PASS row is unreachable on both
models - stage 2 alone can only produce FAIL, HELD_FOR_TAGUCHI or UNDECIDABLE. The
self-check asserts this instead of trusting it.

Refusals (exact messages, checked in this order - the strings are part of the stake):
absent CSV; header sha256 drift against the pinned design hash; harness/frozen-copy
design drift; per-model incomplete design (largest complete block prefix below N_start -
a declared DNF row counts as present, a missing row does not). A model at or past
N_start is scored at its N_complete even while the other model is refused: per-model
verdicts are the point of the two-night split, and a publishable 7B FAIL the morning
after night 1 short-circuits overall P-3. Validity floor: a survivor with fewer than
ceil(0.75 * N_complete) valid ST pairs is UNSCOREABLE and the model's P-3 is VOID, never
guessed.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_REL = "weights/data/doe_sobol_stage2.csv"           # how the stake names the file
CSV_PATH = os.path.join(HERE, "data", "doe_sobol_stage2.csv")
VERDICT_PATH = os.path.join(HERE, "data", "prereg95_sobol_verdict.json")

# The design section-4 CSV header, byte for byte: stage-1 columns with traj,pos,
# changed_factor replaced by block,matrix (30 columns). Column order is load-bearing:
# the refusal below hash-pins it so a drifted schema is refused, never mis-scored.
HEADER = (
    "run_id,model,block,matrix,ngl,ub,t,ctk,mmp,fa,moe_cpu_frac,status,tok_s,"
    "stddev_ts,reps_tok_s,wall_s,settle_s,free_ram_gb_pre,temp_pre,sm_mhz_pre,"
    "mem_mhz_pre,vram_mib_pre,power_w_pre,temp_post,sm_mhz_post,mem_mhz_post,"
    "vram_mib_post,power_w_post,ts_utc,cmd")
HEADER_SHA256 = "95fe4b76345921ea95d6a86cc83b30caad741f799515716d0d07f687c2d89ade"
COLS = HEADER.split(",")

MODELS = ("7B", "30B")

# ---------------------------------------------------------------------------
# Section-5.1 embedded ground truth - QUOTED from the design, never recomputed at score
# time: a later quantprobe release re-classifying this box must not move the goalposts
# of a stake scored against the 2026-08-16 classes. The 30B margin line is recorded as
# context ONLY - the near-tie (51/49) does not widen the mapping set, because widening
# it after seeing Sobol data would be exactly the pick-the-flattering-reading move the
# prereg forbids.
# ---------------------------------------------------------------------------

PLAN_BINDING = {
    "7B": ("binding constraint: BANDWIDTH-BOUND (VRAM bandwidth) - 100% of every "
           "decode token is spent there."),
    "30B": ("binding constraint: BANDWIDTH-BOUND (system RAM bandwidth) - 51% of every "
            "decode token is spent there."),
}   # quantprobe plan, 2026-08-16, auto-detected hw, calibration 2026-07-31 state 2dc97d41
PLAN_MARGIN_30B = ("1.03x - system RAM bandwidth must get that much faster before VRAM "
                   "bandwidth (49%) takes over")
MAPPING_SET = {"7B": ("ctk", "ub"), "30B": ("moe_cpu_frac", "ctk")}
MORRIS_ARGMAX = {"7B": "ngl", "30B": "t"}       # prereg95_verdict.json, frozen
N_START = {"7B": 32, "30B": 40}
N_CAP = 64

N_RESAMPLES = 1000       # bootstrap resamples of Saltelli blocks (design sec 5.4)
GATE_RETENTION = 950     # DECIDED iff modal top factor holds rank 1 in >= 950/1000
FLOOR_FRAC = 0.75        # validity floor: ceil(0.75 * N_complete) valid ST pairs
EXTEND_STEP = 16         # UNDECIDED -> extend the same stream by +16 blocks

# The scope label the kill rule ships on a publishable FAIL - and, per amendment item 7,
# on UNDECIDABLE at N_cap too (the text is literally true in that branch, and perpetual
# undecidability must not become a shield). The scorer prints it; the OPERATOR ships it.
LABEL = "derived from the law, not confirmed by variance attribution (prereg #95 P-3)"
LABEL_SURFACES = (
    "the 'binding constraint:' print in quantprobe plan / report "
    "(quantprobe/plan.py, quantprobe/report.py)",
    "the README example block that quotes it (README.md)",
    "the pipeline chart asset that draws it (weights/data/card_flagship.svg)",
)

# ---------------------------------------------------------------------------
# Stage-2 design constants (design sections 2 and 4). Survivor order is the stage-1
# FACTORS tuple order restricted to survivors - NOT mu_star order - so the seeded
# stream layout cannot drift if a mu_star tie is ever re-ranked. Level lists are the
# stage-1 lists UNCHANGED: narrowing a range after seeing which regime carries the
# variance would quietly change what "top factor" means mid-stake.
# ---------------------------------------------------------------------------

SURVIVORS = {
    "7B": ("ngl", "ub", "t", "ctk"),
    "30B": ("ngl", "t", "ctk", "mmp", "moe_cpu_frac"),
}
# Non-survivors fixed at the stage-1 best-observed levels (design sec 2). Every fixed
# factor is outside both mapping sets, so fixing can only remove FAIL modes.
FIXED = {
    "7B": (("mmp", 0), ("fa", 0)),
    "30B": (("ub", 2048), ("fa", 1)),
}
LEVELS = {
    "7B": {
        "ngl": (0, 9, 19, 99),
        "ub": (128, 512, 1024, 2048),
        "t": (1, 2, 3, 4),
        "ctk": ("f16", "q8_0"),
    },
    "30B": {
        "ngl": (0, 16, 32, 99),
        "t": (1, 2, 3, 4),
        "ctk": ("f16", "q8_0"),
        "mmp": (1, 0),
        "moe_cpu_frac": (0.75, 0.833, 0.917, 1.0),
    },
}
SEED_FMT = "prereg95:stage2:{tag}:20260816"     # the date THIS design was frozen -
BOOTSTRAP_SEED = "prereg95:stage2:bootstrap:20260816"   # stage 1's 20260807 is not reused

# ---------------------------------------------------------------------------
# Staked refusal and verdict messages. Refusal 1 always names the file by its staked
# repo-relative path, whichever physical path was probed: the stake is about the one
# real CSV, and the path parameters below exist only so --self-check can drill these
# code paths on fixtures.
# ---------------------------------------------------------------------------

REFUSED_ABSENT = (
    "REFUSED: weights/data/doe_sobol_stage2.csv not found. Stage 2 has not produced "
    "data; the scorer never invents a verdict.")

REFUSED_DRIFT = (
    "REFUSED: the harness and the frozen scorer no longer regenerate the same design. "
    "The frozen copy wins; un-edit the harness.")


def _refused_header(found):
    return (f"REFUSED: CSV header hash {found} != design hash {HEADER_SHA256}. This "
            "file was not written by the staked harness; scoring it would score a "
            "different experiment.")


def _refused_incomplete(tag, n_complete):
    return (f"REFUSED ({tag}): incomplete design: largest complete prefix is "
            f"{n_complete} blocks, N_start is {N_START[tag]}. Resume "
            "weights/doe_morris.py --stage2; a partial night is not the staked design.")


def _undecided_msg(tag, retention, n_complete):
    # The printed --n-blocks is clamped to N_cap so the command is always one the
    # harness accepts (it refuses --n-blocks outside [N_start, N_cap]).
    nxt = min(n_complete + EXTEND_STEP, N_CAP)
    return (f"UNDECIDED ({tag}): ST argmax rank-1 retention {retention}/1000 < 950. "
            f"Extend the same seeded design: python weights/doe_morris.py --stage2 "
            f"--model {tag} --n-blocks {nxt} (cap 64). No verdict is issued from an "
            "undecided argmax.")


def _held_msg(tag, s):
    return (f"HELD: Morris says {MORRIS_ARGMAX[tag]}, Sobol says {s} on {tag}. Per "
            "prereg #95 kill rule 3, neither ranking is published as a finding until "
            "a Taguchi confirmation run adjudicates. This tool does not pick.")


# ---------------------------------------------------------------------------
# Seeded Saltelli design - regenerated, never stored: same seed => same design,
# forever. This is the FROZEN copy; the harness's future build_stage2_design must
# regenerate it byte-identically or scoring refuses.
# ---------------------------------------------------------------------------

def _level_from_u(levels, u):
    """Design sec 4 level map: 4-level factors take index min(3, floor(u*4)); 2-level
    factors take min(1, floor(u*2)). u comes from random.random() so u in [0,1) and the
    min() guards are formally redundant - they are kept because the design spells them
    out and a guard that exists in the text must exist in the code."""
    if len(levels) == 4:
        return levels[min(3, int(u * 4.0))]
    return levels[min(1, int(u * 2.0))]


def _s2_run(tag, block, matrix, cfg):
    """run_id = sha256(f"{tag}|s2|{block}|{matrix}|{canonical_config_json}")[:16]
    (design sec 4 item 3). The s2 infix keeps stage-2 ids disjoint from stage 1 by
    construction; canonical json = sorted keys, no whitespace, so the id is stable
    across relaunches - which is what makes both the harness's resume skipping and the
    completeness accounting below sound."""
    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    key = f"{tag}|s2|{block}|{matrix}|{canon}"
    rid = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return {"run_id": rid, "model": tag, "block": block, "matrix": matrix, "cfg": cfg}


def build_stage2_design(tag, n_blocks):
    """The seeded Saltelli design as a list of blocks, each (k+2) run dicts.

    Regenerable from the design paragraph alone: seed "prereg95:stage2:{tag}:20260816";
    for block b = 0..N-1, strictly in survivor order from that one stream, k uniforms
    U[0,1) for A_b then k uniforms for B_b. Runs per block, in this order: A_b, B_b,
    then AB_i_b for each survivor i in survivor order, where AB_i_b = A_b with factor
    i's level taken from B_b. Fixed factors are constants in every config (they appear
    in the canonical json the run_id hashes - the harness must do the same). Blocks are
    generated in sequence, so any prefix of blocks is itself a valid smaller-N design -
    that is what makes --n-blocks extension resume-safe, and the self-check asserts it.
    """
    survivors = SURVIVORS[tag]
    k = len(survivors)
    rng = random.Random(SEED_FMT.format(tag=tag))
    fixed = dict(FIXED[tag])
    blocks = []
    for b in range(n_blocks):
        ua = [rng.random() for _ in range(k)]
        ub = [rng.random() for _ in range(k)]
        a_cfg = dict(fixed)
        b_cfg = dict(fixed)
        for f, u in zip(survivors, ua):
            a_cfg[f] = _level_from_u(LEVELS[tag][f], u)
        for f, u in zip(survivors, ub):
            b_cfg[f] = _level_from_u(LEVELS[tag][f], u)
        runs = [_s2_run(tag, b, "A", a_cfg), _s2_run(tag, b, "B", b_cfg)]
        for f in survivors:
            ab_cfg = dict(a_cfg)
            ab_cfg[f] = b_cfg[f]
            runs.append(_s2_run(tag, b, "AB_" + f, ab_cfg))
        blocks.append(runs)
    return blocks


# ---------------------------------------------------------------------------
# CSV loading and the staked refusals 1-2
# ---------------------------------------------------------------------------

def _load_rows(path):
    """(rows by run_id, whole-file sha256, duplicate count) or a staked refusal."""
    if not os.path.exists(path):
        raise SystemExit(REFUSED_ABSENT)
    with open(path, "rb") as fh:
        raw = fh.read()
    header = raw.split(b"\n", 1)[0].rstrip(b"\r")   # exact bytes, no trailing newline
    found = hashlib.sha256(header).hexdigest()
    if found != HEADER_SHA256:
        raise SystemExit(_refused_header(found))
    rows = {}
    dupes = 0
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rid = (row.get("run_id") or "").strip()
            if not rid:
                continue
            if rid in rows:
                # Append-only file + resume skipping should make this impossible; if it
                # happens anyway, the FIRST write is the measurement and later copies
                # are noise. Counted and printed, never silently merged.
                dupes += 1
                continue
            rows[rid] = row
    return rows, hashlib.sha256(raw).hexdigest(), dupes


def _valid_y(row):
    """tok_s as float for a status=ok row, else None.

    An ok row whose tok_s does not parse to a finite number is an invalid endpoint (and
    is reported as such), never a guessed value - same rule as a DNF, applied one level
    deeper.
    """
    if row is None or row.get("status") != "ok":
        return None
    try:
        y = float(row.get("tok_s") or "")
    except ValueError:
        return None
    return y if math.isfinite(y) else None


# ---------------------------------------------------------------------------
# Refusal 3: harness cross-check (design order pins it before the completeness check)
# ---------------------------------------------------------------------------

_CROSS_CHECKED = False


def _cross_check_harness():
    """Actively detect the copy-drift the two-copies design accepts: the harness's live
    build_stage2_design must regenerate this file's frozen run_ids over the full N_cap
    stream, in order, for both models. Today weights/doe_morris.py does not carry
    --stage2 yet (this scorer is committed first, per house rule), so an absent
    function only warns - the frozen copy joined against the CSV is the load-bearing
    gate. Once the harness lands, every scoring run compares and REFUSES on drift.
    Deterministic, so it runs once per process (the fixture drills would otherwise
    re-exec the harness module on every score() call)."""
    global _CROSS_CHECKED
    if _CROSS_CHECKED:
        return
    import importlib.util
    hpath = os.path.join(HERE, "doe_morris.py")
    # the harness does `import runner` (its sibling): HERE must be on sys.path for the
    # exec to resolve it - the stage-1 scorer proved this the hard way, its first
    # version fell into the warning path on EVERY call and would have let drift escape.
    added = HERE not in sys.path
    if added:
        sys.path.insert(0, HERE)
    try:
        spec = importlib.util.spec_from_file_location("doe_morris_s2_livecheck", hpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        build = mod.build_stage2_design
        theirs = []
        for tag in MODELS:
            for item in build(tag, N_CAP):
                if isinstance(item, (list, tuple)):     # per-block nesting, like ours
                    theirs.extend(r["run_id"] for r in item)
                else:                                   # flat list is also acceptable
                    theirs.append(item["run_id"])
    except Exception as e:
        print(f"[prereg95-s2] WARNING: harness cross-check skipped ({e}); scoring on "
              "this file's frozen design copy alone.")
        _CROSS_CHECKED = True
        return
    finally:
        if added and HERE in sys.path:
            sys.path.remove(HERE)
    ours = [r["run_id"] for tag in MODELS
            for blk in build_stage2_design(tag, N_CAP) for r in blk]
    if ours != theirs:
        raise SystemExit(REFUSED_DRIFT)
    _CROSS_CHECKED = True


# ---------------------------------------------------------------------------
# Sobol estimators (design sec 5.3) - Saltelli 2010 Table-2(b) first-order and Jansen
# 1999 total-order, computed from per-block precomputed terms so the full-data estimate
# and every bootstrap resample run through ONE code path (no estimator divergence).
# ---------------------------------------------------------------------------

def sobol_terms(blocks_y, survivors):
    """Precompute per-block estimator terms from per-block y values.

    blocks_y: [{"A": y|None, "B": y|None, "AB": {factor: y|None}}], None = invalid
    endpoint (DNF or unparseable). A None poisons exactly that block's contribution for
    the affected factor - ST needs the pair (A_j, AB_i_j), S needs the triple
    (A_j, B_j, AB_i_j) - and nothing more: the stage-1 poisoning discipline, restated
    for blocks. The A/B pool feeding V(Y) takes every ok A and B value independently.
    """
    pool = []
    st = {f: [] for f in survivors}
    s1 = {f: [] for f in survivors}
    for e in blocks_y:
        a, b = e["A"], e["B"]
        pool.append((a, b))
        for f in survivors:
            ab = e["AB"].get(f)
            st[f].append((a - ab) ** 2 if a is not None and ab is not None else None)
            s1[f].append(b * (ab - a)
                         if a is not None and b is not None and ab is not None else None)
    return {"pool": pool, "st": st, "s1": s1, "n": len(blocks_y)}


def estimate(terms, idx, survivors):
    """S_i and ST_i over the blocks named by idx (with multiplicity - bootstrap
    resamples pass repeated indices; the full-data estimate passes range(N)).

    V(Y) = sample variance (ddof=1) over all ok A and B values pooled. Division order
    is pinned as sum / count / V because the hand-derived fixture below reproduces
    exactly that order on paper; do not "simplify" it into a single division.
    """
    vals = []
    for j in idx:
        a, b = terms["pool"][j]
        if a is not None:
            vals.append(a)
        if b is not None:
            vals.append(b)
    n_pool = len(vals)
    v = None
    if n_pool >= 2:
        mean = sum(vals) / n_pool
        var = sum((x - mean) ** 2 for x in vals) / (n_pool - 1)
        if var > 0.0:
            v = var
    out = {"V": v, "n_pool": n_pool, "S": {}, "ST": {}, "n_S": {}, "n_ST": {}}
    for f in survivors:
        stt = [terms["st"][f][j] for j in idx if terms["st"][f][j] is not None]
        s1t = [terms["s1"][f][j] for j in idx if terms["s1"][f][j] is not None]
        out["n_ST"][f] = len(stt)
        out["n_S"][f] = len(s1t)
        out["ST"][f] = (sum(stt) / (2.0 * len(stt)) / v) if stt and v is not None else None
        out["S"][f] = (sum(s1t) / len(s1t) / v) if s1t and v is not None else None
    return out


def st_argmax(est, survivors):
    """Argmax of ST over survivors with a defined value; an exact tie goes to the first
    factor in survivor order - the same deterministic tie-break stage 1 used for
    mu_star, so the verdict can never depend on dict iteration order."""
    best = None
    for f in survivors:
        v = est["ST"][f]
        if v is None:
            continue
        if best is None or v > est["ST"][best]:
            best = f
    return best


def _percentile(sorted_vals, q):
    """Linear-interpolation percentile on a pre-sorted list (the numpy 'linear'
    definition, restated in stdlib so the dependency budget stays zero)."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _ci95(vals):
    if not vals:
        return None
    s = sorted(vals)
    return [_percentile(s, 0.025), _percentile(s, 0.975)]


def bootstrap(terms, n_blocks, survivors, n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED):
    """1000 resamples of block indices 0..N-1 with replacement (design sec 5.4),
    recomputing every S_i and ST_i per resample; percentile 2.5/97.5 CIs and the
    rank-1 retention count of the modal top factor.

    The RNG is seeded FRESH with the staked string for each model's bootstrap: a
    model's resamples must be bit-identical whether the other model is scored in the
    same run or refused as incomplete, or the night-1 7B verdict would depend on the
    30B's presence. A resample where no factor has a defined ST (all pairs poisoned, or
    a degenerate resampled pool) casts no rank-1 vote: it stays in the /1000
    denominator, which can only push the gate toward UNDECIDED - fail-safe."""
    rng = random.Random(seed)
    argmax_counts = {}
    dist = {f: {"S": [], "ST": []} for f in survivors}
    no_argmax = 0
    for _ in range(n_resamples):
        idx = [rng.randrange(n_blocks) for _ in range(n_blocks)]
        est = estimate(terms, idx, survivors)
        a = st_argmax(est, survivors)
        if a is None:
            no_argmax += 1
        else:
            argmax_counts[a] = argmax_counts.get(a, 0) + 1
        for f in survivors:
            if est["S"][f] is not None:
                dist[f]["S"].append(est["S"][f])
            if est["ST"][f] is not None:
                dist[f]["ST"].append(est["ST"][f])
    modal, modal_n = None, 0
    for f in survivors:                 # survivor order breaks exact count ties
        c = argmax_counts.get(f, 0)
        if c > modal_n:
            modal, modal_n = f, c
    ci = {f: {"S": _ci95(dist[f]["S"]), "ST": _ci95(dist[f]["ST"]),
              "n_S_resamples": len(dist[f]["S"]), "n_ST_resamples": len(dist[f]["ST"])}
          for f in survivors}
    return {"resamples": n_resamples, "argmax_counts": argmax_counts, "modal": modal,
            "retention": modal_n, "no_argmax": no_argmax, "ci": ci}


# ---------------------------------------------------------------------------
# The P-3 decision table (design sec 1.3, verbatim rows)
# ---------------------------------------------------------------------------

def _decide(tag, s):
    m = MORRIS_ARGMAX[tag]
    in_map = s in MAPPING_SET[tag]
    if s == m:
        if in_map:
            return "PASS", "s == m and s in MAP"
        return "FAIL", "s == m and s not in MAP"
    if in_map:
        return "HELD_FOR_TAGUCHI", "s != m and s in MAP"
    return "FAIL", "s != m and s not in MAP"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_model(tag, blocks, rows, n_complete, present_in_cap):
    """Score one model at its complete block prefix; returns the per-model dict."""
    survivors = SURVIVORS[tag]
    blocks_y = []
    n_ok = n_dnf = n_other = 0
    bad_ok = []
    non_ok = []
    for blk in blocks:
        entry = {"A": None, "B": None, "AB": {}}
        for r in blk:
            row = rows[r["run_id"]]
            status = row.get("status") or ""
            y = _valid_y(row)
            if status == "ok":
                n_ok += 1
                if y is None:
                    bad_ok.append(r["run_id"])
            else:
                if status.startswith("dnf"):
                    n_dnf += 1
                else:
                    n_other += 1
                non_ok.append({"run_id": r["run_id"], "block": r["block"],
                               "matrix": r["matrix"], "status": status})
            if r["matrix"] == "A":
                entry["A"] = y
            elif r["matrix"] == "B":
                entry["B"] = y
            else:
                entry["AB"][r["matrix"][3:]] = y
        blocks_y.append(entry)

    terms = sobol_terms(blocks_y, survivors)
    full = estimate(terms, range(n_complete), survivors)
    floor = int(math.ceil(FLOOR_FRAC * n_complete))
    out = {
        "n_start": N_START[tag], "n_cap": N_CAP, "n_complete_blocks": n_complete,
        "validity_floor_pairs": floor,
        "rows": {"ok": n_ok, "dnf": n_dnf, "other_non_ok": n_other,
                 "present_beyond_prefix": present_in_cap - (len(survivors) + 2) * n_complete,
                 "ok_rows_with_unparseable_tok_s": bad_ok},
        "V_Y": full["V"], "n_pool_ok_AB": full["n_pool"],
        "plan_binding": PLAN_BINDING[tag],
        "mapping_set": list(MAPPING_SET[tag]),
        "morris_argmax": MORRIS_ARGMAX[tag],
        "factors": {}, "non_ok_rows": non_ok,
        "st_argmax_full_data": None, "sobol_argmax": None,
        "case": None, "verdict": None,
    }
    for f in survivors:
        out["factors"][f] = {
            "S": full["S"][f], "S_ci95": None, "n_S_triples": full["n_S"][f],
            "ST": full["ST"][f], "ST_ci95": None, "n_ST_pairs": full["n_ST"][f],
            "scoreable": full["n_ST"][f] >= floor,
        }

    if full["V"] is None:
        # Bitwise-identical (or near-empty) ok A/B pool: unreachable with real
        # measurement noise, but every index would be 0/0 on zero total variance.
        # Degenerate data voids the stake - a verdict is never manufactured from it
        # (the stage-1 total-mu*-zero rule, restated for variance).
        out["verdict"] = "VOID"
        out["void_because"] = [
            "degenerate: pooled ok A/B variance is zero or the pool has fewer than 2 "
            "values - no signal to attribute"]
        return out

    unscoreable = [f for f in survivors if full["n_ST"][f] < floor]
    if unscoreable:
        # Design sec 5.2 item 5: a survivor below the floor is UNSCOREABLE and the
        # model's P-3 is VOID, never guessed. An argmax over a factor set with a hole
        # in it would be an argmax over a different experiment.
        out["verdict"] = "VOID"
        out["void_because"] = [f"{f}: {full['n_ST'][f]} valid ST pairs < floor {floor}"
                               for f in unscoreable]
        return out

    boot = bootstrap(terms, n_complete, survivors)
    for f in survivors:
        out["factors"][f]["S_ci95"] = boot["ci"][f]["S"]
        out["factors"][f]["ST_ci95"] = boot["ci"][f]["ST"]
    s_full = st_argmax(full, survivors)
    # DECIDED needs the modal bootstrap top factor at >= 950/1000 AND that factor to BE
    # the full-data argmax: a gate that certified one factor while the point estimate
    # named another would publish an incoherent verdict, so that (pathological,
    # near-unreachable) state is treated as undecided - fail-safe, never a pick.
    decided = boot["retention"] >= GATE_RETENTION and boot["modal"] == s_full
    out["st_argmax_full_data"] = s_full
    out["bootstrap"] = {
        "resamples": boot["resamples"], "seed": BOOTSTRAP_SEED,
        "modal_top": boot["modal"], "retention": boot["retention"],
        "argmax_counts": boot["argmax_counts"],
        "no_argmax_resamples": boot["no_argmax"], "decided": decided,
    }
    if decided:
        out["sobol_argmax"] = s_full
        verdict, case = _decide(tag, s_full)
        out["verdict"] = verdict
        out["case"] = case
        if s_full != MORRIS_ARGMAX[tag]:
            # Design sec 5.5: on ANY s != m - the HELD row and the both-argmaxes-fail
            # FAIL row alike - the hold message prints and the JSON carries it.
            out["held_message"] = _held_msg(tag, s_full)
    elif n_complete >= N_CAP:
        out["verdict"] = "UNDECIDABLE"
        out["case"] = "gate never decides at N_cap"
    elif boot["retention"] >= GATE_RETENTION:
        # retention passed but the modal factor is not the point argmax: the staked
        # "< 950" message would misreport, so this branch states what actually happened.
        out["undecided_message"] = (
            f"UNDECIDED ({tag}): bootstrap modal top {boot['modal']} != full-data ST "
            f"argmax {s_full} despite retention {boot['retention']}/1000 - incoherent "
            "gate state, treated as undecided (fail-safe). Extend the same seeded "
            f"design: python weights/doe_morris.py --stage2 --model {tag} --n-blocks "
            f"{min(n_complete + EXTEND_STEP, N_CAP)} (cap 64). No verdict is issued "
            "from an undecided argmax.")
    else:
        out["undecided_message"] = _undecided_msg(tag, boot["retention"], n_complete)
    return out


def score(csv_path=CSV_PATH, out_path=VERDICT_PATH, chatty=True):
    """Refuse, or score P-3 per model and overall; write the verdict json; return it."""
    rows, csv_sha, dupes = _load_rows(csv_path)     # refusals 1 and 2
    _cross_check_harness()                          # refusal 3 (design order: before 4)

    result_models = {}
    refusal_msgs = []
    all_cap_ids = set()
    for tag in MODELS:
        blocks = build_stage2_design(tag, N_CAP)
        cap_ids = [r["run_id"] for blk in blocks for r in blk]
        all_cap_ids.update(cap_ids)
        n_complete = 0
        for blk in blocks:              # largest complete PREFIX: a later complete
            if all(r["run_id"] in rows for r in blk):   # block cannot heal a hole
                n_complete += 1
            else:
                break
        if n_complete < N_START[tag]:
            msg = _refused_incomplete(tag, n_complete)
            refusal_msgs.append(msg)
            result_models[tag] = {"refused": msg, "n_complete_blocks": n_complete,
                                  "n_start": N_START[tag], "n_cap": N_CAP}
            continue
        present_in_cap = sum(1 for rid in cap_ids if rid in rows)
        result_models[tag] = _score_model(tag, blocks[:n_complete], rows, n_complete,
                                          present_in_cap)

    if len(refusal_msgs) == len(MODELS):
        # Nothing is scoreable: this is a whole-run refusal, exactly like stage 1's -
        # no verdict json is written from a night that produced no complete design.
        raise SystemExit("\n".join(refusal_msgs))

    verdicts = {tag: result_models[tag].get("verdict") for tag in MODELS}
    vlist = list(verdicts.values())
    # Overall aggregation, design sec 1.3: FAIL if either model is a publishable FAIL;
    # else HELD_FOR_TAGUCHI if either model is held or undecidable; PASS only if both
    # models publishably pass. VOID propagates (stage-1 rule: never guessed); PENDING
    # names the branch the design leaves open - a model still refused or undecided
    # while nothing above forced a verdict.
    if "FAIL" in vlist:
        overall = "FAIL"
    elif "HELD_FOR_TAGUCHI" in vlist or "UNDECIDABLE" in vlist:
        overall = "HELD_FOR_TAGUCHI"
    elif "VOID" in vlist:
        overall = "VOID"
    elif all(v == "PASS" for v in vlist):
        overall = "PASS"
    else:
        overall = "PENDING"

    label = None
    reasons = []
    for tag in MODELS:
        v = verdicts[tag]
        if v == "FAIL":
            reasons.append(f"{tag}: publishable P-3 FAIL - the kill rule fires; the "
                           "label ships the same day at full prominence")
        elif v == "UNDECIDABLE":
            reasons.append(f"{tag}: UNDECIDABLE at N_cap {N_CAP} - the label ships "
                           "anyway (amendment item 7: stricter than staked, declared)")
    if reasons:
        label = {"text": LABEL, "surfaces": list(LABEL_SURFACES), "reasons": reasons}

    unresolved = [tag for tag in MODELS
                  if "refused" not in result_models[tag]
                  and result_models[tag].get("verdict") is None]
    rc = 1 if unresolved else 0     # rc 0 only when every non-refused model verdicted

    result = {
        "prereg": 95,
        "stage": 2,
        "scores": "P-3 only (stage-1 stakes are already scored and are not re-scored)",
        "csv": CSV_REL,
        "csv_sha256": csv_sha,
        "scored_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "design": {
            "seeds": [SEED_FMT.format(tag=m) for m in MODELS],
            "bootstrap_seed": BOOTSTRAP_SEED,
            "n_start": dict(N_START), "n_cap": N_CAP, "extend_step": EXTEND_STEP,
            "survivors": {m: list(SURVIVORS[m]) for m in MODELS},
            "fixed": {m: dict(FIXED[m]) for m in MODELS},
            "estimators": {"S": "Saltelli et al. 2010, Table 2, estimator (b)",
                           "ST": "Jansen 1999"},
            "gate": "ST argmax DECIDED iff modal top factor holds rank 1 in >= "
                    f"{GATE_RETENTION}/{N_RESAMPLES} bootstrap resamples",
            "validity_floor": "ceil(0.75 * N_complete) valid ST pairs per survivor",
            "header_sha256": HEADER_SHA256,
        },
        "ground_truth": {
            "plan_binding": dict(PLAN_BINDING),
            "plan_margin_30B": PLAN_MARGIN_30B,
            "mapping_set": {m: list(MAPPING_SET[m]) for m in MODELS},
            "morris_argmax": dict(MORRIS_ARGMAX),
            "source": "quantprobe plan 2026-08-16, auto-detected hw, calibration "
                      "2026-07-31 state 2dc97d41; frozen in this scorer and never "
                      "re-read at score time",
        },
        "rows": {"rows_in_csv": len(rows),
                 "duplicate_run_ids_ignored": dupes,
                 "extra_rows_outside_design": sum(1 for rid in rows
                                                  if rid not in all_cap_ids)},
        "models": result_models,
        "P-3": {
            "rule": "per model: the factor with the highest Sobol total-order index "
                    "must be in the staked mapping set; overall: FAIL if either model "
                    "is a publishable FAIL, else HELD_FOR_TAGUCHI if either model is "
                    "held or undecidable, PASS only if both models publishably pass",
            "overall": overall,
            "per_model": verdicts,
        },
        "exit_code": rc,
    }
    if label:
        result["label"] = label
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    if chatty:
        _print_report(result, out_path)
    return result


def _fmt(v, width=9):
    return " " * (width - 4) + "None" if v is None else f"{v:{width}.4f}"


def _print_report(res, out_path):
    print("prereg #95 stage 2 - Sobol variance attribution on the Morris survivors (P-3)")
    print(f"  csv: {res['csv']}  sha256 {res['csv_sha256'][:16]}..")
    g = res["rows"]
    line = f"  rows: {g['rows_in_csv']} in csv"
    if g["extra_rows_outside_design"]:
        line += f"; {g['extra_rows_outside_design']} outside the cap design ignored"
    if g["duplicate_run_ids_ignored"]:
        line += (f"; {g['duplicate_run_ids_ignored']} duplicate run_ids ignored "
                 "(first write wins)")
    print(line)

    for tag in MODELS:
        mm = res["models"][tag]
        if "refused" in mm:
            print(f"\n  {mm['refused']}")
            continue
        r = mm["rows"]
        print(f"\n  {tag}  N_complete {mm['n_complete_blocks']} blocks "
              f"(N_start {mm['n_start']}, cap {mm['n_cap']}); rows ok {r['ok']}, "
              f"dnf {r['dnf']}, other {r['other_non_ok']}; "
              f"pool {mm['n_pool_ok_AB']} ok A/B values, V(Y) "
              + ("n/a" if mm["V_Y"] is None else f"{mm['V_Y']:.4f}"))
        if r["present_beyond_prefix"]:
            print(f"    note: {r['present_beyond_prefix']} designed rows beyond the "
                  "complete prefix ignored (a partial extension is not scored)")
        if r["ok_rows_with_unparseable_tok_s"]:
            print("    note: ok rows with unparseable tok_s, treated as invalid "
                  "endpoints: " + ", ".join(r["ok_rows_with_unparseable_tok_s"]))
        ranked = sorted(SURVIVORS[tag],
                        key=lambda f: (mm["factors"][f]["ST"]
                                       if mm["factors"][f]["ST"] is not None
                                       else float("-inf")),
                        reverse=True)
        print("    factor        ST        ST 95% CI             nST   "
              "S         S 95% CI              nS")
        for f in ranked:
            e = mm["factors"][f]
            st_ci = ("[" + ", ".join(f"{x:.4f}" for x in e["ST_ci95"]) + "]"
                     if e["ST_ci95"] else "-")
            s_ci = ("[" + ", ".join(f"{x:.4f}" for x in e["S_ci95"]) + "]"
                    if e["S_ci95"] else "-")
            tagtxt = ("" if e["scoreable"] else
                      f"  UNSCOREABLE ({e['n_ST_pairs']} < {mm['validity_floor_pairs']}"
                      " valid ST pairs)")
            print(f"    {f:<13}{_fmt(e['ST'])}  {st_ci:<20}  {e['n_ST_pairs']:>3}  "
                  f"{_fmt(e['S'])}  {s_ci:<20}  {e['n_S_triples']:>3}{tagtxt}")
        if "bootstrap" in mm:
            b = mm["bootstrap"]
            print(f"    bootstrap: modal top {b['modal_top']} holds rank 1 in "
                  f"{b['retention']}/{b['resamples']} resamples -> "
                  + ("DECIDED" if b["decided"] else "not decided")
                  + (f" ({b['no_argmax_resamples']} resamples cast no vote)"
                     if b["no_argmax_resamples"] else ""))
        if mm.get("void_because"):
            print("    VOID: " + "; ".join(mm["void_because"]))
        if mm.get("undecided_message"):
            print("  " + mm["undecided_message"])

    print("\n=== P-3 verdict (mapping staked 2026-08-07; ground truth frozen "
          "2026-08-16; scorer frozen before stage-2 data) ===")
    for tag in MODELS:
        mm = res["models"][tag]
        print(f"  {tag} plan: {PLAN_BINDING[tag]}")
        if tag == "30B":
            print("      margin (context only, never a wider mapping set): "
                  + PLAN_MARGIN_30B)
        if "refused" in mm:
            state = "REFUSED (see above)"
        elif mm.get("verdict"):
            state = mm["verdict"]
        else:
            state = "UNDECIDED (no verdict issued)"
        print(f"      mapping set {{{', '.join(MAPPING_SET[tag])}}}; Morris argmax "
              f"{MORRIS_ARGMAX[tag]}; Sobol argmax "
              f"{mm.get('sobol_argmax') or '-'} -> {state}"
              + (f"  [{mm['case']}]" if mm.get("case") else ""))
        if mm.get("held_message"):
            print("      " + mm["held_message"])
    print(f"  overall P-3: {res['P-3']['overall']}")

    if res.get("label"):
        lab = res["label"]
        print("\n  SCOPE LABEL to ship (operator action, same day, full prominence):")
        for reason in lab["reasons"]:
            print("    " + reason)
        print(f"    label text: {lab['text']}")
        for i, s in enumerate(lab["surfaces"], 1):
            print(f"    {i}. {s}")

    print(f"\n  verdict json: {out_path}")
    print("  (this tool edits no prereg, no README, no plan.py and no chart; the "
          "operator ships the label)")


# ---------------------------------------------------------------------------
# Self-check: pinned strings, seeded-design invariants, a hand-derived fixture,
# refusal drills, and synthetic decided/held/undecided/undecidable/void nights.
# Exit 0 iff all pass. The real CSV and verdict paths are never touched.
# ---------------------------------------------------------------------------

class _Check:
    def __init__(self):
        self.n = 0
        self.failures = []

    def __call__(self, cond, what):
        self.n += 1
        if not cond:
            self.failures.append(what)


def _catch_exit(fn):
    """Run fn, return the SystemExit message it refused with (None if it did not)."""
    try:
        fn()
    except SystemExit as e:
        return str(e)
    return None


def _fixture_csv_row(run, status, tok_s):
    """A full 30-column CSV row for a designed run; telemetry cells stay empty."""
    cells = {c: "" for c in COLS}
    cells.update({"run_id": run["run_id"], "model": run["model"],
                  "block": str(run["block"]), "matrix": run["matrix"],
                  "status": status, "tok_s": tok_s})
    for f, v in run["cfg"].items():
        cells[f] = str(v)
    return [cells[c] for c in COLS]


def _write_fixture_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(HEADER + "\n")
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerows(rows)


def _upos(tag, f, val):
    """Level -> its normalized position in the level list (0..1), for planting
    synthetic responses with a known dominant factor."""
    lv = LEVELS[tag][f]
    return lv.index(val) / (len(lv) - 1.0)


def self_check():
    ck = _Check()

    # --- pinned strings and embedded ground truth --------------------------------
    ck(hashlib.sha256(HEADER.encode("utf-8")).hexdigest() == HEADER_SHA256,
       "pinned header hash reproduces from the header string")
    ck(len(COLS) == 30, "header carries the 30 designed columns")
    ck(REFUSED_ABSENT == "REFUSED: weights/data/doe_sobol_stage2.csv not found. "
                         "Stage 2 has not produced data; the scorer never invents a "
                         "verdict.",
       "refusal 1 string matches the design verbatim")
    ck(REFUSED_DRIFT == "REFUSED: the harness and the frozen scorer no longer "
                        "regenerate the same design. The frozen copy wins; un-edit "
                        "the harness.",
       "refusal 3 string matches the design verbatim")
    ck(PLAN_BINDING == {
        "7B": "binding constraint: BANDWIDTH-BOUND (VRAM bandwidth) - 100% of every "
              "decode token is spent there.",
        "30B": "binding constraint: BANDWIDTH-BOUND (system RAM bandwidth) - 51% of "
               "every decode token is spent there."},
       "plan binding-constraint lines match the design verbatim")
    ck(PLAN_MARGIN_30B == "1.03x - system RAM bandwidth must get that much faster "
                          "before VRAM bandwidth (49%) takes over",
       "30B margin context line matches the design verbatim")
    ck(MAPPING_SET == {"7B": ("ctk", "ub"), "30B": ("moe_cpu_frac", "ctk")},
       "mapping sets match the design table")
    ck(MORRIS_ARGMAX == {"7B": "ngl", "30B": "t"},
       "Morris argmax constants match prereg95_verdict.json as pinned")
    ck(N_START == {"7B": 32, "30B": 40} and N_CAP == 64,
       "N_start / N_cap match the design")
    ck(LABEL == "derived from the law, not confirmed by variance attribution "
                "(prereg #95 P-3)",
       "scope label text matches the design verbatim")
    ck(all(MORRIS_ARGMAX[m] not in MAPPING_SET[m] for m in MODELS),
       "theorem: both Morris argmaxes are outside their mapping sets, so the PASS row "
       "is unreachable out of stage 2 alone")
    ck(all(set(MAPPING_SET[m]) <= set(SURVIVORS[m]) for m in MODELS),
       "every mapping-set factor is a designed survivor (the stake stays winnable)")
    ck(all(f not in MAPPING_SET[m] for m in MODELS for f, _ in FIXED[m]),
       "every fixed factor is outside both mapping sets (fixing removes only FAIL "
       "modes)")
    ck(_refused_incomplete("7B", 12) ==
       "REFUSED (7B): incomplete design: largest complete prefix is 12 blocks, "
       "N_start is 32. Resume weights/doe_morris.py --stage2; a partial night is not "
       "the staked design.",
       "per-model incomplete refusal renders the staked message")
    ck(_undecided_msg("7B", 812, 32) ==
       "UNDECIDED (7B): ST argmax rank-1 retention 812/1000 < 950. Extend the same "
       "seeded design: python weights/doe_morris.py --stage2 --model 7B --n-blocks 48 "
       "(cap 64). No verdict is issued from an undecided argmax.",
       "undecided message renders the staked message with N+16")
    ck(_undecided_msg("30B", 900, 56).count("--n-blocks 64 (cap 64)") == 1,
       "undecided message clamps the extension to N_cap (56+16 -> 64)")
    ck(_held_msg("7B", "ctk") ==
       "HELD: Morris says ngl, Sobol says ctk on 7B. Per prereg #95 kill rule 3, "
       "neither ranking is published as a finding until a Taguchi confirmation run "
       "adjudicates. This tool does not pick.",
       "held message renders the staked message")

    # --- seeded design invariants -------------------------------------------------
    d7 = build_stage2_design("7B", N_START["7B"])
    d30 = build_stage2_design("30B", N_START["30B"])
    ck(len(d7) == 32 and all(len(b) == 6 for b in d7),
       "7B design is 32 blocks x (k+2)=6 runs = 192")
    ck(len(d30) == 40 and all(len(b) == 7 for b in d30),
       "30B design is 40 blocks x (k+2)=7 runs = 280")
    ids = [r["run_id"] for d in (d7, d30) for b in d for r in b]
    ck(len(ids) == 472 and len(set(ids)) == 472,
       "the 472 designed N_start run_ids are unique")
    ck(all(len(rid) == 16 for rid in ids), "run_ids are 16 hex chars")
    ck([r["matrix"] for r in d7[0]] == ["A", "B", "AB_ngl", "AB_ub", "AB_t", "AB_ctk"],
       "7B block layout is A, B, then AB_i in survivor order")
    ck([r["matrix"] for r in d30[0]]
       == ["A", "B", "AB_ngl", "AB_t", "AB_ctk", "AB_mmp", "AB_moe_cpu_frac"],
       "30B block layout is A, B, then AB_i in survivor order")
    ck([r["run_id"] for b in build_stage2_design("7B", 32) for r in b]
       == [r["run_id"] for b in d7 for r in b],
       "design regeneration is deterministic (same seed, same ids)")
    for tag, d in (("7B", d7), ("30B", d30)):
        cap = build_stage2_design(tag, N_CAP)
        ck([r["run_id"] for b in d for r in b]
           == [r["run_id"] for b in cap[:len(d)] for r in b],
           f"{tag}: any block prefix of the N_cap stream is the smaller-N design "
           "(what makes --n-blocks extension resume-safe)")
        fixed = dict(FIXED[tag])
        ck(all(r["cfg"][f] == v for b in d for r in b for f, v in fixed.items()),
           f"{tag}: fixed factors are constants in every config")
        ck(all(set(r["cfg"]) == set(SURVIVORS[tag]) | set(fixed) for b in d for r in b),
           f"{tag}: every config carries all survivors plus all fixed factors")
        ab_ok = lev_ok = True
        for b in d:
            a_cfg, b_cfg = b[0]["cfg"], b[1]["cfg"]
            for f, run in zip(SURVIVORS[tag], b[2:]):
                want = dict(a_cfg)
                want[f] = b_cfg[f]
                if run["cfg"] != want:
                    ab_ok = False
            for r in b:
                if not all(r["cfg"][f] in LEVELS[tag][f] for f in SURVIVORS[tag]):
                    lev_ok = False
        ck(ab_ok, f"{tag}: AB_i is A with exactly factor i taken from B")
        ck(lev_ok, f"{tag}: every drawn level is on the staked level list")
    r0 = d7[0][2]       # recompute one run_id from the design formula by hand
    canon = json.dumps(r0["cfg"], sort_keys=True, separators=(",", ":"))
    ck(r0["run_id"] == hashlib.sha256(
        f"7B|s2|0|AB_ngl|{canon}".encode("utf-8")).hexdigest()[:16],
       "run_id reproduces from sha256(tag|s2|block|matrix|canonical_config_json)[:16]")
    ck(_catch_exit(_cross_check_harness) is None,
       "harness cross-check does not refuse (an absent --stage2 harness warns; a "
       "present one must regenerate the frozen ids)")

    # --- hand-derived fixture -----------------------------------------------------
    # k = 2 factors X, Y; N = 4 Saltelli blocks. Values chosen so every intermediate
    # division is by a power of two or lands exactly on an integer, making each step
    # exact in binary floats - the assertions demand EXACT equality, no tolerances
    # (stage-1 discipline). Derived BY HAND:
    #
    #   block j :   0    1    2    3
    #   f(A_j)  :  12    5    7    9
    #   f(B_j)  :   7    8    8    8
    #   f(AB_X) :  10    6    7    9
    #   f(AB_Y) :  12    9    3    5
    #
    #   V(Y): pooled ok A and B values (8 of them) sum to 64 -> mean 8; deviations
    #   (4, -3, -1, 1 | -1, 0, 0, 0), squares 16+9+1+1+1 = 28; sample variance
    #   (ddof=1) = 28/7 = 4 exactly.
    #
    #   Jansen 1999 total-order, ST_i = sum_j (f(A_j) - f(AB_i_j))^2 / (2*M_i) / V:
    #     X: diffs (2, -1, 0, 0), squares sum 5  -> ST_X = 5/8/4  = 5/32 = 0.15625
    #     Y: diffs (0, -4, 4, 4), squares sum 48 -> ST_Y = 48/8/4 = 3/2  = 1.5
    #
    #   Saltelli 2010 Table-2(b) first-order,
    #   S_i = sum_j f(B_j)*(f(AB_i_j) - f(A_j)) / N_i / V:
    #     X: 7*(-2) + 8*(+1) + 8*0 + 8*0    = -6  -> S_X = -6/4/4  = -3/8 = -0.375
    #     Y: 7*0 + 8*(+4) + 8*(-4) + 8*(-4) = -32 -> S_Y = -32/4/4 = -2.0
    #   (negative first-order estimates are estimator noise made visible; the fixture
    #   pins the arithmetic, not the physics)
    #
    #   ST argmax = Y. Bootstrap gate, reasoned by hand: per-block ST terms are
    #   X: (4, 1, 0, 0) and Y: (0, 16, 16, 16). A resample drawing block 0 c0 times
    #   and block 1 c1 times scores ssd_X = 4*c0 + c1 against ssd_Y = 16*(4 - c0)
    #   (the shared 1/(2M)/V divisor cancels); X outranks Y iff 20*c0 + c1 > 64,
    #   which needs c0 = 4 - ONLY the all-block-0 resample, probability 4^-4 = 1/256
    #   per resample. Y therefore holds rank 1 in ~996/1000 and the gate DECIDES.
    fix_blocks = [
        {"A": 12.0, "B": 7.0, "AB": {"X": 10.0, "Y": 12.0}},
        {"A": 5.0, "B": 8.0, "AB": {"X": 6.0, "Y": 9.0}},
        {"A": 7.0, "B": 8.0, "AB": {"X": 7.0, "Y": 3.0}},
        {"A": 9.0, "B": 8.0, "AB": {"X": 9.0, "Y": 5.0}},
    ]
    fsurv = ("X", "Y")
    terms = sobol_terms(fix_blocks, fsurv)
    est = estimate(terms, range(4), fsurv)
    ck(est["n_pool"] == 8, "fixture pools all 8 ok A/B values")
    ck(est["V"] == 4.0, "fixture V(Y) == 4.0 exactly")
    ck(est["ST"]["X"] == 0.15625, "fixture ST_X == 5/32 == 0.15625 exactly")
    ck(est["ST"]["Y"] == 1.5, "fixture ST_Y == 3/2 == 1.5 exactly")
    ck(est["S"]["X"] == -0.375, "fixture S_X == -3/8 == -0.375 exactly")
    ck(est["S"]["Y"] == -2.0, "fixture S_Y == -2.0 exactly")
    ck(est["n_ST"] == {"X": 4, "Y": 4} and est["n_S"] == {"X": 4, "Y": 4},
       "fixture M_i and N_i counts are 4 everywhere")
    ck(st_argmax(est, fsurv) == "Y", "fixture ST argmax is Y")

    # An invalid AB endpoint poisons exactly that factor's block term; the A/B pool
    # (and so V) is untouched.
    holed = [dict(b, AB=dict(b["AB"])) for b in fix_blocks]
    holed[0]["AB"]["X"] = None
    e2 = estimate(sobol_terms(holed, fsurv), range(4), fsurv)
    ck(e2["n_ST"]["X"] == 3 and e2["n_S"]["X"] == 3
       and e2["n_ST"]["Y"] == 4 and e2["V"] == 4.0,
       "a poisoned AB endpoint costs exactly that factor's pair and triple, nothing "
       "more")
    # An invalid A endpoint poisons every factor's pair for that block AND one pool
    # value.
    holed_a = [dict(b, AB=dict(b["AB"])) for b in fix_blocks]
    holed_a[2]["A"] = None
    e3 = estimate(sobol_terms(holed_a, fsurv), range(4), fsurv)
    ck(e3["n_ST"] == {"X": 3, "Y": 3} and e3["n_pool"] == 7,
       "a poisoned A endpoint costs one pair per factor and one pool value")

    boot = bootstrap(terms, 4, fsurv)
    ck(boot["modal"] == "Y" and boot["retention"] >= GATE_RETENTION,
       "fixture bootstrap: Y holds rank 1 past the 950 gate (hand argument above)")
    ck(boot["retention"] + boot["argmax_counts"].get("X", 0) + boot["no_argmax"]
       == N_RESAMPLES,
       "fixture bootstrap: every resample is a Y vote, an X vote or a no-vote")
    ck(bootstrap(terms, 4, fsurv)["retention"] == boot["retention"],
       "bootstrap is seeded: identical retention on regeneration")
    ci = boot["ci"]["Y"]["ST"]
    ck(ci is not None and ci[0] <= ci[1], "fixture ST_Y CI is ordered")

    tmp = tempfile.mkdtemp(prefix="prereg95_s2_selfcheck_")
    try:
        # --- refusal drills (fixture paths only; the real CSV is never touched) ---
        msg = _catch_exit(lambda: _load_rows(os.path.join(tmp, "absent.csv")))
        ck(msg == REFUSED_ABSENT, "absent CSV refuses with the staked message")

        drift = os.path.join(tmp, "drift.csv")
        with open(drift, "w", encoding="utf-8", newline="") as fh:
            fh.write("run_id,model\nx,7B\n")
        found = hashlib.sha256(b"run_id,model").hexdigest()
        msg = _catch_exit(lambda: _load_rows(drift))
        ck(msg == ("REFUSED: CSV header hash " + found + " != design hash "
                   "95fe4b76345921ea95d6a86cc83b30caad741f799515716d0d07f687c2d89ade. "
                   "This file was not written by the staked harness; scoring it would "
                   "score a different experiment."),
       "drifted header refuses with the staked message and the found hash")

        # Both models under N_start: whole-run refusal, no verdict json. 7B block 0 is
        # complete (prefix 1); 7B block 1 misses one run, so a complete block 2+ could
        # never heal the hole; 30B has nothing (prefix 0).
        d7cap = build_stage2_design("7B", N_CAP)
        partial = os.path.join(tmp, "partial.csv")
        rows_out = [_fixture_csv_row(r, "ok", "12.5") for r in d7cap[0]]
        rows_out += [_fixture_csv_row(r, "ok", "12.5") for r in d7cap[1][:-1]]
        rows_out += [_fixture_csv_row(r, "ok", "12.5") for r in d7cap[2]]
        _write_fixture_csv(partial, rows_out)
        vp = os.path.join(tmp, "v_partial.json")
        msg = _catch_exit(lambda: score(csv_path=partial, out_path=vp, chatty=False))
        ck(msg == _refused_incomplete("7B", 1) + "\n" + _refused_incomplete("30B", 0),
           "both-models-incomplete refuses with both staked messages, prefix counted "
           "up to the first holed block")
        ck(not os.path.exists(vp), "a refused scoring writes no verdict json")

        # --- synthetic night A: 7B complete, ngl-dominant -> publishable FAIL; 30B
        # absent -> per-model refusal; overall FAIL short-circuits (the design's
        # "the 7B can fire the kill rule the morning after night 1"). Hand-reasoned:
        # s = ngl = m, and ngl is not in {ctk, ub} -> row 2 of the decision table.
        def y_a(run):
            c = run["cfg"]
            return (10.0 + 30.0 * _upos("7B", "ngl", c["ngl"])
                    + 1.0 * _upos("7B", "t", c["t"])
                    + 0.5 * _upos("7B", "ctk", c["ctk"])
                    + 0.25 * _upos("7B", "ub", c["ub"]))

        d7s = build_stage2_design("7B", N_START["7B"])
        night_a = os.path.join(tmp, "night_a.csv")
        _write_fixture_csv(night_a, [_fixture_csv_row(r, "ok", repr(y_a(r)))
                                     for b in d7s for r in b])
        va = os.path.join(tmp, "verdict_a.json")
        res = score(csv_path=night_a, out_path=va, chatty=False)
        m7 = res["models"]["7B"]
        ck(m7["verdict"] == "FAIL" and m7["case"] == "s == m and s not in MAP",
           "night A: 7B is a publishable FAIL (methods agree on ngl, mapping refuted)")
        ck(m7["sobol_argmax"] == "ngl" and m7["bootstrap"]["decided"]
           and m7["bootstrap"]["retention"] >= GATE_RETENTION,
           "night A: gate decides on ngl")
        ck("held_message" not in m7,
           "night A: methods agree, so no Taguchi hold is printed")
        ck(res["models"]["30B"].get("refused") == _refused_incomplete("30B", 0),
           "night A: the absent 30B is refused per-model with the staked message")
        ck(res["P-3"]["overall"] == "FAIL",
           "night A: a publishable model-FAIL short-circuits overall P-3")
        ck(res["label"]["text"] == LABEL and len(res["label"]["surfaces"]) == 3,
           "night A: the kill rule ships the label text and all three surfaces")
        ck(res["exit_code"] == 0,
           "night A: rc 0 - every non-refused model reached a verdict")
        with open(va, encoding="utf-8") as fh:
            js = json.load(fh)
        with open(night_a, "rb") as fh:
            ck(js["csv_sha256"] == hashlib.sha256(fh.read()).hexdigest(),
               "verdict json on disk carries the fixture CSV's sha256")
        ck(js["ground_truth"]["plan_binding"]["30B"] == PLAN_BINDING["30B"],
           "verdict json echoes the frozen ground-truth constants")

        # --- synthetic night B: both models complete, both argmaxes overturned INTO
        # the mapping sets -> both HELD_FOR_TAGUCHI (kill rule 3), overall held, no
        # label. Hand-reasoned: 7B s = ctk != ngl, ctk in MAP; 30B s = moe_cpu_frac
        # != t, in MAP.
        def y_b(run):
            c = run["cfg"]
            if run["model"] == "7B":
                return (10.0 + 20.0 * _upos("7B", "ctk", c["ctk"])
                        + 1.0 * _upos("7B", "ngl", c["ngl"])
                        + 0.5 * _upos("7B", "t", c["t"])
                        + 0.25 * _upos("7B", "ub", c["ub"]))
            return (10.0 + 20.0 * _upos("30B", "moe_cpu_frac", c["moe_cpu_frac"])
                    + 1.0 * _upos("30B", "ngl", c["ngl"])
                    + 0.5 * _upos("30B", "t", c["t"])
                    + 0.25 * _upos("30B", "ctk", c["ctk"])
                    + 0.25 * _upos("30B", "mmp", c["mmp"]))

        d30s = build_stage2_design("30B", N_START["30B"])
        night_b = os.path.join(tmp, "night_b.csv")
        _write_fixture_csv(night_b, [_fixture_csv_row(r, "ok", repr(y_b(r)))
                                     for d in (d7s, d30s) for b in d for r in b])
        res = score(csv_path=night_b, out_path=os.path.join(tmp, "verdict_b.json"),
                    chatty=False)
        ck(res["models"]["7B"]["verdict"] == "HELD_FOR_TAGUCHI"
           and res["models"]["7B"]["case"] == "s != m and s in MAP"
           and res["models"]["7B"]["held_message"] == _held_msg("7B", "ctk"),
           "night B: 7B held for Taguchi with the staked message (Sobol says ctk)")
        ck(res["models"]["30B"]["verdict"] == "HELD_FOR_TAGUCHI"
           and res["models"]["30B"]["held_message"] == _held_msg("30B", "moe_cpu_frac"),
           "night B: 30B held for Taguchi with the staked message (Sobol says "
           "moe_cpu_frac)")
        ck(res["P-3"]["overall"] == "HELD_FOR_TAGUCHI" and "label" not in res,
           "night B: overall held, and a hold ships no label")
        ck(res["exit_code"] == 0, "night B: rc 0 - held IS a per-model verdict")

        # --- validity-floor drill: night B's 7B rows with 9 of 32 AB_ctk rows
        # declared DNF -> ctk has 23 valid ST pairs < ceil(0.75*32) = 24 -> ctk is
        # UNSCOREABLE and the 7B P-3 is VOID, never guessed from the remaining factors.
        rows_out = []
        flipped = 0
        for b in d7s:
            for r in b:
                if r["matrix"] == "AB_ctk" and flipped < 9:
                    rows_out.append(_fixture_csv_row(r, "dnf_timeout", ""))
                    flipped += 1
                else:
                    rows_out.append(_fixture_csv_row(r, "ok", repr(y_b(r))))
        void_csv = os.path.join(tmp, "void.csv")
        _write_fixture_csv(void_csv, rows_out)
        res = score(csv_path=void_csv, out_path=os.path.join(tmp, "verdict_void.json"),
                    chatty=False)
        m7 = res["models"]["7B"]
        ck(m7["verdict"] == "VOID"
           and m7["void_because"] == ["ctk: 23 valid ST pairs < floor 24"]
           and not m7["factors"]["ctk"]["scoreable"],
           "validity floor: 9 DNFed AB_ctk rows void the 7B verdict at 23 < 24 pairs")
        ck(m7["rows"]["dnf"] == 9 and len(m7["non_ok_rows"]) == 9,
           "validity floor: the 9 declared DNFs are counted and listed")
        ck(res["P-3"]["overall"] == "VOID" and res["exit_code"] == 0,
           "validity floor: VOID propagates to overall and rc stays 0 (VOID is a "
           "verdict, not a refusal)")

        # --- degenerate drill: every tok_s identical -> V(Y) undefined -> VOID (the
        # stage-1 zero-signal rule restated for variance).
        flat_csv = os.path.join(tmp, "flat.csv")
        _write_fixture_csv(flat_csv, [_fixture_csv_row(r, "ok", "5.0")
                                      for b in d7s for r in b])
        res = score(csv_path=flat_csv, out_path=os.path.join(tmp, "verdict_flat.json"),
                    chatty=False)
        ck(res["models"]["7B"]["verdict"] == "VOID"
           and res["models"]["7B"]["V_Y"] is None,
           "degenerate data (zero pooled variance) voids the model, never a verdict")

        # --- synthetic night C: a planted dead-heat -> UNDECIDED below cap. The
        # response gives ngl the whole effect on even blocks and t on odd blocks, so
        # the full-data STs are near-equal and the resampled argmax flips with the
        # drawn block mix - retention cannot hold 950. No verdict; the staked
        # extension message with N+16 = 48; rc 1.
        def y_c(run):
            c = run["cfg"]
            base = 10.0 + 0.1 * _upos("7B", "ctk", c["ctk"])
            if run["block"] % 2 == 0:
                return base + 14.0 * _upos("7B", "ngl", c["ngl"])
            return base + 14.0 * _upos("7B", "t", c["t"])

        night_c = os.path.join(tmp, "night_c.csv")
        _write_fixture_csv(night_c, [_fixture_csv_row(r, "ok", repr(y_c(r)))
                                     for b in d7s for r in b])
        res = score(csv_path=night_c, out_path=os.path.join(tmp, "verdict_c.json"),
                    chatty=False)
        m7 = res["models"]["7B"]
        ck(m7["verdict"] is None and not m7["bootstrap"]["decided"]
           and m7["bootstrap"]["retention"] < GATE_RETENTION,
           "night C: a planted ngl/t dead heat leaves the gate undecided")
        ck(m7["undecided_message"]
           == _undecided_msg("7B", m7["bootstrap"]["retention"], 32),
           "night C: the staked UNDECIDED message with the +16 extension is issued")
        ck(res["exit_code"] == 1 and res["P-3"]["overall"] == "PENDING",
           "night C: no verdict -> rc 1 and overall pending")

        # --- synthetic night D: the same dead heat at N_cap = 64 -> UNDECIDABLE, and
        # the scope label ships anyway (amendment item 7: stricter than staked,
        # declared - perpetual undecidability must not become a shield).
        night_d = os.path.join(tmp, "night_d.csv")
        _write_fixture_csv(night_d, [_fixture_csv_row(r, "ok", repr(y_c(r)))
                                     for b in d7cap for r in b])
        res = score(csv_path=night_d, out_path=os.path.join(tmp, "verdict_d.json"),
                    chatty=False)
        m7 = res["models"]["7B"]
        ck(m7["verdict"] == "UNDECIDABLE" and m7["n_complete_blocks"] == N_CAP
           and m7["case"] == "gate never decides at N_cap",
           "night D: still undecided at N_cap 64 -> verdict UNDECIDABLE")
        ck(res["label"]["text"] == LABEL
           and any("amendment item 7" in r for r in res["label"]["reasons"]),
           "night D: UNDECIDABLE still ships the scope label, naming amendment item 7")
        ck(res["P-3"]["overall"] == "HELD_FOR_TAGUCHI" and res["exit_code"] == 0,
           "night D: undecidable feeds the held branch of the overall rule; rc 0")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if ck.failures:
        raise SystemExit("self-check FAILED (%d of %d checks):\n  %s"
                         % (len(ck.failures), ck.n, "\n  ".join(ck.failures)))
    print(f"self-check OK: {ck.n} checks passed (pinned strings and ground truth, "
          "seeded design, hand-derived fixture, refusal drills, decided/held/void/"
          "undecided/undecidable synthetic nights)")


def main(argv):
    args = argv[1:]
    if args == ["--self-check"]:
        self_check()
        return 0
    if args:
        raise SystemExit("usage: python weights/prereg95_sobol_score.py [--self-check]")
    return score()["exit_code"]


if __name__ == "__main__":
    sys.exit(main(sys.argv))
