"""Score pre-registration #95 stage 1 - Morris screening: which llama.cpp flags matter.

WRITTEN AND COMMITTED BEFORE THE FIRST STAGE-1 CSV ROW EXISTED. The design doc
(docs/DESIGN_DOE_MORRIS.md, section 4) precommits this scorer the same way prereg #100 and
prereg #102 precommitted theirs, and for the same reason: a verdict computed by code
written in ignorance of the data cannot be tuned to the data. Neither
weights/data/doe_morris_stage1.csv nor the harness that will write it existed when this
file was staked.

    python weights/prereg95_score.py               # score the real CSV (refuses until complete)
    python weights/prereg95_score.py --self-check  # hand-derived fixture + refusal drills, exit 0

Reads  weights/data/doe_morris_stage1.csv (append-only, written by weights/doe_morris.py),
writes weights/data/prereg95_verdict.json and prints the table. It scores ONLY the stage-1
stakes; it refuses everything else. It edits no prereg and no README - humans wire
verdicts into documents.

This module carries its OWN copy of the seeded Morris design generator, and so does the
harness - two copies ON PURPOSE, which is the opposite of the runner.py rule and needs its
reason stated: this copy is the FROZEN precommit expectation, and importing the design
from the harness would let a post-stake harness edit move the expectation with it. The
copies were verified to regenerate identical 150 run_ids at stake time, and score() also
cross-checks the harness's live build_design() on every real scoring run, REFUSING if the
two have drifted apart (an absent harness is a loud warning, not a refusal - the frozen
copy joined against the CSV's run_ids is the load-bearing gate either way).

Stage-1 stakes, thresholds VERBATIM from the prereg
(preregistrations/2026-08-07-doe-flag-screening.md) as pinned by the design doc before
any data existed:

    P-1 (concentration)       PASS iff, on BOTH models, sum of the top-3 mu_star >= 0.70 *
                              sum of all mu_star.
    P-2 (regimes separate)    PASS iff the argmax-mu_star factor differs between 7B and 30B.
    P-4 (interaction warning) PASS iff sigma(-ub) > median(all sigma) on BOTH models - the
                              strict both-models reading, pinned before data exists so a
                              favourable reading cannot be picked afterwards. Per-model
                              values are reported either way.
    P-3                       is a Sobol claim; Morris data cannot score it and this tool
                              never fakes it. Printed verbatim:
                              P-3: deferred to stage 2 (Sobol) - not scored by this tool.

Refusals (exact messages, checked in this order - the strings are part of the stake):
absent CSV; header sha256 drift against the pinned design hash; incomplete design (all
150 seeded run_ids present or nothing - a declared DNF row counts as present, a missing
row does not). Validity floor: a factor with fewer than 6 valid elementary effects (of
R = 10) in a model is UNSCOREABLE there, and every stake touching that model is reported
VOID, never guessed.
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
CSV_REL = "weights/data/doe_morris_stage1.csv"          # how the stake names the file
CSV_PATH = os.path.join(HERE, "data", "doe_morris_stage1.csv")
VERDICT_PATH = os.path.join(HERE, "data", "prereg95_verdict.json")

# The section-3 CSV header, byte for byte. Column order is load-bearing: the refusal
# below hash-pins it so a drifted schema is refused instead of silently mis-scored.
HEADER = (
    "run_id,model,traj,pos,changed_factor,ngl,ub,t,ctk,mmp,fa,moe_cpu_frac,status,tok_s,"
    "stddev_ts,reps_tok_s,wall_s,settle_s,free_ram_gb_pre,temp_pre,sm_mhz_pre,mem_mhz_pre,"
    "vram_mib_pre,power_w_pre,temp_post,sm_mhz_post,mem_mhz_post,vram_mib_post,power_w_post,"
    "ts_utc,cmd")
HEADER_SHA256 = "47ed63b0f4ecfa3c3a7e6a140a79eb38b0713841ee5109c8f49a3c8e27d0c624"
COLS = HEADER.split(",")

# Morris constants from design section 2. Grid arithmetic lives in integer THIRDS
# (grid 0..3 <-> normalized {0, 1/3, 2/3, 1}) so trajectory stepping and elementary
# effects never accumulate float error; delta = 2 thirds is the canonical p/(2(p-1)) = 2/3.
R = 10
P_LEVELS = 4
DELTA_THIRDS = 2
BASE_THIRDS = (0, 1)          # legal starts: the +delta step stays inside the lattice
SEED_FMT = "prereg95:{tag}:20260807"
MODELS = ("7B", "30B")

# Factor order is the RNG-draw order AND the deterministic tie-break for argmax. It
# follows the CSV column order (design section 3). The harness imports this tuple: a
# re-typed copy that disagreed would silently shift every seeded draw.
FACTORS = {
    "7B":  ("ngl", "ub", "t", "ctk", "mmp", "fa"),
    "30B": ("ngl", "ub", "t", "ctk", "mmp", "fa", "moe_cpu_frac"),
}

# Levels from the design section 1 audit. 4-level factors map grid index -> level; the
# 2-level factors (ctk, mmp, fa) read level 0 below the grid midpoint and level 1 above,
# so every delta = 2/3 step flips them and none is ever stranded. mmp lists (1, 0)
# because mmap ON is the build default and sits at the low end of the grid.
LEVELS = {
    "7B": {
        "ngl": (0, 9, 19, 99),
        "ub": (128, 512, 1024, 2048),
        "t": (1, 2, 3, 4),
        "ctk": ("f16", "q8_0"),
        "mmp": (1, 0),
        "fa": (0, 1),
    },
    "30B": {
        "ngl": (0, 16, 32, 99),
        "ub": (128, 512, 1024, 2048),
        "t": (1, 2, 3, 4),
        "ctk": ("f16", "q8_0"),
        "mmp": (1, 0),
        "fa": (0, 1),
        "moe_cpu_frac": (0.75, 0.833, 0.917, 1.0),
    },
}

N_DESIGNED = sum(R * (len(FACTORS[m]) + 1) for m in MODELS)     # 70 + 80 = 150

MIN_VALID_EE = 6     # validity floor: fewer than 6 of R = 10 EEs -> factor UNSCOREABLE
TOP_N = 3            # P-1 counts the top-3 mu_star
CONC_SHARE = 0.70    # P-1 bar
P3_LINE = "P-3: deferred to stage 2 (Sobol) - not scored by this tool."

# Refusal 1 always names the file by its staked repo-relative path, whichever physical
# path was probed: the stake is about the one real CSV, and the path parameters on the
# functions below exist only so --self-check can drill these code paths on fixtures.
REFUSED_ABSENT = (
    "REFUSED: weights/data/doe_morris_stage1.csv not found. Stage 1 has not produced "
    "data; the scorer never invents a verdict.")


def _refused_header(found):
    return (f"REFUSED: CSV header hash {found} != design hash {HEADER_SHA256}. This "
            "file was not written by the staked harness; scoring it would score a "
            "different experiment.")


def _refused_partial(n_present, n_ok, n_dnf):
    return (f"REFUSED: incomplete design: {n_present} of {N_DESIGNED} designed runs "
            f"present ({n_ok} ok, {n_dnf} declared DNF). Resume weights/doe_morris.py; "
            "a partial night is not the staked design.")


# ---------------------------------------------------------------------------
# Seeded design - regenerated, never stored: same seed => same design, forever.
# ---------------------------------------------------------------------------

def config_from_grid(model_tag, grid):
    """{factor: llama-bench value} for a grid vector in thirds (design "numeric mapping")."""
    out = {}
    for f in FACTORS[model_tag]:
        levels = LEVELS[model_tag][f]
        g = grid[f]
        out[f] = levels[g] if len(levels) == P_LEVELS else levels[0 if g < 2 else 1]
    return out


def run_id_for(model_tag, traj, pos, config):
    """sha256(f"{model}|{traj}|{pos}|{canonical_config_json}")[:16] - design section 3.

    Canonical json = sorted keys, no whitespace. Stable across relaunches by
    construction, which is what makes both the harness's resume skipping and our
    completeness refusal sound: the two tools recognise a run by the same 16 hex chars.
    """
    canon = json.dumps(config, sort_keys=True, separators=(",", ":"))
    key = f"{model_tag}|{traj}|{pos}|{canon}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _run(model_tag, traj, pos, changed, grid):
    config = config_from_grid(model_tag, grid)
    return {"rid": run_id_for(model_tag, traj, pos, config), "model": model_tag,
            "traj": traj, "pos": pos, "changed": changed, "grid": dict(grid),
            "config": config}


def morris_design(model_tag):
    """The R = 10 seeded trajectories: a list of trajectories, each k+1 run dicts.

    Draw order per trajectory is pinned by design section 2 - k base draws, then k
    direction draws, then one shuffle - all from the single stream seeded
    "prereg95:{tag}:20260807". rng.choice() consumes one _randbelow() per call
    regardless of the tuple's values, so drawing integer THIRDS here yields the
    identical stream to drawing the design doc's {0, 1/3} floats; string seeding and
    choice/shuffle draws are stable across CPython 3.x.

    A direction of +1 starts the factor at its drawn base and steps up by delta when
    its turn comes; -1 starts at base + delta and steps down - every elementary effect
    crosses the grid midpoint either way.
    """
    factors = FACTORS[model_tag]
    k = len(factors)
    rng = random.Random(SEED_FMT.format(tag=model_tag))
    trajs = []
    for traj in range(R):
        starts = [rng.choice(BASE_THIRDS) for _ in range(k)]        # draw 1: bases
        dirs = [rng.choice((1, -1)) for _ in range(k)]              # draw 2: directions
        order = list(range(k))
        rng.shuffle(order)                                         # draw 3: step order
        cur = {factors[i]: (starts[i] if dirs[i] == 1 else starts[i] + DELTA_THIRDS)
               for i in range(k)}
        runs = [_run(model_tag, traj, 0, None, cur)]
        for pos, i in enumerate(order, start=1):
            cur = dict(cur)
            cur[factors[i]] += dirs[i] * DELTA_THIRDS
            runs.append(_run(model_tag, traj, pos, factors[i], cur))
        trajs.append(runs)
    return trajs


# ---------------------------------------------------------------------------
# CSV loading and the three staked refusals
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
# Morris statistics
# ---------------------------------------------------------------------------

def elementary_effects(trajs, rows):
    """{factor: [signed EE]} with EE = (y_after - y_before) / (g_after - g_before).

    g is the factor's normalized [0,1] grid value, so EEs are tok/s per unit range and
    mu_star is comparable across factors. Computed as dy * 3 / (thirds_after -
    thirds_before): algebraically the design formula, arranged so the division is by an
    exact integer (+-2) instead of the float 2/3 - on the p = 4 grid this keeps every EE
    bit-exact against hand arithmetic, which is what lets --self-check assert equality
    instead of tolerances. A step with a non-ok endpoint yields no EE: a DNF poisons
    exactly the two adjacent effects, nothing more.
    """
    ees = {}
    for runs in trajs:
        for before, after in zip(runs, runs[1:]):
            f = after["changed"]
            ees.setdefault(f, [])
            ya = _valid_y(rows.get(before["rid"]))
            yb = _valid_y(rows.get(after["rid"]))
            if ya is None or yb is None:
                continue
            dg_thirds = after["grid"][f] - before["grid"][f]
            ees[f].append((yb - ya) * 3.0 / dg_thirds)
    return ees


def mu_sigma(values):
    """(mu_star, sigma) = (mean |EE|, stddev of signed EE with ddof=1).

    Implemented by hand instead of statistics.stdev so the arithmetic the self-check
    derives on paper is literally the arithmetic that runs; None where undefined (no
    EEs for mu, fewer than 2 for sigma).
    """
    n = len(values)
    if n == 0:
        return None, None
    mu_star = sum(abs(v) for v in values) / n
    if n < 2:
        return mu_star, None
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mu_star, math.sqrt(var)


def _median(xs):
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

REFUSED_DRIFT = ("REFUSED: design drift - weights/doe_morris.py and this scorer's frozen "
                 "copy no longer regenerate the same 150 run_ids. One factor table was "
                 "edited without the other; adjudicate against git history before scoring "
                 "anything.")


def _cross_check_harness():
    """Added pre-data, 2026-08-16, AFTER the three staked refusals (their order is part of
    the stake and stays untouched): actively detect the copy-drift the two-copies design
    accepts. The harness's live build_design() must regenerate this file's frozen 150
    run_ids in order; a mismatch refuses the scoring run. Import failure only warns -
    the frozen copy joined against the CSV is the load-bearing gate."""
    import importlib.util
    hpath = os.path.join(HERE, "doe_morris.py")
    # the harness does `import runner` (its sibling): HERE must be on sys.path for the
    # exec to resolve it - proven the hard way, the first version of this check fell into
    # the warning path on EVERY call and would have let drift escape silently
    added = HERE not in sys.path
    if added:
        sys.path.insert(0, HERE)
    try:
        spec = importlib.util.spec_from_file_location("doe_morris_livecheck", hpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        theirs = [r["run_id"] for m in MODELS for r in mod.build_design(m)]
    except Exception as e:
        print(f"[prereg95] WARNING: cross-check skipped, {hpath} not importable ({e}); "
              f"scoring on this file's frozen design copy alone.")
        return
    finally:
        if added and HERE in sys.path:
            sys.path.remove(HERE)
    ours = [run["rid"] for m in MODELS for runs in morris_design(m) for run in runs]
    if ours != theirs:
        raise SystemExit(REFUSED_DRIFT)


def score(csv_path=CSV_PATH, out_path=VERDICT_PATH, chatty=True):
    """Refuse, or score the three stage-1 stakes; write the verdict json; return it."""
    rows, csv_sha, dupes = _load_rows(csv_path)

    designs = {m: morris_design(m) for m in MODELS}
    designed = [run for m in MODELS for runs in designs[m] for run in runs]
    present = [run for run in designed if run["rid"] in rows]
    n_ok = sum(1 for run in present if rows[run["rid"]].get("status") == "ok")
    n_dnf = sum(1 for run in present
                if (rows[run["rid"]].get("status") or "").startswith("dnf"))
    if len(present) < N_DESIGNED:
        raise SystemExit(_refused_partial(len(present), n_ok, n_dnf))
    _cross_check_harness()

    result_models = {}
    dnf_list = []
    bad_ok = []
    for m in MODELS:
        ees = elementary_effects(designs[m], rows)
        table = {}
        unscoreable = []
        for f in FACTORS[m]:
            vals = ees.get(f, [])
            mu, sig = mu_sigma(vals)
            scoreable = len(vals) >= MIN_VALID_EE
            if not scoreable:
                unscoreable.append(f)
            table[f] = {"mu_star": mu, "sigma": sig, "n_ee": len(vals),
                        "scoreable": scoreable}
        for runs in designs[m]:
            for run in runs:
                row = rows[run["rid"]]
                status = row.get("status") or ""
                if status != "ok":
                    dnf_list.append({"run_id": run["rid"], "model": m,
                                     "traj": run["traj"], "pos": run["pos"],
                                     "changed_factor": run["changed"] or "base",
                                     "status": status})
                elif _valid_y(row) is None:
                    bad_ok.append(run["rid"])
        result_models[m] = {"factors": table, "unscoreable": unscoreable}

    # Every stage-1 stake touches every factor of both models (P-1 sums all mu_star,
    # P-2 takes an argmax over all, P-4 takes a median over all sigma), so a single
    # unscoreable factor anywhere voids all three verdicts - the design's "any stake
    # touching an unscoreable factor is reported VOID, never guessed". Per-model
    # details are still computed and reported for whichever model IS fully scoreable.
    void_because = {m: list(result_models[m]["unscoreable"]) for m in MODELS
                    if result_models[m]["unscoreable"]}
    p1_detail, p2_detail, p4_detail = {}, {}, {}
    for m in MODELS:
        if result_models[m]["unscoreable"]:
            continue
        table = result_models[m]["factors"]
        mus = {f: table[f]["mu_star"] for f in FACTORS[m]}
        sigs = {f: table[f]["sigma"] for f in FACTORS[m]}
        total = sum(mus.values())
        if total <= 0.0:
            # bitwise-identical tok_s across every run of a model: unreachable with real
            # measurement noise, but "top3 >= 0.70 * 0" would PASS P-1 on zero signal.
            # Degenerate data voids the stakes; a verdict is never manufactured from it.
            void_because.setdefault(m, []).append(
                "ALL FACTORS: total mu_star == 0 (degenerate - no signal in this model)")
            continue
        top3 = sum(sorted(mus.values(), reverse=True)[:TOP_N])
        p1_detail[m] = {"top3_sum": top3, "total_sum": total,
                        "share": (top3 / total) if total else None,
                        "pass": top3 >= CONC_SHARE * total}
        top = max(FACTORS[m], key=lambda f: mus[f])   # exact tie -> first in factor order
        p2_detail[m] = {"top_factor": top, "mu_star": mus[top]}
        med = _median(list(sigs.values()))
        p4_detail[m] = {"sigma_ub": sigs["ub"], "median_sigma": med,
                        "pass": sigs["ub"] > med}

    if void_because:
        v1 = v2 = v4 = "VOID"
    else:
        v1 = "PASS" if all(p1_detail[m]["pass"] for m in MODELS) else "FAIL"
        v2 = ("PASS" if p2_detail["7B"]["top_factor"] != p2_detail["30B"]["top_factor"]
              else "FAIL")
        v4 = "PASS" if all(p4_detail[m]["pass"] for m in MODELS) else "FAIL"

    result = {
        "prereg": 95,
        "stage": 1,
        "csv": CSV_REL,
        "csv_sha256": csv_sha,
        "scored_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "design": {"R": R, "p": P_LEVELS, "delta": "2/3",
                   "seeds": [SEED_FMT.format(tag=m) for m in MODELS],
                   "k": {m: len(FACTORS[m]) for m in MODELS},
                   "n_designed": N_DESIGNED, "header_sha256": HEADER_SHA256},
        "rows": {"present": len(present), "ok": n_ok, "dnf": n_dnf,
                 "other_non_ok": len(present) - n_ok - n_dnf,
                 "duplicate_run_ids_ignored": dupes,
                 "extra_rows_outside_design": len(rows) - len(present),
                 "ok_rows_with_unparseable_tok_s": bad_ok},
        "models": result_models,
        "dnf_list": dnf_list,
        "verdicts": {
            "P-1": {"rule": "sum of top-3 mu_star >= 0.70 * sum of all mu_star, "
                            "on BOTH models",
                    "verdict": v1, "per_model": p1_detail},
            "P-2": {"rule": "argmax-mu_star factor differs between 7B and 30B",
                    "verdict": v2, "per_model": p2_detail},
            "P-4": {"rule": "sigma(ub) > median(all sigma) on BOTH models",
                    "verdict": v4, "per_model": p4_detail},
        },
        "P-3": P3_LINE,
    }
    if void_because:
        result["void_because"] = void_because
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    if chatty:
        _print_report(result, out_path)
    return result


def _print_report(res, out_path):
    print("prereg #95 stage 1 - Morris screening: which llama.cpp flags matter (E-16)")
    print(f"  csv: {res['csv']}  sha256 {res['csv_sha256'][:16]}..")
    rows = res["rows"]
    extra = (f"; {rows['extra_rows_outside_design']} rows outside the design ignored"
             if rows["extra_rows_outside_design"] else "")
    print(f"  rows: {rows['present']} of {N_DESIGNED} designed present "
          f"({rows['ok']} ok, {rows['dnf']} DNF, {rows['other_non_ok']} other non-ok)"
          + extra)
    if rows["duplicate_run_ids_ignored"]:
        print(f"  note: {rows['duplicate_run_ids_ignored']} duplicate run_id rows "
              "ignored (first write wins)")
    if rows["ok_rows_with_unparseable_tok_s"]:
        print("  note: ok rows with unparseable tok_s, treated as invalid endpoints: "
              + ", ".join(rows["ok_rows_with_unparseable_tok_s"]))

    for m in MODELS:
        table = res["models"][m]["factors"]
        print(f"\n  {m}  (k = {len(table)}; mu_star, sigma in tok/s per unit factor "
              "range; ranked by mu_star)")
        ranked = sorted(FACTORS[m],
                        key=lambda f: (table[f]["mu_star"]
                                       if table[f]["mu_star"] is not None
                                       else float("-inf")),
                        reverse=True)
        for f in ranked:
            e = table[f]
            mu = "   unscored" if e["mu_star"] is None else f"{e['mu_star']:11.3f}"
            sg = "          -" if e["sigma"] is None else f"{e['sigma']:11.3f}"
            tag = ("" if e["scoreable"]
                   else f"  UNSCOREABLE ({e['n_ee']} < {MIN_VALID_EE} valid EEs)")
            print(f"    {f:<13} mu* {mu}  sigma {sg}  n_EE {e['n_ee']:>2}{tag}")

    print("\n=== stage-1 verdicts (thresholds staked 2026-08-07; scorer frozen before "
          "data) ===")
    v = res["verdicts"]
    parts = []
    for m in MODELS:
        d = v["P-1"]["per_model"].get(m)
        parts.append(f"{m} top-3 share "
                     + (f"{d['share']:.3f}" if d and d["share"] is not None else "n/a"))
    print(f"  P-1 concentration (bar >= {CONC_SHARE:.2f} both models): "
          + ", ".join(parts) + f"  -> {v['P-1']['verdict']}")
    parts = []
    for m in MODELS:
        d = v["P-2"]["per_model"].get(m)
        parts.append(f"{m} top " + (d["top_factor"] if d else "n/a"))
    print("  P-2 regimes separate (top factor must differ): "
          + ", ".join(parts) + f"  -> {v['P-2']['verdict']}")
    parts = []
    for m in MODELS:
        d = v["P-4"]["per_model"].get(m)
        parts.append(f"{m} sigma(ub) "
                     + (f"{d['sigma_ub']:.3f} vs median {d['median_sigma']:.3f}"
                        if d else "n/a"))
    print("  P-4 interaction warning (sigma(ub) > median sigma, both models): "
          + "; ".join(parts) + f"  -> {v['P-4']['verdict']}")
    if "void_because" in res:
        print("  VOID: unscoreable factors - "
              + "; ".join(m + ": " + ", ".join(fs)
                          for m, fs in res["void_because"].items()))
    print("  " + res["P-3"])
    print(f"\n  verdict json: {out_path}")
    print("  (this tool edits no prereg and no README; humans wire verdicts into "
          "documents)")


# ---------------------------------------------------------------------------
# Self-check: hand-derived fixture, seeded-design invariants, refusal drills,
# and a synthetic full night with hand-reasoned verdicts. Exit 0 iff all pass.
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
    """A full 31-column CSV row for a designed run; telemetry cells stay empty."""
    cells = {c: "" for c in COLS}
    cells.update({"run_id": run["rid"], "model": run["model"],
                  "traj": str(run["traj"]), "pos": str(run["pos"]),
                  "changed_factor": run["changed"] or "base",
                  "status": status, "tok_s": tok_s})
    for f, v in run["config"].items():
        cells[f] = str(v)
    return [cells[c] for c in COLS]


def _write_fixture_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(HEADER + "\n")
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerows(rows)


def self_check():
    ck = _Check()

    # --- pinned strings and constants ------------------------------------------------
    ck(hashlib.sha256(HEADER.encode("utf-8")).hexdigest() == HEADER_SHA256,
       "pinned header hash reproduces from the header string")
    ck(len(COLS) == 31, "header carries the 31 designed columns")
    ck(N_DESIGNED == 150, "designed run count is 70 + 80 = 150")
    ck(REFUSED_ABSENT == "REFUSED: weights/data/doe_morris_stage1.csv not found. "
                         "Stage 1 has not produced data; the scorer never invents a "
                         "verdict.",
       "refusal 1 string matches the design verbatim")
    ck(P3_LINE == "P-3: deferred to stage 2 (Sobol) - not scored by this tool.",
       "P-3 deferral line matches the design verbatim")

    # --- seeded design invariants ----------------------------------------------------
    d7, d30 = morris_design("7B"), morris_design("30B")
    ck(sum(len(t) for t in d7) == 70, "7B design is 10 trajectories x 7 runs = 70")
    ck(sum(len(t) for t in d30) == 80, "30B design is 10 trajectories x 8 runs = 80")
    ids = [r["rid"] for d in (d7, d30) for t in d for r in t]
    ck(len(set(ids)) == 150, "the 150 designed run_ids are unique")
    ck([r["rid"] for t in morris_design("7B") for r in t]
       == [r["rid"] for t in d7 for r in t],
       "design regeneration is deterministic (same seed, same ids)")
    for model, design in (("7B", d7), ("30B", d30)):
        one_step, delta_ok, flips_ok, lattice_ok, cover_ok = True, True, True, True, True
        for t in design:
            stepped = []
            for a, b in zip(t, t[1:]):
                diff = [f for f in FACTORS[model] if a["grid"][f] != b["grid"][f]]
                if len(diff) != 1 or diff[0] != b["changed"]:
                    one_step = False
                    continue
                f = diff[0]
                stepped.append(f)
                if abs(b["grid"][f] - a["grid"][f]) != DELTA_THIRDS:
                    delta_ok = False
                if (len(LEVELS[model][f]) == 2
                        and a["config"][f] == b["config"][f]):
                    flips_ok = False
            if set(stepped) != set(FACTORS[model]):
                cover_ok = False
            if not all(0 <= g <= 3 for r in t for g in r["grid"].values()):
                lattice_ok = False
        ck(one_step, f"{model}: every step changes exactly the recorded factor")
        ck(delta_ok, f"{model}: every step moves by delta = 2 thirds")
        ck(flips_ok, f"{model}: 2-level factors flip level on every step (never stranded)")
        ck(cover_ok, f"{model}: each trajectory steps each factor exactly once")
        ck(lattice_ok, f"{model}: all grids inside the p = 4 lattice")

    # --- hand-derived fixture ---------------------------------------------------------
    # k = 2 factors A, B; R = 2 trajectories; p = 4 grid in thirds (0..3 <-> 0, 1/3,
    # 2/3, 1); delta = 2 thirds = 2/3 of the normalized range. Derived BY HAND:
    #
    #   T0: step order A then B, both directions UP, bases A=0, B=1
    #       pos0 grid (A=0, B=1)  y = 10.0
    #       pos1 grid (A=2, B=1)  y = 16.0    A stepped 0 -> 2 thirds (+2/3 normalized)
    #       pos2 grid (A=2, B=3)  y = 13.0    B stepped 1 -> 3 thirds (+2/3 normalized)
    #       EE_A = (16.0 - 10.0) / (+2/3) =  6.0 * 3 / +2 = +9.0
    #       EE_B = (13.0 - 16.0) / (+2/3) = -3.0 * 3 / +2 = -4.5
    #   T1: step order B then A, both directions DOWN, starts A=3, B=2
    #       pos0 grid (A=3, B=2)  y = 20.0
    #       pos1 grid (A=3, B=0)  y = 23.0    B stepped 2 -> 0 thirds (-2/3 normalized)
    #       pos2 grid (A=1, B=0)  y = 20.0    A stepped 3 -> 1 thirds (-2/3 normalized)
    #       EE_B = (23.0 - 20.0) / (-2/3) =  3.0 * 3 / -2 = -4.5
    #       EE_A = (20.0 - 23.0) / (-2/3) = -3.0 * 3 / -2 = +4.5
    #
    #   mu_star_A = (|+9.0| + |+4.5|) / 2 = 13.5 / 2 = 6.75
    #   sigma_A   = stddev([+9.0, +4.5], ddof=1): mean 6.75, deviations +-2.25,
    #               squares 5.0625 each, var (5.0625 + 5.0625) / (2 - 1) = 10.125,
    #               sigma = sqrt(10.125)   [= 2.25 * sqrt(2) ~= 3.18198]
    #   mu_star_B = (|-4.5| + |-4.5|) / 2 = 4.5
    #   sigma_B   = stddev([-4.5, -4.5], ddof=1) = 0.0 exactly
    #
    # Every arithmetic step above is exact in binary floats (dy integers halved/tripled,
    # 13.5/2, 2.25^2, 10.125 all exactly representable) except the final sqrt, which is
    # IEEE-754 correctly rounded - so the assertions below demand EXACT equality, no
    # tolerances.
    def frun(traj, pos, changed, ga, gb):
        return {"rid": f"fix|{traj}|{pos}", "model": "FIX", "traj": traj, "pos": pos,
                "changed": changed, "grid": {"A": ga, "B": gb}}

    t0 = [frun(0, 0, None, 0, 1), frun(0, 1, "A", 2, 1), frun(0, 2, "B", 2, 3)]
    t1 = [frun(1, 0, None, 3, 2), frun(1, 1, "B", 3, 0), frun(1, 2, "A", 1, 0)]
    fy = {"fix|0|0": "10.0", "fix|0|1": "16.0", "fix|0|2": "13.0",
          "fix|1|0": "20.0", "fix|1|1": "23.0", "fix|1|2": "20.0"}
    fix_rows = {rid: {"status": "ok", "tok_s": y} for rid, y in fy.items()}
    ees = elementary_effects([t0, t1], fix_rows)
    ck(ees["A"] == [9.0, 4.5], "fixture EE_A trajectory values are [+9.0, +4.5] exactly")
    ck(ees["B"] == [-4.5, -4.5], "fixture EE_B trajectory values are [-4.5, -4.5] exactly")
    mu_a, sg_a = mu_sigma(ees["A"])
    mu_b, sg_b = mu_sigma(ees["B"])
    ck(mu_a == 6.75, "fixture mu_star_A == 6.75 exactly")
    ck(sg_a == math.sqrt(10.125), "fixture sigma_A == sqrt(10.125) exactly")
    ck(mu_b == 4.5, "fixture mu_star_B == 4.5 exactly")
    ck(sg_b == 0.0, "fixture sigma_B == 0.0 exactly")

    # A DNF at T0 pos1 must poison exactly the two adjacent effects (T0's EE_A and
    # EE_B) and leave T1 untouched.
    dnf_rows = dict(fix_rows)
    dnf_rows["fix|0|1"] = {"status": "dnf_timeout", "tok_s": ""}
    ees = elementary_effects([t0, t1], dnf_rows)
    ck(ees["A"] == [4.5] and ees["B"] == [-4.5],
       "a DNF endpoint poisons exactly the two adjacent EEs, nothing more")

    tmp = tempfile.mkdtemp(prefix="prereg95_selfcheck_")
    try:
        # --- refusal drills (fixture paths only; the real CSV is never touched) -------
        msg = _catch_exit(lambda: _load_rows(os.path.join(tmp, "absent.csv")))
        ck(msg == REFUSED_ABSENT, "absent CSV refuses with the staked message")

        drift = os.path.join(tmp, "drift.csv")
        with open(drift, "w", encoding="utf-8", newline="") as fh:
            fh.write("run_id,model\nx,7B\n")
        found = hashlib.sha256(b"run_id,model").hexdigest()
        msg = _catch_exit(lambda: _load_rows(drift))
        ck(msg == ("REFUSED: CSV header hash " + found + " != design hash "
                   "47ed63b0f4ecfa3c3a7e6a140a79eb38b0713841ee5109c8f49a3c8e27d0c624. "
                   "This file was not written by the staked harness; scoring it would "
                   "score a different experiment."),
           "drifted header refuses with the staked message and the found hash")

        partial = os.path.join(tmp, "partial.csv")
        _write_fixture_csv(partial, [
            _fixture_csv_row(d7[0][0], "ok", "12.5"),
            _fixture_csv_row(d7[0][1], "ok", "13.5"),
            _fixture_csv_row(d7[0][2], "dnf_timeout", ""),
        ])
        msg = _catch_exit(lambda: score(csv_path=partial,
                                        out_path=os.path.join(tmp, "v.json"),
                                        chatty=False))
        ck(msg == "REFUSED: incomplete design: 3 of 150 designed runs present (2 ok, "
                  "1 declared DNF). Resume weights/doe_morris.py; a partial night is "
                  "not the staked design.",
           "partial CSV refuses with the staked message and the right three counts")
        ck(not os.path.exists(os.path.join(tmp, "v.json")),
           "a refused scoring writes no verdict json")

        # --- synthetic full night with hand-reasoned verdicts -------------------------
        # Response is LINEAR in grid thirds: y = 10 + sum(coef_f * grid_f). Then every
        # EE of factor f equals 3 * coef_f exactly (dy = coef * +-2 thirds, EE = dy * 3
        # / +-2), so mu_star = 3 * coef and sigma = 0.0 for every factor. Hand-reasoned
        # verdicts, staked before running the code below:
        #   P-1 PASS: 7B top-3 share = 3*(30+3+2) / 3*(30+3+2+1+1+1) = 35/38 = 0.921;
        #             30B = (40+30+3)/(40+30+3+2+1+1+1) = 73/78 = 0.936; both >= 0.70.
        #   P-2 PASS: 7B argmax is ngl (mu* 90), 30B argmax is moe_cpu_frac (mu* 120).
        #   P-4 FAIL: every sigma is exactly 0.0, so sigma(ub) > median fails (0 > 0).
        coef = {"7B": {"ngl": 30.0, "ub": 3.0, "t": 2.0, "ctk": 1.0, "mmp": 1.0,
                       "fa": 1.0},
                "30B": {"ngl": 30.0, "ub": 3.0, "t": 2.0, "ctk": 1.0, "mmp": 1.0,
                        "fa": 1.0, "moe_cpu_frac": 40.0}}

        def synth_y(model, grid):
            return 10.0 + sum(coef[model][f] * grid[f] for f in FACTORS[model])

        dnf_run = d7[0][3]      # poisons 7B traj 0 steps pos2->3 and pos3->4 only
        rows_out = []
        for model, design in (("7B", d7), ("30B", d30)):
            for t in design:
                for run in t:
                    if run["rid"] == dnf_run["rid"]:
                        rows_out.append(_fixture_csv_row(run, "dnf_timeout", ""))
                    else:
                        rows_out.append(_fixture_csv_row(
                            run, "ok", repr(synth_y(model, run["grid"]))))
        full = os.path.join(tmp, "full.csv")
        _write_fixture_csv(full, rows_out)
        out_json = os.path.join(tmp, "verdict.json")
        res = score(csv_path=full, out_path=out_json, chatty=False)
        ck(res["verdicts"]["P-1"]["verdict"] == "PASS",
           "synthetic night: P-1 PASS (hand-reasoned shares 0.921 / 0.936)")
        ck(res["verdicts"]["P-2"]["verdict"] == "PASS",
           "synthetic night: P-2 PASS (argmax ngl vs moe_cpu_frac)")
        ck(res["verdicts"]["P-2"]["per_model"]["7B"]["top_factor"] == "ngl"
           and res["verdicts"]["P-2"]["per_model"]["30B"]["top_factor"]
           == "moe_cpu_frac",
           "synthetic night: per-model top factors are the planted ones")
        ck(res["verdicts"]["P-4"]["verdict"] == "FAIL",
           "synthetic night: P-4 FAIL (all sigma exactly 0, strict > fails)")
        ck(res["P-3"] == P3_LINE, "verdict json carries the verbatim P-3 line")
        ck(res["models"]["7B"]["factors"]["ngl"]["mu_star"] == 90.0,
           "synthetic night: 7B ngl mu_star == 3 * coef == 90.0 exactly")
        ck(res["models"]["30B"]["factors"]["moe_cpu_frac"]["mu_star"] == 120.0,
           "synthetic night: 30B moe_cpu_frac mu_star == 120.0 exactly")
        ck(all(res["models"][m]["factors"][f]["sigma"] == 0.0
               for m in MODELS for f in FACTORS[m]),
           "synthetic night: linear response gives sigma == 0.0 for every factor")
        ck([d["run_id"] for d in res["dnf_list"]] == [dnf_run["rid"]],
           "synthetic night: dnf_list carries exactly the injected run")
        poisoned = {d7[0][3]["changed"], d7[0][4]["changed"]}
        ck(all(res["models"]["7B"]["factors"][f]["n_ee"]
               == (9 if f in poisoned else 10) for f in FACTORS["7B"]),
           "synthetic night: the DNF costs exactly one EE from each adjacent factor")
        ck(all(res["models"]["30B"]["factors"][f]["n_ee"] == 10
               for f in FACTORS["30B"]),
           "synthetic night: 30B keeps all 10 EEs per factor")
        with open(out_json, encoding="utf-8") as fh:
            js = json.load(fh)
        with open(full, "rb") as fh:
            ck(js["csv_sha256"] == hashlib.sha256(fh.read()).hexdigest(),
               "verdict json on disk carries the fixture CSV's sha256")

        # --- validity floor drill: DNF all of 30B trajectories 0..4 -------------------
        # Every 30B factor then holds exactly 5 EEs (< 6), the whole model is
        # unscoreable, and all three verdicts must be VOID - never guessed from the
        # 7B side alone. The design is still complete (DNF rows are present), so no
        # refusal fires.
        rows_out = []
        for model, design in (("7B", d7), ("30B", d30)):
            for ti, t in enumerate(design):
                for run in t:
                    if model == "30B" and ti < 5:
                        rows_out.append(_fixture_csv_row(run, "dnf_oom", ""))
                    else:
                        rows_out.append(_fixture_csv_row(
                            run, "ok", repr(synth_y(model, run["grid"]))))
        void_csv = os.path.join(tmp, "void.csv")
        _write_fixture_csv(void_csv, rows_out)
        res = score(csv_path=void_csv, out_path=os.path.join(tmp, "verdict_void.json"),
                    chatty=False)
        ck(all(res["verdicts"][p]["verdict"] == "VOID" for p in ("P-1", "P-2", "P-4")),
           "validity floor: an unscoreable model voids all three verdicts")
        ck(res["models"]["30B"]["unscoreable"] == list(FACTORS["30B"])
           and all(res["models"]["30B"]["factors"][f]["n_ee"] == 5
                   for f in FACTORS["30B"]),
           "validity floor: every 30B factor sits at 5 valid EEs and is unscoreable")
        ck(res.get("void_because") == {"30B": list(FACTORS["30B"])},
           "validity floor: void_because names the model and its factors")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if ck.failures:
        raise SystemExit("self-check FAILED (%d of %d checks):\n  %s"
                         % (len(ck.failures), ck.n, "\n  ".join(ck.failures)))
    print(f"self-check OK: {ck.n} checks passed (pinned strings, seeded design, "
          "hand-derived fixture, refusal drills, synthetic night, validity floor)")


def main(argv):
    args = argv[1:]
    if args == ["--self-check"]:
        self_check()
        return 0
    if args:
        raise SystemExit("usage: python weights/prereg95_score.py [--self-check]")
    score()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
