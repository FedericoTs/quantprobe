"""exp54_binding_constraint.py -- pre-registration #88 (task/experiment #54).

Scores the "name WHICH resource binds" change in quantprobe/plan.py against the kill rules staked
in preregistrations/2026-07-30-binding-constraint.md BEFORE the classifier existed.

WHAT THIS MEASURES.  Nothing on the GPU.  Every number here is deterministic arithmetic over the
shipped preset tables, so C-14 (one cal_id per comparison) does not bind and no machine state is
involved.  What is being tested is (a) whether the classification carries any INFORMATION on the
hardware we actually ship presets for, (b) whether it is SOUND (does it move when a resource the
row does not use is changed?), (c) whether it CHANGES any advice, and (d) whether it agrees with
the one external ground truth we have -- BigMoeOnEdge's laptop, which they measured to be
DRAM-bandwidth-bound with I/O lanes, threads and a faster NVMe all neutral.

THE GRID IS UNSELECTED.  Every shipped model preset x every shipped machine preset, at the tool's
own default bit-width, at ctx 0 and ctx 16384.  340 cells, none chosen by hand.  A hand-picked
grid spanning four regimes would prove only that four cells can be picked.

THE REGRESSION BASELINE IS THE PREVIOUS COMMIT.  `git show <ref>:quantprobe/plan.py` is loaded as
a second module and the identical grid is run through it.  The classifier is a REPORTING layer:
if any predicted tok/s moves, kill rule K-3 fires and the change is reverted.  That check is why
this script needs a git repo and refuses to run without one.

Outputs (all under weights/data/):
  exp54_binding_constraint.json   machine-readable results, every cell
  exp54_binding_constraint.log    human-readable transcript
  exp54_plan_head.py.txt          the extracted baseline module (cached; re-runs are offline)

Idempotent: re-running overwrites the same three files and produces the same verdict.
Exit codes: 0 = PASS, 1 = a kill rule fired, 2 = precondition missing (no number is produced),
3 = INCOMPLETE (a kill rule could not be evaluated; the run is not a pass and must not be cited
    as one).

Run:
  python weights/exp54_binding_constraint.py --baseline-ref <pre-change commit>

ADVERSARIAL-REVIEW ADDENDUM (2026-07-30, after the first scoring run).  An adversarial pass over
this script found four ways it could not fail, all fixed here and all disclosed in the prereg's
section 8:

  1. K-3 auto-passed on any re-run made AFTER the change was committed.  `k3_vacuous` was folded
     into the verdict with `k3 or k3_vacuous`, so the moment HEAD contained `binding_constraint`
     the regression gate silently became a no-op and the script still printed PASS.  A vacuous
     kill rule is now a REFUSAL (exit 2), or -- with an explicit flag -- an INCOMPLETE verdict
     (exit 3).  It can never read as PASS again.
  2. K-4 was a unit test on ONE hand-built row whose `placement` string was the exact literal
     `binding_report` tests for.  It could not fire from the 340-cell sweep no matter what the
     sweep contained.  It is now scored on every all-in-VRAM WINNER in the grid (137 of them at
     2.5 bits), and a run in which zero such winners were checked fails rather than passes.
  3. P-5 claimed the staked numbers were "reproduced by the shipped code" while four of the seven
     were recomputed by this script from formulas copy-pasted out of plan.py -- i.e. the test was
     partly checking itself.  `hot`, `streamable` and `miss` are now INVERTED OUT OF the shipped
     disk row's own terms, and eta_r is imported (plan.ETA_R_MOE) rather than copied.
  4. P-3c counted 35 cells as "over-sold speculation advice" when the shipped speculation block
     never prints on 18 of them.  The population is now gated by `speculation_advice()`, the same
     gate run() uses (17 cells survive; the prediction still passes).

The review also established something no fix can repair, and it is recorded rather than patched:
K-1's 85% bar was never live.  A bit-width sweep (2.0-8.0) is now run and printed for exactly
that reason -- see the P-1 headroom block.

SECOND ADVERSARIAL PASS (2026-07-30, BEFORE this run was published).  A pre-run audit whose only
brief was "construct the input that makes this experiment fail" found five more, all fixed here
and disclosed in the prereg's section 8.5:

  5. K-2's SECOND STAKED CLAUSE WAS NEVER SCORED.  Prereg section 5 K-2 reads "Also fires if any
     row's terms fail to reconstruct its own tok/s to 1e-9", and section 1 R1 calls that
     "smoke-tested, and K-2".  This script never checked it, and the smoke test that does
     (t_p88_terms_reconstruct_every_row) runs a HAND-PICKED six-config grid which produces six of
     the seven row families - it never emits the DENSE SPLIT row, the one with the most complex
     decomposition (three resources, KV split across two tiers) and a WINNER on 8 of the 340
     cells.  Verified by fault injection: dropping the CPU-tier KV term from that row's `ram_bw`
     attribution leaves tok/s bit-identical (K-3 PASS), leaves the phantom and monotonicity arms
     clean (K-2 PASS), and this script printed VERDICT: PASS / exit 0 while the row's terms
     reconstructed 4.94% off its own speed and the PRINTED binding share and ceiling moved
     (64.5% -> 67.7%, 2.82x -> 3.09x).  Reconstruction is now checked on EVERY row of EVERY cell,
     the checks are counted, and an unexecuted pass cannot masquerade as a clean one.
  6. AN ABORT LEFT THE PREVIOUS RUN'S RESULT ON DISK AS THE APPARENT CURRENT ONE.  `die()` exits
     before anything is written, so after a refusal `exp54_binding_constraint.json` still said
     whatever the last completed run said - PASS included - with nothing to distinguish "scored
     and passed" from "refused to run".  The docstring's "idempotent: re-running overwrites the
     same three files" was false on exactly the path where it mattered.  `die()` now overwrites
     both outputs with a REFUSED record first.
  7. K-3 SILENTLY SKIPPED ANY CELL WHOSE ROW LIST WENT EMPTY.  `if res is None: continue` returned
     before the baseline was consulted, so a change that deleted every row for a cell - the
     largest possible regression - was recorded as `klass: null` and never compared.  The baseline
     is now consulted first and a vanished row list is a regression.
  8. A FIRED KILL RULE COULD BE REPORTED AS "INCOMPLETE" RATHER THAN "FAIL".  With a vacuous
     baseline the verdict branch tested `k3_vacuous` FIRST, so a run in which K-1 and K-5 both
     fired exited 3 ("a kill rule could not be evaluated") instead of 1.  Verified by injection.
     FAIL now takes precedence: INCOMPLETE is reserved for a run where everything scored held.
  9. K-3's GRID NEVER EXERCISED THE PINNED-FILE-SIZE OR CODEBOOK PATHS.  `cell_kwargs` fixes
     `true_size_gb=None` and `codebook_share=0.0`, but every one of the 14 locked ladder rows is a
     `--gguf` run, i.e. a real file size pinned (the exact input whose absence this file's own
     comment says "wrongly evicted the all-in-VRAM row") and, for the IQ files, a non-zero
     codebook share.  A regression that only bites when a size is pinned was invisible to K-3.  A
     second deterministic regression arm now replays the identical 340 preset pairs with a pinned
     size and a codebook share, needing no GGUF and no machine state.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "weights", "data")
PREREG = os.path.join(REPO, "preregistrations", "2026-07-30-binding-constraint.md")
OUT_JSON = os.path.join(DATA, "exp54_binding_constraint.json")
OUT_LOG = os.path.join(DATA, "exp54_binding_constraint.log")
HEAD_COPY = os.path.join(DATA, "exp54_plan_head.py.txt")

CTX_GRID = (0, 16384)
BITS = 2.5                       # the tool's own default when --bits is omitted
CLASSES = ("bandwidth-bound", "IO-bound", "capacity-bound", "compute-bound")

# K-2's SECOND clause, staked in prereg section 5 ("Also fires if any row's terms fail to
# reconstruct its own tok/s to 1e-9") and in section 1 R1. Not a new threshold: it is the prereg's
# own number, in the prereg's own words, and it was simply never scored here until the second
# adversarial pass. Applied to EVERY row of EVERY cell, not to the winner only - a mis-attributed
# term on a losing row becomes a wrong printed share the moment that row wins on another machine.
RECON_REL_TOL = 1e-9

# K-3's second arm (second adversarial pass, item 9). The staked 340-cell grid pins
# true_size_gb=None / codebook_share=0.0, so the `--gguf` code paths - which is what every one of
# the 14 locked ladder rows uses - were never compared against the baseline at all. These two
# values replay the same 340 preset pairs through those paths. 0.8 puts the pinned size BELOW the
# bit-derived estimate, which is the prereg-#16 direction (a dense model over-estimated by 4.5/bits
# and its all-in-VRAM row wrongly evicted); 0.5 is the same codebook share P-3b already uses.
LOCKED_PATH_SIZE_SCALE = 0.8
LOCKED_PATH_CODEBOOK_SHARE = 0.5

# Reported, NOT scored. K-1's staked bar is "no class covers >85% of the grid". The adversarial
# review asked the only question that matters about a threshold: was it ever close to firing?
# This sweep answers it in the output rather than leaving a reader to assume. No threshold is
# retuned from it - retuning after the fact is the exact back-fit the protocol forbids - and the
# successor bar is staked FORWARD in the prereg's section 8.
P1_HEADROOM_BITS = (2.0, 2.5, 3.5, 4.5, 6.0, 8.0)

# Every machine preset must supply these outright. `cell_kwargs` claims to rebuild exactly what
# run() computes; run() reaches its `or` fallbacks only when a USER omitted a flag, so a preset
# that leans on one would be priced as a 16 GB / 40 GB/s box without saying so. Checked, not
# assumed - a silently defaulted preset is the "streaming artifact published as an MLA
# measurement" failure in a cheaper costume.
REQUIRED_MACHINE_KEYS = ("vc", "vb", "rc", "rb", "db", "geta")
# Set only by `quantprobe calibrate`. If one ever appears in a preset, run() would route through
# resolve_gpu_eta/resolve_cpu_bw and cell_kwargs would NOT reproduce it.
FORBIDDEN_MACHINE_KEYS = ("_gpu_ratio", "_cpu_ratio", "_cal_active", "_anchor_gpu_act",
                          "_anchor_cpu_act")

_log_lines = []
_quiet = False           # --json: the prereg section 2 documents it; it used to do nothing at all


def say(msg=""):
    if not _quiet:
        print(msg)
    _log_lines.append(msg)          # the .log transcript is written either way


def die(msg):
    """Refuse to run rather than produce a wrong number. A wrong number is worse than none.

    ADVERSARIAL FIX (second pass, item 6). `die()` used to exit before anything was written, and
    the JSON and the log are only written at the END of main(). So after a refusal the two output
    files still held the PREVIOUS run's result - `"verdict": "PASS"` included - with nothing on
    disk or in the file to distinguish "this tree was scored and passed" from "this tree was never
    scored at all". Every consumer of this experiment reads that JSON. A refusal must therefore
    INVALIDATE the outputs, not leave them: the stale-result-after-abort failure is the same shape
    as D-1 (a gate reporting a fact about nothing), one file further downstream.
    """
    print(f"\nPRECONDITION NOT MET -- refusing to run.\n  {msg}\n", file=sys.stderr)
    try:
        os.makedirs(DATA, exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as fh:
            json.dump(dict(prereg=os.path.relpath(PREREG, REPO), experiment=54, prereg_id=88,
                           verdict="REFUSED", refusal=msg,
                           kill_rules=dict(K1=None, K2=None, K3=None, K4=None, K5=None),
                           note="Precondition not met; NOTHING was scored. This file replaced a "
                                "previous result so that it cannot be read as the current one."),
                      fh, indent=1)
        with open(OUT_LOG, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("REFUSED -- precondition not met, nothing was scored.\n  " + msg + "\n")
    except Exception as e:                                     # pragma: no cover - disk problem
        print(f"  (could not invalidate the previous result on disk: {e})", file=sys.stderr)
    sys.exit(2)


# ----------------------------------------------------------------------------------------------
# Preconditions and the staked numbers
# ----------------------------------------------------------------------------------------------

def read_stake():
    """Re-read the staked constants OUT OF the prereg, so this script cannot drift from it."""
    if not os.path.isfile(PREREG):
        die(f"the pre-registration is missing: {PREREG}\n  "
            "Predictions are staked in that file; without it there is nothing to score.")
    text = open(PREREG, encoding="utf-8").read()
    m = re.search(r"```stake\n(.*?)```", text, re.S)
    if not m:
        die(f"{PREREG} has no ```stake block -- the staked numbers cannot be recovered.")
    stake = {}
    for line in m.group(1).splitlines():
        if "=" not in line:
            continue
        k, v = (p.strip() for p in line.split("=", 1))
        try:
            stake[k] = float(v)
        except ValueError:
            stake[k] = v
    need = ["grid_models", "grid_machines", "grid_cells", "p1_min_distinct_classes",
            "p1_max_modal_share", "p4_min_frac_ceiling_lt_2", "p5_hot_gb", "p5_streamable_gb",
            "p5_miss_uniform", "p5_io_s_uniform", "p5_ram_s_uniform", "p5_class_uniform",
            "p5_miss_measured", "p5_io_s_measured", "p5_ram_s_measured", "p5_class_measured",
            "p5_their_compute_floor_s", "p5_kill_rel_error", "cap_promotion_min",
            "cap_shave_max_share", "regression_rel_tol"]
    missing = [k for k in need if k not in stake]
    if missing:
        die(f"the ```stake block is incomplete: missing {missing}")
    return stake


def load_plan():
    sys.path.insert(0, REPO)
    try:
        from quantprobe import plan as planmod
    except Exception as e:                                     # pragma: no cover - env problem
        die(f"cannot import quantprobe.plan from {REPO}: {e}")
    for fn in ("binding_constraint", "capacity_probe", "spec_ceiling", "codebook_bounded_gain",
               "upgrade_advisor", "binding_report", "Row", "resource_times"):
        if not hasattr(planmod, fn):
            die(f"quantprobe.plan has no `{fn}`.\n  "
                "The plan.py change this experiment scores is NOT applied to this working tree, "
                "so there is nothing to measure. Apply it, then re-run.")
    return planmod


def load_baseline(ref):
    """The pre-change plan.py, as a standalone module. K-3 has no meaning without it."""
    try:
        src = subprocess.run(["git", "show", f"{ref}:quantprobe/plan.py"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
    except Exception as e:
        die(f"cannot read quantprobe/plan.py at git ref '{ref}': {e}\n  "
            "K-3 compares every predicted number against the pre-change version; without the "
            "baseline this run could report 'nothing moved' while everything moved.")
    if not src.strip():
        die(f"git show {ref}:quantprobe/plan.py returned nothing.")
    os.makedirs(DATA, exist_ok=True)
    with open(HEAD_COPY, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    loader = importlib.machinery.SourceFileLoader("exp54_plan_baseline", HEAD_COPY)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    try:
        loader.exec_module(mod)
    except Exception as e:
        die(f"the baseline plan.py at {ref} does not import standalone: {e}")
    vacuous = hasattr(mod, "binding_constraint")
    return mod, vacuous


def machine_preset_audit(planmod):
    """Refuse the sweep if a preset would be priced by a DEFAULT instead of by its own numbers.

    `cell_kwargs` asserts it rebuilds exactly the arguments run() computes. run() reaches its
    `or` fallbacks (rc or 16, rb or 40, db 0.5) only when a USER omitted a flag - a machine PRESET
    that leans on one would be silently sized as a 16 GB / 40 GB-per-second box and would then
    carry a binding-constraint label derived from hardware that does not exist. Likewise a preset
    carrying a calibration key would route run() through resolve_gpu_eta / resolve_cpu_bw, which
    cell_kwargs does not call, so the sweep would classify a different machine than the tool does.

    Checked rather than assumed, because this is precisely the class of defect that once let a
    profiler run on a model too big for the card produce a streaming artifact that was nearly
    published as an MLA measurement.
    """
    bad = []
    for name, hw in planmod.MACHINES.items():
        for k in REQUIRED_MACHINE_KEYS:
            if k not in hw:
                bad.append(f"{name}: '{k}' absent - run() would substitute its own default")
            # vc/vb are legitimately 0 on a CPU-only preset (run(): `agg_cap(vc) or 0`).
            # rc/rb/db/geta are NOT: run() replaces a falsy one with 16 / 40 / 0.5 / 0.45.
            elif k not in ("vc", "vb") and not hw.get(k):
                bad.append(f"{name}: '{k}' is {hw.get(k)!r} - run()'s fallback would supply it")
        for k in FORBIDDEN_MACHINE_KEYS:
            if k in hw:
                bad.append(f"{name}: carries calibration key '{k}' - run() would route through "
                           f"resolve_gpu_eta/resolve_cpu_bw and cell_kwargs does not")
    if bad:
        die("the preset tables no longer reconstruct without a default:\n    "
            + "\n    ".join(bad)
            + "\n  Fix the preset or the reconstruction; do NOT let the sweep guess.")


# ----------------------------------------------------------------------------------------------
# The grid
# ----------------------------------------------------------------------------------------------

def cell_kwargs(planmod, model_key, machine_key, ctx):
    """Rebuild EXACTLY the argument set plan.run() would compute for this preset pair."""
    m = planmod.MODELS[model_key]
    hw = planmod.MACHINES[machine_key]
    t = m["t"]
    a = m["a"]
    ne = m["ne"]
    vc = planmod.agg_cap(hw.get("vc")) or 0
    vb = planmod.agg_bw(hw.get("vb"), 0.85) or 0
    rc = hw.get("rc") or 16
    rb = hw.get("rb") or 40
    db = planmod.agg_bw(hw.get("db"), 0.75) or 0.5
    return dict(t=t, a=a, ne=ne, moe=m["moe"], bits=BITS, vc=vc, vb=vb, rc=rc, rb=rb, db=db,
                geta=hw.get("geta", 0.45), gl=hw.get("gl"), ctx=ctx,
                kvp=m.get("kvp", planmod.DEFAULT_KVP), n_layer=m.get("nl"),
                true_size_gb=None, codebook_share=0.0)


def make_ev(planmod, kw):
    """run()'s own `ev` closure, reproduced. Overrides must behave identically or every
    counterfactual (upgrade advisor, capacity probe) measures a different machine than the tool."""
    def ev(**over):
        k = dict(kw)
        if "rc_delta" in over:
            k["rc"] = k["rc"] + over.pop("rc_delta")
        if "true_size_gb_scale" in over:
            s = over.pop("true_size_gb_scale")
            # run() scales `true_size_gb` by `s` when a real file is pinned. A preset sweep pins
            # none, so dropping it is a no-op HERE and nowhere else. Assert that rather than
            # popping it silently: the day a preset carries a size, this dropped scale would turn
            # the capacity shave probe into a measurement of nothing while still reporting a gain.
            if k.get("true_size_gb"):
                k["true_size_gb"] = k["true_size_gb"] * s
        k.update(over)
        return planmod.evaluate(**k)[2]
    return ev


def check_reconstruction(planmod, cell, rows):
    """K-2, second staked clause: every row's terms must reproduce ITS OWN tok/s to 1e-9.

    ADVERSARIAL FIX (second pass, item 5). This clause is staked twice in the prereg - section 1
    R1 ("a row whose terms do not reconstruct its own tok/s to 1e-9 is a bug (smoke-tested, and
    K-2)") and section 5 K-2 ("Also fires if any row's terms fail to reconstruct its own tok/s to
    1e-9") - and it was scored NOWHERE. The smoke suite's version runs a hand-picked six-config
    grid that emits six of the seven row families and never the dense-split row, which is the row
    with the most complex decomposition and a WINNER on 8 of these 340 cells.

    The concrete input that exploited the gap: drop `(1-g)*kv_gb/(ETA_KV*rb)` from that row's
    `ram_bw` attribution. tok/s does not move (K-3 PASS), no phantom or monotonicity check bites
    (K-2 PASS), and the run printed VERDICT: PASS at exit 0 - while the row reconstructed 4.94%
    off its own speed and the PRINTED share and ceiling moved from 64.5%/2.82x to 67.7%/3.09x.

    Scored on EVERY row, not the winner only: a mis-attributed term on a losing row here is a
    wrong printed number on the machine where that row wins. Returns a list of failures.
    """
    bad = []
    for row in rows:
        tt = planmod.resource_times(row)
        if not tt:
            bad.append(dict(cell=cell, row=row[0], why="row carries no decomposition at all"))
            continue
        eff = getattr(row, "eff", None)
        if eff is None:
            bad.append(dict(cell=cell, row=row[0], why="row carries terms but no `eff` numerator"))
            continue
        total = sum(tt.values())
        if total <= 0 or not row[1]:
            bad.append(dict(cell=cell, row=row[0], why=f"degenerate: total={total}, tok_s={row[1]}"))
            continue
        recon = eff / total
        rel = abs(recon / row[1] - 1)
        if rel > RECON_REL_TOL:
            bad.append(dict(cell=cell, row=row[0], tok_s=row[1], reconstructed=recon, rel=rel))
    return bad


def classify_cell(planmod, kw):
    size, act, rows = planmod.evaluate(**kw)
    if not rows:
        return None
    best = rows[0]
    kv_gb = kw["ctx"] * kw["kvp"] / 1e9 if kw["ctx"] > 0 else 0.0
    cap = planmod.capacity_probe(make_ev(planmod, kw), best[1], size, kv_gb, kw["vc"], kw["rc"])
    bc = planmod.binding_constraint(best, capacity=cap)
    return dict(size_gb=size, act_gb=act, rows=rows, best=best, bc=bc, cap=cap, kv_gb=kv_gb)


# ----------------------------------------------------------------------------------------------
# The old upgrade advisor, reproduced verbatim, so P-3a compares like with like
# ----------------------------------------------------------------------------------------------

def old_upgrade_lines(basemod, kw, best_tps):
    """The three shipped call sites, argument for argument -- INCLUDING what they omitted.

    `n_layer` and `true_size_gb` are absent here because they were absent in the shipped code.
    That is the defect: the counterfactual was drawn from a smaller row menu and, at ctx>0, from
    a 32-layer CPU-attention term regardless of the model's real depth.
    """
    alts = []
    common = dict(gl=kw["gl"], ctx=kw["ctx"], kvp=kw["kvp"], codebook_share=kw["codebook_share"])
    if kw["rb"] < 40:
        c2 = basemod.evaluate(kw["t"], kw["a"], kw["ne"], kw["moe"], kw["bits"], kw["vc"],
                              kw["vb"], kw["rc"], 48, kw["db"], kw["geta"], **common)[2]
        if c2 and c2[0][1] > best_tps * 1.08:
            alts.append(("enable XMP (free)", round(c2[0][1], 6)))
    c2 = basemod.evaluate(kw["t"], kw["a"], kw["ne"], kw["moe"], kw["bits"], kw["vc"], kw["vb"],
                          kw["rc"] + 16, kw["rb"], kw["db"], kw["geta"], **common)[2]
    if c2 and c2[0][1] > best_tps * 1.08:
        alts.append(("+16 GB RAM", round(c2[0][1], 6)))
    c2 = basemod.evaluate(kw["t"], kw["a"], kw["ne"], kw["moe"], kw["bits"], kw["vc"], kw["vb"],
                          kw["rc"], kw["rb"], 3.5, kw["geta"], **common)[2]
    if c2 and c2[0][1] > best_tps * 1.08:
        alts.append(("NVMe SSD", round(c2[0][1], 6)))
    return alts


def new_upgrade_lines(planmod, kw, best_tps):
    return [(u["name"], round(u["tps"], 6))
            for u in planmod.upgrade_advisor(make_ev(planmod, kw), best_tps, kw["rb"])]


# ----------------------------------------------------------------------------------------------
# P-5: the external arm (BigMoeOnEdge laptop)
# ----------------------------------------------------------------------------------------------

def external_arm(planmod, stake):
    """Their laptop, their file, our shipped arithmetic.

    C1 is computed by the SHIPPED evaluate: their box as a custom machine, their file's params
    from prereg #86 section 2. C2 substitutes their MEASURED hot-expert cache hit rate (0.80) for
    our capacity-only miss -- that substitution is done HERE, in the script, because the tool
    cannot express it. Disclosed, not hidden: it is the entire distance between the two arms and
    it is exactly the open item tasks #52/#55 exist to close.

    ADVERSARIAL FIX. This function used to recompute `hot`, `streamable` and `miss` from formulas
    copy-pasted out of plan.py, and hard-coded eta_r = 0.38. Four of the seven quantities P-5
    scores were therefore produced by THIS FILE, and the prereg's claim that they were "reproduced
    by the shipped code" was false for them: a sign error inside evaluate's disk row would have
    left every one of them matching its staked value. They are now INVERTED OUT OF the shipped
    row's own `terms`, so if evaluate's arithmetic moves, P-5 breaks -- which is the only reason
    to run it. eta_r is imported from plan.ETA_R_MOE for the same reason.

        io_s   = streamable * miss / db                       -> streamable = io_s * db / miss
        ram_s  = (streamable*(1-miss) + hot) / (eta_r*rb)     -> hot = ram_s*eta_r*rb - ...

    `miss` still comes from the capacity formula, but it is CROSS-CHECKED against the inversion
    below; a mismatch aborts rather than being reported.
    """
    kw = dict(t=35.5053, a=3.0109, ne=1.9791, moe=True, bits=5.02,
              vc=0, vb=0, rc=16, rb=51.0, db=3.0, geta=0.45, gl=None,
              ctx=0, kvp=0.0, n_layer=41, true_size_gb=22.285, codebook_share=0.0)
    size, act, rows = planmod.evaluate(**kw)
    disk = [r for r in rows if r[0].startswith("stream from disk (cold")]
    if not disk:
        die("the disk-streaming row did not fire on the external arm -- the fixture no longer "
            "reproduces prereg #88 section 3 and the comparison would be meaningless.")
    c1_terms = planmod.resource_times(disk[0])
    bc1 = planmod.binding_constraint(disk[0])
    eta_r = getattr(planmod, "ETA_R_MOE", None)
    if not eta_r:
        die("quantprobe.plan has no ETA_R_MOE. P-5 must import the MoE RAM-tier efficiency, not "
            "copy it: a copied constant keeps reproducing the staked number after the shipped one "
            "moves, and P-5 would report agreement with an arithmetic the tool no longer uses.")

    io_s1 = c1_terms.get("io", 0.0)
    ram_s1 = c1_terms.get("ram_bw", 0.0)
    if io_s1 <= 0 or ram_s1 <= 0:
        die("the external arm's disk row carries no io/ram_bw decomposition -- there is nothing "
            "to invert and P-5 cannot be scored against the shipped code.")
    # miss from the capacity formula, then INVERTED back out of the shipped terms as a check.
    miss1 = 1 - (max(kw["rc"] - 4, 1) * 0.9) / size
    streamable = io_s1 * kw["db"] / miss1
    hot = ram_s1 * eta_r * kw["rb"] - streamable * (1 - miss1)
    # Independent reconstruction: streamable + hot must be evaluate's own `act` to 1e-9. If it is
    # not, the inversion assumed a row shape evaluate no longer has and every P-5 number below
    # would be arithmetic about a fiction.
    if abs((streamable + hot) / act - 1) > 1e-9:
        die(f"the inversion of the shipped disk row does not reconstruct its own active bytes: "
            f"hot+streamable = {streamable + hot:.9f} GB vs act = {act:.9f} GB. evaluate's disk "
            f"row no longer has the shape prereg #88 section 3 inverts; P-5 is not scoreable.")

    miss2 = stake["p5_miss_measured"]
    io2 = streamable * miss2 / kw["db"]
    ram2 = (streamable * (1 - miss2) + hot) / (eta_r * kw["rb"])
    row2 = planmod.Row("external C2 (their measured 80% hot-expert hit)",
                       0.95 / (io2 + ram2), None, "-", {"io": io2, "ram_bw": ram2}, eff=0.95)
    bc2 = planmod.binding_constraint(row2)
    return dict(hot_gb=hot, streamable_gb=streamable, act_gb=act, eta_r=eta_r,
                derived_from="inverted out of the shipped disk row's own terms",
                c1=dict(io_s=io_s1, ram_s=ram_s1, miss=miss1,
                        klass=bc1["klass"], share=bc1["share"], margin=bc1["margin_x"],
                        tok_s=disk[0][1]),
                c2=dict(io_s=io2, ram_s=ram2, miss=miss2, klass=bc2["klass"],
                        share=bc2["share"], margin=bc2["margin_x"], tok_s=row2[1]))


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-ref", default="HEAD",
                    help="git ref holding the PRE-change quantprobe/plan.py (default HEAD)")
    ap.add_argument("--allow-grid-drift", action="store_true",
                    help="score even if the preset tables changed since the prereg was staked")
    ap.add_argument("--allow-vacuous-regression", action="store_true",
                    help="proceed when the baseline ref ALREADY contains the change. K-3 is then "
                         "UNCHECKED and the verdict is INCOMPLETE (exit 3), never PASS.")
    ap.add_argument("--json", action="store_true", help="write the JSON only, no transcript")
    args = ap.parse_args()
    if args.json:
        globals()["_quiet"] = True

    stake = read_stake()
    planmod = load_plan()
    machine_preset_audit(planmod)
    basemod, k3_vacuous = load_baseline(args.baseline_ref)
    if k3_vacuous and not args.allow_vacuous_regression:
        # THE defect this review was called to find. K-3 used to be folded into the verdict as
        # `k3 or k3_vacuous`, so the instant the change was committed, `git show HEAD:` handed
        # back a baseline that ALREADY contained the classifier, every cell compared the module
        # to itself, zero regressions were found by construction, and the script printed PASS.
        # A kill rule that turns itself off the moment the work lands is not a kill rule.
        die(f"the baseline ref '{args.baseline_ref}' ALREADY contains `binding_constraint`, so "
            f"K-3 would compare the change against itself.\n  Every cell would match by "
            f"construction and the regression gate would report PASS having checked nothing.\n  "
            f"Pass --baseline-ref <the commit BEFORE this change> (e.g. the parent of the commit "
            f"that added binding_constraint:\n    git log --oneline -S'def binding_constraint' "
            f"-- quantprobe/plan.py\n  ), or pass --allow-vacuous-regression to score the other "
            f"four kill rules with K-3 recorded as UNCHECKED and the verdict INCOMPLETE.")

    models = list(planmod.MODELS)
    machines = list(planmod.MACHINES)
    n_cells = len(models) * len(machines) * len(CTX_GRID)
    if (len(models), len(machines)) != (int(stake["grid_models"]), int(stake["grid_machines"])):
        msg = (f"the preset grid moved since staking: {len(models)}x{len(machines)} now vs "
               f"{int(stake['grid_models'])}x{int(stake['grid_machines'])} staked. "
               f"P-1's denominator is no longer the one the prediction was made against.")
        if not args.allow_grid_drift:
            die(msg + "\n  Re-stake, or pass --allow-grid-drift and disclose it.")
        say(f"[grid drift ACCEPTED by flag] {msg}")

    say("=" * 96)
    say("experiment #54 / pre-registration #88 -- WHICH RESOURCE BINDS")
    say(f"grid: {len(models)} models x {len(machines)} machines x ctx{list(CTX_GRID)} "
        f"= {n_cells} cells @ {BITS:g} bits (unselected)")
    say(f"regression baseline: git {args.baseline_ref}:quantprobe/plan.py")
    say("=" * 96)

    cells = []
    regressions = []
    regressions_locked = []      # K-3's second arm: the pinned-size / codebook paths (item 9)
    upgrade_changes = []
    phantom_failures = []
    monotonic_failures = []
    recon_failures = []          # K-2's second staked clause (item 5)
    # A soundness check that never ran reports "0 failures" and looks identical to a pass. Count
    # the checks so a vacuous P-2 cannot be mistaken for a real one.
    checks = dict(phantom=0, monotonic=0, recon=0, locked_path=0)
    vram_winners = []            # K-4's real population: every all-in-VRAM WINNER on the grid

    def compare_winner(cell, kwargs, bucket):
        """One K-3 comparison: the new module's winner against the baseline's, on the same args.

        Returns the new module's rows so callers can reuse them. A row list that went EMPTY on
        either side is a regression, not a cell to skip - see item 7.
        """
        n_rows = planmod.evaluate(**kwargs)[2]
        b_rows = basemod.evaluate(**kwargs)[2]
        if not n_rows or not b_rows:
            if n_rows or b_rows:
                bucket.append(dict(cell=cell, why=f"row list vanished on one side: "
                                                  f"baseline {len(b_rows)} rows, now {len(n_rows)}"))
            return n_rows
        n, b = n_rows[0], b_rows[0]
        rel = abs(b[1] - n[1]) / b[1] if b[1] else 0.0
        if b[0] != n[0] or b[3] != n[3] or rel > stake["regression_rel_tol"]:
            bucket.append(dict(cell=cell, head=[b[0], b[1], b[3]], now=[n[0], n[1], n[3]], rel=rel))
        return n_rows

    for ctx in CTX_GRID:
        for mk in models:
            for hk in machines:
                cell = f"{mk}/{hk}/ctx{ctx}"
                kw = cell_kwargs(planmod, mk, hk, ctx)
                res = classify_cell(planmod, kw)

                # ---- K-3: the numbers must not have moved -------------------------------------
                # Run BEFORE the `res is None` bail-out. It used to run after, so a change that
                # deleted every row for a cell - the largest regression there is - was filed as
                # `klass: null` and never compared against the baseline at all (item 7).
                n_rows = compare_winner(cell, kw, regressions)

                # ---- K-2, second staked clause: terms must reconstruct their own tok/s ---------
                these = check_reconstruction(planmod, cell, n_rows)
                checks["recon"] += len(n_rows)
                recon_failures.extend(these)

                # ---- K-3, second arm: the `--gguf` code paths the staked grid never touched ----
                # Every one of the 14 locked ladder rows pins a real file size, and the IQ ones
                # carry a codebook share. `cell_kwargs` pins neither, so a regression that only
                # bites on those paths was invisible to the staked arm. Deterministic: no GGUF,
                # no machine state, same 340 preset pairs (item 9).
                if res is not None:
                    locked_kw = dict(kw,
                                     true_size_gb=res["size_gb"] * LOCKED_PATH_SIZE_SCALE,
                                     codebook_share=LOCKED_PATH_CODEBOOK_SHARE)
                    l_rows = compare_winner(cell + "/pinned-size+codebook", locked_kw,
                                            regressions_locked)
                    # Same reconstruction clause on those rows: the codebook share divides eta_r,
                    # so every RAM term is a DIFFERENT number here, and an attribution that only
                    # adds up at codebook_share = 0 would otherwise never be caught.
                    recon_failures.extend(check_reconstruction(
                        planmod, cell + "/pinned-size+codebook", l_rows))
                    checks["recon"] += len(l_rows)
                    checks["locked_path"] += 1

                if res is None:
                    cells.append(dict(model=mk, machine=hk, ctx=ctx, rows=0, klass=None))
                    continue
                bc, best = res["bc"], res["best"]

                # ---- P-2a: no phantom sensitivity to a resource the row does not use ----------
                if "io" not in (planmod.resource_times(best) or {}):
                    alt = classify_cell(planmod, dict(kw, db=5.0 if kw["db"] != 5.0 else 0.45))
                    if alt and alt["best"][0] == best[0]:
                        a_bc = alt["bc"]
                        checks["phantom"] += 1
                        same = (a_bc["resource"] == bc["resource"] and a_bc["klass"] == bc["klass"]
                                and abs(a_bc["share"] - bc["share"]) < 1e-12
                                and (a_bc["margin_x"] or 0) == (bc["margin_x"] or 0))
                        if not same:
                            phantom_failures.append(dict(cell=f"{mk}/{hk}/ctx{ctx}",
                                                         was=bc["klass"], now=a_bc["klass"]))

                # ---- P-2b: shares move the right way when a bandwidth doubles -----------------
                for res_name, over in (("ram_bw", dict(rb=kw["rb"] * 2)),
                                       ("vram_bw", dict(vb=kw["vb"] * 2))):
                    s0 = bc["shares"].get(res_name, 0.0)
                    if s0 <= 0 or len(bc["shares"]) == 1 or not over.get("vb", 1):
                        continue
                    alt = classify_cell(planmod, dict(kw, **over))
                    if not alt or alt["best"][0] != best[0]:
                        continue                       # a different placement won: not comparable
                    s1 = alt["bc"]["shares"].get(res_name, 0.0)
                    checks["monotonic"] += 1
                    if not s1 < s0:
                        monotonic_failures.append(dict(cell=f"{mk}/{hk}/ctx{ctx}",
                                                       resource=res_name, s0=s0, s1=s1))

                # ---- P-3a: does the upgrade advice change once the inputs match? --------------
                old = old_upgrade_lines(basemod, kw, best[1])
                new = new_upgrade_lines(planmod, kw, best[1])
                if old != new:
                    upgrade_changes.append(dict(cell=f"{mk}/{hk}/ctx{ctx}", old=old, new=new))

                # ---- K-4's population: every all-in-VRAM winner, reported the way run() does ----
                if best[0] == "all in VRAM" or best[0].startswith("all in VRAM"):
                    vram_winners.append((f"{mk}/{hk}/ctx{ctx}", bc, best[0]))

                cells.append(dict(
                    model=mk, machine=hk, ctx=ctx, rows=len(res["rows"]),
                    winner=best[0], tok_s=best[1],
                    klass=bc["klass"], resource=bc["resource"], share=bc["share"],
                    margin_x=bc["margin_x"], ceiling_x=bc["ceiling_x"],
                    shares=bc["shares"],
                    spec_ceiling_k2=planmod.spec_ceiling(best, k=2),
                    # run() prints the speculation ceiling ONLY where speculation_advice() fires.
                    # Counting a cell the block never reaches as "over-sold advice" would be the
                    # same error the experiment exists to catch, one level up.
                    spec_advice_prints=bool(planmod.speculation_advice(kw["moe"], best[0])),
                    codebook_bound_at_50pct=planmod.codebook_bounded_gain(best, 0.5),
                    capacity=bool(res["cap"]),
                ))

    scored = [c for c in cells if c.get("klass")]
    counts = {k: sum(1 for c in scored if c["klass"] == k) for k in CLASSES}
    counts = {k: v for k, v in counts.items() if v}
    modal_share = (max(counts.values()) / len(scored)) if scored else 1.0
    distinct = len(counts)

    ceil_lt2 = [c for c in scored if c["ceiling_x"] is not None and c["ceiling_x"] < 2.0]
    frac_ceil_lt2 = len(ceil_lt2) / len(scored) if scored else 0.0

    # P-3b/P-3c probes: only rows that actually put weights on the CPU tier can be told to
    # re-download a file, and only rows the tool prints speculation advice for can be over-sold it.
    # BOTH populations mirror run()'s own gates exactly. P-3b's placement filter is run()'s
    # (`any(k in best[0] for k in (...))`) and the 0.5 codebook share is the SYNTHETIC arm the
    # prereg's P-3b names -- run() additionally requires --iq-share > 0.3, which no preset sets,
    # so these are counterfactual cells and are labelled as such in the report.
    cpu_tier = [c for c in scored
                if any(k in c["winner"] for k in ("->RAM", "CPU", "disk", "split experts"))]
    oversold_codebook = [c for c in cpu_tier
                         if c["codebook_bound_at_50pct"] and c["codebook_bound_at_50pct"] < 1.05]
    # ADVERSARIAL FIX: this used to count every scored cell. 18 of the 35 it counted were cells
    # where run()'s speculation block never prints at all, so they could not be "over-sold"
    # anything. Gated on the same `speculation_advice()` call run() makes.
    spec_shown = [c for c in scored if c["spec_advice_prints"]]
    oversold_spec = [c for c in spec_shown if c["spec_ceiling_k2"] and c["spec_ceiling_k2"] < 1.5]
    oversold_spec_ungated = [c for c in scored
                             if c["spec_ceiling_k2"] and c["spec_ceiling_k2"] < 1.5]

    ext = external_arm(planmod, stake)

    # P-1 headroom (reported, not scored). See P1_HEADROOM_BITS.
    headroom = []
    _bits_saved = globals()["BITS"]
    try:
        for b in P1_HEADROOM_BITS:
            globals()["BITS"] = b
            cc = {}
            for ctx in CTX_GRID:
                for mk in models:
                    for hk in machines:
                        r = classify_cell(planmod, cell_kwargs(planmod, mk, hk, ctx))
                        if r:
                            cc[r["bc"]["klass"]] = cc.get(r["bc"]["klass"], 0) + 1
            tot = sum(cc.values()) or 1
            headroom.append(dict(bits=b, distinct=len(cc), modal_share=max(cc.values()) / tot))
    finally:
        globals()["BITS"] = _bits_saved
    worst_modal = max((h["modal_share"] for h in headroom), default=modal_share)
    min_distinct = min((h["distinct"] for h in headroom), default=distinct)

    # ------------------------------------------------------------------------------------------
    # Verdicts
    # ------------------------------------------------------------------------------------------
    p1 = distinct >= int(stake["p1_min_distinct_classes"]) and modal_share <= stake["p1_max_modal_share"]
    # A soundness check with nothing to check is not a pass.
    # A soundness check with nothing to check is not a pass - and neither is a soundness check that
    # scores two of its three staked clauses. `recon_failures` is K-2's SECOND staked clause
    # ("Also fires if any row's terms fail to reconstruct its own tok/s to 1e-9"), which this
    # script did not score at all until the second adversarial pass (item 5).
    p2 = (not phantom_failures and not monotonic_failures and not recon_failures
          and checks["phantom"] > 0 and checks["monotonic"] > 0 and checks["recon"] > 0)
    p3a, p3b, p3c = bool(upgrade_changes), bool(oversold_codebook), bool(oversold_spec)
    p3 = p3a or p3b or p3c
    p4 = frac_ceil_lt2 >= stake["p4_min_frac_ceiling_lt_2"]

    tol = 2e-3
    def close(x, key):
        return abs(x / stake[key] - 1) < tol if stake[key] else abs(x) < tol
    p5_numbers = all([close(ext["hot_gb"], "p5_hot_gb"),
                      close(ext["streamable_gb"], "p5_streamable_gb"),
                      close(ext["c1"]["miss"], "p5_miss_uniform"),
                      close(ext["c1"]["io_s"], "p5_io_s_uniform"),
                      close(ext["c1"]["ram_s"], "p5_ram_s_uniform"),
                      close(ext["c2"]["io_s"], "p5_io_s_measured"),
                      close(ext["c2"]["ram_s"], "p5_ram_s_measured")])
    p5_floor = abs(ext["c2"]["ram_s"] / stake["p5_their_compute_floor_s"] - 1) < stake["p5_kill_rel_error"]
    p5 = (ext["c1"]["klass"] == stake["p5_class_uniform"]
          and ext["c2"]["klass"] == stake["p5_class_measured"]
          and p5_numbers and p5_floor)

    k1 = p1
    k2 = p2
    # K-3 covers BOTH regression arms: the staked 340 preset cells, and the same 340 pairs replayed
    # through the pinned-file-size / codebook paths every locked ladder row actually uses (item 9).
    # Adding an arm can only make this gate stricter; no threshold moved.
    k3 = not regressions and not regressions_locked
    k4, k4_n, k4_failures = check_k4(planmod, vram_winners)
    k5 = p5

    # ------------------------------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------------------------------
    say("")
    say(f"P-1  informativeness      distinct classes = {distinct} (>= {int(stake['p1_min_distinct_classes'])}), "
        f"modal share = {modal_share:.1%} (<= {stake['p1_max_modal_share']:.0%})   "
        f"{'PASS' if p1 else 'MISS'}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        say(f"       {k:<18} {v:>4} / {len(scored)}  ({v / len(scored):.1%})")
    say(f"P-2  soundness            phantom-sensitivity {len(phantom_failures)} failures / "
        f"{checks['phantom']} checks | monotonicity {len(monotonic_failures)} failures / "
        f"{checks['monotonic']} checks   {'PASS' if p2 else 'MISS'}")
    say(f"                          terms reconstruct their own tok/s to {RECON_REL_TOL:g}: "
        f"{len(recon_failures)} failures / {checks['recon']} rows")
    say(f"                              (K-2's SECOND staked clause, prereg section 5. It was "
        f"scored NOWHERE before the")
    say(f"                               second adversarial pass: the smoke suite's version runs a "
        f"hand-picked six-config")
    say(f"                               grid that never emits the dense-split row - a WINNER on 8 "
        f"of these 340 cells.")
    say(f"                               Fault injection: dropping that row's CPU-tier KV term left "
        f"tok/s bit-identical,")
    say(f"                               every other arm clean, and this script printed PASS at "
        f"exit 0 while the row")
    say(f"                               reconstructed 4.94% off and the PRINTED share/ceiling moved "
        f"64.5%->67.7%, 2.82x->3.09x.)")
    say(f"     P-1 HEADROOM (reported, NOT scored - the adversarial question about any threshold "
        f"is whether it")
    say(f"     was ever close to firing). Modal share over the same 340-cell grid at other "
        f"bit-widths:")
    say("       " + " | ".join(f"{h['bits']:g}b {h['modal_share']:.1%} (d{h['distinct']})"
                               for h in headroom))
    say(f"       Worst modal share anywhere in 2.0-8.0 bits: {worst_modal:.1%} against a staked "
        f"bar of {stake['p1_max_modal_share']:.0%};")
    say(f"       fewest distinct classes anywhere: {min_distinct} against a staked bar of "
        f"{int(stake['p1_min_distinct_classes'])}.")
    say(f"       READ K-1's PASS ACCORDINGLY: on this grid the bar is not live. K-1 could not have "
        f"fired at ANY")
    say(f"       bit-width we ship, so its PASS is weak evidence and is NOT cited as a confirmed "
        f"prediction.")
    say(f"       The bar is not retuned here - back-fitting a threshold after seeing the answer is "
        f"what the")
    say(f"       protocol forbids. A live successor bar is staked FORWARD in the prereg, section 8.")
    say(f"P-3  advice changes       (a) upgrade lines changed on {len(upgrade_changes)} cells   "
        f"{'PASS' if p3a else 'MISS'}")
    say(f"                          (b) codebook warning over-sold (<5%) on {len(oversold_codebook)} "
        f"CPU-tier cells   {'PASS' if p3b else 'MISS'}")
    say(f"                              (SYNTHETIC arm: run() also requires --iq-share > 0.3, which "
        f"no preset sets;")
    say(f"                               these are counterfactual cells, as prereg P-3b states.)")
    say(f"                          (c) speculation ceiling < 1.5x on {len(oversold_spec)} of the "
        f"{len(spec_shown)} cells where run()")
    say(f"                              actually PRINTS speculation advice   {'PASS' if p3c else 'MISS'}")
    say(f"                              (ungated count would be {len(oversold_spec_ungated)}; the "
        f"{len(oversold_spec_ungated) - len(oversold_spec)} extra are cells the")
    say(f"                               shipped block never reaches and so cannot over-sell.)")
    say(f"     P-3 carries NO kill rule (prereg section 5). If all three probes came back empty the "
        f"classification")
    say(f"     would be decoration - prereg section 7.4 names that as a refutation, but nothing in "
        f"section 5 fires on it.")
    say(f"     That asymmetry is a defect of the pre-registration, recorded in section 8 and staked "
        f"forward as K-6.")
    med_share = sorted(c["share"] for c in scored)[len(scored) // 2] if scored else 0.0
    multi = [c for c in scored if len(c["shares"]) > 1]
    med_share_multi = sorted(c["share"] for c in multi)[len(multi) // 2] if multi else 0.0
    say(f"P-4  ceiling non-trivial  ceiling < 2.0x on {len(ceil_lt2)}/{len(scored)} = "
        f"{frac_ceil_lt2:.1%} (>= {stake['p4_min_frac_ceiling_lt_2']:.0%})   {'PASS' if p4 else 'MISS'}")
    if not p4:
        say(f"       MISS, and the reason is worth more than the prediction: the median binding "
            f"share is {med_share:.1%} over")
        say(f"       all {len(scored)} cells and {med_share_multi:.1%} over the {len(multi)} that "
            f"use more than one resource at all.")
        say(f"       The modelled token is dominated by ONE resource almost everywhere, so the "
            f"binding resource's")
        say(f"       own ceiling is >=2x by construction. P-4 staked in advance that a miss here "
            f"costs this")
        say(f"       number its headline space - applied: binding_report now demotes it and says "
            f"so on the page.")
        say(f"       The caps that DO bite are the non-binding ones ('faster storage: NO effect'), "
            f"which is")
        say(f"       the BigMoeOnEdge lesson and is unaffected by this miss. P-4 carries no kill "
            f"rule (section 5).")
    say("")
    say("P-5  external arm (BigMoeOnEdge laptop, retrodiction -- see prereg section 0.3)")
    say(f"       hot {ext['hot_gb']:.6f} GB (staked {stake['p5_hot_gb']:.6f}) | "
        f"streamable {ext['streamable_gb']:.6f} GB (staked {stake['p5_streamable_gb']:.6f})")
    say(f"       C1 our uniform residency : miss {ext['c1']['miss']:.6f}  io {ext['c1']['io_s']:.6f}s  "
        f"ram {ext['c1']['ram_s']:.6f}s  -> {ext['c1']['klass']} "
        f"({ext['c1']['share']:.1%}, margin {ext['c1']['margin']:.2f}x)")
    say(f"       C2 their measured 80% hit: miss {ext['c2']['miss']:.6f}  io {ext['c2']['io_s']:.6f}s  "
        f"ram {ext['c2']['ram_s']:.6f}s  -> {ext['c2']['klass']} "
        f"({ext['c2']['share']:.1%}, margin {ext['c2']['margin']:.2f}x)")
    say(f"       C2 RAM term vs their published compute floor {stake['p5_their_compute_floor_s']:.3f} s: "
        f"{ext['c2']['ram_s'] / stake['p5_their_compute_floor_s'] - 1:+.1%} "
        f"(kill at +/-{stake['p5_kill_rel_error']:.0%})")
    say(f"       staked numbers reproduced: {'yes' if p5_numbers else 'NO'} | "
        f"classes as staked: "
        f"{'yes' if (ext['c1']['klass'] == stake['p5_class_uniform'] and ext['c2']['klass'] == stake['p5_class_measured']) else 'NO'}"
        f"   {'PASS' if p5 else 'MISS'}")
    say(f"       hot/streamable/miss are {ext['derived_from']}, and eta_r is imported "
        f"(plan.ETA_R_MOE = {ext['eta_r']:g}),")
    say(f"       so a change inside evaluate's disk row breaks P-5 instead of leaving it "
        f"self-consistent.")
    say(f"       EPISTEMIC GRADE (prereg 0.3, sharpened by adversarial review): the author computed "
        f"all of section 3")
    say(f"       BY HAND before staking, INCLUDING the -9.2% against their compute floor, and the "
        f"+/-15% band was")
    say(f"       chosen knowing that number. P-5 is therefore a REPRODUCTION check on shipped code "
        f"against arithmetic")
    say(f"       already done, not a blind prediction about the world. K-5 can fire on an "
        f"implementation bug; it")
    say(f"       cannot fire on a surprise from BigMoeOnEdge, because there is no surprise left to "
        f"have.")
    say("")
    say("KILL RULES")
    say(f"  K-1 label carries information            {'PASS' if k1 else 'FIRED -> demote the headline claim'}")
    say(f"  K-2 classification is sound              {'PASS' if k2 else 'FIRED -> revert, nothing ships'}")
    if k3_vacuous:
        say(f"  K-3 no predicted number moved            UNCHECKED -- the baseline ref "
            f"'{args.baseline_ref}' ALREADY contains the change.")
        say(f"      Every cell compared the module to itself, so '{len(regressions)} regressions' "
            f"is a fact about nothing.")
        say(f"      This is NOT a pass and the run is INCOMPLETE. Re-run with --baseline-ref "
            f"<pre-change commit>.")
    else:
        say(f"  K-3 no predicted number moved            "
            f"{f'PASS ({len(scored)}/{n_cells} identical)' if k3 else f'FIRED -> {len(regressions)} staked-grid cells moved, {len(regressions_locked)} pinned-size cells moved'}")
    say(f"      arm 1 (staked, prereg section 2): {n_cells} preset cells, true_size_gb=None, "
        f"codebook_share=0 -> {len(regressions)} moved")
    say(f"      arm 2 (added by the second adversarial pass): the same "
        f"{checks['locked_path']} pairs replayed with a")
    say(f"      PINNED file size (x{LOCKED_PATH_SIZE_SCALE:g}) and codebook_share="
        f"{LOCKED_PATH_CODEBOOK_SHARE:g} -> {len(regressions_locked)} moved. Every one of the 14")
    say(f"      locked ladder rows is a `--gguf` run, i.e. exactly those two inputs; arm 1 pins "
        f"neither, so a")
    say(f"      regression that only bites when a real size is pinned was invisible to the staked "
        f"arm.")
    say(f"  K-4 all-in-VRAM <4.5b carries the caveat  "
        f"{'PASS' if k4 else 'FIRED -> honesty gate'} "
        f"({k4_n} all-in-VRAM WINNERS on the grid checked through binding_report, plus the "
        f">4.5-bit negative control)")
    say(f"  K-5 agrees with the external ground truth {'PASS' if k5 else 'FIRED -> label is not a diagnosis'}")

    for r in (regressions + regressions_locked)[:5]:
        say(f"    regression: {r}")
    for f in (phantom_failures + monotonic_failures)[:5]:
        say(f"    soundness failure: {f}")
    for f in recon_failures[:5]:
        say(f"    reconstruction failure (K-2 clause 2): {f}")
    for f in k4_failures[:5]:
        say(f"    K-4 failure: {f}")
    for u in upgrade_changes[:5]:
        say(f"    upgrade advice changed: {u['cell']}  old={u['old']}  new={u['new']}")

    # K-3 UNCHECKED is not K-3 PASS. `k3 or k3_vacuous` used to collapse the two, which turned the
    # regression gate into a no-op the moment the change was committed and still printed PASS.
    #
    # ADVERSARIAL FIX (second pass, item 8): FAIL now takes precedence over INCOMPLETE. This branch
    # used to test `k3_vacuous` FIRST, so a run with a vacuous baseline in which K-1 and K-5 both
    # FIRED exited 3 - the code the docstring reserves for "a kill rule could not be evaluated" -
    # and a wrapper that treats 3 as "re-run with a proper baseline" would never have surfaced the
    # two rules that died. Verified by injection: constant RESOURCE_CLASS + a vacuous baseline used
    # to print K-1 FIRED, K-5 FIRED, exit 3. INCOMPLETE is now reserved for a run in which every
    # rule that COULD be scored held, and the only thing missing is K-3.
    scored_held = all([k1, k2, k4, k5]) and (k3 if not k3_vacuous else True)
    if not scored_held:
        verdict_word, exit_code = "FAIL", 1
    elif k3_vacuous:
        verdict_word, exit_code = "INCOMPLETE", 3
    else:
        verdict_word, exit_code = "PASS", 0
    say("")
    say("=" * 96)
    if verdict_word == "PASS":
        tail = "every staked kill rule holds."
    elif verdict_word == "INCOMPLETE":
        tail = ("K-3 could not be evaluated (see above). The other four kill rules were scored and "
                "all held, but a run with an unchecked kill rule is NOT a pass and must not be "
                "cited as one. (A run in which any SCORED rule fired now reads FAIL, not "
                "INCOMPLETE - second adversarial pass, item 8.)")
    else:
        tail = ("at least one kill rule fired; see above. Per protocol the miss is published at "
                "equal prominence to a hit and no constant is retuned to recover it.")
    say(f"VERDICT: {verdict_word} -- " + tail)
    say("=" * 96)

    os.makedirs(DATA, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(dict(
            prereg=os.path.relpath(PREREG, REPO), experiment=54, prereg_id=88,
            baseline_ref=args.baseline_ref, k3_vacuous=k3_vacuous,
            grid=dict(models=len(models), machines=len(machines), ctx=list(CTX_GRID),
                      cells=n_cells, bits=BITS),
            stake=stake,
            results=dict(class_counts=counts, distinct_classes=distinct, modal_share=modal_share,
                         p1_headroom=headroom, p1_worst_modal_share_over_bits=worst_modal,
                         p1_min_distinct_over_bits=min_distinct,
                         frac_ceiling_lt_2=frac_ceil_lt2, median_binding_share=med_share,
                         median_binding_share_multi_resource=med_share_multi,
                         n_multi_resource_cells=len(multi), soundness_checks_run=checks,
                         k4_vram_winners_checked=k4_n, k4_failures=k4_failures,
                         upgrade_changes=upgrade_changes,
                         oversold_codebook=[c["model"] + "/" + c["machine"] + f"/ctx{c['ctx']}"
                                            for c in oversold_codebook],
                         n_spec_advice_shown=len(spec_shown),
                         oversold_spec=[c["model"] + "/" + c["machine"] + f"/ctx{c['ctx']}"
                                        for c in oversold_spec],
                         n_oversold_spec_ungated=len(oversold_spec_ungated),
                         regressions=regressions,
                         regressions_locked_path=regressions_locked,
                         locked_path_arm=dict(size_scale=LOCKED_PATH_SIZE_SCALE,
                                              codebook_share=LOCKED_PATH_CODEBOOK_SHARE,
                                              cells=checks["locked_path"]),
                         phantom_failures=phantom_failures,
                         monotonic_failures=monotonic_failures,
                         reconstruction_failures=recon_failures,
                         reconstruction_rel_tol=RECON_REL_TOL,
                         external=ext),
            predictions=dict(P1=p1, P2=p2, P3=p3, P3a=p3a, P3b=p3b, P3c=p3c, P4=p4, P5=p5),
            # K-3 UNCHECKED is recorded as null, NOT true. A consumer that reads this file must be
            # unable to mistake "not evaluated" for "held".
            kill_rules=dict(K1=k1, K2=k2, K3=(None if k3_vacuous else k3), K4=k4, K5=k5),
            kill_rule_notes=dict(
                K1="bar not live on this grid - worst modal share over bits 2.0-8.0 was "
                   f"{worst_modal:.3f} against a staked 0.85; PASS is weak evidence",
                K2="scores all three staked clauses: phantom sensitivity, share monotonicity, and "
                   f"term reconstruction to {RECON_REL_TOL:g} on {checks['recon']} rows (the third "
                   "was unscored until the second adversarial pass)",
                K3=("UNCHECKED - baseline ref already contained the change" if k3_vacuous
                    else f"scored against the pre-change module on {n_cells} staked cells plus "
                         f"{checks['locked_path']} pinned-size/codebook replays"),
                K4=f"scored on {k4_n} all-in-VRAM winners from the grid, not on a synthetic row",
                K5="reproduction check: section 3's arithmetic was done by hand before staking, "
                   "and the +/-15% band was chosen knowing the answer was -9.2%"),
            verdict=verdict_word,
            cells=cells,
        ), fh, indent=1, default=str)
    with open(OUT_LOG, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(_log_lines) + "\n")
    print(f"\nwrote {os.path.relpath(OUT_JSON, REPO)} and {os.path.relpath(OUT_LOG, REPO)}")
    sys.exit(exit_code)


def check_k4(planmod, vram_winners):
    """K-4: an all-in-VRAM winner below 4.5 bits must never be labelled without the unpack caveat.

    Checked on the report itself rather than on a grep of the source: what ships is the printed
    line, and a caveat that exists in a constant but never reaches the page is not a caveat.

    ADVERSARIAL FIX. This used to test ONE hand-built row, constructed with `placement="all in
    VRAM"` -- the exact literal `binding_report` compares against. It was therefore a tautology:
    it asserted that a function given the string it tests for finds that string, and it could not
    fire from the 340-cell sweep however the sweep came out. In particular it would NOT have
    noticed the realistic failure, which is a winning row whose NAME is not that literal (the
    caveat gate is `placement == "all in VRAM"`, not a substring test) -- exactly what would
    happen the day the all-in-VRAM row gains a suffix like "(KV in VRAM)".

    Now scored on the grid: every cell whose winner is an all-in-VRAM placement, reported through
    the same call run() makes. `vram_winners` is [(cell, bc, placement)]. A run that checked ZERO
    of them FAILS -- an honesty gate with an empty population is not a gate, and reporting it as
    PASS is the same error class this whole review exists to catch.

    Returns (ok, n_checked, failures).
    """
    failures = []
    for cell, bc, placement in vram_winners:
        text = "\n".join(planmod.binding_report(bc, bits=BITS, placement=placement))
        if "CAVEAT (#16/#52)" not in text or "SLOWER" not in text:
            failures.append(dict(cell=cell, placement=placement, klass=bc["klass"]))
    if not vram_winners:
        failures.append(dict(cell="<none>", placement="-",
                             klass="NO all-in-VRAM winner in the grid: K-4's population is empty, "
                                   "so the honesty gate was never exercised. Not a pass."))
    # Negative control on the same function: above 4.5 bits the caveat must NOT appear, or the
    # gate is a constant that always prints and proves nothing about the bits < 4.5 cells.
    row = planmod.Row("all in VRAM", 20.0, None, "-ngl 99", {"vram_bw": 0.05})
    bc = planmod.binding_constraint(row)
    high = "\n".join(planmod.binding_report(bc, bits=6.5, placement="all in VRAM"))
    if "CAVEAT" in high:
        failures.append(dict(cell="<negative control @6.5 bits>", placement="all in VRAM",
                             klass="caveat printed ABOVE 4.5 bits - the gate is unconditional"))
    return (not failures), len(vram_winners), failures


if __name__ == "__main__":
    main()
