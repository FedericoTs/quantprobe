"""resweep340_audit.py -- re-score the two defects the 340-cell sweep found, on the CURRENT tree.

Validation item `resweep-340`. Stake: weights/data/resweep340_20260731T1656Z_STAKE.md (written
BEFORE this script ran a single cell).

WHAT IS MEASURED. Nothing on the GPU. Every number is deterministic arithmetic over the shipped
preset tables, so C-14 (one cal_id per comparison) does not bind, nothing is timed, and no probe
runs twice on the same target (there is no cache in the loop to warm).

THE TWO DEFECTS, RE-SCORED CELL BY CELL:

  D-1  the upgrade advisor drew its counterfactuals from a SMALLER ROW MENU and a DIFFERENT MODEL
       SIZE than the baseline it compared them to, because the three shipped call sites passed
       neither `n_layer` nor `true_size_gb`. 31 of 340 cells printed different upgrade lines once
       the inputs matched, and the bug INVENTED 30 recommendations against 3 it suppressed. The
       arithmetic tell of an invented pair is on the record: a RAM-CAPACITY lever and an I/O lever
       printed at an IDENTICAL tok/s.

       Scored here THREE ways, none of them a re-implementation of the old call sites (the old
       P-3a was a re-implementation, and a re-implementation agrees with the reading that produced
       it by construction):
         A-1  no two FIRED upgrades on different resources may report the same tok/s (1e-9 rel).
         A-2  every `evaluate()` call the shipped `upgrade_advisor` actually makes -- fired or not,
              caught at the function boundary -- must carry an argument dict identical to the
              baseline's in every key but the one resource key that upgrade intends to move.
         A-3  each fired upgrade's tok/s and winning row name must be reproduced by an INDEPENDENT
              recomputation: the baseline kwargs plus the override, straight into plan.evaluate.

  D-2  the 4.7x ngram headline is arithmetically unreachable on a row whose token is largely CPU
       attention over the KV cache (a verify batch amortizes weight READS; per-position attention
       does not amortize). Unreachable on 17 of the 127 cells where run() prints the block.

       Scored on the PRINTED PAGE, with the bound recomputed HERE from the row's own terms:
              bound = T_total / (T_bandwidth / R + T_cpu)
       written out rather than obtained by calling plan.spec_reachable_x / plan.spec_ceiling --
       calling the shipped function to check the shipped function's own gate would agree by
       construction. R is the MEASURED per-round amortization (plan.SPEC_X_NGRAM_TUNED /
       plan.SPEC_X_NGRAM_DENSE); those are measurements, not gate logic.

FALSIFICATION (protocol: construct the input that makes the test fail, and verify it exits
non-zero). Two injections, both re-run through the identical auditor:
  --inject upgrade   strips n_layer/true_size_gb from `evaluate` for the duration of
                     `upgrade_advisor` and for nothing else -- exactly what the pre-fix call sites
                     did. A-2 must fire.
  --inject spec      calls speculation_advice(moe, placement, row=None) -- the pre-C-15 call site.
                     The flat headline returns and B-1 must fire.
If either injection leaves the auditor at exit 0, the auditor cannot fail and the clean run has
measured nothing (kill rule K-C in the stake).

Exit codes: 0 = clean, 1 = a kill rule fired, 2 = precondition missing.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "weights", "data")
sys.path.insert(0, REPO)

from quantprobe import plan  # noqa: E402

CTX_GRID = (0, 16384)
BITS = 2.5
REL_TOL = 1e-9
# The baseline argument keys every counterfactual must inherit unchanged. Anything outside the
# single overridden resource key is a counterfactual about a DIFFERENT MODEL, which is D-1.
BASE_KEYS = ("t", "a", "ne", "moe", "bits", "vc", "vb", "rc", "rb", "db", "geta", "gl",
             "ctx", "kvp", "n_layer", "true_size_gb", "codebook_share")
# The empty set is ALLOWED and is not a defect: an override can coincide with the machine's own
# value (every preset that already ships a 3.5 GB/s NVMe makes the `NVMe SSD` counterfactual a
# no-op, and such a row can never clear UPGRADE_MIN_GAIN so it never fires). What must never
# appear is a diff on a key the upgrade does not name - `n_layer` and `true_size_gb` above all,
# which is precisely D-1.
ALLOWED_DIFFS = (set(), {"rb"}, {"rc"}, {"db"})

NOT_REACHABLE = "NOT REACHABLE ON THIS ROW"
RE_AT_MOST = re.compile(r"at most \*\*([0-9.]+)x\*\*")


def args_for(model, machine, ctx):
    return argparse.Namespace(model=model, machine=machine, bits=BITS, ctx=ctx,
                              total=None, active=None, always_active=None, vram=None,
                              vram_bw=None, ram=None, ram_bw=None, disk_bw=None,
                              kv_per_pos=None, n_layer=None, gguf=None)


def override_for(name):
    """The override dict the shipped UPGRADES table carries for this upgrade, translated the way
    run()'s own `ev` closure translates it (rc_delta -> an absolute rc)."""
    for n, resource, over in plan.UPGRADES:
        if n == name:
            return resource, dict(over)
    return None, None


def independent_counterfactual(base_kw, name, real_eval):
    """A-3. Apply the override to the baseline kwargs HERE and evaluate directly."""
    resource, over = override_for(name)
    if over is None:
        return None, None, None
    kw = dict(base_kw)
    if "rc_delta" in over:
        kw["rc"] = kw["rc"] + over.pop("rc_delta")
    kw.update(over)
    rows = real_eval(**kw)[2]
    if not rows:
        return resource, None, None
    return resource, rows[0][0], rows[0][1]


def terms(row):
    """(total, bandwidth, cpu) seconds per token from the row's OWN decomposition, plus the
    reconstruction check that makes those three numbers trustworthy at all."""
    tt = plan.resource_times(row)
    if not tt:
        return None
    cpu = tt.get("cpu_compute", 0.0)
    bw = sum(v for r, v in tt.items() if r != "cpu_compute")
    total = bw + cpu
    eff = getattr(row, "eff", None)
    recon_rel = None
    if eff is not None and total > 0 and row[1]:
        recon_rel = abs((eff / total) / row[1] - 1)
    return dict(total=total, bw=bw, cpu=cpu, recon_rel=recon_rel)


def reachable_bound(row, R):
    """The most a drafter whose measured per-round amortization is R can deliver ON THIS ROW.

    Written out rather than obtained from plan.spec_reachable_x on purpose: this is the number the
    shipped gate is being CHECKED against, and a check that calls the thing it checks agrees by
    construction.  x = T/(T_bw/R + T_cpu)  ->  R at cpu=0, 1.0 at bw=0, monotone between.
    """
    t = terms(row)
    if not t:
        return None, None
    denom = t["bw"] / R + t["cpu"]
    return (t["total"] / denom if denom > 0 else None), t


def sweep(inject=None):
    """Drive the shipped run() over the unselected grid and audit what it actually did."""
    real_eval = plan.evaluate
    real_adv = plan.upgrade_advisor
    real_spec = plan.speculation_advice

    box = {"cell": None, "base": None, "inside": False}
    per_cell = {}

    def spy_eval(*a, **k):
        kk = dict(k)
        if box["inside"] and inject == "upgrade":
            # The pre-fix call sites: neither argument was passed at all.
            kk.pop("n_layer", None)
            kk.pop("true_size_gb", None)
        if box["inside"]:
            per_cell[box["cell"]]["adv_calls"].append(dict(kk))
        elif box["base"] is None and not a:
            box["base"] = dict(kk)
        return real_eval(*a, **kk)

    def spy_adv(ev, best_tps, rb):
        box["inside"] = True
        try:
            res = real_adv(ev, best_tps, rb)
        finally:
            box["inside"] = False
        per_cell[box["cell"]]["fired"] = [
            dict(name=u["name"], resource=u["resource"], tps=u["tps"], row=u["row"],
                 gain=u["gain"]) for u in res]
        per_cell[box["cell"]]["best_tps"] = best_tps
        return res

    def spy_spec(moe, placement, row=None):
        passed = None if inject == "spec" else row
        txt = real_spec(moe, placement, row=passed)
        per_cell[box["cell"]]["spec"] = dict(moe=moe, placement=placement, row=row, text=txt,
                                             row_was_passed=passed is not None)
        return txt

    plan.evaluate = spy_eval
    plan.upgrade_advisor = spy_adv
    plan.speculation_advice = spy_spec
    try:
        for ctx in CTX_GRID:
            for mk in plan.MODELS:
                for hk in plan.MACHINES:
                    cell = f"{mk}/{hk}/ctx{ctx}"
                    box["cell"] = cell
                    box["base"] = None
                    per_cell[cell] = dict(adv_calls=[], fired=None, spec=None, best_tps=None,
                                          base=None, page=None)
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        plan.run(args_for(mk, hk, ctx))
                    per_cell[cell]["base"] = box["base"]
                    per_cell[cell]["page"] = buf.getvalue()
    finally:
        plan.evaluate = real_eval
        plan.upgrade_advisor = real_adv
        plan.speculation_advice = real_spec
    return per_cell, real_eval


def audit(per_cell, real_eval):
    f_pairs, f_identity, f_arith = [], [], []
    f_unreachable, f_spurious_qualifier, f_bound_text = [], [], []
    f_recon = []
    cells = []
    n_spec_printed = 0
    n_oversold_shipped = 0
    n_oversold_k2_old = 0
    checks = dict(cells=0, adv_calls=0, fired=0, spec=0, bound=0)

    for cell in sorted(per_cell):
        rec = per_cell[cell]
        base = rec["base"]
        checks["cells"] += 1
        out = dict(cell=cell)

        # ---------------- D-1 --------------------------------------------------------------
        if base is None:
            f_identity.append(dict(cell=cell, why="no baseline evaluate() call was observed"))
        else:
            for kw in rec["adv_calls"]:
                checks["adv_calls"] += 1
                diff = set()
                for k in BASE_KEYS:
                    if k not in kw:
                        diff.add(f"MISSING:{k}")
                    elif kw[k] != base.get(k):
                        diff.add(k)
                extra = set(kw) - set(base)
                if extra:
                    diff |= {f"EXTRA:{e}" for e in extra}
                if diff not in ALLOWED_DIFFS:
                    f_identity.append(dict(cell=cell, diff=sorted(diff),
                                           base={k: base.get(k) for k in BASE_KEYS},
                                           counterfactual={k: kw.get(k) for k in BASE_KEYS}))
        fired = rec["fired"] or []
        out["fired"] = [[u["name"], round(u["tps"], 6), u["row"]] for u in fired]
        checks["fired"] += len(fired)
        for i in range(len(fired)):
            for j in range(i + 1, len(fired)):
                a, b = fired[i], fired[j]
                if a["resource"] == b["resource"]:
                    continue
                if b["tps"] and abs(a["tps"] / b["tps"] - 1) <= REL_TOL:
                    f_pairs.append(dict(cell=cell, a=[a["name"], a["resource"], a["tps"], a["row"]],
                                        b=[b["name"], b["resource"], b["tps"], b["row"]]))
        if base is not None:
            for u in fired:
                _, rname, rtps = independent_counterfactual(base, u["name"], real_eval)
                ok = (rtps is not None and rname == u["row"]
                      and abs(rtps / u["tps"] - 1) <= REL_TOL)
                if not ok:
                    f_arith.append(dict(cell=cell, upgrade=u["name"], advisor=[u["row"], u["tps"]],
                                        independent=[rname, rtps]))

        # ---------------- D-2 --------------------------------------------------------------
        sp = rec["spec"]
        if sp and sp["text"]:
            n_spec_printed += 1
            checks["spec"] += 1
            row = sp["row"]
            moe, placement, text = sp["moe"], sp["placement"], sp["text"]
            if moe and "split experts" in (placement or ""):
                R = plan.SPEC_X_NGRAM_TUNED
            elif moe:
                R = None
            else:
                R = plan.SPEC_X_NGRAM_DENSE
            out["spec_branch_R"] = R
            qualified = NOT_REACHABLE in text
            out["qualified"] = qualified
            if R is None:
                if qualified:
                    f_spurious_qualifier.append(dict(cell=cell, why="branch quotes no headline "
                                                                    "yet printed NOT REACHABLE"))
            else:
                bound, t = reachable_bound(row, R)
                checks["bound"] += 1
                if t and t["recon_rel"] is not None and t["recon_rel"] > 1e-9:
                    f_recon.append(dict(cell=cell, rel=t["recon_rel"], tok_s=row[1]))
                out["bound_x"] = bound
                out["cpu_share"] = (t["cpu"] / t["total"]) if t and t["total"] else None
                # the OLD sweep's metric, recomputed here for comparability with its 17/127
                ceil_k2 = (t["total"] / (t["bw"] / 3.0 + t["cpu"])) if t and (t["bw"] / 3.0 + t["cpu"]) > 0 else None
                out["ceiling_k2"] = ceil_k2
                oversold_old = bool(ceil_k2 is not None and ceil_k2 < 1.5)
                out["oversold_by_old_k2_metric"] = oversold_old
                n_oversold_k2_old += int(oversold_old)
                oversold = bool(bound is not None
                                and bound < R * (1.0 - plan.SPEC_HEADLINE_TOL))
                out["oversold"] = oversold
                n_oversold_shipped += int(oversold)
                if oversold and not qualified:
                    f_unreachable.append(dict(cell=cell, R=R, bound=bound, placement=placement,
                                              cpu_share=out["cpu_share"]))
                if qualified and not oversold:
                    f_spurious_qualifier.append(dict(cell=cell, R=R, bound=bound,
                                                     why="qualifier printed on a reachable row"))
                if qualified and bound is not None:
                    m = RE_AT_MOST.search(text)
                    if not m:
                        f_bound_text.append(dict(cell=cell, why="qualified text carries no "
                                                                "'at most **N.NNx**' number"))
                    elif abs(float(m.group(1)) - round(bound, 2)) > 5e-3:
                        f_bound_text.append(dict(cell=cell, printed=float(m.group(1)),
                                                 computed=round(bound, 2)))
        else:
            out["spec_printed"] = False
        cells.append(out)

    return dict(cells=cells, checks=checks,
                n_spec_printed=n_spec_printed,
                n_oversold_shipped_gate=n_oversold_shipped,
                n_oversold_old_k2_metric=n_oversold_k2_old,
                invented_pairs=f_pairs, identity_mismatches=f_identity,
                arithmetic_mismatches=f_arith, unreachable_headlines=f_unreachable,
                spurious_qualifiers=f_spurious_qualifier, bound_text_mismatches=f_bound_text,
                reconstruction_failures=f_recon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", choices=("upgrade", "spec"), default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    n_models, n_machines = len(plan.MODELS), len(plan.MACHINES)
    n_cells = n_models * n_machines * len(CTX_GRID)
    per_cell, real_eval = sweep(inject=a.inject)
    res = audit(per_cell, real_eval)
    res["grid"] = dict(models=n_models, machines=n_machines, ctx=list(CTX_GRID),
                       cells=n_cells, bits=BITS)
    res["inject"] = a.inject
    res["grid_as_staked"] = (n_models, n_machines) == (10, 17)

    hard = (res["invented_pairs"] + res["identity_mismatches"] + res["arithmetic_mismatches"]
            + res["unreachable_headlines"] + res["spurious_qualifiers"]
            + res["bound_text_mismatches"] + res["reconstruction_failures"])
    res["verdict"] = "CLEAN" if not hard else "FAILURES"

    print(f"grid {n_models}x{n_machines}x{len(CTX_GRID)} = {n_cells} cells @ {BITS:g} bits"
          f"   inject={a.inject}")
    print(f"  upgrade counterfactuals inspected at the evaluate() boundary : {res['checks']['adv_calls']}")
    print(f"  upgrades FIRED across the grid                              : {res['checks']['fired']}")
    print(f"  cells where run() PRINTS speculation advice                 : {res['n_spec_printed']}")
    print(f"  of those, oversold by the shipped gate (bound < R*(1-tol))  : {res['n_oversold_shipped_gate']}")
    print(f"  of those, oversold by the OLD k=2 ceiling<1.5 metric        : {res['n_oversold_old_k2_metric']}")
    print("")
    print(f"  A-1 invented upgrade pairs (same tok/s, different resource) : {len(res['invented_pairs'])}")
    print(f"  A-2 counterfactual-identity mismatches                      : {len(res['identity_mismatches'])}")
    print(f"  A-3 advisor arithmetic not reproduced independently         : {len(res['arithmetic_mismatches'])}")
    print(f"  B-1 UNREACHABLE headlines printed unqualified               : {len(res['unreachable_headlines'])}")
    print(f"  B-2 qualifier printed on a row that could reach it          : {len(res['spurious_qualifiers'])}")
    print(f"  B-3 printed bound != recomputed bound                       : {len(res['bound_text_mismatches'])}")
    print(f"  B-4 row terms failed to reconstruct their own tok/s         : {len(res['reconstruction_failures'])}")
    for f in hard[:12]:
        print(f"    FAILURE: {f}")
    print(f"\nVERDICT: {res['verdict']}")

    out = a.out or os.path.join(DATA, "resweep340_audit.json")
    os.makedirs(DATA, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    print(f"wrote {out}")
    sys.exit(0 if not hard else 1)


if __name__ == "__main__":
    main()
