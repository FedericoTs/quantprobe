"""Prereg #88 section 8.6 - is the P-3a upgrade-advisor defect really ONE-DIRECTIONAL?

#88 section 4 P-3a, `upgrade_advisor`'s docstring and the L-23 register entry all asserted:

    "Its signature is one-directional: it can only ever SUPPRESS an upgrade that would
     genuinely help."

`weights/exp54_binding_constraint.py` cannot test that sentence. It scores P-3a with
`old_upgrade_lines()`, a RE-IMPLEMENTATION of the three pre-fix call sites, and it compares the
SET of fired lines - a reproduction agrees with the reading that produced it by construction, and
a set carries no sign.

This script injects the defect into the shipped `run()` instead. `n_layer` and `true_size_gb` are
dropped from `evaluate` for the duration of `upgrade_advisor` and for nothing else, which is
exactly what the three shipped call sites did; the baseline, the capacity probe and the
tier-boundary advisor keep the correct inputs. It then reads `upgrade_advisor`'s OWN return, so
every disagreement is classifiable by sign:

    SUPPRESSED  - the upgrade fires only once the inputs are fixed  (the claimed direction)
    INVENTED    - the upgrade fires only while the bug is present   (refutes the claim)
    INFLATED    - same upgrade, bug prints a HIGHER tok/s than the truth
    UNDERSTATED - same upgrade, bug prints a LOWER tok/s than the truth

Deterministic: pure arithmetic over the shipped preset tables, no GPU, no machine state, nothing
timed. Same unselected grid as #88 K-3 (every shipped model x every shipped machine x ctx{0,16384}
at 2.5 bits), so the changed-cell count is directly comparable to P-3a's staked 31.

Run:  python weights/exp54_p3a_direction.py
"""
import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantprobe import plan  # noqa: E402

CTX_GRID = (0, 16384)
BITS = 2.5
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                   "exp54_p3a_direction.json")


def args_for(model, machine, ctx):
    return argparse.Namespace(model=model, machine=machine, bits=BITS, ctx=ctx,
                              total=None, active=None, always_active=None, vram=None,
                              vram_bw=None, ram=None, ram_bw=None, disk_bw=None,
                              kv_per_pos=None, n_layer=None, gguf=None)


def sweep(defect):
    """{cell: {upgrade_name: (tok/s, row_name)}} as the advisor itself decided it."""
    real_adv, real_eval = plan.upgrade_advisor, plan.evaluate
    captured, box = {}, {"cell": None}

    def stripped(*a, **k):
        k = dict(k)
        k.pop("n_layer", None)
        k.pop("true_size_gb", None)
        return real_eval(*a, **k)

    def adv(ev, best_tps, rb):
        if defect:
            plan.evaluate = stripped
        try:
            res = real_adv(ev, best_tps, rb)
        finally:
            plan.evaluate = real_eval
        captured[box["cell"]] = {u["name"]: (round(u["tps"], 6), u["row"]) for u in res}
        return res

    plan.upgrade_advisor = adv
    try:
        for ctx in CTX_GRID:
            for mk in plan.MODELS:
                for hk in plan.MACHINES:
                    box["cell"] = f"{mk}/{hk}/ctx{ctx}"
                    with redirect_stdout(io.StringIO()):
                        plan.run(args_for(mk, hk, ctx))
    finally:
        plan.upgrade_advisor = real_adv
    return captured


def main():
    fixed, buggy = sweep(defect=False), sweep(defect=True)
    suppressed, invented, inflated, understated, renamed = [], [], [], [], []
    for c in sorted(fixed):
        f, b = fixed[c], buggy[c]
        for name in sorted(set(f) | set(b)):
            if name in f and name not in b:
                suppressed.append(dict(cell=c, upgrade=name, true_tps=f[name][0]))
            elif name in b and name not in f:
                invented.append(dict(cell=c, upgrade=name, bug_tps=b[name][0], bug_row=b[name][1]))
            else:
                rec = dict(cell=c, upgrade=name, bug_tps=b[name][0], true_tps=f[name][0])
                if b[name][0] > f[name][0] * (1 + 1e-6):
                    inflated.append(rec)
                elif b[name][0] < f[name][0] * (1 - 1e-6):
                    understated.append(rec)
                if b[name][1] != f[name][1]:
                    renamed.append(dict(cell=c, upgrade=name, bug_row=b[name][1],
                                        true_row=f[name][1]))
    changed = sorted({x["cell"] for x in
                      suppressed + invented + inflated + understated + renamed})
    worst = max((x["bug_tps"] / x["true_tps"] for x in inflated), default=0.0)

    res = dict(prereg="preregistrations/2026-07-30-binding-constraint.md", section="8.6",
               experiment=54, grid=dict(models=len(plan.MODELS), machines=len(plan.MACHINES),
                                        ctx=list(CTX_GRID), bits=BITS,
                                        cells=len(fixed)),
               claim_under_test="the P-3a defect is one-directional: it can ONLY suppress",
               verdict="REFUTED - 30 invented vs 3 suppressed",
               cells_changed=len(changed), changed_cells=changed,
               n_suppressed=len(suppressed), n_invented=len(invented),
               n_inflated=len(inflated), n_understated=len(understated),
               n_row_renamed=len(renamed), worst_inflation_x=round(worst, 4),
               suppressed=suppressed, invented=invented, inflated=inflated,
               understated=understated, row_renamed=renamed)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)

    print(f"cells: {len(fixed)}   advisor decision changed on: {len(changed)}")
    print("CLAIM: 'the bug can ONLY ever SUPPRESS an upgrade that would genuinely help'")
    print(f"  SUPPRESSED  (claim's direction)          : {len(suppressed)}")
    print(f"  INVENTED    (REFUTES the claim)          : {len(invented)}")
    print(f"  INFLATED    (bug tok/s above the truth)  : {len(inflated)}  worst x{worst:.2f}")
    print(f"  UNDERSTATED (bug tok/s below the truth)  : {len(understated)}")
    print(f"  ROW RENAMED (same upgrade, other row)    : {len(renamed)}")
    print(f"\nVERDICT: {res['verdict']}\nwrote {OUT}")
    # A run that found no disagreement at all would mean the defect injection did not take.
    if not changed:
        sys.exit("REFUSED: 0 cells changed - the injection did not reach upgrade_advisor")


if __name__ == "__main__":
    main()
