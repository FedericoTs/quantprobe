"""Confirm the PREVIOUSLY-AFFECTED cells individually, never as an aggregate.

Reads the historical sweep artifact (weights/data/exp54_binding_constraint.json, written at commit
0782477) for the two defect populations it named:

    results.oversold_spec    17 cells - the 4.7x/2.10x headline was arithmetically unreachable
    results.upgrade_changes  31 cells - the upgrade advice moved once the counterfactual inputs
                                        matched the baseline

and the current-tree audit (resweep340_*_audit.json) for what each of those cells does NOW. A cell
named in the old artifact that no longer exists in the grid is reported as UNMEASURABLE, never
dropped from the denominator (stake K-E).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

OLD = json.load(open(os.path.join(DATA, "exp54_binding_constraint.json"), encoding="utf-8"))
NEW = json.load(open(sys.argv[1], encoding="utf-8"))
by_cell = {c["cell"]: c for c in NEW["cells"]}

lines = []


def say(s=""):
    print(s)
    lines.append(s)


say("=" * 110)
say("PREVIOUSLY-AFFECTED CELLS, RE-SCORED ON THE CURRENT TREE - one line per cell, no aggregate")
say(f"old artifact: weights/data/exp54_binding_constraint.json (commit 0782477)")
say(f"new audit   : {os.path.basename(sys.argv[1])}")
say("=" * 110)

say("")
say("D-2  THE 17 CELLS WHOSE SPECULATION HEADLINE WAS UNREACHABLE")
say(f"     {'cell':<34} {'R':>5} {'bound':>7} {'CPU%':>6} {'k2ceil':>7}  oversold?  qualifier printed?")
n_ok = n_bad = n_missing = 0
for cell in OLD["results"]["oversold_spec"]:
    c = by_cell.get(cell)
    if c is None:
        say(f"     {cell:<34}  UNMEASURABLE - cell not present in the current grid")
        n_missing += 1
        continue
    if "bound_x" not in c:
        say(f"     {cell:<34}  UNMEASURABLE - run() no longer prints speculation advice here")
        n_missing += 1
        continue
    ok = (not c["oversold"]) or c["qualified"]
    n_ok += int(ok)
    n_bad += int(not ok)
    say(f"     {cell:<34} {c['spec_branch_R']:>5.2f} {c['bound_x']:>7.3f} "
        f"{c['cpu_share'] * 100:>5.1f}% {c['ceiling_k2']:>7.3f}  "
        f"{str(c['oversold']):<9}  {'YES' if c['qualified'] else 'NO  <-- FAILURE'}")
say(f"     -> {n_ok} qualified or no longer oversold | {n_bad} still printing an unreachable "
    f"headline | {n_missing} unmeasurable")

extra = [c["cell"] for c in NEW["cells"]
         if c.get("oversold") and c["cell"] not in set(OLD["results"]["oversold_spec"])]
say("")
say(f"D-2b CELLS OVERSOLD NOW THAT THE OLD SWEEP DID NOT LIST ({len(extra)}). The old sweep's "
    f"metric was")
say("     `spec_ceiling(row, k=2) < 1.5`; the SHIPPED gate is `bound < R*(1-1%)`, which is the")
say("     stricter and more honest one (it uses the drafter's own measured amortization R, not a")
say("     K=2 draft-model proxy). Every one of these must also carry the qualifier.")
for cell in extra:
    c = by_cell[cell]
    say(f"     {cell:<34} R {c['spec_branch_R']:.2f} bound {c['bound_x']:.3f} "
        f"CPU {c['cpu_share'] * 100:.1f}% k2ceil {c['ceiling_k2']:.3f}  "
        f"qualifier {'YES' if c['qualified'] else 'NO  <-- FAILURE'}")

say("")
say("D-1  THE 31 CELLS WHOSE UPGRADE ADVICE MOVED - what the advisor prints NOW")
say("     (invented-pair = two FIRED upgrades on DIFFERENT resources at the SAME tok/s)")
bad_cells = {f["cell"] for f in NEW["invented_pairs"]} | {f["cell"] for f in NEW["identity_mismatches"]} \
    | {f["cell"] for f in NEW["arithmetic_mismatches"]}
n_ok = n_bad = n_missing = 0
for u in OLD["results"]["upgrade_changes"]:
    cell = u["cell"]
    c = by_cell.get(cell)
    if c is None:
        say(f"     {cell:<34}  UNMEASURABLE - cell not present in the current grid")
        n_missing += 1
        continue
    now = c.get("fired") or []
    flag = "FAILURE" if cell in bad_cells else "clean"
    n_bad += int(cell in bad_cells)
    n_ok += int(cell not in bad_cells)
    old_txt = ", ".join(f"{n} {v:g}" for n, v in u["old"]) or "(none)"
    new_txt = ", ".join(f"{n} {v:g}" for n, v, _ in now) or "(none)"
    say(f"     {cell:<34} {flag:<8} pre-fix[{old_txt}]  ->  now[{new_txt}]")
say(f"     -> {n_ok} clean | {n_bad} with an invented pair or a mismatched counterfactual | "
    f"{n_missing} unmeasurable")

say("")
say("GRID: " + json.dumps(NEW["grid"]))
say(f"as staked (10x17x2=340): {NEW['grid_as_staked']}")

open(os.path.join(DATA, "resweep340_20260731T1656Z_cells.log"), "w",
     encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
