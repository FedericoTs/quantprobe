"""Protocol gate: construct the inputs that MUST make score_final_ladder.py fail, and verify
each exits non-zero. Run BEFORE trusting any PASS from the real ladder.
"""
import copy
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "weights", "data", "ladder_PRE_v124_2dc97d41_backup.json")
TMP = os.path.join(ROOT, "weights", "data", "_falsify_tmp.json")


def run(rows):
    json.dump(rows, open(TMP, "w", encoding="utf-8"), indent=1)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "weights", "score_final_ladder.py"),
                        TMP, REF], capture_output=True, text=True, cwd=ROOT)
    return r.returncode, r.stdout


base = json.load(open(REF, encoding="utf-8"))
cases = {}

# F0 control: the reference scored against itself must PASS (exit 0). If this fails, the
# scorer is broken in the other direction and its FAILs mean nothing either.
cases["F0 reference-vs-itself (must PASS, exit 0)"] = (copy.deepcopy(base), 0)

# F1: force the median out of band by inflating every prediction 12%.
f1 = copy.deepcopy(base)
for r in f1:
    r["err_pct"] = 12.0
cases["F1 median forced to 12.0%"] = (f1, 1)

# F2: drop one measurement -> 13 scored rows.
f2 = copy.deepcopy(base)
f2[5]["measured"] = None
f2[5]["err_pct"] = None
cases["F2 one row unmeasured (13/14)"] = (f2, 1)

# F3: two distinct machine states in one comparison (C-14 violation).
f3 = copy.deepcopy(base)
f3[9]["cal_id"] = "deadbeef"
cases["F3 two cal_ids"] = (f3, 1)

# F4: one prediction moved by 0.1 tok/s -> determinism arm must catch it.
f4 = copy.deepcopy(base)
f4[3]["predicted"] = round(f4[3]["predicted"] + 0.1, 2)
cases["F4 one predicted +0.1"] = (f4, 1)

ok = True
for name, (rows, want) in cases.items():
    rc, out = run(rows)
    good = (rc == want)
    ok &= good
    print(f"{'ok ' if good else 'BAD'} {name}: exit {rc} (wanted {want})")
    if not good:
        print(out)
os.remove(TMP)
print("\nfalsifiability gate:", "PASSED - the scorer can fail" if ok else "BROKEN")
sys.exit(0 if ok else 1)
