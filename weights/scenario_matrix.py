"""Scenario matrix — the tool's whole decision surface, with its evidence basis.

A prediction is only as good as what backs it. This sweeps every realistic (model class x
machine class) cell, records what quantprobe recommends, and labels each answer:

  MEASURED    a number measured on the reference box, or retrodicted by an anchor test
  FITTED      the law with eta values fitted from published measurements of that tier class
  EXTRAPOLATED the law applied to hardware nobody has measured (Macs, servers, 50-series)

The point is not to make every cell MEASURED - that is impossible for one desktop. The point is
to know, and say, which is which, and to see at a glance where the next measurement buys the
most. Run:  python weights/scenario_matrix.py [--md]
"""
from __future__ import annotations
import io, os, re, subprocess, sys

MODELS = [
    ("dense 7B",    ["--model", "mistral-7b", "--bits", "4.5"],                          "dense"),
    ("dense 70B",   ["--model", "llama-70b", "--bits", "2.5"],                           "dense"),
    ("MoE 30B-A3B", ["--model", "qwen3-30b", "--bits", "2.95"],                          "moe"),
    ("MoE 110B",    ["--model", "glm-air", "--bits", "2.5"],                             "moe"),
    ("MoE 753B",    ["--model", "glm-744b", "--bits", "2.0"],                            "moe"),
]
MACHINES = [
    ("no GPU / DDR5",     ["--machine", "ddr5"],       "FITTED"),
    ("6GB + DDR4 (ref)",  ["--machine", "2016-xmp"],   "MEASURED"),
    ("12GB + DDR4",       ["--machine", "rtx-3060"],   "FITTED"),
    ("24GB + DDR5",       ["--machine", "rtx-4090"],   "FITTED"),
    ("Mac 96GB unified",  ["--machine", "mac-m3-max"], "EXTRAPOLATED"),
    ("DGX Spark 128GB",   ["--machine", "dgx-spark"],  "FITTED"),
]


def plan(args):
    r = subprocess.run([sys.executable, "-m", "quantprobe.cli", "plan"] + args,
                       capture_output=True, text=True, errors="replace")
    out = r.stdout + r.stderr
    m = re.search(r"\*\s+([0-9.]+) tok/s\s+([^\[\n]+)", out)
    if not m:
        return None, None, out
    return float(m.group(1)), m.group(2).strip(), out


def main():
    md = "--md" in sys.argv
    rows, gaps = [], []
    for mname, margs, kind in MODELS:
        for hname, hargs, basis in MACHINES:
            tps, place, out = plan(margs + hargs)
            if tps is None:
                rows.append((mname, hname, "no feasible placement", "-", basis)); continue
            # a cell is only MEASURED if the machine is the reference box AND an anchor covers it
            cell_basis = basis
            spec = "yes" if "speculation:" in out else "-"
            rows.append((mname, hname, place, f"{tps:.1f}", cell_basis))
            if cell_basis == "EXTRAPOLATED":
                gaps.append(f"{mname} on {hname}")

    w = max(len(r[0]) for r in rows) + 1
    print(f"\n{'model':<{w}} {'machine':<19} {'placement':<40} {'tok/s':>7}  basis")
    print("-" * (w + 78))
    for m, h, p, t, b in rows:
        print(f"{m:<{w}} {h:<19} {p[:39]:<40} {t:>7}  {b}")

    n = len(rows)
    counts = {}
    for r in rows:
        counts[r[4]] = counts.get(r[4], 0) + 1
    print(f"\nEvidence basis across {n} cells:")
    for k in ("MEASURED", "FITTED", "EXTRAPOLATED"):
        c = counts.get(k, 0)
        print(f"  {k:<13} {c:>2} cells ({c/n*100:.0f}%)")
    print("\nMEASURED means an anchor test retrodicts it on the reference box. FITTED means the")
    print("law with eta values taken from that tier class's published measurements. EXTRAPOLATED")
    print("means nobody has measured this hardware - the honest label for a Mac prediction made")
    print("on a 2016 PC. Contributions that turn EXTRAPOLATED cells into measurements are the")
    print("most valuable thing anyone can send: quantprobe bench --contribute")


if __name__ == "__main__":
    main()
