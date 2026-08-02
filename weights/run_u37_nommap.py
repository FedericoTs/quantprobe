"""U-37 - is the mmap decode penalty RESIDENCY PRESSURE? The quiet-box re-run.

  python weights/run_u37_nommap.py

LAUNCH FROM A PLAIN TERMINAL WITH NO CODING AGENT RUNNING. Memory pressure is the variable
under test, so an agent's own working set pushes the result TOWARD the hypothesis. That is not
a detail here - it is the specific way this experiment would lie to us.

WHAT THE PILOT FOUND AND WHY IT WAS THROWN AWAY. #100b measured decode-only throughput for
mmap vs --no-mmap at two model sizes and got exactly the staked pattern: 1.659x on a model at
86% of RAM, 0.955x on one at 28%. It was NOT counted, for two reasons. It ran under contention
with 2 reps; and its own KR-A - a stability requirement - disqualified it, because both large
arms had rep spreads above 1.5x.

THAT KILL RULE WAS WRONG, AND FIXING IT IS THE POINT OF THIS RE-RUN. The hypothesis predicts a
model at 86% of RAM has NO steady state: each rep faults differently depending on what the
kernel evicted. Instability IS the phenomenon. I had written a rule that discarded any run
exhibiting the effect it was testing for. A cannot-vary guard belongs on the CONTROL, where
stability IS predicted - and the control behaved perfectly (spreads 1.039 and 1.005), so the
harness was never the problem.

STAKED IN U-37, RESTATED HERE UNCHANGED. Decode tok/s only (llama-bench excludes load, so both
arms move the same bytes on the measured path). >= 5 reps. Medians, not means.

  K-1  median(--no-mmap) / median(mmap) >= 1.20 on the BIG model (>=80% of RAM)
  K-2  the same ratio <= 1.10 on the SMALL model (<=30% of RAM). If the small control ALSO
       shows a gap, the mechanism is mmap itself and this hypothesis is WRONG.
  K-3  rep spread on the big mmap arm must EXCEED the small control's spread. The hypothesis
       predicts instability under pressure; a perfectly stable big arm REFUTES it. This is the
       rule the pilot had backwards.
  K-4  free RAM sampled per rep must confirm the big arm ran near capacity. If it did not, the
       premise is absent and no ratio may be claimed however it lands.

  All four must hold. K-1 alone is not a result.
"""
from __future__ import annotations
import json, os, statistics, subprocess, sys, time

def _find_llamacpp():
    """Walk UP looking for tools/llamacpp-b10098 rather than counting directories.

    A fixed level count is wrong depending on where you run from: tools/ sits at the main
    checkout root, which is 2 levels above weights/ there but 5 above it inside a git
    worktree. exp98 and exp100 both carried the same fixed-depth default and only worked
    because QP_LLAMACPP was set on the command line - the default was dead code that looked
    fine. Overridable with QP_LLAMACPP.
    """
    env = os.environ.get("QP_LLAMACPP")
    if env:
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        cand = os.path.join(d, "tools", "llamacpp-b10098")
        if os.path.isdir(cand):
            return cand
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return os.path.join(d, "tools", "llamacpp-b10098")   # give a real path to fail loudly on


B = _find_llamacpp()
BENCH = os.path.join(B, "llama-bench.exe")
BIG = "D:/evo-compress-data/gguf/Qwen3-Coder-30B-A3B-Instruct-Q3_K_M.gguf"
SMALL = "D:/evo-compress-data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STAMP = time.strftime("%Y%m%d_%H%M%S")
OUT = os.path.join(DATA, f"u37_nommap_{STAMP}.json")
REPS, NGEN = 5, 64
CPU_MEAN_CEIL, CPU_MAX_CEIL, GATE_TRIES = 20.0, 45.0, 40


def ps(cmd, timeout=90):
    try:
        return subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                              capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def free_ram_gb():
    o = ps("(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory")
    try:
        return round(float(o) * 1024 / 1e9, 2)
    except ValueError:
        return None


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(OUT.replace(".json", ".log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def gate():
    """No llama-* running, CPU quiet across a window. Fail closed on unreadable."""
    for a in range(1, GATE_TRIES + 1):
        procs = ps("Get-Process | Where-Object { $_.ProcessName -like 'llama*' } | "
                   "Select-Object -ExpandProperty Id")
        if procs.strip():
            log(f"  gate {a}: llama-* running -> wait"); time.sleep(60); continue
        vals = []
        for tok in ps("1..12 | ForEach-Object { (Get-CimInstance Win32_Processor | "
                      "Measure-Object -Property LoadPercentage -Average).Average; "
                      "Start-Sleep -Seconds 5 }", timeout=180).split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass
        if not vals:
            log(f"  gate {a}: cpu unreadable -> treating as BUSY"); time.sleep(30); continue
        mean, mx = sum(vals) / len(vals), max(vals)
        if mean > CPU_MEAN_CEIL or mx > CPU_MAX_CEIL:
            log(f"  gate {a}: cpu mean {mean:.1f}% max {mx:.1f}% -> wait"); continue
        log(f"  gate OPEN: cpu mean {mean:.1f}% max {mx:.1f}%, free RAM {free_ram_gb()} GB")
        return True
    log("  gate NEVER OPENED - aborting rather than measuring a busy box")
    return False


def arm(model, mmap_on, tag):
    """One arm = REPS independent invocations, so each rep re-enters the residency race.

    NOT `-r REPS` inside one process: that would reuse a single already-resident mapping and
    measure the thing this experiment is trying to vary.
    """
    toks, frees = [], []
    for i in range(REPS):
        frees.append(free_ram_gb())
        p = subprocess.run([BENCH, "-m", model, "-ngl", "0", "-t", "4", "-n", str(NGEN),
                            "-p", "0", "-r", "1", "-mmp", "1" if mmap_on else "0", "-o", "json"],
                           capture_output=True, text=True, errors="replace")
        out = p.stdout + p.stderr
        try:
            toks.append(json.loads(out[out.index("["):out.rindex("]") + 1])[0]["avg_ts"])
        except Exception:
            toks.append(None)
        if "out of memory" in out.lower():
            log(f"  {tag}: OOM on rep {i}"); return {"tag": tag, "oom": True, "tok": toks}
    ok = [t for t in toks if t]
    r = {"tag": tag, "tok": [round(t, 4) for t in ok], "free_ram_gb": frees, "oom": False,
         "median": round(statistics.median(ok), 4) if ok else None,
         "spread": round(max(ok) / min(ok), 3) if len(ok) > 1 and min(ok) > 0 else None}
    log(f"  {tag:14} median {r['median']} tok/s  spread {r['spread']}x  "
        f"free RAM {min(x for x in frees if x)}-{max(x for x in frees if x)} GB")
    return r


def main():
    os.makedirs(DATA, exist_ok=True)
    log("=" * 66)
    log(f"U-37 quiet-box re-run  stamp={STAMP}  reps={REPS}  n={NGEN}")
    for m in (BIG, SMALL):
        if not os.path.isfile(m):
            log(f"ABORT: missing {m}"); return 2
    if not gate():
        return 3

    res = {"stamp": STAMP, "reps": REPS, "n_gen": NGEN, "arms": {}}
    log("pre-warming (the load path is not under test)")
    for m in (BIG, SMALL):
        subprocess.run([BENCH, "-m", m, "-ngl", "0", "-t", "4", "-n", "4", "-p", "0", "-r", "1"],
                       capture_output=True, text=True)
    log("BIG 13.7 GB (~86% of RAM):")
    res["arms"]["big_mmap"] = arm(BIG, True, "big_mmap")
    res["arms"]["big_nommap"] = arm(BIG, False, "big_nommap")
    log("SMALL 4.4 GB (~28% of RAM) - the control:")
    res["arms"]["small_mmap"] = arm(SMALL, True, "small_mmap")
    res["arms"]["small_nommap"] = arm(SMALL, False, "small_nommap")

    a = res["arms"]
    def med(k):
        return a[k].get("median")
    res["K1_R_big"] = round(med("big_nommap") / med("big_mmap"), 3) if med("big_mmap") else None
    res["K2_R_small"] = round(med("small_nommap") / med("small_mmap"), 3) if med("small_mmap") else None
    res["K3_big_spread"], res["K3_control_spread"] = a["big_mmap"].get("spread"), a["small_mmap"].get("spread")
    lo = min(x for x in a["big_mmap"]["free_ram_gb"] if x) if a["big_mmap"]["free_ram_gb"] else None
    res["K4_min_free_ram_big"] = lo

    k1 = (res["K1_R_big"] or 0) >= 1.20
    k2 = (res["K2_R_small"] or 9) <= 1.10
    k3 = (res["K3_big_spread"] or 0) > (res["K3_control_spread"] or 0)
    k4 = lo is not None and lo <= 4.0          # big arm really crowded RAM
    log("")
    log(f"  K-1 R_big   {res['K1_R_big']}  (>=1.20) {'PASS' if k1 else 'FAIL'}")
    log(f"  K-2 R_small {res['K2_R_small']}  (<=1.10) {'PASS' if k2 else 'FAIL'}")
    log(f"  K-3 spread big {res['K3_big_spread']} vs control {res['K3_control_spread']} "
        f"(big must EXCEED) {'PASS' if k3 else 'FAIL'}")
    log(f"  K-4 min free RAM during big {lo} GB (<=4.0) {'PASS' if k4 else 'FAIL'}")

    if a["big_nommap"].get("oom"):
        res["verdict"] = "KR: --no-mmap OOMed at 13.7 GB. That IS the result - the lever is infeasible here."
    elif all((k1, k2, k3, k4)):
        res["verdict"] = (f"U-37 CONFIRMED - the mmap decode penalty is RESIDENCY PRESSURE. "
                          f"{res['K1_R_big']}x at 86% of RAM, {res['K2_R_small']}x at 28%, with the "
                          f"big arm more variable than the control as predicted. --no-mmap is a "
                          f"real lever gated on model size relative to RAM; U-23's heuristic gate "
                          f"can become a measured one.")
    elif not k4:
        res["verdict"] = (f"UNINFORMATIVE - K-4: the big arm never ran near capacity (min free RAM "
                          f"{lo} GB). The premise is absent; no ratio may be claimed.")
    elif not k2:
        res["verdict"] = (f"U-37 REFUTED - K-2: the SMALL control also shows a gap "
                          f"({res['K2_R_small']}x). Size is not the variable; the cost is mmap "
                          f"itself, and the residency-pressure story is wrong.")
    elif not k3:
        res["verdict"] = (f"U-37 REFUTED - K-3: the big mmap arm is no more variable than the "
                          f"control ({res['K3_big_spread']} vs {res['K3_control_spread']}). The "
                          f"hypothesis predicts eviction-driven instability and it is absent.")
    else:
        res["verdict"] = (f"NOT CONFIRMED - K-1 {res['K1_R_big']} below 1.20. The pilot's 1.659x "
                          f"does not reproduce on a quiet box; contention likely produced it.")
    log("")
    log(res["verdict"])
    json.dump(res, open(OUT, "w"), indent=1)
    log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
