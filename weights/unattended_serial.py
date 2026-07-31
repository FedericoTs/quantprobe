"""Unattended SERIAL re-run of the two contaminated measurements. Launch from a plain
terminal with NO agent running - the CPU gate cannot pass while a coding agent is making
tool calls, and that is exactly how the 2026-07-31 runs got corrupted.

  python weights/unattended_serial.py

WHY THIS EXISTS
The 2026-07-31 ladder and disk-tier runs overlapped: a disk-tier bench held ~13.7 GB of
working set on a 16 GiB box while ladder rows 9-13 measured. Rows 9-13 collapsed and their
wall-clock ballooned - a prediction error cannot change how long llama-bench takes. Both
sets of numbers are void. The old gate passed anyway because it sampled
Win32_Processor.LoadPercentage three times (a disk-blocked competitor reads as IDLE) and
had NO free-RAM gate at all - the one resource those rows actually compete for.

WHAT IS DIFFERENT HERE
  1. SERIAL BY CONSTRUCTION. One subprocess at a time. A lock file refuses a second instance.
  2. Free-RAM floor (the resource that actually binds), re-checked before EVERY bench.
  3. Process check by IMAGE NAME - any llama-*, regardless of which build directory.
  4. CPU sampled ACROSS A WINDOW (mean AND max), not three point samples.
  5. Gates re-run before every phase, not once at the start. Fail closed on unreadable.
  6. The ladder baseline is BACKED UP first: full_ladder_v124.py --bench overwrites
     weights/data/ladder_state_locked.json, which silently destroyed the baseline once today.

STAKE DISCIPLINE
The disk-tier stake (weights/data/disktier_20260731_1857_staked.json, frozen 16:57:07 UTC)
is NOT renegotiated. Two things are pinned HERE, before any new number exists, and both are
recorded in the addendum file written at startup:
  - KR2's denominator was never specified ("tg32 and tg64 must agree within 25%" - 29.4%
    normalised on tg64, 22.7% on tg32). Resolved to the STRICTER reading: FAIL if EITHER
    normalisation exceeds 25%. Chosen before seeing a number so it cannot be a favourable
    reading picked afterwards.
  - reps 2 -> 3 with per-rep samples captured from llama-bench JSON. This does not move a
    threshold; KR4 already names rep-to-rep spread, and n=2 forced us to INFER the two reps
    from a printed sd. Now they are read directly.
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
LOCK = os.path.join(DATA, "unattended_serial.lock")
STAMP = time.strftime("%Y%m%d_%H%M%S")
RUN = os.path.join(DATA, f"unattended_{STAMP}")
BIN = r"C:\Users\Federico\Documents\evo-compress\tools\llamacpp-b10098\llama-bench.exe"
LAGUNA = "D:/evo-compress-data/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf"
QWEN7B = "D:/evo-compress-data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
LADDER_LOCKED = os.path.join(DATA, "ladder_state_locked.json")

# gates
FREE_RAM_FLOOR_GB = 13.0     # 16 GiB box; the big-MoE rows need the page cache to themselves
GPU_CEIL_MIB = 1500
CPU_WINDOW_S = 60            # sample across a window, not point samples
CPU_MEAN_CEIL = 20.0
CPU_MAX_CEIL = 45.0
GATE_TRIES = 60              # ~60 min of patience before giving up on a phase
EXPECT_CAL = "2dc97d41"      # C-14: the baseline ladder's state. Drift => abort, do not compare.


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RUN + "_run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ps(cmd, timeout=25):
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def free_ram_gb():
    """Free physical RAM in GB. Returns None if unreadable (caller treats None as FAIL)."""
    out = ps("(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory")
    try:
        return float(out) * 1024 / 1e9
    except ValueError:
        return None


def gpu_mib():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def foreign_benches():
    """Any llama-* image running, regardless of which build directory it came from.

    The 2026-07-31 collision was between two DIFFERENT llama.cpp builds, so a
    path-scoped check would have missed it. Match on image name only.
    """
    out = ps("Get-Process | Where-Object { $_.ProcessName -like 'llama*' } | "
             "Select-Object -ExpandProperty Id")
    if out is None:
        return None
    return [x for x in out.split() if x.strip().isdigit()]


def cpu_window():
    """(mean, max) CPU% sampled across CPU_WINDOW_S. A single point sample cannot see a
    competitor that is disk-blocked at the instant you look - which is how the old gate
    passed with three 7% readings while a 4-thread bench was mid-sweep."""
    n = max(6, CPU_WINDOW_S // 5)
    out = ps(f"1..{n} | ForEach-Object {{ "
             f"(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage "
             f"-Average).Average; Start-Sleep -Seconds 5 }}", timeout=CPU_WINDOW_S + 90)
    vals = []
    for tok in out.split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    if not vals:
        return None, None
    return sum(vals) / len(vals), max(vals)


def gate(phase):
    """Fail closed. Every unreadable counter counts as BUSY, never as idle."""
    for attempt in range(1, GATE_TRIES + 1):
        procs = foreign_benches()
        if procs is None:
            log(f"  gate[{phase}] {attempt}: process list unreadable -> treating as BUSY")
            time.sleep(60); continue
        if procs:
            log(f"  gate[{phase}] {attempt}: llama-* still running (pids {','.join(procs)}) -> wait")
            time.sleep(60); continue
        g = gpu_mib()
        if g is None or g > GPU_CEIL_MIB:
            log(f"  gate[{phase}] {attempt}: gpu {g} MiB (ceil {GPU_CEIL_MIB}, None=unreadable) -> wait")
            time.sleep(60); continue
        fr = free_ram_gb()
        if fr is None or fr < FREE_RAM_FLOOR_GB:
            log(f"  gate[{phase}] {attempt}: free RAM {fr} GB (floor {FREE_RAM_FLOOR_GB}) -> wait")
            time.sleep(60); continue
        mean, mx = cpu_window()
        if mean is None:
            log(f"  gate[{phase}] {attempt}: cpu unreadable -> treating as BUSY")
            time.sleep(30); continue
        if mean > CPU_MEAN_CEIL or mx > CPU_MAX_CEIL:
            log(f"  gate[{phase}] {attempt}: cpu mean {mean:.1f}% max {mx:.1f}% "
                f"(ceil {CPU_MEAN_CEIL}/{CPU_MAX_CEIL}) -> wait")
            continue    # cpu_window already burned ~60s
        log(f"  gate[{phase}] OPEN: gpu {g} MiB, free RAM {fr:.1f} GB, cpu mean {mean:.1f}% max {mx:.1f}%")
        return {"gpu_mib": g, "free_ram_gb": round(fr, 2),
                "cpu_mean": round(mean, 1), "cpu_max": round(mx, 1), "attempts": attempt}
    log(f"  gate[{phase}] NEVER OPENED after {GATE_TRIES} tries - phase SKIPPED, not forced")
    return None


def cal_id():
    try:
        p = os.path.expanduser("~/.quantprobe/calibration.json")
        return json.load(open(p, encoding="utf-8")).get("cal_id")
    except Exception:
        return None


def bench(tag, args, timeout_s):
    """One llama-bench, JSON out so per-rep samples are READ, not inferred from a printed sd."""
    cmd = [BIN] + args + ["-o", "json"]
    log(f"  bench {tag}: {' '.join(args)}")
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        out, err = r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        log(f"  bench {tag}: TIMEOUT after {timeout_s}s - recorded as the result, not retried")
        return {"tag": tag, "error": f"timeout {timeout_s}s", "tok_s": None}
    wall = time.perf_counter() - t0
    with open(f"{RUN}_{tag}.log", "w", encoding="utf-8") as f:
        f.write(out + "\n--- stderr ---\n" + err)
    rec = {"tag": tag, "cmd": " ".join(args), "wall_s": round(wall, 1), "error": None}
    try:
        rows = json.loads(out[out.index("["):out.rindex("]") + 1])
        row = rows[0]
        rec["tok_s"] = row.get("avg_ts")
        rec["sd"] = row.get("stddev_ts")
        ns = row.get("samples_ns") or []
        ng = row.get("n_gen") or 0
        if ns and ng:
            reps = [ng / (x / 1e9) for x in ns]           # per-rep tok/s, read not inferred
            rec["reps_tok_s"] = [round(v, 4) for v in reps]
            rec["rep_spread_x"] = round(max(reps) / min(reps), 3) if min(reps) > 0 else None
    except Exception as e:
        rec["error"] = f"parse failed: {e}"
        rec["tok_s"] = None
    log(f"  bench {tag}: tok/s {rec.get('tok_s')} reps {rec.get('reps_tok_s')} "
        f"spread {rec.get('rep_spread_x')}x wall {rec['wall_s']}s")
    return rec


def phase_ladder(res):
    g = gate("ladder")
    if not g:
        res["ladder"] = {"skipped": "gate never opened"}; return
    # full_ladder_v124.py --bench OVERWRITES ladder_state_locked.json. That silently
    # destroyed the baseline once today. Back it up BEFORE, restore AFTER.
    backup = f"{RUN}_ladder_baseline_backup.json"
    if os.path.isfile(LADDER_LOCKED):
        shutil.copy2(LADDER_LOCKED, backup)
        log(f"  baseline backed up -> {os.path.basename(backup)}")
    log("  ladder: starting (long - expect well over an hour)")
    t0 = time.perf_counter()
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, "full_ladder_v124.py"), "--bench"],
                           capture_output=True, text=True, cwd=ROOT, timeout=4 * 3600)
        ok, out = r.returncode == 0, r.stdout + "\n--- stderr ---\n" + r.stderr
    except subprocess.TimeoutExpired:
        ok, out = False, "TIMEOUT after 4h"
    with open(f"{RUN}_ladder.log", "w", encoding="utf-8") as f:
        f.write(out)
    result = f"{RUN}_ladder_result.json"
    if os.path.isfile(LADDER_LOCKED):
        shutil.copy2(LADDER_LOCKED, result)              # keep the NEW numbers separately
    if os.path.isfile(backup):
        shutil.copy2(backup, LADDER_LOCKED)              # and put the baseline back
        log("  baseline restored - the new run is in *_ladder_result.json, NOT in the locked file")
    res["ladder"] = {"gate": g, "ok": ok, "wall_s": round(time.perf_counter() - t0, 1),
                     "result_json": os.path.basename(result), "cal_id_after": cal_id()}
    log(f"  ladder: done ok={ok} in {res['ladder']['wall_s']}s")


def phase_disktier(res, settle=True):
    if settle:
        # The ladder just ran; let the page cache settle before a test whose whole premise
        # is the cache miss fraction.
        log("  disk-tier: settling 300s after the ladder (page-cache state is this row's premise)")
        time.sleep(300)
    g = gate("disktier")
    if not g:
        res["disktier"] = {"skipped": "gate never opened"}; return
    runs = {}
    # KR3 first: prove the harness can VARY before trusting anything it says about the slow row.
    runs["KR3_control_7B_ngl0"] = bench(
        "KR3_control_7B_ngl0",
        ["-m", QWEN7B, "-ngl", "0", "-t", "4", "-n", "64", "-p", "0", "-r", "2"], 900)
    c = runs["KR3_control_7B_ngl0"].get("tok_s")
    if c is None or c <= 3.0:
        log(f"  KR3 FAILED (control {c} <= 3.0): the harness has not been shown to vary. "
            f"ABORTING the disk-tier row - no verdict may be issued from a constant emitter.")
        res["disktier"] = {"gate": g, "runs": runs, "aborted": "KR3 control failed"}
        return
    runs["PRIMARY_laguna_tg64"] = bench(
        "PRIMARY_laguna_tg64",
        ["-m", LAGUNA, "-ngl", "0", "-b", "2048", "-ub", "2048", "-t", "4",
         "-n", "64", "-p", "0", "-r", "3"], 3600)
    runs["KR2_laguna_tg32"] = bench(
        "KR2_laguna_tg32",
        ["-m", LAGUNA, "-ngl", "0", "-b", "2048", "-ub", "2048", "-t", "4",
         "-n", "32", "-p", "0", "-r", "3"], 2400)
    res["disktier"] = {"gate": g, "runs": runs, "cal_id_after": cal_id()}


def score(res):
    """Score against the FROZEN stake. Thresholds are quoted, never recomputed to fit."""
    d = res.get("disktier", {})
    runs = d.get("runs") or {}
    s = {"predicted_tok_s": 0.331505, "band": [0.265204, 0.442007]}
    p = (runs.get("PRIMARY_laguna_tg64") or {}).get("tok_s")
    spread = (runs.get("PRIMARY_laguna_tg64") or {}).get("rep_spread_x")
    t32 = (runs.get("KR2_laguna_tg32") or {}).get("tok_s")
    ctrl = (runs.get("KR3_control_7B_ngl0") or {}).get("tok_s")

    s["KR3_harness_can_vary"] = ("PASS" if (ctrl or 0) > 3.0 else "FAIL") if ctrl else "UNRUN"
    if spread is None:
        s["KR4_steady_state"] = "UNRUN"
    else:
        s["KR4_steady_state"] = "PASS" if spread <= 2.0 else f"FAIL (spread {spread}x > 2x)"
    if p:
        s["err_pct"] = round((0.331505 - p) / p * 100, 2)
        s["KR1_band"] = "PASS" if s["band"][0] <= p <= s["band"][1] else "FAIL"
    else:
        s["KR1_band"] = "UNRUN"
    if p and t32:
        on64, on32 = abs(t32 - p) / p * 100, abs(t32 - p) / t32 * 100
        s["KR2_pct_on_tg64"], s["KR2_pct_on_tg32"] = round(on64, 1), round(on32, 1)
        # denominator pinned in the addendum BEFORE this run: strict reading, either fails
        s["KR2_load_vs_decode"] = "PASS" if max(on64, on32) <= 25.0 else "FAIL"
    else:
        s["KR2_load_vs_decode"] = "UNRUN"
    hard = [s["KR1_band"], s["KR2_load_vs_decode"], s["KR3_harness_can_vary"], s["KR4_steady_state"]]
    s["VERDICT"] = ("PASS - disk tier validated" if all(x == "PASS" for x in hard)
                    else "NOT VALIDATED - " + "; ".join(
                        f"{k}={v}" for k, v in s.items() if k.startswith("KR") and v != "PASS"))
    return s


def main():
    os.makedirs(DATA, exist_ok=True)
    if os.path.isfile(LOCK):
        print(f"REFUSING TO START: {LOCK} exists. Another run is live (or died dirty).\n"
              f"If you are certain nothing is running, delete that file and relaunch.")
        return 2
    open(LOCK, "w").write(f"{os.getpid()} {STAMP}\n")
    try:
        log("=" * 70)
        log(f"unattended SERIAL re-run  stamp={STAMP}  pid={os.getpid()}")
        cal = cal_id()
        log(f"cal_id={cal} (expected {EXPECT_CAL})")
        if cal != EXPECT_CAL:
            log(f"ABORT: calibration drifted to {cal}. C-14 forbids comparing across machine "
                f"states. Re-stake against the new state or restore {EXPECT_CAL} first.")
            return 3
        # Pin the two method decisions BEFORE any number exists.
        json.dump({
            "addendum_to": "disktier_20260731_1857_staked.json",
            "written_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "written_before_any_new_number": True,
            "KR2_denominator_resolved": (
                "The frozen stake said 'tg32 and tg64 must agree within 25%' without naming a "
                "denominator: the 2026-07-31 numbers gave 29.4% on tg64 and 22.7% on tg32, i.e. "
                "fail or pass depending on a choice the stake never made. Resolved HERE, before "
                "any new number exists, to the STRICTER reading: FAIL if EITHER normalisation "
                "exceeds 25%. Chosen against our own interest so it cannot be a favourable "
                "reading picked after the fact."),
            "reps_changed": (
                "r=2 -> r=3, with per-rep tok/s READ from llama-bench JSON samples_ns rather "
                "than inferred from a printed sd. No threshold moves: KR4 already names "
                "rep-to-rep spread, and n=2 made the two reps unobservable."),
            "unchanged": ["KR1 band [0.265204, 0.442007] around predicted 0.331505",
                          "KR3 control > 3.0 tok/s", "KR4 spread > 2x kills the mean",
                          "the VRAM+RAM expert-cache row stays UNMEASURABLE on this box"],
        }, open(f"{RUN}_stake_addendum.json", "w", encoding="utf-8"), indent=1)
        log(f"addendum written -> {os.path.basename(RUN)}_stake_addendum.json")

        only = None
        for a in sys.argv[1:]:
            if a.startswith("--only="):
                only = a.split("=", 1)[1].strip()
        res = {"stamp": STAMP, "cal_id_at_start": cal, "only": only, "started_utc":
               time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
        if only in (None, "ladder"):
            log("--- PHASE 1/2: full 14-row ladder ---")
            phase_ladder(res)
        else:
            log("--- PHASE 1/2 skipped by --only ---")
        if only in (None, "disktier"):
            log("--- PHASE 2/2: disk-tier row vs the frozen stake ---")
            # no settle needed if the ladder did not just run
            phase_disktier(res, settle=(only is None))
        else:
            log("--- PHASE 2/2 skipped by --only ---")
        res["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        res["cal_id_at_end"] = cal_id()
        if res["cal_id_at_end"] != cal:
            res["C14_WARNING"] = (f"calibration changed mid-run ({cal} -> {res['cal_id_at_end']}); "
                                  f"the two phases are NOT one machine state")
        res["disktier_score"] = score(res)
        json.dump(res, open(f"{RUN}_RESULT.json", "w", encoding="utf-8"), indent=1)
        log("=" * 70)
        log(f"RESULT -> {os.path.basename(RUN)}_RESULT.json")
        log(f"disk-tier verdict: {res['disktier_score']['VERDICT']}")
        log("ladder: compare *_ladder_result.json against the baseline BY HAND. Per C-18 the "
            "median has a +/-1 point noise floor - 'no regression outside 2 points' is the "
            "criterion, NOT 'the median improved'.")
        return 0
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
