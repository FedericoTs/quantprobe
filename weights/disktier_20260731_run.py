"""Disk-tier ladder row: gates -> falsifiers -> primary bench. Stake is already written.

Reads  weights/data/disktier_20260731_1857_staked.json  (predictions + kill rules, frozen)
Writes weights/data/disktier_20260731_1857_measured.json + per-run logs.

Nothing in here computes a prediction. The staked numbers are read, never recomputed.
"""
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
STAKE = os.path.join(DATA, "disktier_20260731_1857_staked.json")
OUT = os.path.join(DATA, "disktier_20260731_1857_measured.json")
RUNLOG = os.path.join(DATA, "disktier_20260731_1857_run.log")
BIN = r"<repo>\tools\llamacpp-b10098\llama-bench.exe"
D = "D:/evo-compress-data/gguf"
LAGUNA = D + "/Laguna-S-2.1-UD-Q2_K_XL.gguf"
SMALL = D + "/Qwen2.5-7B-Instruct-Q4_K_M.gguf"


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(RUNLOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def gpu_used():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30).stdout
        return int(o.strip().splitlines()[0])
    except Exception:
        return -1


def cpu_pct():
    try:
        o = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | "
             "Measure-Object -Property LoadPercentage -Average).Average"],
            capture_output=True, text=True, timeout=60).stdout
        m = re.search(r"\d+", o)
        return int(m.group(0)) if m else 100      # unreadable == busy (fail-safe)
    except Exception:
        return 100


def gate(tag, need=3, limit=25, tries_max=60, sleep_s=20):
    subprocess.run(["taskkill", "/IM", "llama-bench.exe", "/F"], capture_output=True)
    subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"], capture_output=True)
    time.sleep(3)
    g = gpu_used()
    if not (0 <= g < 1500):
        log("ABORT %s: GPU busy/unreadable (%s MiB)" % (tag, g))
        return None
    ok = 0
    for i in range(tries_max):
        c = cpu_pct()
        ok = ok + 1 if c <= limit else 0
        log("  gate %s sample %d: cpu %d%% (streak %d/%d), gpu %d MiB"
            % (tag, i + 1, c, ok, need, g))
        if ok >= need:
            return {"gpu_mib": g, "cpu_pct": c, "samples": i + 1}
        time.sleep(sleep_s)
    log("ABORT %s: never idle in %d samples" % (tag, tries_max))
    return None


def start_disk_counter(path):
    """Sample physical-disk read bandwidth for the duration of a bench.

    This is the independent witness that the row really is streaming from disk: the
    planner claims 93% of the token is disk I/O at 0.47 GB/s, and a row that quietly
    served itself from the page cache would show a read rate near zero.
    """
    # Get-Counter '\PhysicalDisk(...)' is LOCALIZED and this box is it-IT, where that path
    # throws "Impossibile trovare l'oggetto specificato". Win32_PerfFormattedData_PerfDisk_
    # PhysicalDisk carries English property names on every locale, and it reports per-volume,
    # so we can watch D: (where the GGUF lives) rather than the whole machine.
    ps = ("$ErrorActionPreference='SilentlyContinue';"
          "while($true){"
          "$d=Get-CimInstance Win32_PerfFormattedData_PerfDisk_PhysicalDisk | "
          "Where-Object {$_.Name -like '*D:*'};"
          "$t=Get-CimInstance Win32_PerfFormattedData_PerfDisk_PhysicalDisk | "
          "Where-Object {$_.Name -eq '_Total'};"
          "\"$([int](Get-Date -UFormat %%s)) $($d.DiskReadBytesPersec) $($t.DiskReadBytesPersec)\""
          " | Out-File -Append -Encoding ascii '%s';"
          "Start-Sleep -Seconds 5}" % path.replace("\\", "\\\\"))
    return subprocess.Popen(["powershell", "-NoProfile", "-Command", ps],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def bench(tag, model, args, n_gen, reps, timeout):
    logf = os.path.join(DATA, "disktier_20260731_1857_%s.log" % tag)
    diskf = os.path.join(DATA, "disktier_20260731_1857_%s_diskcounter.log" % tag)
    cmd = [BIN, "-m", model] + args + ["-n", str(n_gen), "-p", "0", "-r", str(reps),
                                       "--progress"]
    with open(logf, "w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
    log("RUN %s: %s" % (tag, " ".join(cmd)))
    ctr = start_disk_counter(diskf)
    t0 = time.time()
    err = None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        so, se = r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        so = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        se = (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        err = "timeout after %ds" % timeout
    finally:
        ctr.terminate()
    wall = round(time.time() - t0, 1)
    with open(logf, "a", encoding="utf-8") as f:
        f.write(so + "\n--- stderr ---\n" + se + "\n--- wall %ss err=%s ---\n" % (wall, err))
    tok_s = sd = None
    m = re.search(r"tg%d\s*\|\s*([0-9.]+)\s*[^0-9|]*([0-9.]+)?" % n_gen, so)
    if m:
        tok_s = float(m.group(1))
        sd = float(m.group(2)) if m.group(2) else None
    else:
        err = err or "parse-fail"
    log("  -> %s tok/s (sd %s) wall %ss err=%s" % (tok_s, sd, wall, err))
    return {"tag": tag, "cmd": " ".join(cmd), "n_gen": n_gen, "reps": reps,
            "tok_s": tok_s, "sd": sd, "wall_s": wall, "error": err,
            "log": os.path.basename(logf), "disk_counter_log": os.path.basename(diskf)}


def disk_stats(tag):
    # The 2026-07-31 run reported samples=0 / "disk counter unreadable" for all three benches
    # while the logs held 83, 28 and 4 populated rows. The sampler emits an EMPTY timestamp
    # field on this box (PS 5.1, it-IT), so rows arrive as "<read> <write>" - two fields - and
    # the `== 3` test dropped every one of them. The probe then printed a sentence that reads
    # like careful restraint. A silent fallback wearing a disclosure's clothes is worse than a
    # crash: it cost the only independent witness that bytes were actually moving off D:.
    # Fix: parse from the RIGHT (read and write are always the last two columns, timestamp
    # optional), and never claim "unreadable" while unparsed rows exist - say which it was.
    p = os.path.join(DATA, "disktier_20260731_1857_%s_diskcounter.log" % tag)
    dvals, tvals, seen, unparsed = [], [], 0, 0
    if os.path.isfile(p):
        for ln in open(p, encoding="ascii", errors="replace"):
            parts = ln.split()
            if not parts:
                continue
            seen += 1
            if len(parts) < 2:
                unparsed += 1
                continue
            try:
                dvals.append(float(parts[-2]))
                tvals.append(float(parts[-1]))
            except ValueError:
                unparsed += 1
    if not dvals:
        return {"samples": 0, "rows_seen": seen, "rows_unparsed": unparsed,
                "note": ("no counter file at all - NOT substituted with an estimate"
                         if seen == 0 else
                         "counter file has %d rows but NONE parsed - this is a PARSER defect, "
                         "not an absent counter; do not read it as a clean negative" % seen)}

    def stats(v):
        s = sorted(v)
        return {"mean_gbs": round(sum(v) / len(v) / 1e9, 3),
                "median_gbs": round(s[len(s) // 2] / 1e9, 3),
                "p90_gbs": round(s[min(int(len(s) * 0.9), len(s) - 1)] / 1e9, 3),
                "max_gbs": round(s[-1] / 1e9, 3)}
    return {"samples": len(dvals), "D_drive": stats(dvals), "all_disks": stats(tvals)}


def main():
    stake = json.load(open(STAKE, encoding="utf-8"))
    res = {"stake_file": os.path.basename(STAKE), "runs": {},
           "started_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}

    # C-14: the machine state must still be the one the prediction was made under.
    from quantprobe.calibrate import load as _cl
    cal, _ = _cl()
    cur = (cal or {}).get("cal_id", "uncalibrated")
    res["cal_id_at_bench"] = cur
    if cur != stake["machine_state"]["cal_id"]:
        log("ABORT: calibration drifted %s -> %s; staked prediction void (C-14)"
            % (stake["machine_state"]["cal_id"], cur))
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
        sys.exit(2)

    g = gate("pre")
    if not g:
        res["aborted"] = "idle gate failed before any bench"
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
        sys.exit(3)
    res["gate_pre"] = g

    # ---- KR3: prove the harness can vary before trusting it on the slow row ----
    log("KR3 falsifier: same harness on a RAM-resident 7B, must report > 3.0 tok/s")
    res["runs"]["KR3_control_7B_ngl0"] = bench(
        "KR3_control_7B_ngl0", SMALL, ["-ngl", "0", "-t", "4"], 64, 1, 1800)
    res["runs"]["KR3_control_7B_ngl0"]["disk"] = disk_stats("KR3_control_7B_ngl0")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)

    # ---- PRIMARY: the staked disk-tier row ----
    g = gate("primary")
    if not g:
        res["aborted"] = "idle gate failed before the primary bench"
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
        sys.exit(3)
    res["gate_primary"] = g
    la = ["-ngl", "0", "-b", "2048", "-ub", "2048", "-t", "4"]
    res["runs"]["PRIMARY_laguna_tg64"] = bench(
        "PRIMARY_laguna_tg64", LAGUNA, la, 64, 2, 7200)
    res["runs"]["PRIMARY_laguna_tg64"]["disk"] = disk_stats("PRIMARY_laguna_tg64")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)

    # ---- KR2: n-invariance. Load time leaking into timing would drag tg32 below tg64 ----
    res["runs"]["KR2_laguna_tg32"] = bench(
        "KR2_laguna_tg32", LAGUNA, la, 32, 2, 5400)
    res["runs"]["KR2_laguna_tg32"]["disk"] = disk_stats("KR2_laguna_tg32")

    res["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    log("DONE -> %s" % OUT)


if __name__ == "__main__":
    main()
