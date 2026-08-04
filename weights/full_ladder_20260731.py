"""Full E2E ladder re-run, 2026-07-31, under FRESH calibration (C-14).

Phases (run strictly in order; stake writes predictions BEFORE any bench):
  python weights/full_ladder_20260731.py stake     -> weights/data/ladder_20260731_staked.json
  python weights/full_ladder_20260731.py bench     -> benches every staked row SERIALLY,
                                                      unique log per row, results appended
  python weights/full_ladder_20260731.py score     -> weights/data/ladder_20260731_scored.json

Protocol: one cal_id for the whole comparison (C-14); kill rules staked in the JSON header
before the first bench; a failed/OOM row is RECORDED, never re-configured mid-run.
"""
import json
import os
import re
import subprocess
import sys
import time

D = "D:/evo-compress-data/gguf"
BIN = r"C:\Users\Federico\Documents\evo-compress\tools\llamacpp-b10098\llama-bench.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
STAKED = os.path.join(HERE, "data", "ladder_20260731_staked.json")
SCORED = os.path.join(HERE, "data", "ladder_20260731_scored.json")
RUNLOG = os.path.join(HERE, "data", "ladder_20260731_run.log")

MODELS = [
    ("Qwen2.5-0.5B Q8_0", "Qwen2.5-0.5B-Instruct-Q8_0.gguf"),
    ("Qwen3-0.6B Q8_0", "Qwen3-0.6B-Q8_0.gguf"),
    ("Qwen3.5-4B Q4_K_M", "Qwen3.5-4B-Q4_K_M.gguf"),
    ("Qwen2.5-7B IQ4_NL", "Qwen2.5-7B-Instruct.i1-IQ4_NL.gguf"),
    ("Qwen2.5-7B Q4_K_M", "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    ("gemma4-12B depth-aware", "gemma4-12b-B-late12.gguf"),
    ("DS-Lite 16B IQ2_XS", "DeepSeek-Coder-V2-Lite-Base-IQ2_XS.gguf"),
    ("Qwen2.5-14B Q4_K_M", "Qwen2.5-14B-Instruct-Q4_K_M.gguf"),
    ("DS-Lite 16B Q4_K_M", "DeepSeek-Coder-V2-Lite-Base-Q4_K_M.gguf"),
    ("Qwen3-30B-A3B Q2_K", "Qwen3-30B-A3B-Q2_K.gguf"),
    ("Qwen3-Coder-30B Q2_K_L", "Qwen3-Coder-30B-A3B-Instruct-Q2_K_L.gguf"),
    ("Qwen3.5-35B APEX-Mini", "Qwen3.5-35B-A3B-APEX-Mini.gguf"),
    ("Qwen3.6-35B Q2_K_XL", "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"),
    ("Qwen3.6-35B APEX-MTP-Nano", "Qwen3.6-35B-A3B-APEX-MTP-I-Nano.gguf"),
]

# The locked-ladder errors this run is compared against (cal a19aeee4, old model terms).
OLD_ERR = {
    "Qwen2.5-0.5B Q8_0": -18.6, "Qwen3-0.6B Q8_0": -2.6, "Qwen3.5-4B Q4_K_M": 8.7,
    "Qwen2.5-7B IQ4_NL": 8.8, "Qwen2.5-7B Q4_K_M": -5.0, "gemma4-12B depth-aware": 0.0,
    "DS-Lite 16B IQ2_XS": 22.7, "Qwen2.5-14B Q4_K_M": -5.5, "DS-Lite 16B Q4_K_M": -19.1,
    "Qwen3-30B-A3B Q2_K": -10.2, "Qwen3-Coder-30B Q2_K_L": -10.0,
    "Qwen3.5-35B APEX-Mini": -3.8, "Qwen3.6-35B Q2_K_XL": 11.3,
    "Qwen3.6-35B APEX-MTP-Nano": 8.5,
}

PP_SPOTCHECK_ROW = "Qwen3-30B-A3B Q2_K"

KILL_RULES = {
    "staked_utc": None,  # filled at stake time
    "KR1_median": "median |err_pct| over scoreable rows must be <= 8.8% (locked-ladder value) "
                  "to claim parity; a miss is published at equal prominence",
    "KR2_band": "all-in-VRAM rows: measured >= 0.90 x predicted (one-sided floor, the tool's own "
                "printed band) => err_pct <= +11.1%; tiered rows (split/hybrid/CPU/disk): "
                "|err_pct| <= 25% (the tool's printed +/-25% estimate band)",
    "KR3_ub4096_pp": "Qwen3-30B experts->RAM alternative, r=3: pp2048(ub4096)/pp2048(ub2048)-1 "
                     "expected ~+4.3% (L-26 ref 360.76 vs 345.89). REPRODUCED if delta in "
                     "[+1%, +8%]; REFUTED if <= 0; OUTSIDE-BAND otherwise",
    "KR4_bias": "state whether the locked ladder's bias pattern (small-dense over-measured: "
                "0.5B -18.6; DS-Lite Q4KM -19.1; DS-Lite IQ2_XS +22.7) survives the new terms",
    "err_convention": "err_pct = (predicted - measured) / measured * 100  (same as locked ladder)",
}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RUNLOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def gpu_mem_used():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def wait_gpu_idle(limit_mib=1500, timeout_s=120):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        used = gpu_mem_used()
        if 0 <= used < limit_mib:
            return used
        time.sleep(5)
    return gpu_mem_used()


def cal_state():
    from quantprobe.calibrate import load as _cl
    cal, age = _cl()
    return (cal or {}).get("cal_id", "uncalibrated"), cal


def plan(path):
    r = subprocess.run([sys.executable, "-m", "quantprobe", "plan", "--gguf", path],
                       capture_output=True, text=True, timeout=600,
                       encoding="utf-8", errors="replace")
    txt = r.stdout
    lines = txt.splitlines()
    win = next((l for l in lines if l.strip().startswith("*")), "")
    m = re.search(r"([0-9.]+) tok/s\s+(.*?)(?:\s+\[|$)", win)
    emit = next((l.split("run it:", 1)[1].strip() for l in lines if "run it:" in l), "")
    binding = next((l.strip() for l in lines if "binding constraint:" in l), "")
    alt = None
    for l in lines:
        if "Use `" in l:
            mm = re.search(r"Use `([^`]+)`", l)
            if mm:
                alt = mm.group(1).strip()
            break
    band = ("floor: measured >= 0.90 x predicted (one-sided, printed by the tool)"
            if m and m.group(2).strip() == "all in VRAM"
            else "+/-25% (printed eta-band estimate)")
    return dict(predicted=float(m.group(1)) if m else None,
                placement=m.group(2).strip() if m else "?",
                emit=emit, binding=binding, alt_emit=alt, band=band,
                ub4096_in_output=("-ub 4096" in txt),
                stdout=lines)


def stake():
    cal_id, cal = cal_state()
    ambient = gpu_mem_used()
    log(f"STAKE under cal_id {cal_id} (ram {cal.get('ram_bw_measured')} GB/s, "
        f"disk {cal.get('disk_bw_measured')} GB/s, ambient GPU {ambient} MiB)")
    kills = dict(KILL_RULES)
    kills["staked_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    doc = {"cal_id": cal_id, "calibration": cal, "ambient_gpu_mib": ambient,
           "bench_bin": BIN, "kill_rules": kills, "old_errors": OLD_ERR, "rows": []}
    for name, fn in MODELS:
        path = os.path.join(D, fn).replace("\\", "/")
        if not os.path.isfile(path):
            log(f"  SKIP (missing): {name}")
            doc["rows"].append({"name": name, "file": fn, "status": "missing"})
            continue
        p = plan(path)
        row = {"name": name, "file": fn, "cal_id": cal_id, "status": "staked", **p}
        doc["rows"].append(row)
        log(f"  {name:28s} predicted {p['predicted']!s:>6s}  {p['placement']}"
            f"  [ub4096 in output: {p['ub4096_in_output']}]")
        json.dump(doc, open(STAKED, "w", encoding="utf-8"), indent=1)
    log(f"staked {sum(1 for r in doc['rows'] if r['status'] == 'staked')} rows -> {STAKED}")


def flags_from_emit(emit):
    """Emitted llama-server command -> llama-bench args. Unlike v124 this KEEPS -b/-ub:
    the emitted command is the config the prediction targets, so the bench honors it."""
    out, toks = [], emit.split()
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "-ngl":
            out += ["-ngl", toks[i + 1]]; i += 2
        elif t == "-ot":
            out += ["-ot", toks[i + 1].strip('"')]; i += 2
        elif t == "--no-mmap":
            out += ["-mmp", "0"]; i += 1
        elif t == "--threads":
            out += ["-t", toks[i + 1]]; i += 2
        elif t == "-b":
            out += ["-b", toks[i + 1]]; i += 2
        elif t == "-ub":
            out += ["-ub", toks[i + 1]]; i += 2
        else:
            i += 1
    if "-t" not in out:
        out += ["-t", "4"]
    return out


def run_bench(path, args, metric, reps, logfile, timeout=1800):
    cmd = [BIN, "-m", path] + args + (["-n", "64", "-p", "0"] if metric == "tg64"
                                      else ["-n", "0", "-p", "2048"]) + ["-r", str(reps)]
    with open(logfile, "w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write("TIMEOUT\n")
        return None, None, "timeout"
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    m = re.search(re.escape(metric) + r"\s*\|\s*([0-9.]+)(?:\s*\S\s*([0-9.]+))?", r.stdout)
    if not m:
        return None, None, "parse-fail (see log)"
    return float(m.group(1)), (float(m.group(2)) if m.group(2) else None), None


def bench():
    doc = json.load(open(STAKED, encoding="utf-8"))
    cal_id, _ = cal_state()
    if cal_id != doc["cal_id"]:
        log(f"ABORT: calibration drifted ({doc['cal_id']} -> {cal_id}); staked predictions void")
        sys.exit(2)
    for idx, row in enumerate(doc["rows"]):
        if row.get("status") != "staked":
            continue
        if row.get("measured") is not None or row.get("bench_error"):
            continue  # resumable: already done
        slug = re.sub(r"[^A-Za-z0-9]+", "_", row["name"])[:34]
        path = os.path.join(D, row["file"]).replace("\\", "/")
        used = wait_gpu_idle()
        if used >= 1500:
            row["bench_error"] = f"gpu-not-idle ({used} MiB)"
            log(f"  {row['name']}: SKIPPED, GPU not idle ({used} MiB)")
            json.dump(doc, open(STAKED, "w", encoding="utf-8"), indent=1)
            continue
        args = flags_from_emit(row["emit"])
        lf = os.path.join(HERE, "data", f"ladder_20260731_bench_row{idx:02d}_{slug}.log")
        log(f"  bench row {idx:02d} {row['name']} (ambient {used} MiB): tg64 r=2 ...")
        t0 = time.time()
        val, std, err = run_bench(path, args, "tg64", 2, lf)
        row["bench_s"] = round(time.time() - t0)
        row["bench_log"] = os.path.basename(lf)
        if err:
            row["bench_error"] = err
            log(f"    -> FAILED: {err}")
        else:
            row["measured"] = val
            row["measured_std"] = std
            row["err_pct"] = round((row["predicted"] - val) / val * 100, 1)
            log(f"    -> measured {val:.2f} +/- {std if std is not None else 0:.2f} "
                f"({row['err_pct']:+.1f}%)  [{row['bench_s']}s]")
        json.dump(doc, open(STAKED, "w", encoding="utf-8"), indent=1)

        if row["name"] == PP_SPOTCHECK_ROW and not row.get("bench_error"):
            alt = row.get("alt_emit")
            if not alt:
                row["pp_spotcheck"] = {"error": "no alt emit staked"}
            else:
                base = re.sub(r"\s*-b \d+ -ub \d+", "", alt)
                spot = {"alt_emit": alt}
                for tag, bflags in (("ub4096", "-b 4096 -ub 4096"),
                                    ("ub2048", "-b 2048 -ub 2048")):
                    wait_gpu_idle()
                    aargs = flags_from_emit(f"{base} {bflags}")
                    plf = os.path.join(HERE, "data",
                                       f"ladder_20260731_bench_row{idx:02d}_pp_{tag}.log")
                    log(f"  pp spot-check {tag}: -p 2048 r=3 ...")
                    v, s, e = run_bench(path, aargs, "pp2048", 3, plf)
                    spot[tag] = {"pp2048": v, "std": s, "error": e,
                                 "log": os.path.basename(plf)}
                    log(f"    -> {tag}: {v if v else e}")
                if spot.get("ub4096", {}).get("pp2048") and spot.get("ub2048", {}).get("pp2048"):
                    d = spot["ub4096"]["pp2048"] / spot["ub2048"]["pp2048"] - 1
                    spot["delta_pct"] = round(d * 100, 2)
                    spot["expected_pct"] = 4.3
                    log(f"    ub4096 vs ub2048: {spot['delta_pct']:+.2f}% (expected ~+4.3%)")
                row["pp_spotcheck"] = spot
            json.dump(doc, open(STAKED, "w", encoding="utf-8"), indent=1)
    log("bench phase complete")


def score():
    doc = json.load(open(STAKED, encoding="utf-8"))
    rows, errs = [], []
    for r in doc["rows"]:
        if r.get("status") != "staked":
            rows.append({"name": r["name"], "result": r.get("status")})
            continue
        out = {"name": r["name"], "predicted": r.get("predicted"),
               "placement": r.get("placement"), "binding": r.get("binding"),
               "band": r.get("band"), "old_err_pct": OLD_ERR.get(r["name"]),
               "emit": r.get("emit")}
        if r.get("bench_error"):
            out["result"] = "BENCH-FAIL: " + r["bench_error"]
        else:
            e = r["err_pct"]
            errs.append(abs(e))
            inside = (e <= 11.1) if r.get("placement") == "all in VRAM" else (abs(e) <= 25.0)
            out.update(measured=r["measured"], measured_std=r.get("measured_std"),
                       err_pct=e, inside_band=inside, bench_s=r.get("bench_s"),
                       bench_log=r.get("bench_log"))
        if r.get("pp_spotcheck"):
            out["pp_spotcheck"] = r["pp_spotcheck"]
        rows.append(out)
    med = sorted(errs)[len(errs) // 2] if len(errs) % 2 else \
        round((sorted(errs)[len(errs) // 2 - 1] + sorted(errs)[len(errs) // 2]) / 2, 2)
    scored = {"cal_id": doc["cal_id"], "bench_bin": doc["bench_bin"],
              "kill_rules": doc["kill_rules"], "median_abs_err_pct": round(med, 2),
              "n_scoreable": len(errs),
              "inside_band": sum(1 for r in rows if r.get("inside_band")),
              "target_median": 8.8, "rows": rows}
    json.dump(scored, open(SCORED, "w", encoding="utf-8"), indent=1)
    log(f"scored: median |err| {med:.2f}% over {len(errs)} rows "
        f"({scored['inside_band']} inside band) -> {SCORED}")
    for r in rows:
        if "err_pct" in r:
            log(f"  {r['name']:28s} pred {r['predicted']:>6} meas {r['measured']:>7.2f} "
                f"err {r['err_pct']:+6.1f}%  old {r['old_err_pct']:+6.1f}%  "
                f"band={'IN' if r['inside_band'] else 'OUT'}")
        else:
            log(f"  {r['name']:28s} {r.get('result')}")


if __name__ == "__main__":
    ph = sys.argv[1] if len(sys.argv) > 1 else ""
    if ph == "stake":
        stake()
    elif ph == "bench":
        bench()
    elif ph == "score":
        score()
    else:
        print("usage: full_ladder_20260731.py stake|bench|score")
        sys.exit(1)
