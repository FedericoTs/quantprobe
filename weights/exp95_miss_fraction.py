"""Task #52 / prereg #95 - is the disk tier fast because FEWER BYTES MOVE?

D-28 confirmed the disk bandwidth input is correct (0.452-0.459 GB/s measured, 0.47 calibrated)
and killed the access-pattern explanation. If the drive runs at the speed we think, then the
only way C-21's row can beat its prediction by 1.485x is by MOVING FEWER BYTES than the law
budgets. This measures that directly.

llama.cpp exposes no expert-routing counters (-ot and -ncmoe set PLACEMENT, not observation),
so skew cannot be watched. Its consequence can: count the bytes actually read off D: during a
decode run and divide by tokens generated.

STAKED BEFORE RUNNING (2026-08-01):
  predicted disk bytes/token = 1.2530 GB. Provenance: the FROZEN stake file
  disktier_20260731_1857_staked.json records io = 2.66599 s of a 2.86572 s token at
  disk_bw = 0.47 GB/s. 2.66599 x 0.47 = 1.2530. Not re-derived here.

  ratio R = observed_GB_per_token / 1.2530

  P1  R <= 0.80  => H2 SUPPORTED. Materially fewer bytes move than the law budgets, which is
      what a resident hot-expert subset predicts. The QUANTITATIVE hit would be R ~ 0.673
      (= 1/1.485); landing there means the byte deficit explains the WHOLE speed-up.
  P2  R >= 0.92  => H2 REFUTED. The modelled byte volume really is moving, so with the
      bandwidth confirmed and the pattern dead, C-21's 30% miss has NO surviving explanation.
      That is the loud outcome and it must be reported as such, not softened.
  P3  0.80 < R < 0.92 => INCONCLUSIVE. Declared in advance so neither side can claim it.

  KR-A COUNTER READABLE: samples must be > 0 with non-zero reads. A "counter unreadable"
     result is a HARNESS FAILURE, not a finding - this exact silent fallback discarded 83
     populated rows on 2026-07-31 while printing a sentence that read like restraint.
  KR-B CANNOT-VARY: an idle baseline (no bench, 30 s) must read < 0.05 GB/s. If the box
     shows heavy disk traffic with nothing running, every number here is background and no
     verdict may be issued.
  KR-C ATTRIBUTION: background traffic is measured, reported, and SUBTRACTED from the bench
     window. If background exceeds 20% of the bench read rate, the attribution is too weak
     to score and the run is reported UNINFORMATIVE.

  CONTENTION IS CONSERVATIVE HERE, which is why this may run on a non-scrubbed box: competing
  disk traffic ADDS observed bytes and SLOWS token generation, so both terms push R UP, away
  from H2. An R that supports H2 under contention is a lower bound on the effect. An R that
  REFUTES H2 under contention is not trustworthy and must be re-run clean.

  python weights/exp95_miss_fraction.py
"""
from __future__ import annotations
import json, os, re, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BIN = r"C:\Users\Federico\Documents\evo-compress\tools\llamacpp-b10098\llama-bench.exe"
MODEL = "D:/evo-compress-data/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf"
OUT = os.path.join(DATA, "exp95_miss_fraction.json")
PRED_GB_PER_TOK = 1.2530          # frozen stake: io 2.66599 s x 0.47 GB/s
DRIVE_HINT = "D:"

PS_DISK = (
    "$e = Get-CimInstance Win32_PerfFormattedData_PerfDisk_PhysicalDisk | "
    "Where-Object { $_.Name -ne '_Total' }; "
    "foreach ($d in $e) { '{0}|{1}' -f $d.Name, $d.DiskReadBytesPersec }")


def sample_disk():
    """[(name, read_bytes_per_sec)] - English property names on every locale (it-IT included)."""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", PS_DISK],
                           capture_output=True, text=True, timeout=25)
    except Exception:
        return []
    out = []
    for ln in r.stdout.splitlines():
        if "|" in ln:
            name, _, val = ln.partition("|")
            try:
                out.append((name.strip(), float(val.strip())))
            except ValueError:
                pass
    return out


def watch(seconds, stop_when=None, tag=""):
    """Sample until `seconds` elapse or stop_when() returns True. Returns per-drive means."""
    acc, n, t0 = {}, 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        if stop_when and stop_when():
            break
        for name, v in sample_disk():
            acc.setdefault(name, []).append(v)
        n += 1
        time.sleep(1.0)
    span = time.perf_counter() - t0
    res = {k: {"mean_bps": sum(v) / len(v), "n": len(v)} for k, v in acc.items() if v}
    return {"tag": tag, "span_s": round(span, 1), "samples": n, "drives": res}


def pick_drive(w):
    """The volume carrying D:. Falls back to the busiest if the label is not exposed."""
    for k in w["drives"]:
        if DRIVE_HINT.lower() in k.lower():
            return k
    return max(w["drives"], key=lambda k: w["drives"][k]["mean_bps"], default=None)


def main():
    res = {"staked_pred_gb_per_token": PRED_GB_PER_TOK, "model": MODEL,
           "utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}

    print("KR-B baseline: 30 s idle, nothing benching...")
    base = watch(30, tag="idle_baseline")
    res["baseline"] = base
    bd = pick_drive(base)
    base_bps = base["drives"][bd]["mean_bps"] if bd else None
    print(f"  drive {bd}: {(base_bps or 0)/1e9:.4f} GB/s idle")
    if base_bps is None:
        res["verdict"] = "UNINFORMATIVE - KR-A: disk counter returned nothing. Harness failure."
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1); print(res["verdict"]); return
    if base_bps / 1e9 >= 0.05:
        res["verdict"] = (f"UNINFORMATIVE - KR-B cannot-vary guard fired: idle disk traffic "
                          f"{base_bps/1e9:.4f} GB/s >= 0.05. Background dominates; no verdict.")
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1); print(res["verdict"]); return

    n_gen, reps = 64, 2
    cmd = [BIN, "-m", MODEL, "-ngl", "0", "-b", "2048", "-ub", "2048", "-t", "4",
           "-n", str(n_gen), "-p", "0", "-r", str(reps), "-o", "json"]
    print(f"bench: {' '.join(cmd[1:])}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    w = watch(3600, stop_when=lambda: proc.poll() is not None, tag="bench_window")
    out, err = proc.communicate(timeout=120)
    res["bench_window"] = w
    with open(OUT.replace(".json", "_bench.log"), "w", encoding="utf-8") as f:
        f.write(out + "\n--- stderr ---\n" + err)

    tok_s = None
    try:
        row = json.loads(out[out.index("["):out.rindex("]") + 1])[0]
        tok_s = row.get("avg_ts")
        ns = row.get("samples_ns") or []
        res["reps_tok_s"] = [round(n_gen / (x / 1e9), 4) for x in ns]
    except Exception as e:
        res["parse_error"] = str(e)
    res["tok_s"] = tok_s
    if not tok_s:
        res["verdict"] = "UNINFORMATIVE - bench produced no tok/s; nothing to divide by."
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1); print(res["verdict"]); return

    d = pick_drive(w)
    bench_bps = w["drives"][d]["mean_bps"]
    net_bps = bench_bps - base_bps                       # KR-C: subtract background
    res["drive"] = d
    res["bench_read_gbs"] = round(bench_bps / 1e9, 4)
    res["baseline_read_gbs"] = round(base_bps / 1e9, 4)
    res["net_read_gbs"] = round(net_bps / 1e9, 4)
    res["background_share"] = round(base_bps / bench_bps, 4) if bench_bps else None
    print(f"  bench {bench_bps/1e9:.4f} GB/s | idle {base_bps/1e9:.4f} | net {net_bps/1e9:.4f}")
    print(f"  tok/s {tok_s:.4f}  reps {res.get('reps_tok_s')}")

    if res["background_share"] and res["background_share"] > 0.20:
        res["verdict"] = (f"UNINFORMATIVE - KR-C: background is "
                          f"{res['background_share']*100:.0f}% of the bench read rate; "
                          f"attribution too weak to score.")
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1); print(res["verdict"]); return

    obs = (net_bps / 1e9) / tok_s
    R = obs / PRED_GB_PER_TOK
    res["observed_gb_per_token"] = round(obs, 4)
    res["ratio_R"] = round(R, 4)
    res["implied_vs_quantitative_target"] = round(R / 0.673, 3)
    print(f"\n  observed {obs:.4f} GB/token vs staked {PRED_GB_PER_TOK} -> R = {R:.4f}")

    if R <= 0.80:
        res["verdict"] = (f"H2 SUPPORTED - R {R:.3f} <= 0.80. Fewer bytes move than the law "
                          f"budgets, consistent with a resident hot-expert subset. Quantitative "
                          f"target for explaining the WHOLE speed-up was R ~ 0.673; this run sits "
                          f"{R/0.673:.2f}x from it. Contention can only push R up, so this is a "
                          f"lower bound on the effect.")
    elif R >= 0.92:
        res["verdict"] = (f"H2 REFUTED - R {R:.3f} >= 0.92. The modelled byte volume IS moving. "
                          f"With bandwidth confirmed (D-28) and access pattern dead, C-21's 30% "
                          f"miss now has NO surviving explanation. Report loudly; do not soften.")
    else:
        res["verdict"] = (f"INCONCLUSIVE - R {R:.3f} sits in the pre-declared dead band "
                          f"(0.80, 0.92). Neither side may claim it.")
    print(res["verdict"])
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
