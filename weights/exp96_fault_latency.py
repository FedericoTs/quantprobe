"""Prereg #96 - the disk term prices BANDWIDTH but not LATENCY.

WHERE THIS COMES FROM. C-21: the disk row runs 1.485x faster than predicted. D-28: the
bandwidth input is correct (0.452-0.459 measured vs 0.47 calibrated) and access pattern is
irrelevant on this drive. #95: only 0.458 GB/token actually moves against a modelled 1.253 GB,
R = 0.366 - a byte deficit 1.84x LARGER than the speed-up needs. So two errors partially
cancel, and the second one is worth 0.789 s of every 1.995 s token.

WHAT IT IS NOT. The law is ACCURATE on pure-CPU rows with no disk: the anchor predicts 7.09
tok/s and KR3 measured 7.57 (+6.8%). Whatever is missing appears only when pages come off
disk, so it is not unpriced CPU compute - which was the obvious first guess and is hereby
ruled out BEFORE measuring rather than after.

THE HYPOTHESIS. 0.458 GB/token is ~111,816 four-kilobyte page faults. 0.789 s across those
is 7.05 us each - squarely SATA SSD 4K read latency. Law 4 divides bytes by bandwidth and
charges nothing per FAULT. On a resident tier that is fine; on a streaming tier the fault
count is enormous and latency, not bandwidth, may be what binds.

STAKED BEFORE RUNNING (2026-08-01). The page cache warms monotonically across repeated runs,
so disk bytes/token FALLS while the model and compute stay identical. That is the natural
lever - no cache manipulation, no new variable.

  For each run i:  missing_i = t_token_i - (disk_GB_i / 0.455) - 0.19973
                   faults_i  = disk_GB_i * 1e9 / 4096
                   latency_i = missing_i / faults_i

  P1  H4 SUPPORTED: latency_i is constant within 2.0x across runs. The missing cost is
      PER-FAULT, and Law 4's disk tier needs a latency term.
  P2  H4 REFUTED: missing_i is instead constant within 25% while disk bytes vary. Then the
      cost is a fixed per-token overhead, unrelated to fault count, and this hypothesis dies
      the same way the access-pattern one did.
  P3  neither holds => INCONCLUSIVE, declared in advance so neither reading can be claimed.

  KR-A CANNOT-VARY (the guard this whole experiment stands on): max(disk_GB) / min(disk_GB)
      must be >= 1.5. If the cache does not actually warm between runs, disk bytes barely
      move, BOTH P1 and P2 are trivially satisfiable, and the run is UNINFORMATIVE. This is
      the single most likely way for this experiment to produce a fake answer.
  KR-B idle baseline < 0.05 GB/s, subtracted from every window.
  KR-C every run must return real counter samples; "unreadable" is a harness failure.
  KR-D any run whose missing_i is NEGATIVE invalidates the arithmetic (it would mean the
      bandwidth term alone exceeds the measured token) - report and stop, do not clamp.

  CONTENTION: an active agent inflates both disk bytes and token time. It does NOT
  systematically bias a RATIO of latencies across runs measured minutes apart, which is why
  this is scoreable on a non-scrubbed box - but the absolute microseconds are an upper bound
  and are labelled as such.

  python weights/exp96_fault_latency.py
"""
from __future__ import annotations
import json, os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BIN = r"C:\Users\Federico\Documents\evo-compress\tools\llamacpp-b10098\llama-bench.exe"
MODEL = "D:/evo-compress-data/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf"
OUT = os.path.join(DATA, "exp96_fault_latency.json")
BW, RAM_S, PAGE, RUNS, NGEN = 0.455, 0.19973, 4096, 4, 32

PS_DISK = ("$e = Get-CimInstance Win32_PerfFormattedData_PerfDisk_PhysicalDisk | "
           "Where-Object { $_.Name -ne '_Total' }; "
           "foreach ($d in $e) { '{0}|{1}' -f $d.Name, $d.DiskReadBytesPersec }")


def sample():
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", PS_DISK],
                           capture_output=True, text=True, timeout=25)
    except Exception:
        return []
    out = []
    for ln in r.stdout.splitlines():
        if "|" in ln:
            n, _, v = ln.partition("|")
            try:
                out.append((n.strip(), float(v.strip())))
            except ValueError:
                pass
    return out


def watch(limit, stop=None):
    acc, t0 = {}, time.perf_counter()
    while time.perf_counter() - t0 < limit:
        if stop and stop():
            break
        for n, v in sample():
            acc.setdefault(n, []).append(v)
        time.sleep(1.0)
    return {"span_s": round(time.perf_counter() - t0, 1),
            "drives": {k: sum(v) / len(v) for k, v in acc.items() if v}}


def dkey(w):
    for k in w["drives"]:
        if "d:" in k.lower():
            return k
    return max(w["drives"], key=lambda k: w["drives"][k], default=None)


def main():
    res = {"utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
           "bw_gbs": BW, "ram_s": RAM_S, "page_bytes": PAGE, "runs": []}
    print("KR-B baseline 20s...")
    b = watch(20)
    bk = dkey(b)
    base = b["drives"][bk] if bk else None
    if base is None:
        res["verdict"] = "UNINFORMATIVE - KR-C: counter returned nothing."
        json.dump(res, open(OUT, "w"), indent=1); print(res["verdict"]); return
    res["baseline_gbs"] = round(base / 1e9, 4)
    print(f"  {bk}: {base/1e9:.4f} GB/s idle")
    if base / 1e9 >= 0.05:
        res["verdict"] = f"UNINFORMATIVE - KR-B: idle {base/1e9:.4f} GB/s >= 0.05."
        json.dump(res, open(OUT, "w"), indent=1); print(res["verdict"]); return

    for i in range(RUNS):
        cmd = [BIN, "-m", MODEL, "-ngl", "0", "-b", "2048", "-ub", "2048", "-t", "4",
               "-n", str(NGEN), "-p", "0", "-r", "1", "-o", "json"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        w = watch(2400, stop=lambda: p.poll() is not None)
        out, err = p.communicate(timeout=120)
        k = dkey(w)
        net = (w["drives"][k] - base) if k else 0.0
        tok = None
        try:
            tok = json.loads(out[out.index("["):out.rindex("]") + 1])[0].get("avg_ts")
        except Exception:
            pass
        r = {"i": i, "tok_s": tok, "window_s": w["span_s"],
             "read_gbs": round((w["drives"][k] if k else 0) / 1e9, 4),
             "net_gbs": round(net / 1e9, 4)}
        if tok:
            r["t_token_s"] = round(0.95 / tok, 4)
            r["disk_gb_per_token"] = round((net / 1e9) / tok, 4)
            r["faults"] = round(r["disk_gb_per_token"] * 1e9 / PAGE)
            r["bw_term_s"] = round(r["disk_gb_per_token"] / BW, 4)
            r["missing_s"] = round(r["t_token_s"] - r["bw_term_s"] - RAM_S, 4)
            r["latency_us"] = (round(r["missing_s"] / r["faults"] * 1e6, 3)
                               if r["faults"] else None)
        res["runs"].append(r)
        print(f"  run {i}: tok/s {tok} disk {r.get('disk_gb_per_token')} GB/tok "
              f"missing {r.get('missing_s')}s latency {r.get('latency_us')}us")
        json.dump(res, open(OUT, "w"), indent=1)

    ok = [r for r in res["runs"] if r.get("latency_us") is not None]
    if len(ok) < 2:
        res["verdict"] = "UNINFORMATIVE - fewer than 2 scoreable runs."
    elif any(r["missing_s"] < 0 for r in ok):
        res["verdict"] = ("INVALID - KR-D: a run has NEGATIVE missing time, so the bandwidth "
                          "term alone exceeds the measured token. Arithmetic is wrong "
                          "somewhere; not clamped, not scored.")
    else:
        gb = [r["disk_gb_per_token"] for r in ok]
        spread = max(gb) / min(gb) if min(gb) > 0 else 0
        res["disk_bytes_spread_x"] = round(spread, 3)
        lat = [r["latency_us"] for r in ok]
        mis = [r["missing_s"] for r in ok]
        res["latency_spread_x"] = round(max(lat) / min(lat), 3) if min(lat) > 0 else None
        res["missing_spread_x"] = round(max(mis) / min(mis), 3) if min(mis) > 0 else None
        if spread < 1.5:
            res["verdict"] = (f"UNINFORMATIVE - KR-A cannot-vary fired: disk bytes/token spread "
                              f"only {spread:.2f}x (< 1.5). The cache did not warm enough to "
                              f"separate a per-fault cost from a per-token one; both hypotheses "
                              f"are trivially satisfiable and neither may be claimed.")
        elif res["latency_spread_x"] and res["latency_spread_x"] <= 2.0:
            res["verdict"] = (f"H4 SUPPORTED - disk bytes varied {spread:.2f}x while per-fault "
                              f"latency held within {res['latency_spread_x']:.2f}x "
                              f"(mean {sum(lat)/len(lat):.2f} us). The missing cost is PER-FAULT: "
                              f"Law 4's disk tier prices bandwidth and charges nothing for "
                              f"latency. Absolute microseconds are an upper bound (contention).")
        elif res["missing_spread_x"] and res["missing_spread_x"] <= 1.25:
            res["verdict"] = (f"H4 REFUTED - missing time held within "
                              f"{res['missing_spread_x']:.2f}x while disk bytes varied "
                              f"{spread:.2f}x. The cost is a fixed per-token overhead, not "
                              f"per-fault. This hypothesis dies like the access-pattern one.")
        else:
            res["verdict"] = (f"INCONCLUSIVE - disk spread {spread:.2f}x, latency spread "
                              f"{res['latency_spread_x']}x, missing spread "
                              f"{res['missing_spread_x']}x. Neither pre-declared pattern holds.")
    print("\n" + res["verdict"])
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
