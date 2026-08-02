"""Prereg #96b - per-fault cost, measured with a lever that actually moves.

#96 FAILED ITS OWN CANNOT-VARY GUARD and this is the redesign. Laguna is 37 GB against 16 GB
of RAM, so the working set always exceeds the cache: four runs moved disk bytes/token only
1.18x (0.4262 -> 0.3619) against a required 1.5x. At that spread BOTH staked patterns passed
at once - latency constant within 1.14x (P1 threshold 2.0) and missing time constant within
1.11x (P2 threshold 1.25). Without KR-A it would have been published as H4 SUPPORTED.

THE FIX IS A DIFFERENT MODEL, NOT A LOOSER GUARD. Qwen3-Coder-30B-A3B Q3_K_M is 13.7 GB and
FITS in this box's RAM, so it goes from cold to fully resident and the fault count collapses
by an order of magnitude instead of 18%.

BETTER STATISTICAL FORM. #96 forced an either/or between "per-fault" and "per-token". That
was the wrong shape - both can be present. Regressing token time on fault count decomposes
them instead:

        t_token(f) = INTERCEPT + LAMBDA * f

    LAMBDA    = per-fault cost, the thing Law 4 charges nothing for
    INTERCEPT = fault-free token time: compute + RAM, i.e. what the law already models

STAKED BEFORE RUNNING (2026-08-01):
  P1  H4 SUPPORTED: LAMBDA lands in 1-50 us/fault AND the regression is not flat
      (fault term explains the majority of the spread in t_token).
  P2  H4 REFUTED: LAMBDA ~ 0 - token time does not scale with fault count at all. Then
      streaming cost is not per-fault and this hypothesis dies like the access-pattern one.
  P3  otherwise INCONCLUSIVE, declared in advance.

  KR-A CANNOT-VARY, tightened from 1.5x to 3.0x: max(faults)/min(faults) must be >= 3.0.
      This is the guard that killed #96 and it is stricter here, not weaker.
  KR-E OUT-OF-SAMPLE CHECK - the real falsifier. The fitted INTERCEPT must match the
      MEASURED fully-resident token time within 25%. The fit is never shown that number;
      it has to predict it. A line can always be drawn through points, but only a correct
      model reproduces a measurement it was not fitted to. If KR-E fails, LAMBDA is
      curve-fitting and is reported as such no matter how plausible it looks.
  KR-B idle baseline < 0.05 GB/s, subtracted.
  KR-D negative implied fault counts or non-monotone nonsense => report, do not clamp.

  CONTENTION: inflates absolute times, so LAMBDA is an UPPER bound. It does not manufacture
  a correlation between fault count and token time that is not there.

  python weights/exp96b_fault_latency.py
"""
from __future__ import annotations
import json, os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BIN = r"<repo>\tools\llamacpp-b10098\llama-bench.exe"
MODEL = "D:/evo-compress-data/gguf/Qwen3-Coder-30B-A3B-Instruct-Q3_K_M.gguf"
FLUSH = "D:/evo-compress-data/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf"
OUT = os.path.join(DATA, "exp96b_fault_latency.json")
PAGE, RUNS, NGEN, FLUSH_GB = 4096, 5, 32, 15

PS_DISK = ("$e = Get-CimInstance Win32_PerfFormattedData_PerfDisk_PhysicalDisk | "
           "Where-Object { $_.Name -ne '_Total' }; "
           "foreach ($d in $e) { '{0}|{1}' -f $d.Name, $d.DiskReadBytesPersec }")


def sample():
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", PS_DISK],
                           capture_output=True, text=True, timeout=25)
    except Exception:
        return []
    o = []
    for ln in r.stdout.splitlines():
        if "|" in ln:
            n, _, v = ln.partition("|")
            try:
                o.append((n.strip(), float(v.strip())))
            except ValueError:
                pass
    return o


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


def flush_cache():
    """Evict the test model by streaming an unrelated large file through the page cache."""
    print(f"  flushing {FLUSH_GB} GB through page cache to evict the test model...")
    left = FLUSH_GB << 30
    with open(FLUSH, "rb", buffering=0) as f:
        while left > 0:
            b = f.read(1 << 24)
            if not b:
                break
            left -= len(b)


def bench(tag, base):
    cmd = [BIN, "-m", MODEL, "-ngl", "0", "-t", "4", "-n", str(NGEN), "-p", "0", "-r", "1",
           "-o", "json"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    w = watch(1800, stop=lambda: p.poll() is not None)
    out, err = p.communicate(timeout=120)
    k = dkey(w)
    net = max(0.0, (w["drives"][k] - base)) if k else 0.0
    tok = None
    try:
        tok = json.loads(out[out.index("["):out.rindex("]") + 1])[0].get("avg_ts")
    except Exception:
        pass
    r = {"tag": tag, "tok_s": tok, "window_s": w["span_s"], "net_gbs": round(net / 1e9, 4)}
    if tok:
        r["t_token_s"] = round(1.0 / tok, 5)
        r["disk_gb_per_token"] = round((net / 1e9) / tok, 5)
        r["faults"] = round(r["disk_gb_per_token"] * 1e9 / PAGE)
    print(f"  {tag}: tok/s {tok} disk {r.get('disk_gb_per_token')} GB/tok "
          f"faults {r.get('faults')} t_token {r.get('t_token_s')}s")
    return r


def fit(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None, None, None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    icept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (icept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return slope, icept, r2


def main():
    res = {"utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
           "model": MODEL, "page_bytes": PAGE, "runs": []}
    print("KR-B baseline 20s...")
    b = watch(20); bk = dkey(b)
    base = b["drives"][bk] if bk else None
    if base is None or base / 1e9 >= 0.05:
        res["verdict"] = f"UNINFORMATIVE - KR-B/KR-C: idle {base}. "
        json.dump(res, open(OUT, "w"), indent=1); print(res["verdict"]); return
    res["baseline_gbs"] = round(base / 1e9, 4)
    print(f"  {bk}: {base/1e9:.4f} GB/s idle")

    flush_cache()
    for i in range(RUNS):
        res["runs"].append(bench(f"run{i}" + ("_cold" if i == 0 else ""), base))
        json.dump(res, open(OUT, "w"), indent=1)

    ok = [r for r in res["runs"] if r.get("faults") is not None and r.get("t_token_s")]
    if len(ok) < 3:
        res["verdict"] = "UNINFORMATIVE - fewer than 3 scoreable runs."
        json.dump(res, open(OUT, "w"), indent=1); print("\n" + res["verdict"]); return

    f = [r["faults"] for r in ok]
    spread = max(f) / min(f) if min(f) > 0 else float("inf")
    res["fault_spread_x"] = round(spread, 2) if spread != float("inf") else "inf"
    # KR-E holdout: the most-resident run is EXCLUDED from the fit and predicted.
    resident = min(ok, key=lambda r: r["faults"])
    fitset = [r for r in ok if r is not resident]
    slope, icept, r2 = fit([r["faults"] for r in fitset], [r["t_token_s"] for r in fitset])
    res["holdout_resident"] = {"faults": resident["faults"], "t_token_s": resident["t_token_s"]}
    if slope is None:
        res["verdict"] = "UNINFORMATIVE - fit degenerate (no variation in fault count)."
        json.dump(res, open(OUT, "w"), indent=1); print("\n" + res["verdict"]); return
    lam_us = slope * 1e6
    pred_resident = icept + slope * resident["faults"]
    kre = abs(pred_resident - resident["t_token_s"]) / resident["t_token_s"]
    res.update({"lambda_us_per_fault": round(lam_us, 3), "intercept_s": round(icept, 5),
                "r2": round(r2, 4) if r2 is not None else None,
                "predicted_resident_t_token_s": round(pred_resident, 5),
                "KR_E_rel_error": round(kre, 4)})
    print(f"\n  fit (holding out the most-resident run): lambda {lam_us:.2f} us/fault, "
          f"intercept {icept:.4f}s, r2 {r2}")
    print(f"  KR-E: predicts resident t_token {pred_resident:.4f}s vs measured "
          f"{resident['t_token_s']:.4f}s -> {kre*100:.1f}% error")

    if spread < 3.0:
        res["verdict"] = (f"UNINFORMATIVE - KR-A cannot-vary fired again: fault spread "
                          f"{spread:.2f}x < 3.0. Same failure as #96; the lever still does not "
                          f"move. Do NOT loosen the guard - find a bigger lever.")
    elif kre > 0.25:
        res["verdict"] = (f"H4 NOT ESTABLISHED - KR-E failed: the fit predicts the held-out "
                          f"resident token time to {kre*100:.1f}% (limit 25%). lambda "
                          f"{lam_us:.2f} us/fault is curve-fitting, not a model, and is not "
                          f"claimed however plausible it looks.")
    elif 1.0 <= lam_us <= 50.0 and (r2 or 0) >= 0.80:
        # NOT identifiable as a per-fault cost. Faults are DERIVED as bytes/PAGE, so
        # lambda*faults == bytes/(PAGE/lambda): the per-fault model and a lower-effective-
        # bandwidth model are the same equation in different units. Report the bandwidth,
        # which is what was actually measured, and refuse the mechanism claim. The original
        # verdict string said "H4 SUPPORTED" and was wrong; it is kept in the register as a
        # withdrawn claim rather than deleted.
        eff = PAGE / (slope) / 1e9 if slope else None
        distinct = len({r["faults"] for r in fitset})
        res["effective_stream_gbs"] = round(eff, 4) if eff else None
        res["distinct_x_in_fit"] = distinct
        res["verdict"] = (
            f"MECHANISM NOT IDENTIFIABLE - but the magnitude is. lambda {lam_us:.2f} us per "
            f"{PAGE}-byte page is arithmetically identical to an effective streaming bandwidth "
            f"of {eff:.4f} GB/s, because fault counts were DERIVED from byte counts and never "
            f"observed. Per-fault latency and plain lower throughput cannot be separated by "
            f"this design. WHAT IS ESTABLISHED: llama.cpp streams this drive at {eff:.4f} GB/s "
            f"effective against {0.455:.3f} GB/s for raw reads (D-28) - a "
            f"{0.455/eff:.2f}x gap, the right size to be C-21's missing cost. CAVEATS THE r2 "
            f"HIDES: the fit had only {distinct} distinct x values, so r2 {r2:.4f} is "
            f"decorative; and KR-E held out a zero-fault run, so its {kre*100:.1f}% only tests "
            f"run-to-run stability of the resident time, not the fault model. Settling the "
            f"mechanism needs real hard-fault counters or a --no-mmap arm.")
    elif abs(lam_us) < 1.0:
        res["verdict"] = (f"H4 REFUTED - lambda {lam_us:.3f} us/fault is ~0: token time does "
                          f"not scale with fault count. Streaming cost is not per-fault.")
    else:
        res["verdict"] = (f"INCONCLUSIVE - lambda {lam_us:.2f} us/fault, r2 {r2}, spread "
                          f"{spread:.2f}x. Pre-declared bands not met; nothing claimed.")
    print("\n" + res["verdict"])
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
