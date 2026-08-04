"""AUTOTUNE sweep - the staked experiment behind a future `quantprobe autotune`
(prereg 2026-08-04-autotune.md). Productization into the CLI happens ONLY if P1/P2 pass.

  python weights/autotune_sweep.py --plan       # enumerate + price, NO measurement (reviewable)
  python weights/autotune_sweep.py --run        # coordinate sweep -> compose -> KR-A reproduce

Design, per the stake:
- coordinate sweep around the planner's own emitted config (one axis at a time), then greedy
  composition of per-axis winners + their pairwise combos: ~30 configs/row, 3 rows, all
  deterministic, best-predicted-first inside each axis;
- fixed 180s wall cap per config (llama-bench tg64 r=2, load included) - a config that busts
  the cap is recorded `over_budget`, never silently skipped (KR-C's no-silent-anything rule);
- flags the law cannot price carry predicted=null and are EXCLUDED from P2's Spearman (KR-C);
- any winner must reproduce at r=3 in a fresh session before it may be published (KR-A);
- refuses to start while ANY runner lock exists (.p0_lock included) - one measurement owns
  the box (KR-B).
"""
from __future__ import annotations
import argparse, itertools, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
LOCKS = [os.path.join(DATA, ".p0_lock"), os.path.join(DATA, ".autotune_lock")]
MYLOCK = os.path.join(DATA, ".autotune_lock")
LLAMA = os.environ.get("QUANTPROBE_LLAMA_DIR",
                       "C:/Users/Federico/Documents/evo-compress/tools/llamacpp-b10098")
GGUF = "D:/evo-compress-data/gguf"
WALL_CAP = 180.0
SEED_ORDER = 20260804

ROWS = {
    "0.6B_aiv": os.path.join(GGUF, "Qwen3-0.6B-Q8_0.gguf"),
    "7B_aiv":   os.path.join(GGUF, "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    "30B_split": os.path.join(GGUF, "Qwen3-30B-A3B-Q2_K.gguf"),
}


def planner_baseline(gguf):
    from weights.p0_lanes import qp_flags          # one source for the planner call
    best, flags = qp_flags(gguf)
    return best, flags


def parse_flags(flags):
    """server-style flags list -> dict of the axes we sweep."""
    d, i = {}, 0
    while i < len(flags):
        f = flags[i]
        if f in ("-ngl", "-t", "-ub", "-b", "-ctk", "-ctv", "-ot"):
            d[f] = flags[i + 1]; i += 2
        elif f in ("-fa", "--no-mmap"):
            d[f] = True; i += 1
        else:
            i += 1
    return d


def bench_args(cfg):
    """axis dict -> llama-bench argv fragment. llama-bench dialect: -fa 1, -mmp 0."""
    out = []
    for k in ("-ngl", "-t", "-ub", "-b", "-ctk", "-ctv", "-ot"):
        if k in cfg and cfg[k] is not None:
            out += [k, str(cfg[k])]
    out += ["-fa", "1" if cfg.get("-fa") else "0"]
    if cfg.get("--no-mmap"):
        out += ["-mmp", "0"]
    return out


def variants(base, row):
    """Coordinate sweep: one axis changed at a time. Deterministic order.
    Each variant: (name, cfg, priceable) - priceable=False -> predicted null (KR-C)."""
    base = {k: (v if isinstance(v, bool) else str(v)) for k, v in base.items()}
    vs = []
    ngl0 = int(base.get("-ngl", 99))
    for t in ("4", "6", "8"):
        vs.append((f"t{t}", {**base, "-t": t}, False))
    for ub in ("256", "512", "1024", "2048"):
        vs.append((f"ub{ub}", {**base, "-ub": ub}, False))
    for b in ("512", "2048"):
        vs.append((f"b{b}", {**base, "-b": b}, False))
    vs.append(("kv_q8", {**base, "-ctk": "q8_0", "-ctv": "q8_0", "-fa": True}, False))
    vs.append(("fa_flip", {**base, "-fa": not base.get("-fa", False)}, False))
    vs.append(("nommap_flip", {**base, "--no-mmap": not base.get("--no-mmap", False)}, False))
    # -ngl variants: swept but NOT priced in this arm. Pricing an arbitrary offload count
    # needs plan.evaluate with a forced gl, which the public API does not expose today -
    # inventing predictions through a side door would make P2 a fiction. Stake amendment
    # (recorded in the prereg before any measurement): if every row's priceable subset is
    # degenerate, P2 is UNTESTABLE and the best-predicted-first ordering must not ship until
    # the -ngl/-ot axes are genuinely priceable.
    if ngl0 < 99:
        for d in (-2, -1, 1, 2):
            if ngl0 + d >= 0:
                vs.append((f"ngl{ngl0+d}", {**base, "-ngl": ngl0 + d}, False))
    return [(n, c, p) for n, c, p in vs if c != base]   # a variant equal to baseline is a re-measure, not a variant


def price(gguf, cfg, priceable, cache={}):
    """Predicted tok/s where the law can honestly price the variant; null otherwise (KR-C).
    In this arm only the baseline carries a prediction - see the stake amendment above."""
    return None


def measure(gguf, cfg, n=64, r=2, cap=WALL_CAP):
    exe = os.path.join(LLAMA, "llama-bench.exe")
    cmd = [exe, "-m", gguf, "-n", str(n), "-p", "0", "-r", str(r)] + bench_args(cfg)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=cap)
    except subprocess.TimeoutExpired:
        return None, None, "over_budget", round(time.time() - t0, 1)
    for line in p.stdout.splitlines():
        if f"tg{n}" in line and "|" in line:
            try:
                val = line.split("|")[-2].strip().split()
                return float(val[0]), float(val[2]) if len(val) > 2 else 0.0, "ok", \
                    round(time.time() - t0, 1)
            except (ValueError, IndexError):
                pass
    return None, None, "parse_fail", round(time.time() - t0, 1)


def gpu_state(tag, log):
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        log(f"[gpu {tag}] {q.stdout.strip()}")
    except Exception as e:
        log(f"[gpu {tag}] unavailable: {e}")


def spearman(pairs):
    """pairs: [(pred, meas)] with no Nones. Plain implementation, no scipy."""
    if len(pairs) < 4:
        return None                                  # degenerate subset - scored as untestable
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        rk = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            for k in range(i, j + 1):
                rk[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return rk
    rp, rm = ranks([p for p, _ in pairs]), ranks([m for _, m in pairs])
    n = len(pairs)
    mp, mm = sum(rp) / n, sum(rm) / n
    num = sum((a - mp) * (b - mm) for a, b in zip(rp, rm))
    den = (sum((a - mp) ** 2 for a in rp) * sum((b - mm) ** 2 for b in rm)) ** 0.5
    return round(num / den, 3) if den else None


def run(dry=False):
    os.makedirs(DATA, exist_ok=True)
    for l in LOCKS:
        if os.path.isdir(l) and l != MYLOCK:
            print(f"REFUSED: {l} exists - another measurement owns the box"); return 3
    if not dry:
        try:
            os.mkdir(MYLOCK)
        except FileExistsError:
            print(f"REFUSED: {MYLOCK} exists"); return 3
    logp = os.path.join(DATA, "autotune_run.log")
    lf = open(logp, "a", encoding="utf-8")
    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True); lf.write(line + "\n"); lf.flush()
    try:
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
        subprocess.run(["taskkill", "/F", "/IM", "llama-bench.exe"], capture_output=True)
        time.sleep(2)
        all_out = {}
        for row, gguf in ROWS.items():
            log(f"=== row {row} ===")
            gpu_state(f"{row} pre", log)
            best, flags = planner_baseline(gguf)
            base = parse_flags(flags)
            log(f"[{row}] planner: {best[0]} predicted {best[1]:.1f} tok/s  flags={base}")
            ledger = []
            bm, be, st, dt = (None, None, "dry", 0) if dry else measure(gguf, base)
            ledger.append(dict(name="baseline", cfg=base, predicted=round(best[1], 2),
                               measured=bm, err=be, status=st, wall=dt))
            log(f"[{row}] baseline measured: {bm} ({st}, {dt}s)")
            for name, cfg, priceable in variants(base, row):
                pred = price(gguf, cfg, priceable)
                if dry:
                    ledger.append(dict(name=name, cfg=cfg, predicted=pred, measured=None,
                                       err=None, status="dry", wall=0))
                    continue
                m, e, st, dt = measure(gguf, cfg)
                ledger.append(dict(name=name, cfg=cfg, predicted=pred, measured=m, err=e,
                                   status=st, wall=dt))
                log(f"[{row}] {name}: {m} tok/s ({st}, {dt}s)"
                    + (f" pred {pred}" if pred else ""))
            if not dry:
                # greedy compose: take every axis whose lone variant beat baseline, together
                okv = [l for l in ledger[1:] if l["status"] == "ok" and l["measured"]]
                basem = ledger[0]["measured"] or 0
                winners = [l for l in okv if l["measured"] > basem * 1.02]
                if winners:
                    comp = dict(base)
                    for w in sorted(winners, key=lambda l: -l["measured"])[:4]:
                        for k, v in w["cfg"].items():
                            if base.get(k) != v:
                                comp[k] = v
                    m, e, st, dt = measure(gguf, comp)
                    ledger.append(dict(name="compose", cfg=comp, predicted=None, measured=m,
                                       err=e, status=st, wall=dt))
                    log(f"[{row}] compose: {m} tok/s ({st}, {dt}s)")
                # KR-A: reproduce the single best config at r=3, fresh run
                cand = max((l for l in ledger if l["status"] == "ok" and l["measured"]),
                           key=lambda l: l["measured"])
                m3, e3, st3, dt3 = measure(gguf, cand["cfg"], r=3, cap=WALL_CAP * 2)
                cand["repro_r3"] = dict(measured=m3, err=e3, status=st3, wall=dt3)
                log(f"[{row}] KR-A reproduce '{cand['name']}': {m3} tok/s ({st3})")
            pairs = [(l["predicted"], l["measured"]) for l in ledger
                     if l["predicted"] is not None and l["measured"] is not None]
            rho = spearman(pairs)
            all_out[row] = dict(planner=dict(placement=best[0], predicted=round(best[1], 2)),
                                ledger=ledger, spearman_priceable=rho,
                                priceable_n=len(pairs))
            gpu_state(f"{row} post", log)
        with open(os.path.join(DATA, "autotune_results.json"), "w", encoding="utf-8") as fh:
            json.dump(all_out, fh, indent=1)
        log("results -> autotune_results.json")
        if not dry:
            log("=== STAKED GATES (score after inspection) ===")
            for row, o in all_out.items():
                led = [l for l in o["ledger"] if l["status"] == "ok" and l["measured"]]
                if not led:
                    log(f"  {row}: no clean measurements"); continue
                basem = o["ledger"][0]["measured"] or 0
                top = max(led, key=lambda l: l["measured"])
                gain = (top["measured"] / basem - 1) * 100 if basem else 0
                rep = top.get("repro_r3", {})
                log(f"  {row}: best '{top['name']}' {top['measured']:.2f} vs baseline "
                    f"{basem:.2f} ({gain:+.1f}%)  repro_r3={rep.get('measured')}  "
                    f"P2 rho={o['spearman_priceable']} (n={o['priceable_n']})")
        return 0
    finally:
        if not dry:
            try:
                os.rmdir(MYLOCK)
            except OSError:
                pass
        lf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="enumerate + price only, no measurement")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    sys.exit(run(dry=not a.run) if (a.plan or a.run) else 0)
