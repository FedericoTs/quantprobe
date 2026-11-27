"""Score pre-registration #115 - should a small GPU's VRAM hold whole layers of experts, or a
cache of the hot ones?

Committed BEFORE any arm ran. Reads weights/data/prereg115.json (weights/prereg115_run.py) and
replays the recorded routing traces through weights/moe_cache_replay.py.

    python weights/prereg115_score.py --results weights/data/prereg115.json
    python weights/prereg115_score.py --self-check   # every branch reachable, no data needed
"""

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ARMS = ["A", "C1008", "C504", "C1008s"]
SPEEDUP = 1.20  # P-1: the best equal-VRAM cache arm over the static split
COUNTER_TOL = 0.001  # P-2: replayed counters within 0.1% of the lookups llama.cpp logged
HIT_TOL = 2.0  # P-3: percentage points of hit rate
MIN_REPS = 2
CURVE_MIB = [126, 252, 504, 1008, 1512, 2016, 3024, 4032]


def replay_small(trace, mib, share):
    """The "ubatch <= 8" counters llama.cpp would have logged for this trace at (mib, share)."""
    import moe_cache_replay as mcr

    r = mcr.replay(mcr.read_trace(trace), mib, share)
    s = r["small"]
    return {"hits": s["hits"], "misses": s["misses"], "cpu": s["cpu"], "uploaded_mib": s["bytes"] / 2**20}


def rate(c):
    n = c["hits"] + c["misses"]
    return 100.0 * c["hits"] / n if n else float("nan")


def score(d, replay=replay_small):
    """-> (lines, (kind, hits, misses, decision)). Pure given `replay`, so --self-check can drive
    every branch with a stand-in."""
    out, hits, misses = [], [], []
    runs = list((d.get("runs") or {}).values())
    ok = {a: [r for r in runs if r.get("arm") == a and r.get("rc") == 0 and r.get("decode_tps")] for a in ARMS}
    if d.get("status") != "complete" or any(len(v) < MIN_REPS for v in ok.values()):
        got = {a: len(v) for a, v in ok.items()}
        return [f"VOID: every arm needs >= {MIN_REPS} completed runs; got {got}."], ("VOID-incomplete", [], [], "")
    cache_runs = [r for a in ARMS[1:] for r in ok[a]]
    if any(not r.get("trace") or not r.get("small") for r in cache_runs):
        return ["VOID: a cache run lacks its routing trace or llama.cpp's counters."], ("VOID-no-trace", [], [], "")

    med = {a: statistics.median(r["decode_tps"] for r in ok[a]) for a in ARMS}
    out.append("median decode tok/s: " + "  ".join(f"{a} {med[a]:.2f}" for a in ARMS))

    # P-1: the lever
    best = max(("C1008", "C1008s"), key=lambda a: med[a])
    ratio = med[best] / med["A"]
    out.append(f"P-1: best equal-VRAM cache arm {best} is {ratio:.3f}x the static split (staked >= {SPEEDUP})")
    if ratio >= SPEEDUP:
        hits.append("P-1")
        decision = "INTEGRATE"
        out.append("  HIT - stake the planner integration: --moe-cache-mib where the build has it.")
    else:
        misses.append("P-1")
        decision = "NO-GAIN" if ratio >= 1.0 else "STATIC-WINS"
        out.append("  MISS - " + ("the cache helps, but not by enough to change the plan." if ratio >= 1.0
                                 else "the static split is faster on this machine; the plan stands."))

    # P-2: the port reproduces llama.cpp's own counters
    worst = 0.0
    for r in cache_runs:
        rep, log = replay(r["trace"], r["mib"], r["share"]), r["small"]
        n = log["hits"] + log["misses"]
        dev = max(abs(rep["hits"] - log["hits"]) / n, abs(rep["misses"] - log["misses"]) / n,
                  abs(rep["cpu"] - log["cpu"]) / n,
                  abs(rep["uploaded_mib"] - log["uploaded_mib"]) / max(log["uploaded_mib"], 1.0))
        worst = max(worst, dev)
    out.append(f"P-2: worst replay deviation from llama.cpp's counters over {len(cache_runs)} runs: "
               f"{worst * 100:.4f}% (staked <= {COUNTER_TOL * 100:.1f}%)")
    (hits if worst <= COUNTER_TOL else misses).append("P-2")

    # P-3: one trace predicts the other configurations
    src = min(ok["C1008"], key=lambda r: r["rep"])
    devs = []
    for arm, mib, share in (("C504", 504, 0.0), ("C1008s", 1008, 0.95)):
        pred = rate(replay(src["trace"], mib, share))
        meas = statistics.median(rate(r["small"]) for r in ok[arm])
        devs.append(abs(pred - meas))
        out.append(f"P-3: C1008 r{src['rep']}'s trace at {mib} MiB, share {share}: predicted hit rate "
                   f"{pred:.2f}% vs {arm} measured {meas:.2f}%")
    (hits if max(devs) <= HIT_TOL else misses).append("P-3")

    out.append(f"\ndescriptive: hit-rate curve from C1008 r{src['rep']}'s trace, share 0")
    out.append("  " + "  ".join(f"{m} MiB {rate(replay(src['trace'], m, 0.0)):.1f}%" for m in CURVE_MIB))
    n = len(hits) + len(misses)
    out.append(f"\nVERDICT: {len(hits)}/{n} predictions hit; decision {decision}")
    return out, ("SCORED", hits, misses, decision)


# ---------------------------------------------------------------------------- self-check
def _data(tps, n_ok=3, traces=True, status="complete"):
    runs = {}
    for a in ARMS:
        for rep in range(1, n_ok + 1):
            mib = {"A": 0, "C1008": 1008, "C504": 504, "C1008s": 1008}[a]
            r = {"arm": a, "rep": rep, "rc": 0, "decode_tps": tps[a], "mib": mib,
                 "share": 0.95 if a == "C1008s" else 0.0}
            if a != "A":
                r["trace"] = f"{a}_r{rep}.mct1" if traces else None
                r["small"] = {"hits": 7000 if a != "C504" else 6000, "misses": 3000 if a != "C504" else 4000,
                              "cpu": 2000 if a == "C1008s" else 0, "uploaded_mib": 900.0}
            runs[f"{a}_r{rep}"] = r
    return {"status": status, "runs": runs}


def _fake(exact=True, cross_off=0.0):
    """A stand-in replay: exact (or 1% off) on a run's own configuration, cross_off points off elsewhere."""
    def f(trace, mib, share):
        own = trace.split("_")[0]
        same = (own == "C1008" and mib == 1008 and share == 0.0) or (own == "C504" and mib == 504) or \
               (own == "C1008s" and share == 0.95)
        if same:
            hits = 6000 if mib == 504 else 7000
            k = 1.0 if exact else 1.01
            return {"hits": round(hits * k), "misses": 10000 - round(hits * k),
                    "cpu": 2000 if share == 0.95 else 0, "uploaded_mib": 900.0}
        target = 60.0 if mib == 504 else 70.0
        hits = round(100 * (target + cross_off) * 1.0)
        return {"hits": hits, "misses": 10000 - hits, "cpu": 0, "uploaded_mib": 900.0}
    return f


FAST = {"A": 14.0, "C1008": 18.0, "C504": 16.0, "C1008s": 17.0}
SELF = [
    ("incomplete", _data(FAST, n_ok=1), _fake()),
    ("no traces", _data(FAST, traces=False), _fake()),
    ("all hit", _data(FAST), _fake()),
    ("P-1 small gain", _data({**FAST, "C1008": 15.0, "C1008s": 15.5}), _fake()),
    ("P-1 static wins", _data({**FAST, "C1008": 12.0, "C1008s": 11.0}), _fake()),
    ("P-2 port off", _data(FAST), _fake(exact=False)),
    ("P-3 cross off", _data(FAST), _fake(cross_off=5.0)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        seen = set()
        for name, data, fake in SELF:
            _, (kind, h, m, dec) = score(data, fake)
            tag = f"{kind}:{','.join(h)}|{','.join(m)}|{dec}"
            seen.add(tag)
            print(f"  {name:16s} -> {tag}")
        print(f"\n{len(SELF)} fixtures, {len(seen)} distinct verdicts")
        assert len(seen) == len(SELF), f"{len(seen)} distinct outcomes for {len(SELF)} fixtures"
        return 0
    if not a.results:
        sys.exit("pass --results <file.json> or --self-check")
    lines, _ = score(json.load(open(a.results, encoding="utf-8")))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
