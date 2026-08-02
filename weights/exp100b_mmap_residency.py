"""Prereg #100b - is the mmap decode penalty about RESIDENCY PRESSURE, not load speed?

#100 FAILED ITS OWN GUARDS and this is the redesign. I compared total WALL CLOCK between mmap
and --no-mmap, but the two modes move different amounts of data: --no-mmap reads all 13.7 GB
upfront while mmap with -n 8 on a MoE faults in only the experts those 8 tokens route to. Not
the same bytes, so not a bandwidth comparison. KR-A caught it (cold/warm mmap 1.008x - mmap
barely read more when cold because it never reads much at all) and KR-C caught the consequence
(warm arms disagreed 1.80x).

WHAT THE DISCARDED RUN SUGGESTED, post-hoc and unstaked: warm DECODE was 0.5472 tok/s under
mmap and 0.9274 under --no-mmap, a 1.69x gap - LARGER than L-28's 1.388x page-touch penalty.
If faulting-vs-reading were the whole story it could not exceed it.

THE HYPOTHESIS THAT WOULD EXPLAIN AN EXCESS. A 13.7 GB mmap on a 16 GiB box competes with the
page cache for the same pages. Under pressure the kernel evicts file-backed pages, so a "warm"
mmap keeps re-faulting the model it already read. malloc-backed memory (--no-mmap) is anonymous
and is not evicted the same way. If that is right, the lever is about RESIDENCY STABILITY NEAR
CAPACITY, and it should VANISH on a model far below capacity.

That gives a control, which is what #100 lacked.

STAKED BEFORE RUNNING (2026-08-02). Decode throughput ONLY - tok/s from llama-bench, which
excludes load - so the two arms are compared on the one path where they move the same bytes.
Both models pre-warmed; -n 64 -r 2 so a steady state exists.

    BIG   Qwen3-Coder-30B-A3B Q3_K_M  13.7 GB   = 86% of 16 GiB  -> pressure
    SMALL Qwen2.5-7B-Instruct Q4_K_M   4.4 GB   = 28% of 16 GiB  -> no pressure

  R_big   = tok/s(no-mmap, BIG)   / tok/s(mmap, BIG)
  R_small = tok/s(no-mmap, SMALL) / tok/s(mmap, SMALL)

  P1 RESIDENCY PRESSURE IS THE MECHANISM: R_big >= 1.20 AND R_small <= 1.10. The penalty
     appears only where the model crowds RAM. Then --no-mmap is a real lever, gated on model
     size relative to RAM, and it is about eviction rather than load.
  P2 IT IS MMAP ITSELF: both >= 1.20. Then size does not matter, faulting is simply slower per
     access, and the gate should be on mmap alone.
  P3 THE 1.69x WAS AN ARTEFACT: both <= 1.10. The discarded run's signal came from n=8 and does
     not survive a steady state. L-28's practical half stays unproven.
  P4 anything else => INCONCLUSIVE, declared in advance.

  KR-A rep spread within any arm > 1.5x => that arm has no steady state and is not scored.
  KR-B free RAM is sampled before every arm and reported. If the BIG arms do not actually run
     near capacity the premise is absent and P1 cannot be claimed whatever the ratio says.
  KR-C --no-mmap OOM on BIG => report it as the result; do not silently drop to SMALL only.

  CONTENTION: an agent is active, adding memory and disk pressure to every arm. Pressure is the
  very variable under test, so contention pushes TOWARD P1. A P1 result here therefore needs the
  free-RAM numbers read alongside it, and a quiet re-run before anything is wired.

  python weights/exp100b_mmap_residency.py
"""
from __future__ import annotations
import json, os, subprocess, time

def _find_llamacpp():
    """Walk UP for tools/llamacpp-b10098 instead of counting directories. A fixed level count
    is wrong depending on where you run from - tools/ is 2 levels above weights/ in the main
    checkout but 5 above it inside a git worktree. This default was previously dead code that
    only ever worked because QP_LLAMACPP was set on the command line."""
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
    return os.path.join(d, "tools", "llamacpp-b10098")


B = _find_llamacpp()
BENCH = os.path.join(B, "llama-bench.exe")
BIG = "D:/evo-compress-data/gguf/Qwen3-Coder-30B-A3B-Instruct-Q3_K_M.gguf"
SMALL = "D:/evo-compress-data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
OUT = "weights/data/exp100b_mmap_residency.json"


def free_ram_gb():
    try:
        o = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                            "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                           capture_output=True, text=True, timeout=25).stdout.strip()
        return round(float(o) * 1024 / 1e9, 2)
    except Exception:
        return None


def run(model, mmap_on, tag, n=64, reps=2):
    cmd = [BENCH, "-m", model, "-ngl", "0", "-t", "4", "-n", str(n), "-p", "0",
           "-r", str(reps), "-mmp", "1" if mmap_on else "0", "-o", "json"]
    fr = free_ram_gb()
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    out = p.stdout + p.stderr
    open(os.path.join("weights", "data", f"exp100b_{tag}.log"), "w", encoding="utf-8").write(out)
    tok = spread = None
    try:
        row = json.loads(out[out.index("["):out.rindex("]") + 1])[0]
        tok = row.get("avg_ts")
        ns = row.get("samples_ns") or []
        if len(ns) > 1:
            r = [n / (x / 1e9) for x in ns]
            spread = round(max(r) / min(r), 3)
    except Exception:
        pass
    oom = "out of memory" in out.lower() or ("alloc" in out.lower() and "fail" in out.lower())
    print(f"  {tag:20} tok/s {tok}   rep-spread {spread}   free RAM before {fr} GB {'OOM!' if oom else ''}")
    return {"tok_s": tok, "rep_spread": spread, "free_ram_before_gb": fr,
            "oom": oom, "rc": p.returncode}


def main():
    res = {"utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "arms": {}}
    print("pre-warming both models (the load path is not what is under test)...")
    run(BIG, True, "prewarm_big", n=4, reps=1)
    run(SMALL, True, "prewarm_small", n=4, reps=1)
    print("\nBIG (13.7 GB, 86% of RAM):")
    res["arms"]["big_mmap"] = run(BIG, True, "big_mmap")
    res["arms"]["big_nommap"] = run(BIG, False, "big_nommap")
    print("SMALL (4.4 GB, 28% of RAM) - the control:")
    res["arms"]["small_mmap"] = run(SMALL, True, "small_mmap")
    res["arms"]["small_nommap"] = run(SMALL, False, "small_nommap")

    a = res["arms"]
    def ratio(x, y):
        p, q = a[x]["tok_s"], a[y]["tok_s"]
        return round(q / p, 3) if p and q else None
    res["R_big"] = ratio("big_mmap", "big_nommap")
    res["R_small"] = ratio("small_mmap", "small_nommap")
    print(f"\n  R_big   (no-mmap / mmap, 13.7 GB) : {res['R_big']}")
    print(f"  R_small (no-mmap / mmap,  4.4 GB) : {res['R_small']}")

    bad = [k for k, v in a.items() if not k.startswith("prewarm")
           and v.get("rep_spread") and v["rep_spread"] > 1.5]
    if a["big_nommap"]["oom"] or a["big_nommap"]["rc"] != 0:
        res["verdict"] = ("KR-C: --no-mmap failed on the 13.7 GB model. That is the result - the "
                          "lever is infeasible at this size on this box.")
    elif bad:
        res["verdict"] = (f"UNINFORMATIVE - KR-A: no steady state in {bad} (rep spread > 1.5x). "
                          f"Those arms are not scored.")
    elif res["R_big"] is None or res["R_small"] is None:
        res["verdict"] = "UNINFORMATIVE - an arm produced no tok/s."
    elif res["R_big"] >= 1.20 and res["R_small"] <= 1.10:
        res["verdict"] = (f"P1 SUPPORTED - the penalty is RESIDENCY PRESSURE, not mmap itself: "
                          f"{res['R_big']}x on a model at 86% of RAM, {res['R_small']}x on one at "
                          f"28%. A large mmap competes with page cache and keeps re-faulting what "
                          f"it already read; malloc-backed memory does not. The lever is real and "
                          f"gates on model size relative to RAM. Contention pushes toward this "
                          f"result - read the free-RAM figures alongside it and re-run quiet "
                          f"before wiring.")
    elif res["R_big"] >= 1.20 and res["R_small"] >= 1.20:
        res["verdict"] = (f"P2 SUPPORTED - mmap itself is slower per access ({res['R_big']}x big, "
                          f"{res['R_small']}x small). Size does not matter; the gate belongs on "
                          f"mmap alone.")
    elif res["R_big"] <= 1.10 and res["R_small"] <= 1.10:
        res["verdict"] = (f"P3 SUPPORTED - the 1.69x from the discarded run was an n=8 ARTEFACT. "
                          f"At steady state mmap costs {res['R_big']}x / {res['R_small']}x, i.e. "
                          f"nothing. L-28's practical half stays unproven. Published as a miss.")
    else:
        res["verdict"] = (f"INCONCLUSIVE - R_big {res['R_big']}, R_small {res['R_small']} match no "
                          f"pre-declared pattern.")
    print("\n" + res["verdict"])
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
