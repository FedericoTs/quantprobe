"""Prereg #100 - does L-28's 1.388x mmap penalty actually transfer to llama.cpp?

L-28 measured, on a bare page-touch loop: sequential read() 0.5501 GB/s vs sequential
mmap-and-touch 0.3962 GB/s, a 1.388x penalty for faulting pages in rather than reading them.
That was Python touching bytes. Whether it survives contact with a real runtime is a separate
question, and the honest way to find out is to predict something llama.cpp will do and check.

THIS PREDICTION RUNS AGAINST llama.cpp's OWN DOCUMENTATION, which is what makes it worth
running. `llama-cli --help` says of --no-mmap: "slower load but may reduce pageouts if not
using mlock". Their framing is about LOAD alone, and it is correct on its own terms - mmap
returns instantly and defers the work. But deferring is not avoiding: those pages still fault
in during the first forward pass. TOTAL cold time to produce N tokens is load PLUS decode, and
if read() moves bytes 1.388x faster than faulting, --no-mmap should WIN on total even while
losing on the load line. The two views make opposite predictions about the quantity a user
actually waits for.

STAKED BEFORE RUNNING (2026-08-02). Model: Qwen3-Coder-30B-A3B Q3_K_M, 13.7 GB - large enough
for a clear signal and small enough to fit 16 GiB so --no-mmap can allocate it at all.

  ratio = cold_wall(mmap) / cold_wall(no-mmap)

  P1 L-28 TRANSFERS: ratio >= 1.15. Faulting really is the slower way to get bytes into a
     model, the penalty survives a real runtime, and --no-mmap is a genuine cold-start lever
     for models that fit RAM.
  P2 L-28 DOES NOT TRANSFER: ratio <= 1.00 - --no-mmap is no faster or actively slower on
     total. Then llama.cpp's documentation is right end-to-end, my page-touch measurement does
     not describe what a real loader does, and L-28's practical half is REFUTED. This is the
     outcome I consider less likely, which is exactly why it is written down first.
  P3 1.00 < ratio < 1.15 => INCONCLUSIVE, declared in advance.

  KR-A CANNOT-VARY: cold must exceed warm by >= 1.5x on the mmap arm. If eviction did not
     work, both arms read from page cache, no bytes move, and the comparison is meaningless.
  KR-B --no-mmap must not OOM. If it does, that IS the result: the lever is infeasible at this
     size on this box, report it and do not silently fall back to a smaller model.
  KR-C warm arms must agree within 10%. Steady-state decode does not touch the load path, so a
     difference there would mean the arms differ in something other than page acquisition and
     the cold comparison is confounded.

  CONTENTION: an agent is active. It adds disk traffic to BOTH cold arms roughly equally, which
  compresses the ratio toward 1.0 - i.e. toward P2, against the hypothesis. A P1 result is
  therefore conservative.

  python weights/exp100_nommap.py
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
MODEL = "D:/evo-compress-data/gguf/Qwen3-Coder-30B-A3B-Instruct-Q3_K_M.gguf"
EVICTOR = "D:/evo-compress-data/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf"
OUT = "weights/data/exp100_nommap.json"
EVICT_GB = 16


def evict():
    """Push the test model out of page cache by streaming an unrelated file through it."""
    left = EVICT_GB << 30
    with open(EVICTOR, "rb", buffering=0) as f:
        while left > 0:
            b = f.read(1 << 24)
            if not b:
                break
            left -= len(b)


def run(mmap_on, tag):
    """Wall clock of the WHOLE invocation: load + decode. That is what a user waits for."""
    cmd = [BENCH, "-m", MODEL, "-ngl", "0", "-t", "4", "-n", "8", "-p", "0", "-r", "1",
           "-mmp", "1" if mmap_on else "0", "-o", "json"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    wall = time.perf_counter() - t0
    out = p.stdout + p.stderr
    open(os.path.join("weights", "data", f"exp100_{tag}.log"), "w", encoding="utf-8").write(out)
    tok = None
    try:
        tok = json.loads(out[out.index("["):out.rindex("]") + 1])[0].get("avg_ts")
    except Exception:
        pass
    oom = ("alloc" in out.lower() and "fail" in out.lower()) or "out of memory" in out.lower()
    print(f"  {tag:22} wall {wall:7.2f}s   tok/s {tok}   {'OOM!' if oom else ''}")
    return {"wall_s": round(wall, 2), "tok_s": tok, "oom": oom, "rc": p.returncode}


def main():
    res = {"utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
           "model": os.path.basename(MODEL), "arms": {}}
    print("evicting, then COLD mmap...")
    evict();  res["arms"]["cold_mmap"] = run(True, "cold_mmap")
    print("evicting, then COLD no-mmap...")
    evict();  res["arms"]["cold_nommap"] = run(False, "cold_nommap")
    print("warm arms (no eviction) - the load path is already paid...")
    res["arms"]["warm_mmap"] = run(True, "warm_mmap")
    res["arms"]["warm_nommap"] = run(False, "warm_nommap")

    A = res["arms"]["cold_mmap"]["wall_s"]
    Bw = res["arms"]["cold_nommap"]["wall_s"]
    wm = res["arms"]["warm_mmap"]["wall_s"]
    wn = res["arms"]["warm_nommap"]["wall_s"]
    res["ratio_cold_mmap_over_nommap"] = round(A / Bw, 3) if Bw else None
    res["cold_over_warm_mmap"] = round(A / wm, 3) if wm else None
    res["warm_agreement"] = round(max(wm, wn) / min(wm, wn), 3) if min(wm, wn) else None
    print(f"\n  cold mmap/no-mmap : {res['ratio_cold_mmap_over_nommap']}x   (L-28 page-touch said 1.388x)")
    print(f"  cold/warm (mmap)  : {res['cold_over_warm_mmap']}x   (KR-A needs >= 1.5)")
    print(f"  warm agreement    : {res['warm_agreement']}x   (KR-C needs <= 1.10)")

    r = res["ratio_cold_mmap_over_nommap"]
    if res["arms"]["cold_nommap"]["oom"] or res["arms"]["cold_nommap"]["rc"] != 0:
        res["verdict"] = ("KR-B: --no-mmap failed/OOMed at 13.7 GB on this box. That IS the "
                          "result - the lever is infeasible at this size here. Not retried on a "
                          "smaller model, which would answer a different question.")
    elif (res["cold_over_warm_mmap"] or 0) < 1.5:
        res["verdict"] = (f"UNINFORMATIVE - KR-A: cold is only {res['cold_over_warm_mmap']}x warm, "
                          f"so eviction did not work and both arms read from cache. No bytes "
                          f"moved; the comparison means nothing.")
    elif (res["warm_agreement"] or 9) > 1.10:
        res["verdict"] = (f"UNINFORMATIVE - KR-C: warm arms disagree by {res['warm_agreement']}x. "
                          f"Steady-state decode should not touch the load path, so the arms "
                          f"differ in something else and the cold comparison is confounded.")
    elif r >= 1.15:
        res["verdict"] = (f"P1 SUPPORTED - --no-mmap is {r}x faster cold, against llama.cpp's own "
                          f"documentation which frames --no-mmap as 'slower load'. It is: but "
                          f"deferring page faults is not avoiding them, and on TOTAL time to "
                          f"first tokens read() wins. L-28's 1.388x page-touch penalty survives "
                          f"contact with a real runtime. Contention biases toward 1.0, so this is "
                          f"a lower bound.")
    elif r <= 1.00:
        res["verdict"] = (f"P2 SUPPORTED - L-28 DOES NOT TRANSFER. --no-mmap is {r}x, i.e. no "
                          f"faster or slower. llama.cpp's documentation is right end to end, my "
                          f"page-touch measurement does not describe what a real loader does, and "
                          f"L-28's practical half is REFUTED. Published as a miss.")
    else:
        res["verdict"] = (f"INCONCLUSIVE - {r}x sits in the pre-declared dead band (1.00, 1.15).")
    print("\n" + res["verdict"])
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
