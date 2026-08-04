"""Phase A baseline grid (prereg 2026-08-05-benchmark-sanctity-and-baseline-grid.md).

  python weights/gridbench.py --selftest    # KR-A1 both benches full, KR-A4 cross-bench, no GPU
  python weights/gridbench.py --run         # all rows, resumable; ~2 nights at worst

Rows: {0.6B, 4B, 7B} x {single, lanes16} + 30B single, on MBPP+ (378) and HumanEval+ (164).
30B lanes excluded per U-39 (MoE offload batching caps ~2.0x; ~10 GPU-hours for nothing).
4B lanes deferred to last (not needed by any staked prediction).

Inherited from P0 verbatim: sandbox, selection honesty (base tests pick, plus tests score),
truncation quarantine, mkdir-or-refuse lock, sequential rows, GPU state logging.
New here, with reasons in the stake:
- Qwen3-family rows get the thinking soft-switch off (/no_think) + npredict 1024 - the P0
  0.6B-lanes truncation lesson fixed, not re-suffered (KR-A2 still measures it).
- Lanes rows run q8_0 KV + -fa: 16 slots x 2048 ctx of f16 KV on the 7B is ~1.8 GB and does
  not fit beside 4.7 GB of weights on a 6 GB card. U-01 measured the speed side (+37% at
  depth), autotune measured +1.6% here; the config is recorded in every row JSON and P-A1's
  +/-6 band is what absorbs the P0-vs-grid config delta.
- Resumable: one JSON per (bench, model, mode); a complete file is never re-measured.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
from p0_lanes import (ROWS, ask, extract_code, run_candidate, expected_outputs,   # noqa: E402
                      start_server, stop_server, gpu_state, LOCK as P0_LOCK)

GRID_LOCK = os.path.join(DATA, ".grid_lock")
AUTOTUNE_LOCK = os.path.join(DATA, ".autotune_lock")
K = 16                        # inherited from P0's pilot on this same box/7B; not re-swept
NPRED = 1024
GGUF_4B = "D:/evo-compress-data/gguf/Qwen3.5-4B-Q4_K_M.gguf"
THINKING_FAMILY = {"0.6B", "4B"}          # Qwen3/Qwen3.5 - soft-switch capable

MODELS = dict(ROWS)
MODELS["4B"] = GGUF_4B


def load_bench(name):
    if name == "mbpp":
        from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
        return get_mbpp_plus(), get_mbpp_plus_hash()
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
    return get_human_eval_plus(), get_human_eval_plus_hash()


def prompt_for(task, model):
    p = ("Complete this Python function. Return ONLY the complete function definition "
         "in a ```python code block, no explanation.\n\n```python\n"
         + task["prompt"].rstrip() + "\n```")
    if model in THINKING_FAMILY:
        p += "\n/no_think"
    return p


def kr_a1_reference_check(bench, tasks):
    """Every reference must pass its own plus tests HERE. Excluded-and-counted, >10% blocks."""
    excluded = []
    for tid, t in tasks.items():
        try:
            allin = t["base_input"] + t["plus_input"]
            exp = expected_outputs(t, allin)
            code = t["prompt"] + t["canonical_solution"]
            n, tot = run_candidate(code, t["entry_point"], allin, exp, timeout=60)
            if n != tot:
                excluded.append((tid, f"reference passes {n}/{tot}"))
        except Exception as e:
            excluded.append((tid, f"reference crashed: {str(e)[:80]}"))
    return excluded


def row_path(bench, model, mode):
    return os.path.join(DATA, f"grid_{bench}_{model}_{mode}.json")


def row_complete(bench, model, mode, n_tasks):
    p = row_path(bench, model, mode)
    if not os.path.isfile(p):
        return False
    try:
        d = json.load(open(p, encoding="utf-8"))
        return len(d.get("records", [])) >= n_tasks
    except Exception:
        return False


def score_row(bench, model, mode, tasks, log):
    """One (bench, model, mode) cell. Server up -> every task -> JSON. P0's score_arm shape."""
    from concurrent.futures import ThreadPoolExecutor
    k = 1 if mode == "single" else K
    extra = () if mode == "single" else ("-ctk", "q8_0", "-ctv", "q8_0", "-fa")
    ctx = 4096 if mode == "single" else 2048
    gpu_state(f"{bench}/{model}/{mode} pre", log)
    proc, cmd = start_server(MODELS[model], k, ctx_per_slot=ctx, extra=extra)
    if proc is None:
        log(f"[{bench}/{model}/{mode}] SERVER FAILED - row recorded unrunnable")
        json.dump({"unrunnable": True, "cmd": [str(c) for c in cmd]},
                  open(row_path(bench, model, mode), "w"), indent=1)
        return
    log(f"[{bench}/{model}/{mode}] server up (k={k}, ctx/slot={ctx})")
    recs = []
    t_row = time.time()
    for n, (tid, t) in enumerate(sorted(tasks.items())):
        t0 = time.time()
        prompt = prompt_for(t, model)
        if k == 1:
            outs = [ask(prompt, 0.0, 20260805, npredict=NPRED)]
        else:
            with ThreadPoolExecutor(max_workers=k) as ex:
                outs = list(ex.map(lambda i: ask(prompt, 0.8, 1000 + i, npredict=NPRED),
                                   range(k)))
        cands = [(extract_code(txt), tr) for txt, tr in outs]
        n_trunc = sum(1 for _, tr in cands if tr)
        if k == 1:
            winner = cands[0][0]
        else:
            scored = []
            for code, tr in cands:
                if tr or not code:
                    continue
                nb, _ = run_candidate(code, t["entry_point"], t["base_input"], t["_exp_base"])
                scored.append((nb, -len(code), code))          # base tests ONLY pick
            scored.sort(reverse=True)
            winner = scored[0][2] if scored else (cands[0][0] if cands else "")
        np_, tot = run_candidate(winner, t["entry_point"], t["plus_input"], t["_exp_plus"]) \
            if winner else (0, len(t["plus_input"]))
        recs.append(dict(tid=tid, solved=(np_ == tot and tot > 0), wall=round(time.time()-t0, 2),
                         plus=f"{np_}/{tot}", truncated=n_trunc))
        if (n + 1) % 25 == 0 or n + 1 == len(tasks):
            el = time.time() - t_row
            log(f"  [{bench}/{model}/{mode}] {n+1}/{len(tasks)} "
                f"({sum(r['solved'] for r in recs)} solved, {el/60:.0f}m elapsed, "
                f"ETA {el/(n+1)*(len(tasks)-n-1)/60:.0f}m)")
    stop_server(proc)
    gpu_state(f"{bench}/{model}/{mode} post", log)
    solved = sum(r["solved"] for r in recs)
    walls = sorted(r["wall"] for r in recs)
    degraded_tasks = sum(1 for r in recs if k > 1 and r["truncated"] >= k / 2)
    out = dict(bench=bench, model=model, mode=mode, k=k, ctx_per_slot=ctx,
               kv=("q8_0+fa" if k > 1 else "f16"), npredict=NPRED,
               solved=solved, total=len(recs), rate=round(solved/len(recs), 4),
               median_wall=walls[len(walls)//2],
               degraded=bool(degraded_tasks > 0.20 * len(recs)),
               degraded_tasks=degraded_tasks, records=recs)
    json.dump(out, open(row_path(bench, model, mode), "w"), indent=1)
    log(f"[{bench}/{model}/{mode}] {solved}/{len(recs)} = {100*solved/len(recs):.1f}%  "
        f"median {out['median_wall']:.1f}s"
        + ("  [DEGRADED - floor per KR-A2]" if out["degraded"] else ""))


def selftest():
    """KR-A1 on BOTH full benches + KR-A4 cross-bench disjointness. CPU only."""
    import hashlib
    ok = True
    subsets = {}
    for bench in ("mbpp", "humaneval"):
        tasks, h = load_bench(bench)
        excluded = kr_a1_reference_check(bench, tasks)
        keep = {i: t for i, t in tasks.items() if i not in {x[0] for x in excluded}}
        print(f"{bench}+: {len(tasks)} tasks, hash {h}, KR-A1 excluded {len(excluded)}"
              + (f" {excluded[:4]}" if excluded else ""))
        if len(excluded) > 0.10 * len(tasks):
            print(f"  KR-A1 PRECONDITION-BLOCK on {bench}"); ok = False
        subsets[bench] = keep
    hs = {b: {hashlib.sha256(t["prompt"].encode()).hexdigest() for t in s.values()}
          for b, s in subsets.items()}
    inter = hs["mbpp"] & hs["humaneval"]
    print(f"KR-A4 cross-bench prompt-hash overlap: {len(inter)}")
    if inter:
        ok = False
    json.dump({b: sorted(s.keys()) for b, s in subsets.items()},
              open(os.path.join(DATA, "grid_included_tasks.json"), "w"), indent=1)
    print("selftest", "OK" if ok else "FAILED", "-> grid_included_tasks.json")
    return 0 if ok else 1


def run():
    for l in (P0_LOCK, AUTOTUNE_LOCK):
        if os.path.isdir(l):
            print(f"REFUSED: {l} exists"); return 3
    try:
        os.mkdir(GRID_LOCK)
    except FileExistsError:
        print(f"REFUSED: {GRID_LOCK} exists"); return 3
    logp = os.path.join(DATA, "grid_run.log")
    lf = open(logp, "a", encoding="utf-8")
    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True); lf.write(line + "\n"); lf.flush()
    try:
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
        time.sleep(3)
        if selftest() != 0:
            return 2
        included = json.load(open(os.path.join(DATA, "grid_included_tasks.json")))
        benches = {}
        for b in ("mbpp", "humaneval"):
            tasks, _ = load_bench(b)
            keep = {i: tasks[i] for i in included[b]}
            log(f"precomputing expected outputs: {b} ({len(keep)})")
            for t in keep.values():
                t["_exp_base"] = expected_outputs(t, t["base_input"])
                t["_exp_plus"] = expected_outputs(t, t["plus_input"])
            benches[b] = keep
        # singles first (P-A3 scoreable early), then lanes by staked priority; 4B lanes LAST
        rows = ([(b, m, "single") for b in ("mbpp", "humaneval")
                 for m in ("7B", "30B", "0.6B", "4B")]
                + [("mbpp", "7B", "lanes"), ("humaneval", "7B", "lanes"),
                   ("mbpp", "0.6B", "lanes"), ("humaneval", "0.6B", "lanes"),
                   ("mbpp", "4B", "lanes"), ("humaneval", "4B", "lanes")])
        for bench, model, mode in rows:
            if row_complete(bench, model, mode, len(benches[bench])):
                log(f"[{bench}/{model}/{mode}] already complete - skipped (resume)")
                continue
            score_row(bench, model, mode, benches[bench], log)
        log("grid pass complete")
        return 0
    finally:
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
        try:
            os.rmdir(GRID_LOCK)
        except OSError:
            pass
        lf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else (run() if a.run else 0))
