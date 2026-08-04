"""P0 - k verified lanes of a 7B vs one shot of the Coder-30B (prereg 2026-08-04-p0).

  python weights/p0_lanes.py --selftest     # harness checks, no server, no GPU
  python weights/p0_lanes.py --run          # pilot -> arms A/B/C/D -> verdicts (hours)

Discipline carried in from the scars this repo already paid for:
- lock is mkdir-or-refuse (an existing lock REFUSES; `ls || mkdir` accepted stale locks and a
  revived parent loop double-ran S-1's phase 1);
- servers are Popen objects killed by the SAME python process that owns them (bash $! is not a
  Windows PID);
- selection reads BASE tests only, scoring reads PLUS tests only (KR-D, staked);
- every reference solution must pass its own plus tests here before its task may score (KR-B);
- truncated generations are quarantined, never scored as wrong-shaped answers.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
LOCK = os.path.join(DATA, ".p0_lock")
GGUF = "D:/evo-compress-data/gguf"
LLAMA = os.environ.get("QUANTPROBE_LLAMA_DIR", "D:/evo-compress-data/llama-b10242")
PORT = 8093
SEED = 20260804
N_TASKS = 120
K_MAX = 16
NPRED = 768
EXEC_TIMEOUT = 10.0

ROWS = {
    "0.6B": os.path.join(GGUF, "Qwen3-0.6B-Q8_0.gguf"),
    "7B":   os.path.join(GGUF, "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    "30B":  os.path.join(GGUF, "Qwen3-Coder-30B-A3B-Instruct-Q2_K_L.gguf"),
}


# ---------------------------------------------------------------- tasks + sandbox
def load_tasks():
    """Seeded 120-task subset of MBPP+, with the base/plus split made explicit."""
    from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
    import random
    all_t = get_mbpp_plus()
    ids = sorted(all_t.keys())
    random.Random(SEED).shuffle(ids)
    picked = sorted(ids[:N_TASKS])
    tasks = {i: all_t[i] for i in picked}
    return tasks, get_mbpp_plus_hash()


def run_candidate(code, entry, inputs, expected, timeout=EXEC_TIMEOUT):
    """Execute candidate code against (inputs -> expected) pairs in a subprocess.
    Returns (n_pass, n_total). Any crash/timeout/mismatch = fail on that input set."""
    harness = (
        "import json, sys, math\n"
        + code + "\n"
        "def _same(a, b):\n"
        "    if isinstance(a, float) or isinstance(b, float):\n"
        "        try:\n"
        "            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)\n"
        "        except TypeError:\n"
        "            return a == b\n"
        "    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):\n"
        "        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))\n"
        "    return a == b\n"
        "_IN = json.loads(sys.argv[1]); _EXP = json.loads(sys.argv[2])\n"
        "ok = 0\n"
        "for _i, _e in zip(_IN, _EXP):\n"
        f"    try:\n"
        f"        _r = {'{}'}\n"
        "    except Exception:\n"
        "        continue\n"
        "    if _same(_r, _e):\n"
        "        ok += 1\n"
        "print(ok)\n"
    )
    # the call line is injected per-entry-point; arguments splat from the input row
    harness = harness.replace("_r = {}", f"_r = {entry}(*_i)")
    try:
        p = subprocess.run([sys.executable, "-c", harness,
                            json.dumps(inputs), json.dumps(expected)],
                           capture_output=True, text=True, timeout=timeout)
        n = int(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else 0
    except Exception:
        n = 0
    return n, len(inputs)


def expected_outputs(task, inputs):
    """Ground truth = the reference solution executed HERE, on our sandbox, our floats.
    Comparing candidates against locally-computed expectations removes every serialization
    ambiguity between us and upstream evalplus internals."""
    code = task["prompt"] + task["canonical_solution"]
    harness = (
        "import json, sys\n" + code + "\n"
        "_IN = json.loads(sys.argv[1])\n"
        "out = []\n"
        "for _i in _IN:\n"
        f"    out.append({task['entry_point']}(*_i))\n"
        "print(json.dumps(out, default=str))\n"
    )
    p = subprocess.run([sys.executable, "-c", harness, json.dumps(inputs)],
                       capture_output=True, text=True, timeout=60)
    return json.loads(p.stdout.strip().splitlines()[-1])


def extract_code(txt):
    m = re.search(r"```(?:python)?\s*(.*?)```", txt, flags=re.S)
    if m:
        return m.group(1).strip()
    i = txt.find("def ")
    return txt[i:].strip() if i >= 0 else txt.strip()


# ---------------------------------------------------------------- server + ask
def qp_flags(gguf):
    """The tool plans its own experiment: quantprobe's best_flags for this file, this box."""
    from quantprobe.cli import build_parser
    from quantprobe import runtime
    a = build_parser().parse_args(["run", "--gguf", gguf])
    best, flags = runtime.best_flags(a)
    return best, [f for f in flags if f]


def start_server(gguf, np_, ctx_per_slot=1024, extra=()):
    exe = os.path.join(LLAMA, "llama-server.exe")
    _, flags = qp_flags(gguf)
    cmd = ([exe, "-m", gguf] + flags
           + ["-c", str(ctx_per_slot * np_), "-np", str(np_), "--port", str(PORT),
              "--host", "127.0.0.1"] + list(extra))
    logp = os.path.join(DATA, f"p0_server_{os.path.basename(gguf)[:20]}_np{np_}.log")
    proc = subprocess.Popen(cmd, stdout=open(logp, "w"), stderr=subprocess.STDOUT)
    import urllib.request
    for _ in range(180):
        time.sleep(2)
        if proc.poll() is not None:
            return None, cmd            # died (OOM etc.) - that is DATA, not an error
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as fh:
                if json.loads(fh.read().decode()).get("status") == "ok":
                    return proc, cmd
        except Exception:
            pass
    proc.kill()
    return None, cmd


def stop_server(proc):
    if proc and proc.poll() is None:
        proc.kill()
        try:
            proc.wait(timeout=15)
        except Exception:
            pass
    time.sleep(3)


def ask(prompt, temp, seed, npredict=NPRED):
    import urllib.request
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": npredict, "temperature": temp,
                       "top_p": 0.95 if temp > 0 else 1.0, "seed": seed}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as fh:
        d = json.loads(fh.read().decode("utf-8", "replace"))
    txt = d["choices"][0]["message"].get("content") or ""
    if "<think>" in txt:
        txt = re.sub(r"<think>.*?(?:</think>|$)", "", txt, flags=re.S)
    fin = d["choices"][0].get("finish_reason", "")
    return txt.strip(), (fin == "length" and not extract_code(txt).strip())


def prompt_for(task):
    return ("Complete this Python function. Return ONLY the complete function definition "
            "in a ```python code block, no explanation.\n\n```python\n"
            + task["prompt"].rstrip() + "\n```")


# ---------------------------------------------------------------- phases
def gpu_state(tag, log):
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        log(f"[gpu {tag}] {q.stdout.strip()}")
    except Exception as e:
        log(f"[gpu {tag}] unavailable: {e}")


def selftest():
    """KR-B + KR-D on the full subset, plus one mutation check - before any server exists."""
    tasks, h = load_tasks()
    print(f"MBPP+ hash {h}; subset {len(tasks)} tasks, seed {SEED}")
    excluded, superset = [], 0
    for tid, t in tasks.items():
        base, plus = t["base_input"], t["plus_input"]
        allin = base + plus
        if len(plus) > 0:
            superset += 1
        try:
            exp = expected_outputs(t, allin)
            code = t["prompt"] + t["canonical_solution"]
            n, tot = run_candidate(code, t["entry_point"], allin, exp, timeout=60)
            if n != tot:
                excluded.append((tid, f"reference passes {n}/{tot}"))
        except Exception as e:
            excluded.append((tid, f"reference crashed: {e}"))
    print(f"KR-B: {len(excluded)} excluded of {len(tasks)}"
          + (f" -> {excluded[:5]}" if excluded else ""))
    if len(excluded) > 0.10 * len(tasks):
        print("KR-B PRECONDITION-BLOCK: >10% of references fail our sandbox"); return 2
    if superset < 0.90 * len(tasks):
        print("KR-D PRECONDITION-BLOCK: plus set not a strict extension on >=90%"); return 2
    # mutation check: a broken candidate must fail the same harness the reference passed
    t = tasks[sorted(tasks)[0]]
    exp = expected_outputs(t, t["base_input"])
    n, tot = run_candidate("def " + t["entry_point"] + "(*a, **k):\n    return None\n",
                           t["entry_point"], t["base_input"], exp)
    if n != 0:
        print("MUTATION FAIL: null candidate passed tests"); return 1
    ok_ids = [i for i in tasks if i not in {x[0] for x in excluded}]
    json.dump({"seed": SEED, "hash": h, "task_ids": ok_ids,
               "excluded": excluded}, open(os.path.join(DATA, "p0_subset.json"), "w"), indent=1)
    print(f"selftest OK -> p0_subset.json ({len(ok_ids)} tasks)")
    return 0


def score_arm(tag, tasks, temp, k, log):
    """Run one arm against the live server. k=1 -> single greedy shot; k>1 -> lanes with
    base-test selection. Returns per-task records."""
    from concurrent.futures import ThreadPoolExecutor
    recs = []
    for n, (tid, t) in enumerate(sorted(tasks.items())):
        t0 = time.time()
        prompt = prompt_for(t)
        if k == 1:
            outs = [ask(prompt, 0.0, SEED)]
        else:
            with ThreadPoolExecutor(max_workers=k) as ex:
                outs = list(ex.map(lambda i: ask(prompt, temp, 1000 + i), range(k)))
        cands = [(extract_code(txt), trunc) for txt, trunc in outs]
        n_trunc = sum(1 for _, tr in cands if tr)
        exp_base = t["_exp_base"]; exp_plus = t["_exp_plus"]
        if k == 1:
            winner = cands[0][0]
        else:
            scored = []
            for code, tr in cands:
                if tr or not code:
                    continue
                nb, _ = run_candidate(code, t["entry_point"], t["base_input"], exp_base)
                scored.append((nb, -len(code), code))       # KR-D: BASE tests only
            scored.sort(reverse=True)
            winner = scored[0][2] if scored else (cands[0][0] if cands else "")
        np_, tot = run_candidate(winner, t["entry_point"], t["plus_input"], exp_plus) \
            if winner else (0, len(t["plus_input"]))
        solved = (np_ == tot and tot > 0)
        wall = time.time() - t0
        recs.append(dict(tid=tid, solved=solved, wall=round(wall, 2),
                         plus=f"{np_}/{tot}", truncated=n_trunc))
        log(f"  [{tag}] {n+1}/{len(tasks)} {tid} {'PASS' if solved else 'fail'} "
            f"({wall:.1f}s, trunc {n_trunc}/{max(k,1)})")
    return recs


def pilot(log):
    """-np sweep on the 7B row: k = largest np <= 16 with aggregate >= 0.8x peak (staked)."""
    import urllib.request
    agg = {}
    for np_ in (2, 4, 8, 12, 16):
        proc, cmd = start_server(ROWS["7B"], np_)
        if proc is None:
            log(f"[pilot] np={np_}: server did not come up (OOM?) - recorded, skipping")
            agg[np_] = 0.0
            continue
        t0 = time.time()
        from concurrent.futures import ThreadPoolExecutor
        prompt = "Write a Python function that reverses a string. Code only."
        with ThreadPoolExecutor(max_workers=np_) as ex:
            outs = list(ex.map(lambda i: ask(prompt, 0.8, 500 + i, npredict=256), range(np_)))
        dt = time.time() - t0
        toks = np_ * 256          # upper bound; all lanes run to budget on this prompt class
        agg[np_] = toks / dt
        log(f"[pilot] np={np_}: {agg[np_]:.1f} tok/s aggregate ({dt:.1f}s)")
        stop_server(proc)
    peak = max(agg.values()) if agg else 0.0
    ks = [n for n, v in agg.items() if peak and v >= 0.8 * peak]
    k = max(ks) if ks else 1
    log(f"[pilot] chosen k={k} (peak {peak:.1f}, rule: largest np with >=0.8x peak)")
    return k, agg


def run():
    os.makedirs(DATA, exist_ok=True)
    try:
        os.mkdir(LOCK)                      # mkdir-or-refuse: an existing lock REFUSES
    except FileExistsError:
        print(f"REFUSED: {LOCK} exists - another measurement owns the box"); return 3
    logp = os.path.join(DATA, "p0_run.log")
    lf = open(logp, "a", encoding="utf-8")
    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True); lf.write(line + "\n"); lf.flush()
    try:
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                       capture_output=True)  # orphans die BEFORE state is logged
        time.sleep(3)
        rc = selftest()
        if rc != 0:
            return rc
        sub = json.load(open(os.path.join(DATA, "p0_subset.json")))
        tasks, _ = load_tasks()
        tasks = {i: tasks[i] for i in sub["task_ids"]}
        log(f"precomputing expected outputs for {len(tasks)} tasks")
        for t in tasks.values():
            t["_exp_base"] = expected_outputs(t, t["base_input"])
            t["_exp_plus"] = expected_outputs(t, t["plus_input"])
        k, agg = pilot(log)
        results = {"k": k, "pilot_agg": agg, "arms": {}}
        arms = [("B_7B_single", "7B", 0.0, 1), ("C_7B_lanes", "7B", 0.8, k),
                ("A_30B_single", "30B", 0.0, 1),
                ("D_06B_single", "0.6B", 0.0, 1), ("D_06B_lanes", "0.6B", 0.8, k)]
        for tag, row, temp, kk in arms:
            gpu_state(f"{tag} pre", log)
            proc, cmd = start_server(ROWS[row], kk)
            if proc is None:
                log(f"[{tag}] SERVER FAILED - arm recorded as unrunnable");
                results["arms"][tag] = {"unrunnable": True, "cmd": cmd}
                continue
            log(f"[{tag}] server up: {' '.join(map(str, cmd[2:]))[:160]}")
            recs = score_arm(tag, tasks, temp, kk, log)
            stop_server(proc)
            gpu_state(f"{tag} post", log)
            solved = sum(r["solved"] for r in recs)
            walls = sorted(r["wall"] for r in recs)
            med = walls[len(walls)//2]
            results["arms"][tag] = {"solved": solved, "total": len(recs),
                                    "rate": solved/len(recs), "median_wall": med,
                                    "records": recs}
            log(f"[{tag}] {solved}/{len(recs)} = {100*solved/len(recs):.1f}%  "
                f"median {med:.1f}s/task")
        A = results["arms"].get("A_30B_single", {})
        B = results["arms"].get("B_7B_single", {})
        C = results["arms"].get("C_7B_lanes", {})
        if all(x and not x.get("unrunnable") for x in (A, B, C)):
            p1 = C["rate"] >= A["rate"] + 0.05
            p2 = C["median_wall"] <= A["median_wall"]
            p3 = C["rate"] >= B["rate"] + 0.10
            kra_void = not (B["rate"] <= A["rate"] - 0.05)
            results["verdicts"] = {
                "P1_C_beats_A_5pts": {"C": C["rate"], "A": A["rate"], "pass": p1},
                "P2_wallclock": {"C": C["median_wall"], "A": A["median_wall"], "pass": p2},
                "P3_selection_real": {"C": C["rate"], "B": B["rate"], "pass": p3},
                "KRA_no_headroom_voids_P1": kra_void,
            }
            log("=== STAKED VERDICTS ===")
            log(f"  P1 (C >= A+5pts): C {C['rate']:.1%} vs A {A['rate']:.1%} -> "
                + ("PASS" if p1 else "FAIL"))
            log(f"  P2 (wall-clock):  C {C['median_wall']:.1f}s vs A {A['median_wall']:.1f}s -> "
                + ("PASS" if p2 else "FAIL"))
            log(f"  P3 (C >= B+10pts): C {C['rate']:.1%} vs B {B['rate']:.1%} -> "
                + ("PASS" if p3 else "FAIL"))
            log(f"  KR-A: 7B-vs-30B gap "
                + ("TOO SMALL - P1 VOID as beat-the-teacher" if kra_void else "real - P1 stands"))
        json.dump(results, open(os.path.join(DATA, "p0_results.json"), "w"), indent=1)
        log("results -> p0_results.json")
        return 0
    finally:
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
        try:
            os.rmdir(LOCK)
        except OSError:
            pass
        lf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else (run() if a.run else 0))
