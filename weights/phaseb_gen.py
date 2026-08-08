"""Phase B4: test-generation + committee problems -> the screened, verified training corpus.

  python weights/phaseb_gen.py --probe    # 3 live samples end-to-end, then stop (box check)
  python weights/phaseb_gen.py --run      # full overnight pass -> phaseb_corpus.jsonl + verdict

Design choices, recorded per the stake (gates are staked; mechanisms are engineering):
- FEED 1 (corpus verification): the 4B writes tests - it out-scored the 7B on HumanEval+
  (Phase A promotion) and is ~10x cheaper per sample than the 30B. Weak tests are caught
  mechanically, not stylistically: every generated test set must FAIL a null-stub candidate
  or the sample is dropped and counted (the P-B2 spirit applied to feed 1 as well).
- FEED 2 (committee problems): the Coder-30B invents problem+reference+tests from seeded
  diversity templates. Gates per problem: reference passes its tests, a null stub fails
  them, a return-mutated reference fails them. Drop rates published; >30% blocks (KR-B2).
- EVERY final sample text re-passes the decontamination screen - text our own models
  generate can echo bench idioms, and the law does not care who wrote the byte.
- One measurement owner: shared lock family, servers sequential, probe before bulk.
"""
from __future__ import annotations
import argparse, json, os, random, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from p0_lanes import start_server, stop_server, ask, gpu_state, extract_code   # noqa: E402
from gridbench import MODELS, tk                                                # noqa: E402
import decon                                                                     # noqa: E402

import runner  # noqa: E402
DATA = os.path.join(HERE, "data")
SEED = 20260805
FEED1_N = 5000
FEED2_N = 500
SLICE = (0, FEED1_N)          # --continue-slice sets a disjoint slice of the seeded shuffle
TAG = ""
EXEC_TIMEOUT = 10.0

TOPICS = ["string parsing", "list and dict transformations", "date and interval arithmetic",
          "graph or tree traversal", "regex extraction", "csv/record munging",
          "numeric methods without numpy", "state machines", "interval scheduling",
          "text diffing", "caching and memoization", "validation of structured input"]
LEVELS = ["easy", "medium", "hard-but-testable"]


def run_asserts(code, asserts_src, timeout=EXEC_TIMEOUT):
    """True iff code + asserts execute cleanly. Candidate text is hostile input - and it
    WRITES FILES: B4's generated code left 44 artifacts in the repo root before cwd
    isolation (2026-08-06). Side-effects now land in a temp dir and die with it."""
    import tempfile
    child = "import math, re, json, itertools, collections, functools, datetime\n" \
            + code + "\n" + asserts_src + "\nprint('PB_OK')\n"
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True,
                               timeout=timeout, cwd=td)
        return "PB_OK" in p.stdout
    except Exception:
        return False


def null_stub(code):
    """Replace every function body with `return None` - the candidate no test should pass."""
    out, indent = [], None
    for line in code.splitlines():
        if re.match(r"\s*def \w+\(", line):
            out.append(line); indent = (len(line) - len(line.lstrip())) + 4
            out.append(" " * indent + "return None")
        elif indent is not None and (not line.strip() or len(line) - len(line.lstrip()) > indent - 4):
            continue                       # drop original body lines
        else:
            indent = None; out.append(line)
    return "\n".join(out)


def fn_name(code):
    m = re.search(r"^def (\w+)\(", code, flags=re.M) or re.search(r"def (\w+)\(", code)
    return m.group(1) if m else None


# ------------------------------------------------------------------ feed 1: verify corpus
def feed1_prompt(instruction, response, fname):
    return (f"A student solved this task:\n\nTASK: {instruction}\n\nSOLUTION:\n```python\n"
            f"{response}\n```\n\nWrite EXACTLY 5 assert statements that test `{fname}` "
            f"thoroughly (edge cases included). Return ONLY the asserts in a ```python "
            f"block - no imports, no prose, no function definitions.")


def feed1(rows, k, log, probe=False):
    kept, dropped = [], {"no_fn": 0, "gen_fail": 0, "null_passed": 0, "ref_failed": 0,
                         "screen": 0}
    for n, r in enumerate(rows):
        code = extract_code(r["response"]) or r["response"]
        fname = fn_name(code)
        if not fname:
            dropped["no_fn"] += 1; continue
        txt, trunc = ask(feed1_prompt(r["instruction"], code, fname), 0.3, SEED + n,
                         npredict=512, template_kwargs=tk("4B"))
        asserts = extract_code(txt)
        if trunc or asserts.count("assert") < 3:
            dropped["gen_fail"] += 1; continue
        if run_asserts(null_stub(code), asserts):        # tests must REJECT a gutted solution
            dropped["null_passed"] += 1; continue
        if not run_asserts(code, asserts):               # and ACCEPT the real one
            dropped["ref_failed"] += 1; continue
        sample = dict(source="self-oss-instruct/" + str(r["id"]),
                      instruction=r["instruction"], response=code, tests=asserts)
        ok, why = decon.screen_one(sample["instruction"] + "\n" + code + "\n" + asserts,
                                   *SCREEN)
        if not ok:
            dropped["screen"] += 1; continue
        kept.append(sample)
        if (n + 1) % 100 == 0 or probe:
            log(f"  [feed1] {n+1}/{len(rows)} kept {len(kept)} dropped {dropped}")
        if probe and len(kept) >= 2:
            break
    return kept, dropped


# ------------------------------------------------------------------ feed 2: committee problems
def feed2_prompt(topic, level, i):
    # Step 1 of two: problem + solution only. Tests come from a SECOND low-temp call written
    # AGAINST the solution (the feed-1 pattern) - the probe measured one-shot
    # problem+solution+tests at temp 0.8 self-contradicting (ref_failed), and coherence by
    # construction beats coherence by luck.
    return (f"Invent ONE original {level} Python programming problem about {topic}. "
            f"Respond with ONLY a JSON object, no markdown fence, with keys: "
            f'"problem" (the task statement, self-contained, names the required function) and '
            f'"solution" (a complete function definition solving it). Variation seed: {i}.')


def mutate_returns(code):
    """Nullify EVERY return value. The probe caught first-return-only mutation hitting a guard
    clause and changing nothing - a mutation the tests cannot see is not a mutation."""
    return re.sub(r"return\s+.+", "return None", code)


def feed2(n_problems, log, probe=False):
    rng = random.Random(SEED)
    kept, dropped = [], {"parse": 0, "no_fn": 0, "gen_fail": 0, "ref_failed": 0,
                         "null_passed": 0, "mut_passed": 0, "screen": 0}
    combos = [(t, l) for t in TOPICS for l in LEVELS]
    for i in range(n_problems):
        topic, level = combos[rng.randrange(len(combos))]
        txt, trunc = ask(feed2_prompt(topic, level, i), 0.8, SEED + 10000 + i, npredict=1024)
        try:
            m = re.search(r"\{.*\}", txt, flags=re.S)
            d = json.loads(m.group(0))
            prob, sol = d["problem"], d["solution"]
        except Exception:
            dropped["parse"] += 1; continue
        fname = fn_name(sol)
        if not fname:
            dropped["no_fn"] += 1; continue
        ttxt, ttr = ask(feed1_prompt(prob, sol, fname), 0.3, SEED + 20000 + i,
                        npredict=512)
        tests = extract_code(ttxt)
        if ttr or tests.count("assert") < 3:
            dropped["gen_fail"] += 1; continue
        if not run_asserts(sol, tests):
            # v2 (KR-B2 remedy, staked in the Phase B verdict): ONE repair pass targeting the
            # measured pathology - test-writer overconfidence (54.8% of the 30B's own tests
            # asserted wrong expected values). The model re-derives each expected value by
            # executing its own solution mentally; no second repair, no bar-lowering.
            rtxt, rtr = ask(
                f"These asserts test the function below, but some EXPECTED VALUES are wrong.\n\n"
                f"```python\n{sol}\n```\n\nASSERTS:\n```python\n{tests}\n```\n\n"
                f"Recompute each expected value by carefully executing the function step by "
                f"step on each input. Return ONLY the 5 corrected assert statements in a "
                f"```python block.", 0.2, SEED + 30000 + i, npredict=512)
            tests2 = extract_code(rtxt)
            if rtr or tests2.count("assert") < 3 or not run_asserts(sol, tests2):
                dropped["ref_failed"] += 1; continue
            tests = tests2
            dropped["repaired"] = dropped.get("repaired", 0) + 1
        if run_asserts(null_stub(sol), tests):
            dropped["null_passed"] += 1; continue
        if run_asserts(mutate_returns(sol), tests):
            dropped["mut_passed"] += 1; continue
        ok, why = decon.screen_one(prob + "\n" + sol + "\n" + tests, *SCREEN)
        if not ok:
            dropped["screen"] += 1; continue
        kept.append(dict(source=f"committee/{topic}/{level}/{i}", instruction=prob,
                         response=sol, tests=tests))
        if (i + 1) % 50 == 0 or probe:
            log(f"  [feed2] {i+1}/{n_problems} kept {len(kept)} dropped {dropped}")
        if probe and kept:
            break
    return kept, dropped


SCREEN = None


def main(probe=False):
    global SCREEN
    # Shared lock discipline (weights/runner.py): this file guarded 4 of 5 and missed
    # .ev1_lock, so it could have started mid-EV-1-night and contended for the GPU.
    try:
        ctx = runner.owns_the_box(".phaseb_lock", DATA, kill=False)
        ctx.__enter__()
    except runner.BoxBusy as e:
        print(e); return 3
    logp = os.path.join(DATA, "phaseb_gen.log")
    lf = open(logp, "a", encoding="utf-8")
    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True); lf.write(line + "\n"); lf.flush()
    try:
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
        time.sleep(2)
        h, g, meta = decon.load_protected()
        SCREEN = (h, g)
        log(f"screen loaded: {meta['n_grams']:,} grams (KR-B1 satisfied for this run)")
        import pyarrow.parquet as pq
        ledger = json.load(open(os.path.join(DATA, "phaseb_screen_ledger.json")))
        keep_ids = set(ledger["kept_ids"])
        tab = pq.read_table("D:/evo-compress-data/corpora/self-oss-instruct/data/"
                            "train-00000-of-00001.parquet")
        rows = [r for r in tab.select(["id", "instruction", "response"]).to_pylist()
                if r["id"] in keep_ids]
        random.Random(SEED).shuffle(rows)
        lo, hi = SLICE
        rows = rows[lo:lo + 6] if probe else rows[lo:hi]
        log(f"feed1 slice [{lo}:{hi}] of {len(keep_ids)} screen-clean rows (seed {SEED}, "
            f"disjoint from prior slices by shuffle-order construction)")

        gpu_state("feed1 pre", log)
        proc, _ = start_server(MODELS["4B"], 8, ctx_per_slot=2048,
                               extra=("-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on"))
        if proc is None:
            log("feed1 SERVER FAILED"); return 1
        f1, d1 = feed1(rows, 8, log, probe=probe)
        stop_server(proc)
        gpu_state("feed1 post", log)
        log(f"feed1: kept {len(f1)}, dropped {d1}")

        gpu_state("feed2 pre", log)
        proc, _ = start_server(MODELS["30B"], 2, ctx_per_slot=3072)
        if proc is None:
            log("feed2 SERVER FAILED"); return 1
        f2, d2 = feed2(2 if probe else FEED2_N, log, probe=probe)
        stop_server(proc)
        gpu_state("feed2 post", log)
        log(f"feed2: kept {len(f2)}, dropped {d2}")

        out = os.path.join(DATA, f"phaseb_corpus{TAG}.jsonl" if not probe else "phaseb_probe.jsonl")
        with open(out, "w", encoding="utf-8") as fh:
            for s in f1 + f2:
                fh.write(json.dumps(s) + "\n")
        n_total = len(f1) + len(f2)
        f2_attempted = (2 if probe else FEED2_N)
        drop_rate2 = 1 - len(f2) / max(1, f2_attempted)
        verdict = dict(feed1_kept=len(f1), feed1_dropped=d1, feed2_kept=len(f2),
                       feed2_dropped=d2, total=n_total,
                       PB1_ge_3000=(n_total >= 3000), PB2_ge_500=(len(f2) >= 500),
                       KRB2_drop_rate=round(drop_rate2, 3),
                       KRB2_blocks=(drop_rate2 > 0.30), probe=probe)
        json.dump(verdict, open(os.path.join(DATA, f"phaseb_verdict{TAG}.json"), "w"), indent=1)
        log("=== STAKED GATES ===")
        log(f"  P-B1 (>=3000 verified): {n_total} -> " + ("PASS" if verdict['PB1_ge_3000'] else "FAIL/pending"))
        log(f"  P-B2 (>=500 committee): {len(f2)} -> " + ("PASS" if verdict['PB2_ge_500'] else "FAIL/pending"))
        log(f"  KR-B2 (drop rate): {drop_rate2:.1%} -> " + ("BLOCKS" if verdict['KRB2_blocks'] else "ok"))
        return 0
    finally:
        runner.kill_orphans()
        ctx.__exit__(None, None, None)       # releases the lock, even on exception
        lf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--continue-slice", action="store_true",
                    help="B4b: feed1 rows [5000:7500] + feed2 v2 (repair loop), outputs _b files")
    a = ap.parse_args()
    if a.continue_slice:
        # set the RUNNING module's globals - the import-myself trick creates a second module
        # instance under __main__ and main() would still read the stale values
        globals()["SLICE"] = (5000, 7500)
        globals()["TAG"] = "_b"
    sys.exit(main(probe=a.probe) if (a.probe or a.run or a.continue_slice) else 0)
