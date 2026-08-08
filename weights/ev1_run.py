"""EV-1 night runner - standard benches via lm-eval against our llama-server placements.

  python weights/ev1_run.py --night1    # 0.6B then 4B x {math500, aime24, aime25, ifeval}, + gsm8k on 0.6B
  python weights/ev1_run.py --night2    # 7B, 30B rows + gsm8k completions + gpqa attempt

Prereg 2026-08-06-ev1-standard-benches. Protocol pinned there: lm-eval 0.4.12,
thinking-as-served, temp 0, full sets, one model row per server session, resume by
existing results dir. Windows lesson baked in: the child gets PYTHONUTF8=1 because
lm-eval's results table prints U+2191 and cp1252 dies AFTER saving results.
"""
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from p0_lanes import start_server, stop_server, gpu_state   # noqa: E402
from gridbench import MODELS, THINKING_FAMILY                # noqa: E402
import runner                                                # noqa: E402

DATA = os.path.join(HERE, "data")
PORT = 8093

# A slot's context holds PROMPT + GENERATION. Asking for max_gen_toks == ctx_per_slot leaves
# ZERO room for the prompt, and the row wedges: 30B AIME24 ran 90 minutes, stopped advancing,
# and the watchdog killed it at 102m. MATH-500 survived the identical slot plan only because
# its budget (3072) left 5120 tokens of headroom. Budgets are therefore DERIVED from the slot
# with a reserve, never written next to it and left to drift apart.
PROMPT_RESERVE = 1024          # measured AIME prompts run a few hundred tokens; 1024 is slack


def gen_budget(task):
    ctx = CTX_PER_SLOT.get(task, 4096)
    return min(_GEN_WANT.get(task, 2048), ctx - PROMPT_RESERVE)


_GEN_WANT = {"gsm8k_cot_zeroshot": 2048, "math500_boxed": 3072,
             "aime24_boxed": 8192, "aime25_boxed": 8192,
             "ifeval": 2048, "gpqa_main_zeroshot": 2048}


class _GenMap(dict):
    """GEN[task] stays a string for every caller, but the number is derived, not declared."""
    def __getitem__(self, task):
        return f"max_gen_toks={gen_budget(task)}"

    def __contains__(self, task):
        return task in _GEN_WANT


GEN = _GenMap()

# Protocol v3 (amended in the prereg BEFORE any v3 row ran). Two fixes, both forced by reading
# the v2 outputs rather than the v2 exit codes:
#
# 1. hendrycks_math500 -> minerva_math500. SAME 500 items (both pull HuggingFaceH4/MATH-500),
#    but hendrycks' process_results slices the answer as "everything between the FIRST $ and
#    the LAST $" of the response and never inspects \boxed{} in the model's output at all. Any
#    model that shows its work in LaTeX therefore scores 0 by construction - which is exactly
#    what we measured: 89.4% of 0.6B responses carried a \boxed answer, scorer said 0.00%.
#    minerva_math500 extracts properly and adds a sympy-equivalence metric.
# 2. SYSTEM_INSTRUCTION, applied uniformly to every model and every generative task. AIME's
#    extractor is fine (it tries $...$, then \boxed, then is_equiv) - but nothing in its
#    zero-shot prompt ASKS for either, so the 4B wrote a bare "Answer: 49" in 30 of 30 items
#    and scored 0 while the 0.6B happened to box out of habit. Standardising the answer format
#    is what every published AIME/MATH eval does; stating it here keeps it protocol, not a
#    thumb on the scale.
SYSTEM_INSTRUCTION = ("Solve the problem. Put your final answer inside \\boxed{}.")

# ...but ONLY on tasks that are graded by extracting a boxed answer. IFEval grades literal
# instruction compliance ("respond in all lowercase", "no commas", "wrap the whole reply in
# quotes"), so a standing system instruction to box the answer is a competing instruction and
# would depress the score for a reason that has nothing to do with the model. GSM8K extracts
# from its own "The answer is X" convention and GPQA is multiple choice; neither wants it.
BOXED_TASKS = {"math500_boxed", "aime24_boxed", "aime25_boxed"}

# Tasks carrying few-shot examples need them delivered as chat turns when a chat template is
# applied; lm-eval refuses the combination otherwise. (math500_boxed is zero-shot - the answer
# format is asked for in the prompt instead of demonstrated, which is cheaper and clearer.)
FEWSHOT_TASKS = set()

# Our task definitions live in the repo so any row here is reproducible by a stranger.
TASK_PATH = os.path.join(HERE, "lm_eval_tasks")

# THE REAL LENGTH LIMIT. A slot's context holds prompt AND generation, so ctx_per_slot - prompt
# is the true generation ceiling and max_gen_toks above it is decorative. Measured directly:
# one AIME item ran 11,386 chars at max_gen_toks=4096 and 10,970 at 8192 - unchanged, because
# both runs were actually stopped by ctx_per_slot=4096 (~3,900 tokens after the prompt).
# Raising the harness budget alone was a fix that fixed nothing.
#
# Long-reasoning tasks therefore get a wider slot, paid for by halving concurrency so total KV
# (slots x ctx) stays flat and the placement does not move - changing VRAM pressure mid-suite
# would break C-14 (one machine state per comparison) far more expensively than truncation does.
# AIME goes to ONE slot. 30B/aime24 deadlocked at 8192x2 after generating all 30 answers:
# server showed 30/30 completions and released every task, lm-eval held NO tcp connection to
# 8093 and burned 0.00s CPU over 60s - nothing pending to time out, so waiting could not
# recover it. MATH-500 survived the identical 8192x2 plan; the only difference is generation
# LENGTH (3072 vs 7168 budget), which points at long concurrent generations, not the slots.
# One in-flight request removes the race entirely. slots x ctx stays 16384 either way, so KV
# footprint and placement do not move - the C-14 comparability the smoke test enforces holds.
# Cost is wall-clock: serial instead of 2-way, on a 30-item set.
CTX_PER_SLOT = {"aime24_boxed": 16384, "aime25_boxed": 16384, "math500_boxed": 8192}
CONCURRENT = {"aime24_boxed": 1, "aime25_boxed": 1, "math500_boxed": 2}


def slot_plan(task):
    """(ctx_per_slot, concurrent) for a task. Product is constant, so KV footprint is too."""
    return CTX_PER_SLOT.get(task, 4096), CONCURRENT.get(task, 4)


def out_dir(model, task, tag=""):
    return os.path.join(DATA, "ev1" + tag, model, task)


def done(model, task, tag=""):
    d = out_dir(model, task, tag)
    if not os.path.isdir(d):
        return False
    for root, _, files in os.walk(d):
        if any(f.startswith("results_") for f in files):
            return True
    return False


def build_cmd(model, task, concurrent=None, limit=None, tag=""):
    """The lm-eval invocation for one row, as data so it can be asserted without a GPU.

    Extracted because the last protocol change - scoping the boxed-answer instruction to the
    tasks that are graded by extracting a boxed answer - is exactly the kind of edit that is
    invisible until a row has already been spent on it. IFEval grades literal instruction
    compliance, so a stray system instruction there is a silent scoring bug, not a crash.
    """
    concurrent = concurrent or slot_plan(task)[1]
    # The HTTP timeout must cover the SLOWEST model finishing the LONGEST budget, or the row
    # fails without measuring anything: 600s buys ~4,900 tokens at the 30B's measured 8.2 t/s,
    # AIME budgets 8,192 - every long item timed out, retried 3x, and 166 minutes produced
    # rc=1. Sized from the budget at a 3 t/s floor (well under any measured rate), min 600.
    budget = int(GEN[task].split("=")[1])
    req_timeout = max(600, -(-budget // 3))     # ceil division: the floor-rate guarantee must hold exactly
    cmd = [sys.executable, "-m", "lm_eval", "--model", "local-chat-completions",
           "--model_args",
           f"model={model},base_url=http://127.0.0.1:{PORT}/v1/chat/completions,"
           f"num_concurrent={concurrent},max_retries=3,tokenized_requests=False,"
           f"timeout={req_timeout}",
           "--tasks", task, "--gen_kwargs", GEN[task],
           "--include_path", TASK_PATH,
           "--apply_chat_template", "--seed", "0",
           "--output_path", out_dir(model, task, tag), "--log_samples"]
    if task in BOXED_TASKS:
        cmd += ["--system_instruction", SYSTEM_INSTRUCTION]
    if task in FEWSHOT_TASKS:
        cmd.append("--fewshot_as_multiturn")
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd


STALL_MIN = 25          # no forward progress for this long = wedged, not slow
HEARTBEAT_MIN = 15      # how often a healthy row reports that it is still moving


def _progress():
    """A monotone count of finished generations, read from the live server logs.

    Deliberately measures the SERVER, not the harness: lm-eval prints nothing until a row
    completes, so the only honest evidence that work is happening is requests retiring.
    """
    n = 0
    for p in glob.glob(os.path.join(DATA, "p0_server_*.log")):
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                n += fh.read().count("print_timing")
        except OSError:
            pass
    return n


def run_watched(cmd, env, proc, model, task, log, t0):
    """Run a row with NO wall-clock cap, killed only if progress actually stops.

    A cap is wrong in both directions - it killed a healthy 30B row 103 minutes from the end,
    and it would have let a wedged row burn the same six hours doing nothing. Progress is the
    right signal: a row that is moving is never interrupted, and a row that is stuck is caught
    in ~25 minutes instead of hours. Returns the CompletedProcess, or None if it was killed
    (row skipped, night continues).
    """
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, encoding="utf-8", errors="replace", env=env)
    last_n, last_move, last_beat = _progress(), time.time(), time.time()
    while child.poll() is None:
        time.sleep(30)
        now, n = time.time(), _progress()
        if n > last_n:
            last_n, last_move = n, now
        if now - last_beat >= HEARTBEAT_MIN * 60:
            last_beat = now
            log(f"  [{model}/{task}] alive: {n} generations done, "
                f"{(now-t0)/60:.0f}m elapsed, last progress {(now-last_move)/60:.0f}m ago")
        if now - last_move >= STALL_MIN * 60:
            child.kill()
            out, err = child.communicate()
            stop_server(proc)
            gpu_state(f"{model}/{task} post (STALLED)", log)
            log(f"[{model}/{task}] STALLED - no progress for {STALL_MIN}m after "
                f"{(now-t0)/60:.0f}m and {n} generations. Row killed, night continues. "
                f"Nothing was written: lm-eval saves only on completion.")
            open(os.path.join(DATA, f"ev1_fail_{model}_{task}.txt"), "w",
                 encoding="utf-8").write(f"STALLED after {(now-t0)/60:.0f} min, "
                                         f"{n} generations\n{(out or '')[-2000:]}\n"
                                         f"{(err or '')[-2000:]}")
            return None
    out, err = child.communicate()
    return subprocess.CompletedProcess(cmd, child.returncode, out, err)


def server_extra(model):
    """Server flags for one model. v2: thinking off SERVER-SIDE for the thinking family."""
    return ("--reasoning", "off") if model in THINKING_FAMILY else ()


def run_row(model, task, log, concurrent=None, limit=None, tag=""):
    if not limit and done(model, task):
        log(f"[{model}/{task}] already complete - skipped (resume)")
        return
    ctx, conc = slot_plan(task)
    concurrent = concurrent or conc
    gpu_state(f"{model}/{task} pre (ctx {ctx} x {concurrent} slots)", log)
    # Protocol v2 (amended in the prereg BEFORE re-runs): thinking off SERVER-SIDE via
    # --reasoning off. v1's "thinking-as-served" buried thought in reasoning_content, which
    # lm-eval never sees - budgets burned invisibly, answers truncated, scores were floors.
    proc, _ = start_server(MODELS[model], concurrent, ctx_per_slot=ctx,
                           extra=server_extra(model))
    if proc is None:
        log(f"[{model}/{task}] SERVER FAILED - row recorded unrunnable")
        return
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    cmd = build_cmd(model, task, concurrent, limit, tag)
    t0 = time.time()
    # 6h was not enough and killed a row 103 minutes from the finish line. The 30B MATH-500
    # row runs 500 items at ~8 tok/s on a split placement with concurrency halved to 2 for
    # context - measured 1.08 items/min, i.e. ~7.7h. A timeout shorter than the work is not a
    # safety net, it is a silent row-shredder: six GPU-hours spent, nothing saved, and the
    # runner moves on. Sized to 12h with the arithmetic recorded so the next person can check it.
    # NO WALL-CLOCK CAP. A cap is the wrong instrument in both directions: it killed a healthy
    # 30B row 103 minutes from the finish, and it would happily let a wedged row burn the same
    # six hours making zero progress. What matters is not how long a row has taken - it is
    # whether it is STILL MOVING. So: watch progress, kill only on a stall, and never interrupt
    # work that is flowing. (Federico, 2026-08-08.)
    r = run_watched(cmd, env, proc, model, task, log, t0)
    if r is None:
        return
    stop_server(proc)
    gpu_state(f"{model}/{task} post", log)
    ok = done(model, task, tag)
    log(f"[{model}/{task}] rc={r.returncode} results={'saved' if ok else 'MISSING'} "
        f"({(time.time()-t0)/60:.0f}m)")
    if not ok:
        # The tail is the ONLY diagnostic for a failed row - it must survive a child that
        # emits bytes the parent cannot decode. (It did not: the parent's text=True used
        # cp1252, the reader thread died mid-stream, and stdout came back None.)
        tailp = os.path.join(DATA, f"ev1_fail_{model}_{task}.txt")
        blob = (r.stdout or "") + (r.stderr or "") or "(no output captured)"
        open(tailp, "w", encoding="utf-8").write(blob[-4000:])
        log(f"  failure tail -> {tailp}")


def main(night, probe=0):
    # Lock discipline lives in runner.py so the guarded set cannot drift. It had: this file
    # checked all five locks while autotune_sweep checked two, so autotune could start during
    # an EV-1 night and contend for the GPU - the overlap that voided the 2026-07-31 ladder.
    log, close_log = runner.make_log(os.path.join(DATA, "ev1_run.log"))
    try:
        ctx = runner.owns_the_box(".ev1_lock", DATA)
        ctx.__enter__()
    except runner.BoxBusy as e:
        print(e); close_log(); return 3
    try:
        if probe:
            # Checkpoint before the night owns the box: v2 was stopped only after five rows
            # had already been spent proving a protocol wrong. Scores land in ev1_probe/,
            # never in the scored tree, so a probe can never be mistaken for a published row.
            #
            # A CROSS, not a matrix. Two dimensions can break independently - the task config
            # (prompt, scorer, extra flags) and the model path (server flags, chat template,
            # placement) - so we walk each once instead of paying 20 cells. Every task runs on
            # the cheapest model, and the cheapest task runs on every model. The expensive
            # failure this exists to prevent is a 30B row dying at hour 12 of a 15-hour night
            # for a reason a 3-item run would have shown in two minutes.
            rows = ([("0.6B", t) for t in ("math500_boxed", "aime24_boxed", "aime25_boxed", "ifeval",
                                            "gsm8k_cot_zeroshot")]
                    + [(m, "gsm8k_cot_zeroshot") for m in ("4B", "7B", "30B")])
            for model, task in rows:
                run_row(model, task, log, limit=probe, tag="_probe")
            log(f"probe cross complete (limit={probe}, {len(rows)} cells)")
            return 0
        if night == 1:
            rows = ([("0.6B", t) for t in ("math500_boxed", "aime24_boxed", "aime25_boxed", "ifeval",
                                            "gsm8k_cot_zeroshot")]
                    + [("4B", t) for t in ("aime24_boxed", "aime25_boxed", "ifeval", "math500_boxed")])
        else:
            rows = ([("4B", "gsm8k_cot_zeroshot")]
                    + [("7B", t) for t in ("math500_boxed", "aime24_boxed", "aime25_boxed", "ifeval",
                                            "gsm8k_cot_zeroshot")]
                    + [("30B", t) for t in ("math500_boxed", "aime24_boxed", "aime25_boxed", "ifeval",
                                             "gsm8k_cot_zeroshot")]
                    + [(m, "gpqa_main_zeroshot") for m in ("0.6B", "4B", "7B", "30B")])
        for model, task in rows:
            run_row(model, task, log)
        log(f"night {night} pass complete")
        return 0
    finally:
        ctx.__exit__(None, None, None)     # kills orphans and releases the lock
        close_log()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--night1", action="store_true")
    ap.add_argument("--night2", action="store_true")
    ap.add_argument("--probe", type=int, default=0,
                    help="validate a protocol change on N items per task before a night runs")
    a = ap.parse_args()
    if a.probe:
        sys.exit(main(1, probe=a.probe))
    sys.exit(main(1 if a.night1 else 2) if (a.night1 or a.night2) else 0)
