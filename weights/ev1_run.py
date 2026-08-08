"""EV-1 night runner - standard benches via lm-eval against our llama-server placements.

  python weights/ev1_run.py --night1    # 0.6B then 4B x {math500, aime24, aime25, ifeval}, + gsm8k on 0.6B
  python weights/ev1_run.py --night2    # 7B, 30B rows + gsm8k completions + gpqa attempt

Prereg 2026-08-06-ev1-standard-benches. Protocol pinned there: lm-eval 0.4.12,
thinking-as-served, temp 0, full sets, one model row per server session, resume by
existing results dir. Windows lesson baked in: the child gets PYTHONUTF8=1 because
lm-eval's results table prints U+2191 and cp1252 dies AFTER saving results.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from p0_lanes import start_server, stop_server, gpu_state   # noqa: E402
from gridbench import MODELS, THINKING_FAMILY                # noqa: E402

DATA = os.path.join(HERE, "data")
LOCKS = [os.path.join(DATA, n) for n in (".p0_lock", ".autotune_lock", ".grid_lock", ".phaseb_lock")]
MYLOCK = os.path.join(DATA, ".ev1_lock")
PORT = 8093

GEN = {"gsm8k_cot_zeroshot": "max_gen_toks=2048",
       "math500_boxed": "max_gen_toks=3072",
       # 8192 here is NOT what governs length - see CTX_PER_SLOT below. Kept generous so the
       # harness is never the binding limit; the server is, deliberately and measurably.
       "aime24_boxed": "max_gen_toks=8192", "aime25_boxed": "max_gen_toks=8192",
       "ifeval": "max_gen_toks=2048",
       "gpqa_main_zeroshot": "max_gen_toks=2048"}

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
CTX_PER_SLOT = {"aime24_boxed": 8192, "aime25_boxed": 8192, "math500_boxed": 8192}
CONCURRENT = {"aime24_boxed": 2, "aime25_boxed": 2, "math500_boxed": 2}


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
    cmd = [sys.executable, "-m", "lm_eval", "--model", "local-chat-completions",
           "--model_args",
           f"model={model},base_url=http://127.0.0.1:{PORT}/v1/chat/completions,"
           f"num_concurrent={concurrent},max_retries=3,tokenized_requests=False,timeout=600",
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
    # A timeout must cost ONE ROW, not the night. Uncaught, TimeoutExpired walks out of here,
    # out of the row loop, past the finally that kills the server, and terminates the run - so
    # a single slow row silently cancels every row queued behind it AND skips the failure-tail
    # write below, because that line sits after the one that raised. Caught, recorded, skipped.
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env, timeout=12 * 3600)
    except subprocess.TimeoutExpired as e:
        stop_server(proc)
        gpu_state(f"{model}/{task} post (TIMED OUT)", log)
        log(f"[{model}/{task}] TIMED OUT after {(time.time()-t0)/60:.0f}m - row skipped, "
            f"night continues. Nothing was written: lm-eval saves only on completion.")
        open(os.path.join(DATA, f"ev1_fail_{model}_{task}.txt"), "w",
             encoding="utf-8").write(f"TIMEOUT after {(time.time()-t0)/60:.0f} min\n"
                                     f"{(e.stdout or b'')[-2000:]!r}\n{(e.stderr or b'')[-2000:]!r}")
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
    for l in LOCKS:
        if os.path.isdir(l):
            print(f"REFUSED: {l} exists"); return 3
    try:
        os.mkdir(MYLOCK)
    except FileExistsError:
        print(f"REFUSED: {MYLOCK} exists"); return 3
    lf = open(os.path.join(DATA, "ev1_run.log"), "a", encoding="utf-8")
    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True); lf.write(line + "\n"); lf.flush()
    try:
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
        time.sleep(2)
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
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
        try:
            os.rmdir(MYLOCK)
        except OSError:
            pass
        lf.close()


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
