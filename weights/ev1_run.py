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
       "aime24": "max_gen_toks=4096", "aime25": "max_gen_toks=4096",
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
BOXED_TASKS = {"math500_boxed", "aime24", "aime25"}

# Tasks carrying few-shot examples need them delivered as chat turns when a chat template is
# applied; lm-eval refuses the combination otherwise. (math500_boxed is zero-shot - the answer
# format is asked for in the prompt instead of demonstrated, which is cheaper and clearer.)
FEWSHOT_TASKS = set()

# Our task definitions live in the repo so any row here is reproducible by a stranger.
TASK_PATH = os.path.join(HERE, "lm_eval_tasks")


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


def run_row(model, task, log, concurrent=4, limit=None, tag=""):
    if not limit and done(model, task):
        log(f"[{model}/{task}] already complete - skipped (resume)")
        return
    gpu_state(f"{model}/{task} pre", log)
    # Protocol v2 (amended in the prereg BEFORE re-runs): thinking off SERVER-SIDE via
    # --reasoning off. v1's "thinking-as-served" buried thought in reasoning_content, which
    # lm-eval never sees - budgets burned invisibly, answers truncated, scores were floors.
    extra = ("--reasoning", "off") if model in THINKING_FAMILY else ()
    proc, _ = start_server(MODELS[model], concurrent, ctx_per_slot=4096, extra=extra)
    if proc is None:
        log(f"[{model}/{task}] SERVER FAILED - row recorded unrunnable")
        return
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
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
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=6 * 3600)
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
            # had already been spent proving a protocol wrong. A protocol change gets a
            # 20-item smoke on BOTH families first - scores land in ev1_probe/, never in the
            # scored tree, so a probe can never be mistaken for a published row.
            for model, task in (("0.6B", "math500_boxed"), ("4B", "aime24")):
                run_row(model, task, log, limit=probe, tag="_probe")
            log(f"probe pass complete (limit={probe})")
            return 0
        if night == 1:
            rows = ([("0.6B", t) for t in ("math500_boxed", "aime24", "aime25", "ifeval",
                                            "gsm8k_cot_zeroshot")]
                    + [("4B", t) for t in ("aime24", "aime25", "ifeval", "math500_boxed")])
        else:
            rows = ([("4B", "gsm8k_cot_zeroshot")]
                    + [("7B", t) for t in ("math500_boxed", "aime24", "aime25", "ifeval",
                                            "gsm8k_cot_zeroshot")]
                    + [("30B", t) for t in ("math500_boxed", "aime24", "aime25", "ifeval",
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
