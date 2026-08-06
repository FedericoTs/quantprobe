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
       "hendrycks_math500": "max_gen_toks=2048",
       "aime24": "max_gen_toks=4096", "aime25": "max_gen_toks=4096",
       "ifeval": "max_gen_toks=2048",
       "gpqa_main_zeroshot": "max_gen_toks=2048"}


def out_dir(model, task):
    return os.path.join(DATA, "ev1", model, task)


def done(model, task):
    d = out_dir(model, task)
    if not os.path.isdir(d):
        return False
    for root, _, files in os.walk(d):
        if any(f.startswith("results_") for f in files):
            return True
    return False


def run_row(model, task, log, concurrent=4):
    if done(model, task):
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
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-m", "lm_eval", "--model", "local-chat-completions",
         "--model_args",
         f"model={model},base_url=http://127.0.0.1:{PORT}/v1/chat/completions,"
         f"num_concurrent={concurrent},max_retries=3,tokenized_requests=False,timeout=600",
         "--tasks", task, "--gen_kwargs", GEN[task],
         "--apply_chat_template", "--seed", "0",
         "--output_path", out_dir(model, task), "--log_samples"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, timeout=6 * 3600)
    stop_server(proc)
    gpu_state(f"{model}/{task} post", log)
    ok = done(model, task)
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


def main(night):
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
        if night == 1:
            rows = ([("0.6B", t) for t in ("hendrycks_math500", "aime24", "aime25", "ifeval",
                                            "gsm8k_cot_zeroshot")]
                    + [("4B", t) for t in ("aime24", "aime25", "ifeval", "hendrycks_math500")])
        else:
            rows = ([("4B", "gsm8k_cot_zeroshot")]
                    + [("7B", t) for t in ("hendrycks_math500", "aime24", "aime25", "ifeval",
                                            "gsm8k_cot_zeroshot")]
                    + [("30B", t) for t in ("hendrycks_math500", "aime24", "aime25", "ifeval",
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
    a = ap.parse_args()
    sys.exit(main(1 if a.night1 else 2) if (a.night1 or a.night2) else 0)
