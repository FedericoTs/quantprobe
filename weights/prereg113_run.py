"""Run pre-registration #113 end to end: reference -> imatrix -> builds -> evaluations.

    python weights/prereg113_run.py            # resumable: each stage skips if its output exists
    python weights/prereg113_run.py --plan     # print every command, run nothing

Order is deliberate. The reference run comes FIRST because it doubles as the protocol check:
if it does not reproduce #112's 5.5950 within 0.5%, the run is VOID and stops before spending
~6 hours computing an importance matrix. Builds reuse prereg112_build.cmd_for() verbatim and
only add `--imatrix`, so "everything held fixed except the imatrix" is true by construction.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prereg112_build as b112  # noqa: E402

LLAMA = b112.LLAMA
SRC = b112.SRC
TEST = "D:/evo-compress-data/eval/wiki.test.raw"
TRAIN = "D:/evo-compress-data/eval/wikitext-2-raw/wiki.train.raw"
WORK = "D:/evo-compress-data/prereg113"
IMAT = f"{WORK}/qwen38-27b-wikitrain-128.imatrix.gguf"
# Disk is tight by design (no models were deleted for this). D: holds the two equal-bytes arms
# that decide the result; C: holds the base logits (~4 GB expected, read once per evaluation, so
# the faster system SSD) and the descriptive UNIFORM-I arm, which SKIPS rather than aborting the
# run when space is short - it must never cost the two arms that matter.
BASE = "C:/qp113/qwen38-27b-q8_0.kld-base"
ARMS = {
    "OURS-I": f"{WORK}/Qwen3.8-27B-OURS-I.gguf",
    "SPREAD-I": f"{WORK}/Qwen3.8-27B-SPREAD-I.gguf",
    "UNIFORM-I": "C:/qp113/Qwen3.8-27B-UNIFORM-I.gguf",
}
OPTIONAL = {"UNIFORM-I"}
NEED_GB = {"OURS-I": 14, "SPREAD-I": 14, "UNIFORM-I": 12}
RESULTS = os.path.join(HERE, "data", "prereg113.json")
LOGS = os.path.join(HERE, "data")
IMAT_CHUNKS, EVAL_CHUNKS, NGL = 128, 32, 12
REF_112, REF_TOL = 5.5950, 0.005
LOCK = f"{WORK}/.running"


def tool(name):
    return os.path.join(LLAMA, name + ".exe")


def commands():
    c = {
        "reference": [tool("llama-perplexity"), "-m", SRC, "-f", TEST, "--chunks", str(EVAL_CHUNKS),
                      "-ngl", str(NGL), "--kl-divergence-base", BASE],
        "imatrix": [tool("llama-imatrix"), "-m", SRC, "-f", TRAIN, "-o", IMAT,
                    "--chunks", str(IMAT_CHUNKS), "-ngl", str(NGL)],
    }
    for arm, out in ARMS.items():
        if arm == "UNIFORM-I":
            c[f"build {arm}"] = [tool("llama-quantize"), "--allow-requantize", "--imatrix", IMAT,
                                 SRC, out, "Q2_K", "8"]
        else:
            base = b112.cmd_for(arm.replace("-I", ""), out)  # #112's exact flags
            c[f"build {arm}"] = base[:2] + ["--imatrix", IMAT] + base[2:]
        c[f"eval {arm}"] = [tool("llama-perplexity"), "-m", out, "-f", TEST,
                            "--chunks", str(EVAL_CHUNKS), "-ngl", str(NGL),
                            "--kl-divergence-base", BASE, "--kl-divergence"]
    return c


def load():
    if os.path.isfile(RESULTS):
        return json.load(open(RESULTS, encoding="utf-8"))
    return {"arms": {}, "protocol": {"imatrix_chunks": IMAT_CHUNKS, "eval_chunks": EVAL_CHUNKS,
                                     "ngl": NGL, "calibration": TRAIN, "eval": TEST,
                                     "llama": os.path.basename(LLAMA)}}


def save(d):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f, indent=2)
        f.write("\n")


def run(stage, cmd):
    log = os.path.join(LOGS, f"prereg113_{stage.replace(' ', '_')}.log")
    print(f"[{time.strftime('%H:%M:%S')}] {stage}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, errors="replace")
    txt = open(log, encoding="utf-8", errors="replace").read()
    mins = (time.time() - t0) / 60
    print(f"[{time.strftime('%H:%M:%S')}] {stage}: exit {p.returncode} after {mins:.1f} min "
          f"-> {log}", flush=True)
    if p.returncode != 0:
        sys.exit(f"ABORT: {stage} exited {p.returncode}. Tail:\n" +
                 "\n".join(txt.strip().splitlines()[-8:]))
    return txt


def num(pattern, txt):
    m = re.search(pattern, txt)
    return float(m.group(1)) if m else None


def free_gb(path):
    return shutil.disk_usage(os.path.splitdrive(os.path.abspath(path))[0] + os.sep).free / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    cmds = commands()
    if a.plan:
        for k, v in cmds.items():
            print(f"--- {k}\n{' '.join(v)}\n")
        return 0
    for p in (SRC, TEST, TRAIN):
        if not os.path.isfile(p):
            sys.exit(f"missing input: {p}")
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(os.path.dirname(ARMS["UNIFORM-I"]), exist_ok=True)
    if os.path.exists(LOCK):
        sys.exit(f"ABORT: {LOCK} exists - another run may be live. Delete it only if none is.")
    open(LOCK, "w").write(str(os.getpid()))
    try:
        d = load()

        # 1. reference + base logits: the protocol check, before anything expensive
        if d.get("ref_ppl_measured") is None or not os.path.isfile(BASE):
            txt = run("reference", cmds["reference"])
            ref = num(r"Final estimate: PPL = ([0-9.]+)", txt)
            if ref is None or not os.path.isfile(BASE):
                sys.exit("ABORT: reference run produced no PPL or no base-logits file")
            d["ref_ppl_measured"] = ref
            d["ref_bytes_base"] = os.path.getsize(BASE)
            save(d)
        drift = abs(d["ref_ppl_measured"] - REF_112) / REF_112
        print(f"reference PPL {d['ref_ppl_measured']:.4f} vs #112 {REF_112} ({drift * 100:.2f}%)")
        if drift > REF_TOL:
            d["status"] = "VOID: reference did not reproduce #112"
            save(d)
            sys.exit("VOID: protocol drifted - stopping before the imatrix")

        # 2. importance matrix (the long pole)
        if not (os.path.isfile(IMAT) and os.path.getsize(IMAT) > 0):
            run("imatrix", cmds["imatrix"])
            if not (os.path.isfile(IMAT) and os.path.getsize(IMAT) > 0):
                sys.exit("ABORT: llama-imatrix produced no output file")
        d["imatrix_bytes"] = os.path.getsize(IMAT)
        save(d)

        # 3. builds, each refusing to start without room (a truncated arm is not a measurement)
        skipped = []
        for arm, out in ARMS.items():
            if os.path.isfile(out):
                continue
            if free_gb(out) < NEED_GB[arm]:
                msg = f"only {free_gb(out):.1f} GB free for {arm}; need {NEED_GB[arm]}"
                if arm in OPTIONAL:
                    print(f"SKIP {arm} (descriptive only): {msg}", flush=True)
                    skipped.append(arm)
                    continue
                sys.exit(f"ABORT: {msg}")
            run(f"build {arm}", cmds[f"build {arm}"])
            if not os.path.isfile(out):
                sys.exit(f"ABORT: build {arm} produced no file")
        if skipped:
            d["skipped"] = skipped
            save(d)

        # 4. evaluations against the saved base logits
        for arm, out in ARMS.items():
            if arm in skipped or (d["arms"].get(arm) or {}).get("kld_mean") is not None:
                continue
            txt = run(f"eval {arm}", cmds[f"eval {arm}"])
            d["arms"][arm] = {
                "bytes": os.path.getsize(out),
                "ppl": num(r"Mean PPL\(Q\)\s*:\s*([0-9.]+)", txt),
                "kld_mean": num(r"Mean\s+KLD:\s*([0-9.]+)", txt),
                "kld_p99": num(r"99\.0%\s+KLD:\s*([0-9.]+)", txt),
                "top1": num(r"Same top p:\s*([0-9.]+)", txt),
            }
            save(d)
            if d["arms"][arm]["kld_mean"] is None:
                sys.exit(f"ABORT: could not parse KLD for {arm}; see its log")
        d["status"] = "complete"
        save(d)
        print("\nall stages complete -> python weights/prereg113_score.py --results " + RESULTS)
        return 0
    finally:
        if os.path.exists(LOCK):
            os.remove(LOCK)


if __name__ == "__main__":
    sys.exit(main())
