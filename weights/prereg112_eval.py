"""Evaluate one arm of pre-registration #112 and append it to the results JSON.

Same protocol as prereg #104 so the two are comparable: llama-perplexity, 32 chunks, the same
held-out WikiText-2 file, llama.cpp b10098, no imatrix on any arm.

    python weights/prereg112_eval.py --arm OURS
    python weights/prereg112_eval.py --arm REF --gguf D:/.../Qwen3.8-27B-Q8_0.gguf

Writes weights/data/prereg112.json incrementally, so a crashed arm never destroys earlier work
and the scorer can be run against whatever has completed.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

LLAMA = os.environ.get(
    "QUANTPROBE_LLAMA_DIR", "C:/Users/Federico/Documents/evo-compress/tools/llamacpp-b10098"
)
EVAL = "D:/evo-compress-data/eval/wiki.test.raw"
OUT_DIR = os.environ.get("PREREG112_OUT", "C:/qp112")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "prereg112.json")
CHUNKS = 32
NGL = 12  # the placement that measured cleanly on this box for a >RAM model (prereg #106)


def run_ppl(gguf, log_path):
    perp = os.path.join(LLAMA, "llama-perplexity.exe")
    cmd = [perp, "-m", gguf, "-f", EVAL, "--chunks", str(CHUNKS), "-ngl", str(NGL)]
    print("  $ " + " ".join(cmd), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    blob = p.stdout + p.stderr
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(blob)
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", blob)
    if not m:
        tail = "\n".join(blob.strip().splitlines()[-6:])
        print(f"  NO PPL PARSED (rc={p.returncode}). tail:\n{tail}", flush=True)
        return None, time.time() - t0
    return float(m.group(1)), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--gguf")
    a = ap.parse_args()

    gguf = a.gguf or os.path.join(OUT_DIR, f"Qwen3.8-27B-{a.arm}.gguf")
    if not os.path.isfile(gguf):
        sys.exit(f"missing: {gguf}")
    if not os.path.isfile(EVAL):
        sys.exit(f"missing eval corpus: {EVAL}")

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    d = {}
    if os.path.isfile(RESULTS):
        d = json.load(open(RESULTS, encoding="utf-8"))
    d.setdefault("arms", {})
    d.setdefault("protocol", {
        "chunks": CHUNKS, "ngl": NGL, "eval": EVAL, "llama": os.path.basename(LLAMA),
        "imatrix": None, "note": "same protocol as prereg #104 - no imatrix on any arm",
    })

    size = os.path.getsize(gguf)
    print(f"[{a.arm}] {os.path.basename(gguf)} ({size:,} B)", flush=True)
    log = os.path.join(os.path.dirname(RESULTS), f"prereg112_ppl_{a.arm}.log")
    ppl, secs = run_ppl(gguf, log)
    if ppl is None:
        print(f"[{a.arm}] FAILED after {secs / 60:.1f} min - recorded as incomplete, not dropped")
        d["arms"][a.arm] = {"bytes": size, "ppl": None, "incomplete": True, "log": log}
    else:
        print(f"[{a.arm}] PPL = {ppl:.4f}  ({secs / 60:.1f} min)")
        if a.arm == "REF":
            d["ref_ppl"] = ppl
            d["ref_bytes"] = size
        else:
            d["arms"][a.arm] = {"bytes": size, "ppl": ppl, "log": log}
    with open(RESULTS, "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f, indent=2)
        f.write("\n")
    print(f"  -> {RESULTS}")
    return 0 if ppl is not None else 1


if __name__ == "__main__":
    sys.exit(main())
