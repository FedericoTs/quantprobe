"""The full end-to-end re-run: every model on this box through the SHIPPED tool (calibrated,
anchored, with the L-19 CPU-attention term), prediction captured BEFORE its measurement, then
benched on the tool's own emitted placement.

Run:  python weights/full_ladder_v124.py            (predictions only, fast)
      python weights/full_ladder_v124.py --bench    (predict + measure; long)
"""
import json
import os
import re
import subprocess
import sys
import time

D = "D:/evo-compress-data/gguf"
BIN = r"C:\Users\Federico\Documents\evo-compress\tools\llama.cpp-pristine\build\bin\llama-bench.exe"
CUDA = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ladder_state_locked.json")

MODELS = [
    ("Qwen2.5-0.5B Q8_0", "Qwen2.5-0.5B-Instruct-Q8_0.gguf"),
    ("Qwen3-0.6B Q8_0", "Qwen3-0.6B-Q8_0.gguf"),
    ("Qwen3.5-4B Q4_K_M", "Qwen3.5-4B-Q4_K_M.gguf"),
    ("Qwen2.5-7B IQ4_NL", "Qwen2.5-7B-Instruct.i1-IQ4_NL.gguf"),
    ("Qwen2.5-7B Q4_K_M", "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    ("gemma4-12B depth-aware", "gemma4-12b-B-late12.gguf"),
    ("DS-Lite 16B IQ2_XS", "DeepSeek-Coder-V2-Lite-Base-IQ2_XS.gguf"),
    ("Qwen2.5-14B Q4_K_M", "Qwen2.5-14B-Instruct-Q4_K_M.gguf"),
    ("DS-Lite 16B Q4_K_M", "DeepSeek-Coder-V2-Lite-Base-Q4_K_M.gguf"),
    ("Qwen3-30B-A3B Q2_K", "Qwen3-30B-A3B-Q2_K.gguf"),
    ("Qwen3-Coder-30B Q2_K_L", "Qwen3-Coder-30B-A3B-Instruct-Q2_K_L.gguf"),
    ("Qwen3.5-35B APEX-Mini", "Qwen3.5-35B-A3B-APEX-Mini.gguf"),
    ("Qwen3.6-35B Q2_K_XL", "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"),
    ("Qwen3.6-35B APEX-MTP-Nano", "Qwen3.6-35B-A3B-APEX-MTP-I-Nano.gguf"),
]


def plan(path):
    r = subprocess.run([sys.executable, "-m", "quantprobe", "plan", "--gguf", path],
                       capture_output=True, text=True, timeout=600)
    txt = r.stdout
    win = next((l for l in txt.splitlines() if l.strip().startswith("*")), "")
    m = re.search(r"([0-9.]+) tok/s\s+(.*?)(?:\s+\[|$)", win)
    run = next((l.split("run it:", 1)[1].strip() for l in txt.splitlines() if "run it:" in l), "")
    return (float(m.group(1)) if m else None, m.group(2).strip() if m else "?", run)


def flags_from_emit(emit):
    """Translate the emitted llama-server command into llama-bench arguments."""
    out, toks = [], emit.split()
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "-ngl":
            out += ["-ngl", toks[i + 1]]; i += 2
        elif t == "-ot":
            pat = toks[i + 1].strip('"'); out += ["-ot", pat]; i += 2
        elif t == "--no-mmap":
            out += ["-mmp", "0"]; i += 1
        elif t == "--threads":
            out += ["-t", toks[i + 1]]; i += 2
        else:
            i += 1
    if "-t" not in out:
        out += ["-t", "4"]
    return out


def bench(path, emit):
    env = dict(os.environ, PATH=CUDA + os.pathsep + os.environ["PATH"])
    cmd = [BIN, "-m", path] + flags_from_emit(emit) + ["-n", "64", "-p", "0", "-r", "2"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    m = re.search(r"tg64\s*\|\s*([0-9.]+)", r.stdout)
    return float(m.group(1)) if m else None


def main():
    do_bench = "--bench" in sys.argv
    # C-14: lock the whole run to ONE machine state and record it. Predictions and measurements
    # that carry different ids must never be scored against each other.
    from quantprobe.calibrate import load as _cl
    _cal, _ = _cl()
    STATE = (_cal or {}).get("cal_id", "uncalibrated")
    print(f"machine state: {STATE}  (ram {(_cal or {}).get('ram_bw_measured')} GB/s, "
          f"disk {(_cal or {}).get('disk_bw_measured')} GB/s)\n")
    rows = []
    for name, fn in MODELS:
        path = os.path.join(D, fn).replace("\\", "/")
        if not os.path.isfile(path):
            print(f"  SKIP (missing): {name}"); continue
        pred, place, emit = plan(path)
        row = {"name": name, "file": fn, "predicted": pred, "placement": place, "emit": emit,
               "cal_id": STATE}
        print(f"  {name:28s} predicted {pred!s:>6s}  {place}")
        if do_bench and pred:
            t0 = time.time()
            row["measured"] = bench(path, emit)
            row["bench_s"] = round(time.time() - t0)
            if row["measured"]:
                row["err_pct"] = round((pred - row["measured"]) / row["measured"] * 100, 1)
                print(f"      -> measured {row['measured']:.2f}  ({row['err_pct']:+.1f}%)")
            else:
                print("      -> BENCH FAILED")
        rows.append(row)
        json.dump(rows, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
