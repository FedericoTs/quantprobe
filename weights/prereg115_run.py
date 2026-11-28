"""Run pre-registration #115 - should a small GPU's VRAM hold whole layers of experts, or a
cache of the hot ones?

    python weights/prereg115_run.py           # resumable; arms interleaved across repetitions
    python weights/prereg115_run.py --plan    # print every command, run nothing

Every arm is a single llama-completion process that exits by itself, so llama.cpp's MoE cache
prints its own counters when the context closes (a killed server never does on Windows). The
arms differ only in what the 2,268 MiB of VRAM that quantprobe's plan gives to experts holds:
whole layers (A) or fewer layers plus a cache (C*). Each run writes its log to weights/data and
its routing trace (LLAMA_MOE_CACHE_TRACE) to D:, where the scorer replays it.
"""

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA = "D:/evo-compress-data/tools/llamacpp-b12815"
MODEL = "D:/evo-compress-data/gguf/Qwen3.6-35B-A3B-OURS-depthaware.gguf"
PROMPT = os.path.join(HERE, "data", "prereg115_prompt.txt")
WORK = "D:/evo-compress-data/prereg115"
RESULTS = os.path.join(HERE, "data", "prereg115.json")
LOCK = f"{WORK}/.running"
REPS = 3
N_PREDICT = 512
TIMEOUT_S = 1200

COMMON = ["-m", MODEL, "-ngl", "99", "-lm", "none", "-b", "1024", "-ub", "1024", "--threads", "4",
          "-c", "2048", "-n", str(N_PREDICT), "--temp", "0.7", "--seed", "42", "--ignore-eos",
          "-no-cnv", "-f", PROMPT,
          # the cache prints its slots and counters only at log verbosity 4 (llama.cpp's own tuner
          # uses -lv 4 too); a pilot showed it adds start-up and exit lines only, nothing per token
          "-lv", "4"]


def ot(first):
    """The experts of layers first..39 in host RAM; layers below `first` keep theirs in VRAM."""
    return ["-ot", "blk\\.(" + "|".join(str(i) for i in range(first, 40)) + ")\\.ffn_.*_exps\\.=CPU"]


# name -> (flags, cache MiB, CPU share). Layers 0-8 hold 2,268 MiB of experts; layers 0-4 hold
# 1,260 MiB, so 1,008 MiB of cache keeps the VRAM spent on experts equal to arm A's.
ARMS = {
    "A": (ot(9), 0, 0.0),  # quantprobe's printed plan for this model on this machine
    "C1008": (ot(5) + ["--moe-cache-mib", "1008", "--moe-cache-cpu-share", "0"], 1008, 0.0),
    "C504": (ot(5) + ["--moe-cache-mib", "504", "--moe-cache-cpu-share", "0"], 504, 0.0),
    "C1008s": (ot(5) + ["--moe-cache-mib", "1008", "--moe-cache-cpu-share", "0.95"], 1008, 0.95),
}

EVAL = re.compile(r"(?<!prompt )eval time\s*=\s*([\d.]+) ms /\s*(\d+) (?:runs|tokens)\s*\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)")
PROMPT_EVAL = re.compile(r"prompt eval time\s*=\s*([\d.]+) ms /\s*(\d+) tokens\s*\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)")
STATS = re.compile(r"llama_moe_cache: (ubatch <= 8|ubatch  > 8): hits = (\d+), misses = (\d+) \((\d+) on the CPU\), hit rate = ([\d.]+)%, uploaded = ([\d.]+) MiB")
SLOTS = re.compile(r"(\d+) layers, (\w+): +(\d+) slots")


def avail_ram_gib():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30


def command(arm):
    return [os.path.join(LLAMA, "llama-completion.exe")] + COMMON + ARMS[arm][0]


def parse(txt):
    out = {}
    m = [x for x in EVAL.finditer(txt)]
    if m:
        ms, n, _, tps = m[-1].groups()
        out.update(eval_ms=float(ms), eval_n=int(n), decode_tps=float(tps))
    p = PROMPT_EVAL.search(txt)
    if p:
        out.update(prompt_n=int(p.group(2)), prompt_tps=float(p.group(4)))
    for s in STATS.finditer(txt):
        key = "small" if "<=" in s.group(1) else "large"
        out[key] = {"hits": int(s.group(2)), "misses": int(s.group(3)), "cpu": int(s.group(4)),
                    "hit_rate": float(s.group(5)), "uploaded_mib": float(s.group(6))}
    out["slots"] = [[int(a), t, int(b)] for a, t, b in SLOTS.findall(txt)]
    return out


def load():
    if os.path.isfile(RESULTS):
        return json.load(open(RESULTS, encoding="utf-8"))
    return {"protocol": {"llama": os.path.basename(LLAMA), "model": os.path.basename(MODEL),
                         "model_bytes": os.path.getsize(MODEL), "reps": REPS, "n_predict": N_PREDICT,
                         "common": COMMON, "arms": {k: {"flags": v[0], "mib": v[1], "share": v[2]}
                                                    for k, v in ARMS.items()}},
            "runs": {}}


def save(d):
    with open(RESULTS, "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f, indent=1)
        f.write("\n")


def others_running():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama*"], capture_output=True, text=True,
                         errors="replace").stdout
    return "llama" in out.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    if a.plan:
        for arm in ARMS:
            print(f"--- {arm}\n{subprocess.list2cmdline(command(arm))}\n")
        return 0
    for p in (MODEL, PROMPT, os.path.join(LLAMA, "llama-completion.exe")):
        if not os.path.isfile(p):
            sys.exit(f"missing: {p}")
    os.makedirs(WORK, exist_ok=True)
    if os.path.exists(LOCK):
        sys.exit(f"ABORT: {LOCK} exists - another run may be live. Delete it only if none is.")
    if others_running():
        sys.exit("ABORT: a llama.cpp process is already running; a shared GPU is not a measurement")
    open(LOCK, "w").write(str(os.getpid()))
    try:
        d = load()
        for rep in range(1, REPS + 1):
            for arm, (_, mib, share) in ARMS.items():
                key = f"{arm}_r{rep}"
                if key in d["runs"] and d["runs"][key].get("decode_tps") is not None:
                    continue
                log = os.path.join(HERE, "data", f"prereg115_{key}.log")
                trace = f"{WORK}/prereg115_{key}.mct1" if mib else None
                env = dict(os.environ)
                if trace:
                    env["LLAMA_MOE_CACHE_TRACE"] = trace
                ram = avail_ram_gib()
                print(f"[{time.strftime('%H:%M:%S')}] {key}: {ram:.1f} GiB RAM available", flush=True)
                t0 = time.time()
                try:
                    with open(log, "w", encoding="utf-8", errors="replace") as f:
                        p = subprocess.run(command(arm), stdout=f, stderr=subprocess.STDOUT, env=env,
                                           stdin=subprocess.DEVNULL, timeout=TIMEOUT_S)
                    rc = p.returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
                txt = open(log, encoding="utf-8", errors="replace").read()
                rec = {"arm": arm, "rep": rep, "mib": mib, "share": share, "rc": rc,
                       "seconds": round(time.time() - t0, 1), "ram_avail_gib": round(ram, 2),
                       "trace": trace if trace and os.path.isfile(trace) else None, **parse(txt)}
                if rc != 0 or rec.get("decode_tps") is None:
                    rec["tail"] = txt.strip().splitlines()[-8:]
                d["runs"][key] = rec
                save(d)
                print(f"[{time.strftime('%H:%M:%S')}] {key}: rc={rc} decode "
                      f"{rec.get('decode_tps')} tok/s, hit rate {(rec.get('small') or {}).get('hit_rate')}", flush=True)
        d["status"] = "complete"
        save(d)
        print(f"\nall runs complete -> python weights/prereg115_score.py --results {RESULTS}")
        return 0
    finally:
        if os.path.exists(LOCK):
            os.remove(LOCK)


if __name__ == "__main__":
    sys.exit(main())
