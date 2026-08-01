"""Prereg #93 - speculation x KV-quant interaction (2x2 + draft-KV disambiguation arm).

Staked design: preregistrations/2026-07-31-speculation-x-kvquant.md. Read it first; every
gate below is a numbered kill rule there (K-1..K-7) and none is renegotiable at run time.

Arms (one session, frozen placement -ngl 16 / -ngld 0, -c 16896, deep wikitext prompt):

    T00  llama-bench   no draft   target KV f16    tg128 @ d0 and d16384
    T01  llama-bench   no draft   target KV q8_0   tg128 @ d0 and d16384
    T10  llama-spec    K=2        f16 / f16        tok/s + acceptance at ~15.5k depth
    T11  llama-spec    K=2        q8_0 / q8_0      tok/s + acceptance
    T11h llama-spec    K=2        q8_0 / f16       the acceptance-attribution arm

Scored: G = T01/T00, S = T10/T00, C = T11/T00, I = (T11/T10)/(T01/T00).

The script REFUSES (non-zero exit) rather than print a wrong number:
    exit 2 - precondition / validity failure (K-2..K-5, K-7 is scored not exited)
    exit 3 - UNMEASURABLE (K-1 depth gate) or UNINFORMATIVE (K-6)
    exit 0 - valid completed run; per-prediction PASS/FAIL printed and written to JSON

Run:        python weights/exp57_spec_x_kvquant.py
Self-test:  python weights/exp57_spec_x_kvquant.py --self-test      (no GPU, exercises gates)
Dry run:    python weights/exp57_spec_x_kvquant.py --dry-run        (prints commands only)
Failing in: python weights/exp57_spec_x_kvquant.py --prompt-file \
                weights/data/exp57_FAILING_INPUT_shallow_prompt.txt   -> must exit 2
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

TOOLS = "<repo>/tools"
BENCH = TOOLS + "/llama.cpp-pristine/build/bin/llama-bench.exe"
SPEC = TOOLS + "/llama.cpp-pristine/build/bin/llama-speculative.exe"
TOKENIZE = TOOLS + "/llamacpp-b10098/llama-tokenize.exe"
TARGET = "D:/evo-compress-data/gguf/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
DRAFT = "D:/evo-compress-data/gguf/Qwen2.5-0.5B-Instruct-Q8_0.gguf"
DRAFT_MTIME_FLOOR = "2026-07-28"  # the #69 lineage incident: June file, same size, crashes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "weights", "data")
CORPUS = os.path.join(DATA, "wikitext2_test.txt")
PROMPT = os.path.join(DATA, "exp57_prompt_deep.txt")
OUT_JSON = os.path.join(DATA, "exp57_spec_x_kvquant.json")

NGL, NGLD = 16, 0
CTX, DEPTH = 16896, 16384
TOKEN_BAND = (14336, 16384)
CHAR_FLOOR = int(TOKEN_BAND[0] * 3.2)  # conservative chars-per-token lower bound
GEN_SPEC, K_DRAFT = 256, 2
THREADS, SEED = 4, 42
REPS_SPEC, REPS_BENCH = 3, 3
DEPTH_GATE = 0.88          # K-1: tg(deep)/tg(d0) must be <= this
UNINFORMATIVE_EPS = 0.01   # K-6
MIN_DECODED = 128          # K-5
SPEC_TIMEOUT, BENCH_TIMEOUT = 2400, 3600

SPEC_ARMS = {  # tag -> (target KV, draft KV)
    "T10": ("f16", "f16"),
    "T11": ("q8_0", "q8_0"),
    "T11h": ("q8_0", "f16"),
}


class GateFailure(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


# ---------------------------------------------------------------- gates (pure, testable)

def gate_prompt_chars(path):
    """K-2 pre-tokenizer floor: fires before any model load or GPU touch."""
    n = os.path.getsize(path)
    if n < CHAR_FLOOR:
        raise GateFailure(2, "K-2 prompt gate (char floor): %s is %d chars < %d - a shallow "
                             "prompt would make the KV factor a measurement that cannot vary"
                          % (path, n, CHAR_FLOOR))
    return n


def gate_token_band(n_tokens, path):
    if not (TOKEN_BAND[0] <= n_tokens <= TOKEN_BAND[1]):
        raise GateFailure(2, "K-2 prompt gate: %s tokenizes to %d, outside staked band %s"
                          % (path, n_tokens, TOKEN_BAND))


def gate_depth(tg_d0, tg_deep):
    """K-1: the cannot-vary guard."""
    ratio = tg_deep / tg_d0
    if ratio > DEPTH_GATE:
        raise GateFailure(3, "K-1 depth gate: tg(d%d)/tg(d0) = %.3f > %.2f - the KV term "
                             "does not bind at this placement; every downstream ratio would "
                             "be a clean-looking null. UNMEASURABLE-AT-THIS-PLACEMENT."
                          % (DEPTH, ratio, DEPTH_GATE))
    return ratio


def gate_spec_kv_types(log_text, expect_target, expect_draft, tag):
    """K-3 for spec arms: exactly two contexts, target-then-draft order, staked types."""
    kinds = re.findall(r"K \((\w+)\):", log_text)
    if not kinds:
        kinds = re.findall(r"type_k\s*=\s*'?([A-Za-z0-9_]+)'?", log_text)
    if len(kinds) != 2:
        raise GateFailure(2, "K-3 KV-type gate [%s]: expected exactly 2 KV contexts "
                             "(target, draft), parsed %r - fail closed" % (tag, kinds))
    if kinds[0] != expect_target or kinds[1] != expect_draft:
        raise GateFailure(2, "K-3 KV-type gate [%s]: staked (target=%s, draft=%s), log shows "
                             "(%s, %s)" % (tag, expect_target, expect_draft, kinds[0], kinds[1]))


def gate_spec_placement(log_text, tag):
    """K-4 for spec arms: -ngl 16 target, -ngld 0 draft, both actually applied."""
    offs = re.findall(r"offloaded (\d+)/\d+ layers", log_text)
    if len(offs) != 2:
        raise GateFailure(2, "K-4 placement gate [%s]: expected 2 offload lines "
                             "(target, draft), found %d" % (tag, len(offs)))
    if int(offs[0]) != NGL or int(offs[1]) != NGLD:
        raise GateFailure(2, "K-4 placement gate [%s]: staked %d/%d GPU layers, log shows "
                             "%s/%s - no per-arm renegotiation" % (tag, NGL, NGLD, offs[0], offs[1]))


def parse_spec_log(log_text, tag):
    """K-5: acceptance AND enough decoded tokens, else the rep is not scoreable."""
    acc = re.findall(r"accept\s*=\s*([\d.]+)\s*%", log_text)
    spd = re.findall(r"speed:\s*([\d.]+)\s*t/s", log_text)
    dec = re.findall(r"decoded\s+(\d+)\s+tokens", log_text)
    if not acc:
        raise GateFailure(2, "K-5 acceptance gate [%s]: no 'accept = X%%' line - tok/s "
                             "without acceptance must not be scored" % tag)
    if not spd:
        raise GateFailure(2, "K-5 [%s]: no 'speed: X t/s' line" % tag)
    if not dec or int(dec[-1]) < MIN_DECODED:
        raise GateFailure(2, "K-5 [%s]: decoded %s tokens < %d" % (tag, dec, MIN_DECODED))
    return float(acc[-1]), float(spd[-1]), int(dec[-1])


def parse_bench_json(stdout_text, expect_ctk, expect_ctv, tag):
    """Bench arms: K-3 + K-4 from the -o json fields; returns {depth: (avg_ts, std_ts)}."""
    try:
        rows = json.loads(stdout_text)
    except ValueError:
        raise GateFailure(2, "bench [%s]: stdout is not JSON (crash/OOM upstream?)" % tag)
    out = {}
    for r in rows:
        if r.get("n_gen", 0) <= 0:
            continue
        if r.get("n_gpu_layers") != NGL:
            raise GateFailure(2, "K-4 placement gate [%s]: bench n_gpu_layers=%r != %d"
                              % (tag, r.get("n_gpu_layers"), NGL))
        if r.get("type_k") != expect_ctk or r.get("type_v") != expect_ctv:
            raise GateFailure(2, "K-3 KV-type gate [%s]: staked %s/%s, bench row says %r/%r"
                              % (tag, expect_ctk, expect_ctv, r.get("type_k"), r.get("type_v")))
        if "avg_ts" in r:
            avg, std = float(r["avg_ts"]), float(r.get("stddev_ts", 0.0))
        elif "avg_ns" in r:
            avg = r["n_gen"] * 1e9 / float(r["avg_ns"])
            std = 0.0
        else:
            raise GateFailure(2, "bench [%s]: no avg_ts/avg_ns in row %r" % (tag, r))
        out[int(r.get("n_depth", 0))] = (avg, std)
    for want in (0, DEPTH):
        if want not in out:
            raise GateFailure(2, "bench [%s]: missing tg row at depth %d" % (tag, want))
    return out


def interaction(t00, t01, t10, t11):
    g = t01 / t00
    s = t10 / t00
    c = t11 / t00
    i = (t11 / t10) / g
    return g, s, c, i


# ---------------------------------------------------------------- machine preconditions

def run_cmd(cmd, timeout):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def gate_files():
    missing = [p for p in (BENCH, SPEC, TOKENIZE, TARGET, DRAFT, CORPUS)
               if not os.path.isfile(p)]
    if missing:
        raise GateFailure(2, "precondition: missing file(s): %s" % ", ".join(missing))
    mtime = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(DRAFT)))
    if mtime < DRAFT_MTIME_FLOOR:
        raise GateFailure(2, "precondition: draft %s mtime %s predates %s - the #69 "
                             "same-size/wrong-lineage file. Re-fetch (quantprobe fetch --force)."
                          % (DRAFT, mtime, DRAFT_MTIME_FLOOR))


def gate_gpu_idle():
    try:
        p = run_cmd(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], 30)
        used = int(p.stdout.strip().splitlines()[0])
    except Exception as e:
        raise GateFailure(2, "precondition: cannot read GPU state (%s)" % e)
    if used > 1500:
        raise GateFailure(2, "precondition: GPU busy (%d MiB used) - kill orphans first" % used)
    p = run_cmd(["tasklist"], 60)
    orphans = [ln.split()[0] for ln in p.stdout.splitlines()
               if ln.lower().startswith("llama")]
    if orphans:
        raise GateFailure(2, "precondition: llama processes already running: %s" % orphans)


def clocks():
    try:
        p = run_cmd(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,memory.used",
                     "--format=csv,noheader,nounits"], 30)
        return p.stdout.strip()
    except Exception:
        return "unavailable"


# ---------------------------------------------------------------- prompt construction

def count_tokens(path):
    p = run_cmd([TOKENIZE, "-m", TARGET, "-f", path], 900)
    blob = p.stdout + p.stderr
    m = re.search(r"[Tt]otal number of tokens:\s*(\d+)", blob)
    if m:
        return int(m.group(1))
    n = len(re.findall(r"^\s*\d+\s+->", p.stdout, re.M))
    if n == 0:
        raise GateFailure(2, "K-2: llama-tokenize output unparsable for %s (exit %d)"
                          % (path, p.returncode))
    return n


def build_prompt():
    with open(CORPUS, encoding="utf-8", errors="replace") as f:
        text = f.read()
    target_mid = (TOKEN_BAND[0] + TOKEN_BAND[1]) // 2
    chars, n = int(target_mid * 4.2), -1
    for _ in range(6):
        chars = min(chars, len(text))
        with open(PROMPT, "w", encoding="utf-8") as f:
            f.write(text[:chars])
        n = count_tokens(PROMPT)
        if TOKEN_BAND[0] <= n <= TOKEN_BAND[1]:
            print("prompt: %s = %d tokens (band %s)" % (PROMPT, n, (TOKEN_BAND,)))
            return n
        chars = int(chars * target_mid / max(n, 1))
    raise GateFailure(2, "K-2: could not land prompt in band %s (last count %d)"
                      % (TOKEN_BAND, n))


# ---------------------------------------------------------------- arm runners

def bench_cmd(ctk):
    return [BENCH, "-m", TARGET, "-p", "0", "-n", "128", "-d", "0,%d" % DEPTH,
            "-r", str(REPS_BENCH), "-ngl", str(NGL), "-t", str(THREADS),
            "-fa", "on", "-mmp", "0", "-ctk", ctk, "-ctv", ctk, "-o", "json"]


def spec_cmd(ctk, ctkd):
    return [SPEC, "-m", TARGET, "-md", DRAFT, "-ngl", str(NGL), "-ngld", str(NGLD),
            "-c", str(CTX), "-n", str(GEN_SPEC), "-t", str(THREADS),
            "--temp", "0", "-s", str(SEED), "-f", PROMPT, "--no-mmap", "-fa", "on",
            "--spec-draft-n-max", str(K_DRAFT),
            "-ctk", ctk, "-ctv", ctk, "-ctkd", ctkd, "-ctvd", ctkd]


def run_bench_arm(tag, ctk):
    print("=== %s (llama-bench, KV %s) ===  clocks: %s" % (tag, ctk, clocks()))
    try:
        p = run_cmd(bench_cmd(ctk), BENCH_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise GateFailure(2, "K-4 [%s]: bench timeout - whole run INVALID" % tag)
    with open(os.path.join(DATA, "exp57_bench_%s.log" % tag), "w", encoding="utf-8") as f:
        f.write("CMD: %s\n\n--- stdout ---\n%s\n--- stderr ---\n%s\n"
                % (" ".join(bench_cmd(ctk)), p.stdout, p.stderr))
    if p.returncode != 0:
        raise GateFailure(2, "K-4 [%s]: bench exit %d (OOM/crash) - whole run INVALID"
                          % (tag, p.returncode))
    rows = parse_bench_json(p.stdout, ctk, ctk, tag)
    print("    tg128 @ d0     = %.2f +/- %.2f" % rows[0])
    print("    tg128 @ d%d = %.2f +/- %.2f" % (DEPTH, rows[DEPTH][0], rows[DEPTH][1]))
    return rows


def run_spec_arm(tag, ctk, ctkd, rep):
    print("=== %s rep %d (llama-speculative, target %s / draft %s) ===  clocks: %s"
          % (tag, rep, ctk, ctkd, clocks()))
    try:
        p = run_cmd(spec_cmd(ctk, ctkd), SPEC_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise GateFailure(2, "K-4 [%s r%d]: spec timeout - whole run INVALID" % (tag, rep))
    log = p.stdout + "\n" + p.stderr
    with open(os.path.join(DATA, "exp57_spec_%s_r%d.log" % (tag, rep)), "w",
              encoding="utf-8") as f:
        f.write("CMD: %s\n\n%s" % (" ".join(spec_cmd(ctk, ctkd)), log))
    if p.returncode != 0:
        raise GateFailure(2, "K-4 [%s r%d]: llama-speculative exit %d (the #69 lineage-crash "
                             "signature is 'invalid vector subscript') - whole run INVALID"
                          % (tag, rep, p.returncode))
    gate_spec_kv_types(log, ctk, ctkd, "%s r%d" % (tag, rep))
    gate_spec_placement(log, "%s r%d" % (tag, rep))
    acc, spd, dec = parse_spec_log(log, "%s r%d" % (tag, rep))
    print("    accept = %.2f%%   speed = %.3f t/s   decoded = %d" % (acc, spd, dec))
    return acc, spd, dec


# ---------------------------------------------------------------- scoring

def score(t00_rows, t01_rows, spec):
    t00_d0, t00 = t00_rows[0][0], t00_rows[DEPTH][0]
    t01 = t01_rows[DEPTH][0]
    med = {tag: statistics.median(s for _, s, _ in reps) for tag, reps in spec.items()}
    acc = {tag: statistics.median(a for a, _, _ in reps) for tag, reps in spec.items()}
    g, s, c, i = interaction(t00, t01, med["T10"], med["T11"])
    naive = g * s

    # K-6 uninformative guard (depth gate already passed by the time we are here)
    if abs(g - 1.0) < UNINFORMATIVE_EPS and abs(med["T11"] / med["T10"] - 1.0) < UNINFORMATIVE_EPS:
        raise GateFailure(3, "K-6: depth binds but both paired ratios are flat (<1%%) - "
                             "UNINFORMATIVE, not a pass for 'no interaction'")

    preds = {}
    preds["P-1"] = dict(value=round(g, 4), gate=">= 1.06", verdict="PASS" if g >= 1.06 else "FAIL",
                        note="does NOT transfer - scope the banked +37%" if g < 1.03 else "")
    preds["P-2"] = dict(value=round(s, 4), acc=round(acc["T10"], 2), gate=">= 1.15 and acc >= 60",
                        verdict="PASS" if (s >= 1.15 and acc["T10"] >= 60.0) else "FAIL")
    k7 = med["T11"] < max(t01, med["T10"])
    preds["P-3"] = dict(value=round(c, 4), naive=round(naive, 4), gate=">= 0.90 x naive",
                        verdict="KILLED-K7-ANTI-COMPOSITION" if k7
                        else ("PASS" if c >= 0.90 * naive else "FAIL"))
    branch = ("SUB-MULTIPLICATIVE" if i < 0.97 else
              "INDEPENDENT-WITHIN-RESOLUTION" if i <= 1.03 else
              "SUPER-MULTIPLICATIVE-UNEXPLAINED")
    preds["P-4"] = dict(value=round(i, 4), staked_point=0.98, gate="<= 1.00", branch=branch,
                        verdict="PASS" if i <= 1.00 else "FAIL")
    d_acc = acc["T11"] - acc["T10"]
    if abs(d_acc) <= 3.0:
        attribution = "acceptance-neutral"
    elif abs(acc["T11h"] - acc["T10"]) <= 1.0:
        attribution = "DRAFT-KV damage (ship target-q8_0 + draft-f16)"
    else:
        attribution = "TARGET-side damage (no KV split rescues it)"
    preds["P-5"] = dict(delta_pts=round(d_acc, 2), gate="|delta| <= 3.0",
                        acc=dict(T10=round(acc["T10"], 2), T11=round(acc["T11"], 2),
                                 T11h=round(acc["T11h"], 2)),
                        verdict="PASS" if abs(d_acc) <= 3.0 else "FAIL",
                        attribution=attribution)
    measured = dict(T00_d0=round(t00_d0, 3), T00=round(t00, 3), T01=round(t01, 3),
                    T10=round(med["T10"], 3), T11=round(med["T11"], 3),
                    T11h=round(med["T11h"], 3),
                    depth_ratio=round(t00 / t00_d0, 4))
    return measured, preds


# ---------------------------------------------------------------- self-test (no GPU)

def expect_gate(fn, code, label):
    try:
        fn()
    except GateFailure as e:
        assert e.code == code, "%s: wrong exit code %d != %d" % (label, e.code, code)
        print("  self-test PASS: %s fires (exit %d): %s" % (label, e.code, str(e)[:80]))
        return
    raise AssertionError("%s DID NOT FIRE - an unfalsifiable gate" % label)


def self_test():
    print("self-test (no GPU, fabricated inputs):")
    # 1. K-2 char floor on a genuinely short prompt file
    fd, short = tempfile.mkstemp(suffix=".txt")
    os.write(fd, b"x" * 2000)
    os.close(fd)
    expect_gate(lambda: gate_prompt_chars(short), 2, "K-2 char floor (512-token prompt)")
    os.unlink(short)
    # 2. K-2 token band
    expect_gate(lambda: gate_token_band(512, "fake"), 2, "K-2 token band")
    # 3. K-1 depth gate: shallow-config tg pair that cannot vary
    expect_gate(lambda: gate_depth(5.54, 5.40), 3, "K-1 depth gate (ratio 0.975)")
    assert gate_depth(4.7, 3.5) < DEPTH_GATE, "K-1 must pass a binding depth"
    # 4. K-3: both contexts f16 although q8_0 staked
    twin = "K (f16): 1056.0 MiB, V (f16): 1056.0 MiB\nK (f16): 198.0 MiB, V (f16): 198.0 MiB"
    expect_gate(lambda: gate_spec_kv_types(twin, "q8_0", "q8_0", "T11"), 2,
                "K-3 twin-f16 logs where q8_0 staked")
    # 5. K-3: order swap (draft-first) must fail closed
    swap = "K (f16): 198.0 MiB, V (f16): x\nK (q8_0): 561.0 MiB, V (q8_0): x"
    expect_gate(lambda: gate_spec_kv_types(swap, "q8_0", "f16", "T11h"), 2,
                "K-3 target/draft order swap")
    ok = "K (q8_0): 561.0 MiB, V (q8_0): x\nK (f16): 198.0 MiB, V (f16): x"
    gate_spec_kv_types(ok, "q8_0", "f16", "T11h")
    print("  self-test PASS: K-3 accepts the staked assignment")
    # 6. K-4 placement drift
    drift = "offloaded 28/49 layers to GPU\noffloaded 0/25 layers to GPU"
    expect_gate(lambda: gate_spec_placement(drift, "T10"), 2, "K-4 ngl drift (28 != 16)")
    gate_spec_placement("offloaded 16/49 layers to GPU\noffloaded 0/25 layers to GPU", "T10")
    print("  self-test PASS: K-4 accepts the frozen placement")
    # 7. K-5 missing acceptance
    noacc = "decoded  256 tokens in 60.0 seconds, speed: 4.27 t/s"
    expect_gate(lambda: parse_spec_log(noacc, "T11"), 2, "K-5 log without accept line")
    a, sp, d = parse_spec_log("accept    = 76.316%\ndecoded  256 tokens in 60.0 seconds, "
                              "speed: 4.27 t/s", "T10")
    assert (a, sp, d) == (76.316, 4.27, 256)
    print("  self-test PASS: K-5 parses the #69 log format")
    # 8. bench JSON: KV-type mismatch + happy path
    rows = json.dumps([dict(n_gen=128, n_depth=0, n_gpu_layers=16, type_k="f16", type_v="f16",
                            avg_ts=4.71, stddev_ts=0.03),
                       dict(n_gen=128, n_depth=DEPTH, n_gpu_layers=16, type_k="f16",
                            type_v="f16", avg_ts=3.52, stddev_ts=0.04)])
    expect_gate(lambda: parse_bench_json(rows, "q8_0", "q8_0", "T01"), 2,
                "K-3 bench JSON type mismatch")
    parsed = parse_bench_json(rows, "f16", "f16", "T00")
    assert parsed[0] == (4.71, 0.03) and parsed[DEPTH] == (3.52, 0.04)
    print("  self-test PASS: bench JSON parser")
    # 9. interaction arithmetic against a hand-computed sub-multiplicative case
    g, s, c, i = interaction(3.50, 3.99, 4.20, 4.55)
    assert abs(g - 1.14) < 1e-9 and abs(s - 1.20) < 1e-9
    assert abs(i - (4.55 / 4.20) / 1.14) < 1e-12 and i < 0.97
    print("  self-test PASS: interaction I = %.4f (hand value 0.9503)" % i)
    print("SELF-TEST: ALL GATES FIRE AND ALL PARSERS AGREE - exit 0")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Prereg #92 harness")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prompt-file", default=None,
                    help="override the deep prompt (still gated by K-2)")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if args.dry_run:
        print("bench T00:", " ".join(bench_cmd("f16")))
        print("bench T01:", " ".join(bench_cmd("q8_0")))
        for tag, (ctk, ctkd) in SPEC_ARMS.items():
            print("spec %-4s x%d:" % (tag, REPS_SPEC), " ".join(spec_cmd(ctk, ctkd)))
        return

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    gate_files()

    global PROMPT
    if args.prompt_file:
        PROMPT = os.path.abspath(args.prompt_file)
        gate_prompt_chars(PROMPT)          # fires BEFORE tokenizer/GPU on a shallow prompt
        gate_gpu_idle()
        n_tokens = count_tokens(PROMPT)
        gate_token_band(n_tokens, PROMPT)
    else:
        gate_gpu_idle()
        n_tokens = build_prompt()

    # K-1 first: T00 doubles as the depth gate, before any other arm burns time.
    t00_rows = run_bench_arm("T00", "f16")
    depth_ratio = gate_depth(t00_rows[0][0], t00_rows[DEPTH][0])
    print("K-1 depth gate PASSED: ratio %.3f <= %.2f (KV term binds)" % (depth_ratio, DEPTH_GATE))

    t01_rows = run_bench_arm("T01", "q8_0")

    spec = {tag: [] for tag in SPEC_ARMS}
    for rep in range(1, REPS_SPEC + 1):           # round-robin: thermal drift shared
        for tag, (ctk, ctkd) in SPEC_ARMS.items():
            spec[tag].append(run_spec_arm(tag, ctk, ctkd, rep))

    measured, preds = score(t00_rows, t01_rows, spec)

    print("\n================ PREREG #92 SCORE ================")
    for k, v in measured.items():
        print("  %-11s %s" % (k, v))
    overall = True
    for name in ("P-1", "P-2", "P-3", "P-4", "P-5"):
        p = preds[name]
        overall &= p["verdict"] == "PASS"
        print("  %s  %-28s -> %s   %s" % (name, "value=%s gate %s" % (p.get("value",
              p.get("delta_pts")), p["gate"]), p["verdict"],
              p.get("branch", p.get("attribution", p.get("note", "")))))
    print("OVERALL: %s (a FAIL here is a scored miss, not an invalid run)"
          % ("ALL PREDICTIONS PASS" if overall else "AT LEAST ONE STAKED PREDICTION MISSED"))

    result = dict(prereg="preregistrations/2026-07-31-speculation-x-kvquant.md",
                  experiment=57, started=started,
                  finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                  config=dict(target=TARGET, draft=DRAFT, ngl=NGL, ngld=NGLD, ctx=CTX,
                              depth=DEPTH, prompt_tokens=n_tokens, gen=GEN_SPEC, k=K_DRAFT,
                              threads=THREADS, seed=SEED, fa="on", mmap=0,
                              reps_spec=REPS_SPEC, reps_bench=REPS_BENCH,
                              bench_bin=BENCH, spec_bin=SPEC),
                  clocks_end=clocks(), measured=measured, predictions=preds,
                  raw_spec={t: [dict(accept=a, tps=s, decoded=d) for a, s, d in r]
                            for t, r in spec.items()},
                  raw_bench=dict(T00={str(k): v for k, v in t00_rows.items()},
                                 T01={str(k): v for k, v in t01_rows.items()}))
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print("wrote %s" % OUT_JSON)


if __name__ == "__main__":
    try:
        main()
    except GateFailure as e:
        print("REFUSED: %s" % e)
        sys.exit(e.code)
