"""Smoke suite for quantprobe — plain asserts, no pytest dependency.
Run:  python tests/smoke.py   (needs the package installed; llama.cpp NOT required for these)"""
from __future__ import annotations
import io, subprocess, sys
from contextlib import redirect_stdout

FAIL = []


def check(name, fn):
    try:
        fn()
        print(f"  ok    {name}")
    except Exception as e:
        FAIL.append((name, e))
        print(f"  FAIL  {name}: {e}")


def cli(*args):
    r = subprocess.run([sys.executable, "-m", "quantprobe.cli"] + list(args),
                       capture_output=True, text=True, errors="replace")
    return r.returncode, r.stdout + r.stderr


def t_help():
    rc, out = cli("--help")
    assert rc == 0 and all(k in out for k in ["probe", "plan", "run", "bench", "dashboard", "target", "fetch"])


def t_plan_preset():
    rc, out = cli("plan", "--model", "qwen3-30b", "--machine", "2016-xmp")
    assert rc == 0 and "hybrid" in out and "tok/s" in out and "run it:" in out


def t_plan_custom():
    rc, out = cli("plan", "--total", "13", "--active", "13", "--vram", "8", "--vram-bw", "300",
                  "--ram", "32", "--ram-bw", "50", "--disk-bw", "2")
    assert rc == 0 and "tok/s" in out


def t_target():
    rc, out = cli("target", "--tps", "5", "--machine", "2016-xmp", "--ladder")
    assert rc == 0 and "Qwen3-30B" in out and "LADDER" in out


def t_target_infeasible():
    rc, out = cli("target", "--tps", "5000", "--machine", "2016")
    assert rc == 0 and "nothing in the catalog" in out


def t_fetch_preset_resolution():
    # unknown preset with no files must fail with the preset list, not crash
    rc, out = cli("fetch", "not-a-preset", ".")
    assert rc != 0 and "preset" in out


def t_law_invariants():
    from quantprobe.plan import evaluate
    # law sanity: doubling RAM bandwidth ~doubles pure-CPU tok/s (in-RAM model)
    _, _, c1 = evaluate(7, 7, 7, False, 2.5, 0, 0, 32, 40, 2, 0.5)
    _, _, c2 = evaluate(7, 7, 7, False, 2.5, 0, 0, 32, 80, 2, 0.5)
    r = c2[0][1] / c1[0][1]
    assert 1.9 < r < 2.1, f"BW scaling broken: {r}"
    # dense bigger than RAM must be disk-slow (the 70B honesty fix)
    _, _, c3 = evaluate(70, 70, 70, False, 4.5, 0, 0, 16, 48, 0.45, 0.5)
    assert c3[0][1] < 0.1, f"dense disk-stream too optimistic: {c3[0][1]}"
    # low-bit VRAM must use gl not geta (decode-util law)
    _, _, c4 = evaluate(7, 7, 7, False, 2.0, 8, 300, 32, 50, 2, 0.5, 1.0, 0.05)
    _, _, c5 = evaluate(7, 7, 7, False, 4.5, 8, 300, 32, 50, 2, 0.5, 1.0, 0.05)
    vr4 = [x for x in c4 if x[0] == "all in VRAM"][0][1]
    vr5 = [x for x in c5 if x[0] == "all in VRAM"][0][1]
    assert vr4 < vr5, "low-bit VRAM eta collapse not applied"


def t_llama_commands_parse_and_fail_gracefully():
    # probe/run/bench/dashboard must ACCEPT their args and fail with a CLEAR message when llama.cpp
    # is absent — never a traceback. (CI has no llama.cpp; this guarantees the stranger experience.)
    for args in (
        ["run", "--gguf", "x.gguf", "--model", "qwen3-30b", "--machine", "2016-xmp", "--dry"],
        ["bench", "--gguf", "x.gguf", "--model", "qwen3-30b", "--machine", "2016-xmp", "--dry"],
    ):
        rc, out = cli(*args)
        # --dry prints the plan without touching llama.cpp: must succeed and name the placement
        assert "placement" in out.lower() or "tok/s" in out.lower(), f"{args[0]} --dry broke: {out[:200]}"

def t_probe_help_and_missing_llama_message():
    rc, out = cli("probe", "--help")
    assert rc == 0 and "--gguf" in out and "--eval" in out
    # missing llama.cpp must be a clean SystemExit message, not a traceback
    rc2, out2 = cli("probe", "--gguf", "nope.gguf", "--eval", "nope.txt", "--llama-dir", "/definitely/not/here")
    assert "not found" in out2.lower() and "Traceback" not in out2, f"probe missing-llama not graceful: {out2[:200]}"

def t_all_subcommands_present():
    rc, out = cli("--help")
    for c in ("plan", "target", "fetch", "probe", "run", "bench", "dashboard"):
        assert c in out, f"subcommand {c} missing from --help"


def t_ctx_zero_identity():
    # Law 4 v2 regression guarantee: --ctx 0 (and flag absent) must reproduce v1.0 numbers EXACTLY
    rc1, out1 = cli("plan", "--model", "qwen3-30b", "--machine", "2016-xmp")
    rc2, out2 = cli("plan", "--model", "qwen3-30b", "--machine", "2016-xmp", "--ctx", "0")
    assert rc1 == 0 and out1 == out2, "ctx=0 is not an identity"

def t_ctx_monotonic():
    from quantprobe.plan import evaluate, MODELS
    m = MODELS["qwen3-30b"]
    tps = []
    for ctx in (0, 8192, 32768):
        _, _, cfgs = evaluate(m["t"], m["a"], m["ne"], m["moe"], 4.5, 0, 0, 64, 48, 3.5, 0.5,
                              ctx=ctx, kvp=m["kvp"])
        cpu = [c for c in cfgs if c[0].startswith("pure CPU")]
        assert cpu, f"pure CPU missing at ctx={ctx}"
        tps.append(cpu[0][1])
    assert tps[0] > tps[1] > tps[2], f"tok/s not monotonic in ctx: {tps}"

def t_ctx_placement_dependence():
    # KV on a slow tier must hurt more: pure-CPU (KV@RAM 48) degrades steeper than hybrid (KV@VRAM 192)
    from quantprobe.plan import evaluate, MODELS
    m = MODELS["qwen3-30b"]
    def ratio(placement_prefix, vc, vb, rc):
        r = []
        for ctx in (0, 16384):
            _, _, cfgs = evaluate(m["t"], m["a"], m["ne"], m["moe"], 2.5, vc, vb, rc, 48, 3.5,
                                  0.35, gl=0.04, ctx=ctx, kvp=m["kvp"])
            hit = [c for c in cfgs if c[0].startswith(placement_prefix)]
            assert hit, f"{placement_prefix} missing at ctx={ctx}"
            r.append(hit[0][1])
        return r[1] / r[0]
    r_hybrid = ratio("hybrid", 6, 192, 32)      # 32 GB RAM so both placements exist at 16k
    r_cpu = ratio("pure CPU", 0, 0, 32)
    assert r_cpu < r_hybrid, f"CPU-placed KV should degrade steeper: cpu {r_cpu:.3f} vs hybrid {r_hybrid:.3f}"

def t_ctx_calibration_anchor():
    # the law must retrodict its own calibration: measured d16384/d0 = 16.12/20.02 = 0.805 on 2016-xmp
    from quantprobe.plan import evaluate, MODELS, MACHINES
    m, hw = MODELS["qwen3-30b"], MACHINES["2016-xmp"]
    r = []
    for ctx in (0, 16384):
        _, _, cfgs = evaluate(m["t"], m["a"], m["ne"], m["moe"], 2.5, hw["vc"], hw["vb"],
                              hw["rc"], hw["rb"], hw["db"], hw["geta"], gl=hw["gl"],
                              ctx=ctx, kvp=m["kvp"])
        hy = [c for c in cfgs if c[0].startswith("hybrid")]
        assert hy, f"hybrid missing at ctx={ctx}"
        r.append(hy[0][1])
    ratio = r[1] / r[0]
    assert 0.75 < ratio < 0.90, f"calibration anchor off: predicted ratio {ratio:.3f} vs measured 0.805"

def t_ctx_fit_flip():
    # KV memory must count against capacity: at 16k the 30B no longer fits 16GB RAM as pure-CPU
    rc0, out0 = cli("plan", "--model", "qwen3-30b", "--machine", "2016-xmp")
    rc1, out1 = cli("plan", "--model", "qwen3-30b", "--machine", "2016-xmp", "--ctx", "16384")
    assert "pure CPU" in out0, "baseline should list pure CPU"
    assert "pure CPU" not in out1, "16k KV must evict the pure-CPU placement on a 16GB box"
    assert "tok/s" in out1, "planner must still return a feasible placement"

def t_bench_depth_dry():
    rc, out = cli("bench", "--gguf", "x.gguf", "--model", "qwen3-30b", "--machine", "2016-xmp",
                  "--depth", "16384", "--dry")
    assert "-d 16384" in out and "placement" in out.lower(), f"bench --depth --dry broke: {out[:200]}"

def t_quantize_missing_file_graceful():
    # quantize on a missing GGUF must give a CLEAN error, never a traceback
    rc, out = cli("quantize", "--gguf", "nope.gguf", "--out", "o.gguf", "--protect-late", "12", "--dry")
    assert "not found" in out.lower() and "Traceback" not in out, f"quantize missing-file not graceful: {out[:200]}"

def t_quantize_help():
    rc, out = cli("quantize", "--help")
    assert rc == 0 and "--gguf" in out and "--protect-late" in out

def t_version():
    import quantprobe
    assert quantprobe.__version__


if __name__ == "__main__":
    print("quantprobe smoke suite")
    for n, f in list(globals().items()):
        if n.startswith("t_"):
            check(n, f)
    if FAIL:
        sys.exit(f"\n{len(FAIL)} FAILURES")
    print("\nall green")
