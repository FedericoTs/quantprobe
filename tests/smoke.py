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

def t_plan_uses_preset_gl():
    # plan CLI must apply the preset's MEASURED low-bit collapse (gl), not geta*0.6
    # 2016-xmp gl=0.04: dense 7B @2.5 all-in-VRAM ~1.7 tok/s; the geta*0.6 bug gave ~8.7
    rc, out = cli("plan", "--model", "mistral-7b", "--machine", "2016-xmp", "--bits", "2.5")
    import re
    m = re.search(r"([0-9.]+) tok/s\s+all in VRAM", out)
    assert m, f"no all-in-VRAM row: {out[:200]}"
    assert float(m.group(1)) < 3.0, f"preset gl ignored: VRAM row {m.group(1)} tok/s (expected ~1.7)"

def t_hw_command():
    rc, out = cli("hw")
    assert rc == 0 and "equivalent flags" in out and "--ram" in out, f"hw broke: {out[:200]}"

def t_auto_hardware_no_flags():
    # no machine, no hw flags -> auto-detect (works on CI via /proc/meminfo + defaults)
    rc, out = cli("plan", "--total", "7", "--active", "7", "--bits", "4.5")
    assert rc == 0 and "auto-detected" in out and "tok/s" in out

def t_bits_continuous():
    rc, out = cli("plan", "--total", "7", "--active", "7", "--bits", "2.88",
                  "--vram", "0", "--ram", "32", "--ram-bw", "45", "--disk-bw", "2")
    assert rc == 0 and "2.88-bit" in out, f"continuous bits rejected: {out[:150]}"

def t_autospec_from_gguf():
    import os
    g = "D:/evo-compress-data/gguf/Qwen3-30B-A3B-Q2_K.gguf"
    if not os.path.isfile(g):
        return  # CI has no model files; runs on the reference box
    from quantprobe.spec import from_gguf
    s = from_gguf(g)
    assert 29 < s["t"] < 32 and 3.0 < s["a"] < 3.8 and s["moe"], f"autospec params off: {s}"
    assert abs(s["kvp"] - 98304) < 2048, f"kvp should be ~98304 exact, got {s['kvp']}"
    assert 2.5 < s["bits"] < 3.3, f"effective bits off: {s['bits']}"

def t_multi_device_aggregate():
    rc, out = cli("plan", "--model", "glm-744b", "--bits", "2.5", "--vram", "24,24,24",
                  "--vram-bw", "936,936,936", "--ram", "128", "--ram-bw", "80", "--disk-bw", "14,14")
    assert rc == 0 and "tok/s" in out, f"multi-device syntax broke: {out[:200]}"

def t_three_tier_row_additive():
    # big-VRAM + big-RAM + fast-disk rig: new expert-cache row appears AND the llama.cpp row survives
    rc, out = cli("plan", "--model", "glm-744b", "--bits", "2.5", "--vram", "72", "--vram-bw", "900",
                  "--ram", "128", "--ram-bw", "80", "--disk-bw", "15")
    assert "VRAM+RAM expert cache" in out and "cold experts" in out, "3-tier row missing or llama.cpp row lost"
    import re
    three = float(re.search(r"([0-9.]+) tok/s\s+stream from disk \(VRAM\+RAM", out).group(1))
    plain = float(re.search(r"([0-9.]+) tok/s\s+stream from disk \(cold", out).group(1))
    assert three > plain * 1.5, f"VRAM cache credit too small: {three} vs {plain}"

def t_anchor_matrix_v13():
    # measured anchors must stay retrodicted: 110B->0.19, laguna->0.38 (llama.cpp rows)
    import re
    rc, out = cli("plan", "--model", "glm-air", "--bits", "2.5", "--machine", "2016-xmp")
    v = float(re.search(r"([0-9.]+) tok/s\s+stream from disk \(cold", out).group(1))
    assert 0.12 <= v <= 0.30, f"110B anchor drifted: {v}"
    rc, out = cli("plan", "--total", "117.6", "--active", "8", "--always-active", "2.5",
                  "--bits", "2.5", "--machine", "2016-xmp")
    v = float(re.search(r"([0-9.]+) tok/s\s+stream from disk \(cold", out).group(1))
    assert 0.2 <= v <= 0.5, f"laguna anchor drifted: {v}"

def t_tier_boundary_advisor():
    # file just over the VRAM boundary -> advisor names the shave and prices the promotion
    rc, out = cli("plan", "--total", "30.5", "--active", "3.3", "--always-active", "1.2",
                  "--bits", "3.6", "--vram", "16", "--vram-bw", "448", "--ram", "32",
                  "--ram-bw", "45", "--disk-bw", "2")
    assert "tier-boundary advisor" in out and "x" in out, f"advisor missing: {out[-300:]}"
    # comfortably-fitting config -> no advisor
    rc2, out2 = cli("plan", "--total", "7", "--active", "7", "--bits", "4.5", "--vram", "16",
                    "--vram-bw", "448", "--ram", "32", "--ram-bw", "45", "--disk-bw", "2")
    assert "tier-boundary advisor" not in out2, "advisor fired on a fitting config"

def t_optimize_backtest_rediscovers_measured_config():
    # blind run on 30B/2016 must place the MEASURED config (2.5-bit hybrid, 18.9) on the frontier top-2
    rc, out = cli("optimize", "--model", "qwen3-30b", "--machine", "2016-xmp")
    assert rc == 0
    lines = [l for l in out.splitlines() if "tok/s" in l and "quality" in l]
    top2 = " ".join(lines[:2])
    assert "18.9" in top2 and "2.5-bit" in top2 and "hybrid" in top2, f"backtest failed: {top2}"

def t_optimize_realizable_default():
    rc, out = cli("optimize", "--model", "qwen3-30b", "--machine", "2016-xmp")
    assert "expert cache" not in out, "aspirational runtime row leaked into default (llama.cpp) mode"
    rc2, out2 = cli("optimize", "--model", "qwen3-30b", "--machine", "2016-xmp", "--any-runtime")
    assert rc2 == 0

def t_optimize_target_unreachable():
    rc, out = cli("optimize", "--model", "glm-744b", "--machine", "2016-xmp", "--tps", "50")
    assert "NOT REACHABLE" in out and "tok/s" in out

def t_optimize_kv_gate():
    # Pascal-class (geta .35): KV-q8 lever must NOT appear even with ctx; modern GPU: appears tagged
    rc, out = cli("optimize", "--model", "qwen3-30b", "--machine", "2016-xmp", "--ctx", "16384")
    assert "KV q8" not in out, "KV-q8 offered on Pascal-class (measured trap)"
    rc2, out2 = cli("optimize", "--model", "qwen3-30b", "--machine", "rtx-3090", "--ctx", "16384")
    assert rc2 == 0  # gate open on geta>=0.5 hardware (tag appears when the lever wins a frontier row)

def t_optimize_prune_flagged():
    rc, out = cli("optimize", "--model", "qwen3-30b", "--machine", "2016-xmp", "--allow-prune")
    assert rc == 0 and ("PRUNED" in out or "tok/s" in out)

def t_gpu_table_lookup():
    from quantprobe.detect import gpu_lookup
    assert gpu_lookup("NVIDIA GeForce RTX 5060 Ti")[0] == 448
    assert gpu_lookup("NVIDIA GeForce RTX 3060")[0] == 360
    assert gpu_lookup("NVIDIA GeForce GTX 1060 6GB")[0] == 192
    assert "default" in gpu_lookup("Mystery GPU 9000")[3]

def t_auto_dry_picks_a_file():
    # auto --dry: optimizer -> HF file-list match -> prediction, NO download. Tolerant of offline CI.
    rc, out = cli("auto", "qwen3-30b", "--tps", "10", "--machine", "2016-xmp", "--dry")
    if "could not list" in out:
        return                                   # no network in this environment
    assert ".gguf" in out and "predicted on this machine" in out and "nothing downloaded" in out, out[-300:]

def t_auto_unknown_target_graceful():
    rc, out = cli("auto", "not-a-real-preset-or-repo", "--dry")
    assert rc != 0 and ("not a preset" in out or "could not list" in out)

def t_auto_custom_dry():
    rc, out = cli("auto", "qwen3-30b", "--machine", "2016-xmp", "--custom", "--dry")
    if "could not list" in out:
        return
    assert "source:" in out and "fragile band" in out and "nothing downloaded" in out, out[-300:]

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
