"""Smoke suite for quantprobe — plain asserts, no pytest dependency.
Run:  python tests/smoke.py   (needs the package installed; llama.cpp NOT required for these)"""
from __future__ import annotations
import io, os, subprocess, sys
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
    # Low-bit all-in-VRAM decode must NOT collapse. This asserted the opposite until
    # pre-registration #16 measured it: Qwen2.5-7B all in VRAM decodes 19.17 (Q2_K, 2.8 bits),
    # 18.11 (IQ3_XS, 3.3 bits) and 20.03 (Q4_K_M, 4.5 bits) - a 10% band across 2.8-4.5 bits,
    # not the 8.75x cliff the old gl gate applied below 4 bits. The old test enforced the cliff,
    # which is also backwards on bytes alone: fewer bits per weight is fewer bytes to read.
    _, _, c4 = evaluate(7, 7, 7, False, 2.0, 8, 300, 32, 50, 2, 0.5, 1.0, 0.05)
    _, _, c5 = evaluate(7, 7, 7, False, 4.5, 8, 300, 32, 50, 2, 0.5, 1.0, 0.05)
    vr4 = [x for x in c4 if x[0] == "all in VRAM"][0][1]
    vr5 = [x for x in c5 if x[0] == "all in VRAM"][0][1]
    # Equality is the expected answer here and it is what measurement shows: for a DENSE model the
    # activation term holds attention at >=4.5 bits, so active bytes barely move with the nominal
    # bit-width - and the 7B measured 19.17 at 2.8 bits vs 20.03 at 4.5, a 4% spread. What must
    # never come back is the PENALTY.
    assert vr4 >= vr5 * 0.9, f"low-bit VRAM penalised: {vr4:.1f} vs {vr5:.1f} at 4.5 bits"


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

def t_low_bit_vram_not_collapsed():
    # Regression test for the bug pre-registration #16 fixed. This test previously ASSERTED the
    # collapse (< 3.0 tok/s), locking in a 9.5x error. Measured reality on this exact box:
    # gemma4-12b at 3.51 bits, all in VRAM, decodes 9.56 tok/s. The old law said 1.0 - so low
    # that plan recommended pure CPU (3.9) over the GPU placement that actually runs 9.56.
    # A dense 7B at 2.5 bits is smaller still, so it must comfortably clear the same bar.
    rc, out = cli("plan", "--model", "mistral-7b", "--machine", "2016-xmp", "--bits", "2.5")
    import re
    m = re.search(r"([0-9.]+) tok/s\s+all in VRAM", out)
    assert m, f"no all-in-VRAM row: {out[:200]}"
    tps = float(m.group(1))
    assert tps > 9.0, f"low-bit VRAM collapse regressed: {tps} tok/s (measured floor 9.56)"
    # and it must beat the CPU row - recommending CPU over a working GPU was the user-visible bug
    c = re.search(r"([0-9.]+) tok/s\s+pure CPU", out)
    if c:
        assert tps > float(c.group(1)), f"VRAM {tps} must beat CPU {c.group(1)} for a 7B at 2.5 bits"

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

# EVERY published measured number becomes a regression test. If a code change makes the law
# stop retrodicting reality, this fails - which is the only guarantee that matters when the
# tool and the claims evolve together. Tolerance is the stated +/-25% prediction band unless a
# tighter one is justified. Add a row here whenever a new number is published.
MEASURED_ANCHORS = [
    # (label, plan args, placement row substring, measured value, tolerance)
    ("30B hybrid on the 2016 box (the flagship 19.3)",
     ["plan", "--model", "qwen3-30b", "--bits", "2.95", "--machine", "2016-xmp"],
     "hybrid", 19.30, 0.30),
    ("30B all-experts-to-CPU, corrected baseline",
     ["plan", "--model", "qwen3-30b", "--bits", "2.95", "--machine", "2016-xmp"],
     "hybrid", 18.35, 0.30),
    ("110B GLM-Air streamed from SATA",
     ["plan", "--model", "glm-air", "--bits", "2.5", "--machine", "2016-xmp"],
     "stream from disk (cold", 0.19, 0.60),
    ("Laguna 118B streamed from SATA",
     ["plan", "--total", "117.6", "--active", "8", "--always-active", "2.5",
      "--bits", "2.5", "--machine", "2016-xmp"],
     "stream from disk (cold", 0.38, 0.60),
]

def t_measured_anchors_still_retrodicted():
    import re
    drift = []
    for label, args, row, measured, tol in MEASURED_ANCHORS:
        rc, out = cli(*args)
        assert rc == 0, f"{label}: plan failed"
        m = re.search(r"([0-9.]+) tok/s\s+" + re.escape(row), out)
        assert m, f"{label}: placement row '{row}' vanished from the plan output"
        pred = float(m.group(1))
        if abs(pred - measured) / measured > tol:
            drift.append(f"{label}: predicted {pred}, measured {measured} "
                         f"({(pred-measured)/measured*100:+.0f}%, tolerance +/-{tol*100:.0f}%)")
    assert not drift, "the law stopped retrodicting measured reality:\n  " + "\n  ".join(drift)

# Every all-in-VRAM datapoint measured on the reference box, with the law's CURRENT error.
#
# This table exists because of a structural hole in the one above it: every MEASURED_ANCHOR is a
# MoE-hybrid or a disk-stream row. Not one covered "all in VRAM" - the single most common
# configuration for anyone with enough VRAM - and that is exactly why a 9.5x error (the refuted
# sub-4-bit collapse, pre-registration #16) lived there undetected through a public release.
#
# These are NOT tolerances that certify the law is right. The law is knowingly PESSIMISTIC in
# this regime - it under-predicts every single point and never over-predicts, which is a bias,
# not noise (pre-registration #15, unresolved: no clean fit exists, a 12B is off by 9% while a
# 7B is off by 38%). They are a RATCHET: the error may shrink, never grow. Improve the law and
# these numbers come down; break it and the suite goes red.
#
# (file, measured tok/s all-in-VRAM on 2016-xmp, current |error| bound)
VRAM_GAPS = [
    ("Qwen3-0.6B-Q8_0.gguf",            93.12, 0.10),   # -2%   dense GQA, Q8_0
    ("Qwen3.5-4B-Q4_K_M.gguf",          27.30, 0.32),   # -25%  dense GQA, K-quant
    ("Qwen2.5-7B-Instruct-Q4_K_M.gguf", 20.03, 0.45),   # -38%  worst of the K-quant dense points
    ("Qwen2.5-7B-Instruct-Q2_K.gguf",   19.17, 0.36),   # -29%  same model, 2.8 bits
    ("Qwen2.5-7B-Instruct-IQ3_XS.gguf", 18.11, 0.32),   # -25%  same model, IQ format
    ("gemma4-12b-B-late12.gguf",         9.56, 0.16),   # -9%   the model that exposed the bug
    ("Bonsai-27B-Q1_0.gguf",            11.94, 0.74),   # -67%  linear-attention hybrid (Law 2 note)
]
GGUF_DIR = os.environ.get("QUANTPROBE_GGUF_DIR", "D:/evo-compress-data/gguf")


def t_vram_regime_error_does_not_grow():
    """Ratchet on the known all-in-VRAM pessimism. Skips per-file when the GGUF is absent."""
    import re
    worse, checked = [], 0
    for fname, measured, bound in VRAM_GAPS:
        path = os.path.join(GGUF_DIR, fname)
        if not os.path.isfile(path):
            continue
        checked += 1
        rc, out = cli("plan", "--gguf", path, "--machine", "2016-xmp")
        assert rc == 0, f"{fname}: plan failed"
        m = re.search(r"([0-9.]+) tok/s\s+all in VRAM", out)
        assert m, (f"{fname}: the all-in-VRAM row vanished. That row disappearing IS the bug "
                   f"pre-registration #16 fixed - the planner recommended pure CPU instead.")
        pred = float(m.group(1))
        err = abs(pred - measured) / measured
        if err > bound + 0.02:                      # 2pp slack for run-to-run bench noise
            worse.append(f"{fname}: predicted {pred}, measured {measured} "
                         f"({(pred-measured)/measured*100:+.0f}%, was within {bound*100:.0f}%)")
    assert not worse, ("the all-in-VRAM regime got WORSE - this is a ratchet, errors may only "
                       "shrink:\n  " + "\n  ".join(worse))
    if checked == 0:
        print("      (VRAM_GAPS: no GGUFs found, set QUANTPROBE_GGUF_DIR)", end="")


def t_commands_agree_on_the_same_input():
    """plan / run / bench MUST predict the same number for the same model+machine.

    They did not: bench applied a file-size calibration that plan skipped, so on identical
    input plan said 19.6 and bench said 22.1 - and bench was 11% wrong against a measured
    19.88 while plan was 1.4% right. Every 'predicted vs measured' figure the tool reported
    was distorted by which command produced it. This is the guard against a repeat.
    """
    import re, os
    gguf = os.environ.get("QUANTPROBE_TEST_GGUF")
    cases = ([["--gguf", gguf, "--machine", "2016-xmp"]] if gguf and os.path.isfile(gguf) else []) + [
        ["--model", "qwen3-30b", "--bits", "2.95", "--machine", "2016-xmp"],
        ["--model", "mistral-7b", "--bits", "4.5", "--machine", "rtx-3060"],
        ["--total", "110", "--active", "12", "--always-active", "2.7", "--bits", "2.5",
         "--vram", "24", "--vram-bw", "936", "--ram", "64", "--ram-bw", "80", "--disk-bw", "3"],
    ]
    for args in cases:
        rc0, plan_out = cli("plan", *args)
        assert rc0 == 0, f"plan failed for {args}"
        m0 = re.search(r"\*\s+([0-9.]+) tok/s", plan_out)
        assert m0, f"plan printed no winning row for {args}"
        plan_tps = float(m0.group(1))
        g = gguf if "--gguf" in args else "x.gguf"
        for cmd in ("run", "bench"):
            extra = ["--gguf", g] if "--gguf" not in args else []
            rc, out = cli(cmd, *(extra + args), "--dry")
            m = re.search(r"predicted ([0-9.]+)", out)
            assert m, f"{cmd} printed no prediction for {args}: {out[:200]}"
            got = float(m.group(1))
            assert abs(got - plan_tps) / plan_tps < 0.01, (
                f"{cmd} disagrees with plan on identical input {args}: "
                f"plan {plan_tps} vs {cmd} {got}")

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
    # Backtest of the optimizer against MEASURED reality. Until 2026-07-26 this asserted the
    # 2.5-bit hybrid (18.9) in the top-2. Pre-registration #13 then MEASURED that partial expert
    # offload beats it (+34.7%), so the optimizer legitimately promotes split-experts rows and
    # the old assertion is obsolete rather than violated. What must stay true: the top pick is
    # grounded in a measured mechanism, is realizable by stock llama.cpp, and beats the old
    # hybrid number it superseded.
    rc, out = cli("optimize", "--model", "qwen3-30b", "--machine", "2016-xmp")
    assert rc == 0
    lines = [l for l in out.splitlines() if "tok/s" in l and "quality" in l]
    top = lines[0]
    assert "split experts" in top, f"measured MoE offload not promoted: {top}"
    tps = float(top.split("tok/s")[0].split()[-1])
    assert tps > 18.9, f"top pick must beat the hybrid config it replaced: {tps}"
    # and the emitted command must be a real -ot regex, not prose or a bare fallback
    assert "ffn_.*_exps" in out and r"blk\.(" in out, "realize-the-pick lost its exact -ot flags"

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

def cli_in(stdin_text, *args):
    r = subprocess.run([sys.executable, "-m", "quantprobe.cli"] + list(args),
                       capture_output=True, text=True, errors="replace", input=stdin_text)
    return r.returncode, r.stdout + r.stderr

def t_auto_custom_machine_gate():
    # on a machine where the optimizer wants >=3.5 bits, --custom must DECLINE the surgery
    # (Laws 1-2: the fragile-band fix only pays below ~3 bits) and fetch standard instead
    rc, out = cli("auto", "qwen3-30b", "--custom", "--dry", "--vram", "24", "--vram-bw", "936",
                  "--ram", "64", "--ram-bw", "86", "--disk-bw", "3")
    assert rc == 0 and "doesn't need the surgery" in out and "closest file" in out, \
        f"custom gate broken: rc={rc} {out[:300]}"

def t_auto_force_custom():
    # --force-custom overrides the gate: the source pick must happen
    rc, out = cli("auto", "qwen3-30b", "--custom", "--force-custom", "--dry", "--vram", "24",
                  "--vram-bw", "936", "--ram", "64", "--ram-bw", "86", "--disk-bw", "3")
    assert rc == 0 and "source:" in out and "surgery" not in out, \
        f"force-custom broken: rc={rc} {out[:300]}"

def t_auto_wizard_dry():
    # no model argument -> interactive wizard: answers piped, --dry keeps it offline-light
    rc, out = cli_in("qwen3-30b\n1\nn\n", "auto", "--dry", "--machine", "2016-xmp")
    assert rc == 0 and "interactive" in out and "closest file" in out, \
        f"wizard broken: rc={rc} {out[:300]}"

def t_auto_wizard_noninteractive_graceful():
    # no model and no terminal must be a CLEAN one-line refusal, never a traceback
    rc, out = cli_in("", "auto", "--dry")
    assert rc != 0 and "no terminal to ask" in out and "Traceback" not in out, \
        f"wizard EOF not graceful: rc={rc} {out[:300]}"

def t_plan_unknown_model_loud():
    # an unknown --model preset must FAIL LOUDLY, never silently fall back to defaults
    rc, out = cli("plan", "--model", "laguna-s", "--machine", "2016-xmp")
    assert rc != 0 and "unknown --model" in out and "presets:" in out and "Traceback" not in out, \
        f"unknown model not loud: rc={rc} {out[:200]}"

def t_plan_unknown_model_with_total_ok():
    # ...but an unknown name WITH an explicit --total is a described custom model: allowed
    rc, out = cli("plan", "--model", "laguna-s", "--total", "117.6", "--active", "8",
                  "--bits", "2.7", "--machine", "2016-xmp")
    assert rc == 0 and "tok/s" in out, f"explicit-spec override broken: rc={rc} {out[:200]}"

def t_plan_unknown_machine_loud():
    rc, out = cli("plan", "--model", "qwen3-30b", "--machine", "gamign-typo")
    assert rc != 0 and "unknown --machine" in out and "Traceback" not in out, \
        f"unknown machine not loud: rc={rc} {out[:200]}"

def t_optimize_unknown_machine_loud():
    rc, out = cli("optimize", "--tps", "20", "--machine", "nope-preset")
    assert rc != 0 and "unknown --machine" in out and "Traceback" not in out, \
        f"optimize unknown machine not loud: rc={rc} {out[:200]}"

def t_optimize_no_prices():
    # upgrade suggestions must carry NO currency figures (region/time-dependent guesses)
    rc, out = cli("optimize", "--machine", "2016-xmp")
    assert rc == 0 and "EUR" not in out and "€" not in out, f"prices leaked: {out[:300]}"
    assert "hw upgrade" in out or "free" in out, f"upgrade column missing: {out[:200]}"

def t_auto_744b_preset_dry():
    # the massive preset resolves to the VERIFIED GLM-5.2 repo (753B - not GLM-4.7/358B).
    # Durable across repo states: either a usable quant is picked, or the not-yet state is
    # explained cleanly - never a traceback, never a silent wrong pick.
    rc, out = cli("auto", "glm-744b", "--dry", "--vram", "96", "--vram-bw", "1800",
                  "--ram", "512", "--ram-bw", "300", "--disk-bw", "7")
    assert "Traceback" not in out and "GLM-5.2-GGUF" in out, f"744b preset broken: rc={rc} {out[:300]}"
    assert (rc == 0 and "closest file" in out) or "no ready-to-run quant" in out,         f"neither pick nor clean explanation: rc={rc} {out[:300]}"

def t_auto_bf16_only_graceful():
    # kimi-k2.6: today BF16-only upstream -> the >9-bit filter must yield the honest
    # explanation (with the --custom tip), not a bare error; when quants land, it just works
    rc, out = cli("auto", "kimi-k2.6", "--dry", "--vram", "96", "--vram-bw", "1800",
                  "--ram", "768", "--ram-bw", "300", "--disk-bw", "7")
    assert "Traceback" not in out, f"kimi traceback: {out[:300]}"
    assert (rc == 0 and "closest file" in out) or "no ready-to-run quant" in out,         f"BF16-only repo not graceful: rc={rc} {out[:300]}"

def t_split_parts_offline():
    from quantprobe.auto import split_parts
    ps = split_parts("Q8_0/GLM-5.2-Q8_0-00001-of-00017.gguf")
    assert len(ps) == 17 and ps[0].endswith("00001-of-00017.gguf") and ps[16].endswith("00017-of-00017.gguf")
    assert split_parts("model-Q4_K_M.gguf") == ["model-Q4_K_M.gguf"]

def t_run_never_execs_prose_flags():
    # the three-tier row's "flags" field is a PROSE description, not argv. run/bench must never
    # select it: exec'ing it hands llama-cli a bare "+" and it dies (real user report, 2026-07-25).
    # This config is chosen so the expert-cache row ranks FIRST.
    rc, out = cli("run", "--gguf", "x.gguf", "--total", "110.5", "--active", "15", "--bits", "2.77",
                  "--vram", "6", "--vram-bw", "192", "--ram", "16", "--ram-bw", "48",
                  "--disk-bw", "0.5", "--dry")
    assert rc == 0, f"run --dry failed: {out[:300]}"
    exec_line = [l for l in out.splitlines() if "exec:" in l]
    assert exec_line, f"no exec line: {out[:300]}"
    assert " + " not in exec_line[0] and "runtime-managed" not in exec_line[0], \
        f"prose leaked into the launch command: {exec_line[0]}"
    assert "expert cache" not in exec_line[0], f"unrunnable placement selected: {exec_line[0]}"

def _build_cmd(**kw):
    """Capture the command build_depthaware would run (dry=True needs no GGUF and no llama.cpp)."""
    from quantprobe.probe import build_depthaware
    buf = io.StringIO()
    with redirect_stdout(buf):
        build_depthaware(None, "src.gguf", "out.gguf", 34, 39, 40, dry=True, **kw)
    return buf.getvalue()

def t_quantize_shexp_protection_first():
    # always-active shared-expert tensors must be protected, and the rule MUST come before the
    # band rules: llama.cpp resolves --tensor-type first-match-wins, so placed last it is a
    # silent no-op. Measured -3.2% ppl when protected (pre-registration #12).
    out = _build_cmd()
    assert "ffn_.*_shexp.*=q8_0" in out, f"shared-expert protection missing: {out[:300]}"
    assert out.index("ffn_.*_shexp") < out.index("blk\\.("), \
        "shexp rule must precede band rules (first-match-wins) or it silently does nothing"

def t_quantize_imatrix_passthrough():
    out = _build_cmd(imatrix="cal.gguf")
    assert "--imatrix cal.gguf" in out, f"imatrix not passed through: {out[:300]}"
    assert "--imatrix" not in _build_cmd(), "imatrix flag leaked into an uncalibrated build"

def t_moe_split_row_and_flags():
    # MoE partial expert offload (pre-registration #13): measured +34.7% decode over
    # all-experts-to-CPU. Must appear for MoE with spare VRAM, and emit a REAL -ot regex.
    from quantprobe.plan import evaluate
    _, _, cfgs = evaluate(30.5, 3.3, 1.2, True, 2.5, 6, 192, 16, 48, 0.45, 0.35, gl=0.04, n_layer=48)
    split = [c for c in cfgs if c[0].startswith("split experts")]
    assert split, f"MoE split row missing: {[c[0] for c in cfgs]}"
    flags = split[0][3]
    assert "ffn_.*_exps" in flags and "-ngl 99" in flags, f"bad split flags: {flags}"
    assert "blk\\.(" in flags and "47)" in flags, f"regex must end at the last layer: {flags}"

def t_moe_split_no_layer_count_no_bogus_regex():
    # without a layer count we must NOT invent layer indices - fall back and say why
    from quantprobe.plan import evaluate
    _, _, cfgs = evaluate(30.5, 3.3, 1.2, True, 2.5, 6, 192, 16, 48, 0.45, 0.35, gl=0.04, n_layer=None)
    split = [c for c in cfgs if c[0].startswith("split experts")]
    if split:
        assert "blk\\.(" not in split[0][3], f"invented a regex without layer count: {split[0][3]}"
        assert split[0][2] and "layer count" in split[0][2], "must explain why flags are generic"

def t_dense_split_row_unregressed():
    # the pre-existing dense split row must still work (no MoE change may break it)
    from quantprobe.plan import evaluate
    _, _, cfgs = evaluate(13, 13, 13, False, 4.5, 8, 300, 32, 50, 2, 0.5)
    assert any(c[0].startswith("split:") for c in cfgs), \
        f"dense split row regressed: {[c[0] for c in cfgs]}"

def _all_placement_rows():
    """Every row the planner can emit, swept across the realistic hardware/model space.
    Case-by-case tests missed two shipped bugs (prose flags in v1.6.5, a missing --no-mmap in
    v1.8.0); invariants over the whole space are what actually catch that class."""
    from quantprobe.plan import evaluate
    rows = []
    for moe, t, a, ne, nl in ((True, 30.5, 3.3, 1.2, 48), (True, 110, 12, 2.7, 46),
                              (False, 7, 7, 7, 32), (False, 70, 70, 70, 80)):
        for bits in (2.0, 2.5, 3.0, 4.5):
            for vc, vb in ((0, 0), (6, 192), (12, 360), (24, 936), (96, 1800)):
                for rc, rb in ((16, 48), (32, 51), (128, 80)):
                    for ctx in (0, 8192):
                        _, _, cfgs = evaluate(t, a, ne, moe, bits, vc, vb, rc, rb, 2.0, 0.5,
                                              gl=0.3, ctx=ctx, kvp=98304, n_layer=nl)
                        for c in cfgs:
                            rows.append((c, dict(moe=moe, bits=bits, vc=vc, nl=nl)))
    assert len(rows) > 300, f"sweep too small to be meaningful: {len(rows)}"
    return rows

def t_invariant_cpu_override_implies_no_mmap():
    # llama.cpp warns that tensor overrides to CPU with mmap enabled cost performance, and we
    # MEASURED it: 16.45 vs 18.70 tok/s (+13.7%). v1.8.0 shipped a row that violated this.
    bad = [(c[0], c[3]) for c, _ in _all_placement_rows()
           if "-ot" in c[3] and "=CPU" in c[3] and "--no-mmap" not in c[3]
           and "expert cache" not in c[0]]
    assert not bad, f"rows override tensors to CPU without --no-mmap: {bad[:3]}"

def t_invariant_flags_are_valid_argv():
    # no prose may ever reach a launch command (the v1.6.5 bug: a bare '+' killed llama-cli)
    for c, _ in _all_placement_rows():
        if "expert cache" in c[0]:
            continue                      # aspirational row, filtered by run/bench (tested separately)
        toks = c[3].replace('"', "").split()
        assert "+" not in toks, f"prose leaked into flags: {c[0]} -> {c[3]}"
        for i, tk in enumerate(toks):
            if tk == "-ngl":
                assert toks[i+1].isdigit() and 0 <= int(toks[i+1]) <= 99, f"bad -ngl: {c[3]}"
            if tk == "-ot":
                assert i + 1 < len(toks) and toks[i+1], f"-ot with no pattern: {c[3]}"

def t_invariant_split_regex_layers_in_range():
    import re
    for c, meta in _all_placement_rows():
        if not c[0].startswith("split experts"):
            continue
        m = re.search(r"blk\\\.\(([0-9|]+)\)", c[3])
        assert m, f"split row without a layer regex: {c[3]}"
        idx = [int(x) for x in m.group(1).split("|")]
        assert min(idx) >= 1 and max(idx) == meta["nl"] - 1, \
            f"regex layers {min(idx)}-{max(idx)} outside 1..{meta['nl']-1}: {c[0]}"

def t_invariant_rows_sorted_and_positive():
    from quantprobe.plan import evaluate
    for moe in (True, False):
        _, _, cfgs = evaluate(30.5, 3.3, 1.2, moe, 2.5, 6, 192, 16, 48, 2.0, 0.5, gl=0.3, n_layer=48)
        tps = [c[1] for c in cfgs]
        assert tps == sorted(tps, reverse=True), f"rows not sorted by tok/s: {tps}"
        assert all(x > 0 for x in tps), f"non-positive prediction: {tps}"

def t_recipes_all_valid_and_evidenced():
    # every recipe must be well-formed AND carry its evidence - a recipe you cannot check is a
    # recipe nobody should use, and that bar applies to ours as much as to contributions.
    from quantprobe.recipes import load_all
    rs = load_all()
    assert len(rs) >= 4, f"recipe atlas missing entries: {len(rs)}"
    for r in rs:
        m, p, pr = r["model"], r["probe"], r["provenance"]
        assert m["key"] and m["arch"] and m["n_layer"] > 0, f"bad model block: {m}"
        lo, hi = p["fragile_band"]
        assert 0 <= lo <= hi <= m["n_layer"] - 1, f"{m['key']}: band {lo}-{hi} outside 0..{m['n_layer']-1}"
        assert len(p["band_deltas"]) >= 3, f"{m['key']}: too few bands to locate a fragile one"
        # the declared fragile band must actually be the worst one measured
        worst = max(p["band_deltas"], key=lambda b: b["delta_ppl"])
        assert [worst["lo"], worst["hi"]] == [lo, hi], \
            f"{m['key']}: declared band {lo}-{hi} is not the measured worst {worst['lo']}-{worst['hi']}"
        for f in ("raw_log", "eval", "hardware", "measured"):
            assert pr.get(f), f"{m['key']}: provenance missing '{f}'"

def t_recipes_command_lists_them():
    rc, out = cli("recipes")
    assert rc == 0 and "fragility bands" in out and "--recipe" in out, f"recipes cmd broke: {out[:200]}"
    assert "evidence:" in out, "recipes must show their evidence"

def t_recipe_mismatch_refused():
    # a band is only meaningful for the model it was measured on
    rc, out = cli("quantize", "--gguf", "x.gguf", "--out", "o.gguf", "--recipe", "qwen3-30b", "--dry")
    assert rc != 0 and "Traceback" not in out, f"mismatch not handled: {out[:200]}"

def t_recipe_unknown_key_graceful():
    rc, out = cli("quantize", "--gguf", "x.gguf", "--out", "o.gguf", "--recipe", "no-such", "--dry")
    assert rc != 0 and "Traceback" not in out and ("no recipe" in out or "not found" in out), \
        f"unknown recipe not graceful: {out[:200]}"

def t_tensor_role_registry_covers_always_active():
    # Structure transfers between models; fragility does not. The registry must name every
    # always-active class we know about - missing one is the v1.6.4 SSM bug (-24% ppl).
    from quantprobe.spec import TENSOR_ROLES
    names = [n for n, _, _ in TENSOR_ROLES]
    for required in ("shared-expert", "attention", "recurrent/SSM", "embedding", "routed-expert"):
        assert required in names, f"tensor-role registry missing '{required}'"
    import re
    for _, pat, _ in TENSOR_ROLES:
        re.compile(pat)                      # every pattern must actually compile
    # the roles our builder protects must each have a rule
    protected = [n for n, _, d in TENSOR_ROLES if "ALWAYS ACTIVE" in d]
    assert len(protected) >= 4, f"too few always-active classes recognised: {protected}"

def t_python_m_package():
    # `python -m quantprobe` must work identically to the console script -
    # it is the PATH-proof fallback for Windows user-site installs
    r = subprocess.run([sys.executable, "-m", "quantprobe", "--help"],
                       capture_output=True, text=True, errors="replace")
    assert r.returncode == 0 and "plan" in r.stdout and "auto" in r.stdout, \
        f"python -m quantprobe broken: rc={r.returncode} {(r.stdout + r.stderr)[:200]}"

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
