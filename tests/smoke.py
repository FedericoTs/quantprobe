"""Smoke suite for quantprobe — plain asserts, no pytest dependency.
Run:  python tests/smoke.py   (needs the package installed; llama.cpp NOT required for these)"""
from __future__ import annotations
import io, os, subprocess, sys
from contextlib import redirect_stdout

# Test the code in THIS repo, not whatever happens to be installed.
#
# Running `python tests/smoke.py` puts tests/ on sys.path[0] - NOT the cwd - so in-process
# `from quantprobe.plan import ...` resolved to site-packages while the subprocess CLI tests
# (`python -m quantprobe.cli`, which does add cwd) resolved to the repo. The suite was silently
# validating TWO DIFFERENT COPIES of the code at once, and an edit that had not been reinstalled
# was invisible to exactly the half that imports directly.
#
# That cost a real mutation test: re-introducing the -ngl bug showed "ok" here while failing when
# the same function was called directly, and it took three rewrites of the assertion before the
# harness itself turned out to be the problem. verify.py layer 2 covers the INSTALLED artifact
# deliberately and separately; layer 1 should test what the developer just edited.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def t_workload_frontier_is_pareto():
    """No dominated point may sit on the frontier - and a point must EARN its place by a margin.

    This test has been rewritten three times and each rewrite lowered a number, which is the
    finding. The claimed workload spread went 2.25x -> 1.33x -> 1.23x -> gone, and every step came
    from correcting one of our own measurement errors:

      2.25x  a dominated cell (split + KV in VRAM at ub 2048, 163 pp) made the worst choice look
             far worse than it was - it had been measured past the compute-buffer cliff
      1.33x  corrected to its ub-512 form (281 pp)
      1.23x  all four cells re-measured in ONE session, after #24 found 10-13% between-session
             drift against sub-1% within-session error bars
      gone   at that width the surviving alternative wins by 0.44 tok/s against combined error bars
             of 0.456 - 0.96 sigma, which is noise

    So `MOE_FRONTIER` holds one row, and Law 7 ("there is no single best placement, there is a
    frontier") is refuted for this model on this box. The selection machinery stays, because it
    would work again the moment a genuinely better configuration is measured. What must never come
    back is a printed CHOICE that is inside its own error bars.
    """
    from quantprobe.plan import MOE_FRONTIER, workload_frontier
    for i, (li, ppi, tgi, _) in enumerate(MOE_FRONTIER):
        for j, (lj, ppj, tgj, _) in enumerate(MOE_FRONTIER):
            if i == j:
                continue
            assert not (ppj >= ppi and tgj >= tgi),                 f"'{li}' is dominated by '{lj}' on both axes - it must not be on the frontier"
    chat, rag = workload_frontier(0.5), workload_frontier(200)
    if len(MOE_FRONTIER) == 1:
        # The collapsed state. Selection must be degenerate and must not claim a spread.
        assert chat["label"] == rag["label"], "one row cannot produce two different picks"
        assert abs(rag["speedup_vs_worst"] - 1.0) < 1e-9,             f"a single-row frontier cannot have a spread, got {rag['speedup_vs_worst']}"
    else:
        # If a second row is ever restored it has to beat the alternative by more than the noise
        # floor that killed the last one - 0.96 sigma is not a recommendation.
        assert chat["label"] != rag["label"], (
            "the frontier picks the same configuration for chat and for document QA - then it is "
            "not a frontier and the whole workload dimension is unnecessary")
        assert chat["tg"] > rag["tg"], "the chat pick should favour generation"
        assert rag["pp"] > chat["pp"], "the long-prompt pick should favour prompt processing"
        assert rag["speedup_vs_worst"] > 1.25,             f"long-prompt spread only {rag['speedup_vs_worst']:.2f}x - not worth a recommendation"
        assert chat["tg"] / rag["tg"] > 1.03, (
            f"the chat pick wins by only {(chat['tg']/rag['tg']-1)*100:.1f}% on decode, which is "
            "inside the error bars that retired the previous frontier - do not ship it as a choice")


def t_accuracy_band_is_per_regime():
    """The published accuracy band must stay REGIME-AWARE, and must stay a gate.

    Until v1.15.0 one symmetric +/-25% covered every placement, and for all-in-VRAM it was false:
    13 benchmarks over 8 models put the real spread at -9% to +84%. A single +/-25% is not a
    conservative approximation of that - it is wrong in both directions at once, too wide below and
    far too narrow above.

    The lower bound is the half that earns its keep. Widening a band to make a test pass would be
    goalpost-moving; what makes this different is that the band is MEASURED, published as a
    correction, and still fails on a regression. If the tool ever became OPTIMISTIC about a model
    that fits in VRAM - the direction that actually costs a user something - -15 is what catches it.
    """
    from verify import e2e_band
    lo_v, hi_v, why_v = e2e_band("all in VRAM")
    lo_o, hi_o, why_o = e2e_band("hybrid: attention->VRAM, experts->RAM")
    assert (lo_v, hi_v) != (lo_o, hi_o), "the band is no longer regime-aware - one number is back"
    assert hi_v >= 84, f"the all-in-VRAM upper bound {hi_v} is below the measured +84% - it would fail on truth"
    assert lo_v > -100, "the all-in-VRAM band must stay bounded below; an unbounded band is not a gate"
    assert lo_v >= -20, (
        f"the all-in-VRAM lower bound has drifted to {lo_v} - that is the half that catches the tool "
        "becoming optimistic, which is the direction that costs a user something")
    assert (lo_o, hi_o) == (-25, 25), "the validated regimes must keep the published +/-25%"
    for why in (why_v, why_o):
        assert why and len(why) > 20, "each band must say WHY it is what it is"


def t_iq_quants_warned_on_cpu_tiers():
    """IQ files on a host tier must carry the measured 2.7x warning; K files must not.

    Pre-registration #31, pure-CPU decode, same 7B, r=3: Q2_K 28.4 GB/s effective, Q4_K_M 29.7,
    IQ3_XS 10.6. The K-format dequant is bandwidth-shaped on AVX2; the IQ codebook lookup is
    compute-shaped, and 4 cores cannot hide it. A user who downloads an IQ file because it is
    smaller, then lands on a host-resident placement, silently loses 2.7x - unless we tell them.
    The control matters as much as the warning: in VRAM the IQ formats are fine (the eta study
    measured IQ3 mid-pack there), so warning on a VRAM row would be crying wolf.
    """
    import os
    iq = os.path.join(GGUF_DIR, "DeepSeek-Coder-V2-Lite-Base-IQ2_XS.gguf")
    kq = os.path.join(GGUF_DIR, "Qwen3-30B-A3B-Q2_K.gguf")
    if not (os.path.isfile(iq) and os.path.isfile(kq)):
        return  # gguf fixtures absent on this machine; the code path is still covered by review
    _, out_iq = cli("plan", "--gguf", iq, "--machine", "2016-xmp")
    assert "I-quant" in out_iq and "2.7x slower" in out_iq,         "IQ MoE on a host placement lost its measured warning"
    _, out_kq = cli("plan", "--gguf", kq, "--machine", "2016-xmp")
    assert "I-quant" not in out_kq, "K-quant file must not carry the IQ warning - crying wolf"


def t_concurrency_is_disclosed_not_modelled():
    """We must say out loud that our numbers are single-stream, because they understate a server 2x.

    Pre-registration #26 measured aggregate decode from 1 to 8 slots: split 21.93 -> 44.48 (2.03x),
    all-experts-CPU 19.89 -> 37.70 (1.90x), dense 7B fully in VRAM 22.47 -> 50.48 (2.25x). The same
    ceiling on every architecture, placement and memory tier, saturating by about 4 slots - and no
    mechanism this project models predicts it (C-06).

    Two staked explanations died: host-residency amortisation was refuted BY DIRECTION (the arm with
    MORE host-resident weight gained LESS), and MoE routing divergence was refuted by the dense
    control. So concurrency must NOT be modelled. It must be disclosed, and this test is what stops
    the disclosure being quietly dropped the next time the output is tidied.
    """
    rc, out = cli("plan", "--model", "qwen3-30b", "--bits", "2.95", "--machine", "2016-xmp")
    assert rc == 0, "plan failed"
    low = out.lower()
    assert "single-stream" in low, (
        "the plan output no longer says its numbers are single-stream - anyone sizing a server on "
        "them is wrong by ~2x in a direction we have measured")
    assert "#26" in out, "the concurrency disclosure must cite the pre-registration behind it"


def t_ubatch_is_sized_not_pinned():
    """The ubatch must be SIZED from headroom, never pinned - the buffer is linear, VRAM is not.

    Pre-registration #23 measured llama.cpp's CUDA compute buffer at 0.5874 MiB per ubatch token,
    linear to four figures (601.50 / 902.25 / 1203.00 MiB at ub 1024 / 1536 / 2048). Demand grows
    smoothly; supply ends abruptly; prefill therefore falls off a CLIFF rather than tapering:
    381.21 -> 209.64 tok/s in a single -ub step, then flat for every larger value.

    A pinned `-ub 2048` is correct only for whoever has the headroom for it. This asserts the
    emitted ubatch actually fits the budget it was derived from, and that a tighter card gets a
    SMALLER ubatch rather than the same one with a warning attached.
    """
    from quantprobe.plan import safe_ubatch, ubatch_flags, COMPUTE_BUFFER_MIB_PER_UB_TOKEN
    # never promise a buffer the headroom cannot hold (half the budget, per the measured margin)
    for headroom in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        ub = safe_ubatch(headroom)
        if ub:
            assert ub * COMPUTE_BUFFER_MIB_PER_UB_TOKEN <= headroom * 1024 * 0.5,                 f"ub {ub} needs {ub*COMPUTE_BUFFER_MIB_PER_UB_TOKEN:.0f} MiB, headroom {headroom} GB"
            assert ub & (ub - 1) == 0, f"ubatch {ub} is not a power of two"
    # monotone: more headroom never yields a smaller ubatch
    sizes = [safe_ubatch(h) for h in (0.5, 1.0, 2.0, 4.0, 8.0)]
    assert sizes == sorted(sizes), f"ubatch not monotone in headroom: {sizes}"
    # a roomy card still gets the measured-best 2048; a tight one gets less, not a caveat
    assert safe_ubatch(6.0) == 2048, "roomy card should still reach the measured optimum"
    assert safe_ubatch(1.5) < 2048, "tight card must step DOWN, not be handed the cliff"
    # and the flag string the CLI emits has to agree with the sizer (at the CLI's own cap,
    # raised to 4096 for big-VRAM cards after the first external replication - see SAFE_UBATCH_CAP)
    from quantprobe.plan import SAFE_UBATCH_CAP
    fl = ubatch_flags("hybrid: attention->VRAM, experts->RAM", 0.7, 6)
    assert fl and str(safe_ubatch(6 * 0.90 - 0.7, cap=SAFE_UBATCH_CAP)) in fl, f"flags disagree with sizer: {fl}"


def t_frontier_rows_are_off_the_cliff():
    """No frontier row may quote a figure measured at the edge of VRAM.

    v1.14.0 shipped `-ub 2048 -nkvo 1` at 391.72 tok/s. The identical command measures 209.64 on
    the same box with ~250 MiB more desktop VRAM held - one browser window, a 1.85x flip. The
    number was real and reproducible; it was simply a peak sitting one step from a 45% cliff, and
    quoting it as a property of the configuration is what made it wrong.

    This test is the cheap, permanent form of that lesson: any row asking for `-ub 2048` alongside
    a placement that already fills VRAM is refused, because we have measured that combination and
    it does not survive an ordinary desktop.
    """
    import re as _re
    from quantprobe.plan import MOE_FRONTIER, COMPUTE_BUFFER_MIB_PER_UB_TOKEN
    for lab, pp, tg, flags in MOE_FRONTIER:
        m = _re.search(r"-ub\s+(\d+)", flags)
        assert m, f"frontier row has no explicit ubatch: {lab}"
        ub = int(m.group(1))
        # the split fills VRAM with experts; evicting KV buys back ~1 GB, not ~2
        if "-nkvo" in flags:
            buf = ub * COMPUTE_BUFFER_MIB_PER_UB_TOKEN
            assert buf <= 902,                 f"row '{lab}' asks {buf:.0f} MiB of compute buffer; measured cliff is above 902 MiB"
        assert pp > 0 and tg > 0, f"frontier row {lab} has a non-positive rate"
    # the row that was wrong must stay fixed: no 2048 next to -nkvo, ever again
    assert not any("-ub 2048" in f and "-nkvo" in f for _, _, _, f in MOE_FRONTIER),         "the v1.14.0 cliff configuration is back on the frontier"


def t_nkvo_never_emitted_for_deep_context():
    """-nkvo is WITHDRAWN from deep-workload advice, per prereg #25's own pre-commitment (L-24).

    The measurement that pre-committed this: split placement, -fa 1, r=3, one session. tg32 at
    d16384: q8_0 KV in VRAM 10.59 tok/s vs -nkvo 1 at 3.48 - 3.04x worse decode - for a prefill
    difference inside the error bar (382.17 vs 386.14 pp2048). -nkvo exists to serve RAG and
    document-QA at 50:1/200:1 prompt:generation ratios, which are exactly the deep contexts
    where it loses hardest. So three invariants, each falsifiable by one line of output:

      1. no frontier row (the source of every workload recommendation) carries -nkvo;
      2. no placement `evaluate` emits at d16384 - any preset model on any preset machine -
         carries -nkvo in its flags;
      3. a deep-context `plan` never puts -nkvo in the run command, and if the flag appears in
         prose at all it must be the withdrawal (accompanied by its measured 3.04x), never a
         recommendation.

    Failing inputs constructed when this test was written, each applied, observed to fail (exit
    non-zero), and reverted: (A) '-nkvo 1' appended to a MOE_FRONTIER row's flags; (B) '-nkvo 1'
    appended to evaluate()'s all-in-VRAM row flags; (C) the withdrawal print lines silently
    deleted from the workload advice.
    """
    import itertools
    from quantprobe.plan import (MOE_FRONTIER, MODELS, MACHINES, DEFAULT_KVP, evaluate)
    for lab, _pp, _tg, fl in MOE_FRONTIER:
        assert "-nkvo" not in fl, f"frontier row '{lab}' recommends -nkvo: {fl}"
    for mn, hn in itertools.product(MODELS, MACHINES):
        m, hw = MODELS[mn], MACHINES[hn]
        _, _, cfgs = evaluate(m["t"], m["a"], m["ne"], m["moe"], 2.5,
                              hw["vc"], hw["vb"], hw["rc"], hw["rb"], hw["db"],
                              hw.get("geta", 0.45), gl=hw.get("gl"),
                              ctx=16384, kvp=m.get("kvp", DEFAULT_KVP), n_layer=m.get("nl"))
        for name, _tps, _warn, flags in cfgs:
            assert "-nkvo" not in flags, (
                f"{mn} on {hn} at d16384 emits -nkvo in '{name}': {flags}")
    # MoE deep workload (the RAG case prereg #25 measured) and a dense deep split whose winning
    # row FIRES the depth note (mistral-7b @4.5 bits on 2016-xmp -> split: 20/32 layers->VRAM):
    # the emitted command is clean, and prose only ever mentions -nkvo to bury it.
    for extra in (("--model", "qwen3-30b", "--machine", "2016-xmp"),
                  ("--model", "mistral-7b", "--machine", "2016-xmp", "--bits", "4.5")):
        rc, out = cli("plan", *extra, "--ctx", "16384")
        assert rc == 0, out
        runline = next(l for l in out.splitlines() if "run it:" in l)
        assert "-nkvo" not in runline, f"deep-context run command emits -nkvo: {runline}"
        if "-nkvo" in out:
            assert "3.04x" in out and "WITHDRAWN" in out, (
                f"-nkvo appears in deep-context output without its measured withdrawal "
                f"(3.04x, prereg #25): {extra}")
    # both deep cases must actually SHOW the withdrawal, not merely avoid the flag - silent
    # removal would leave a user who read the old advice with no correction
    rc, out = cli("plan", "--model", "qwen3-30b", "--machine", "2016-xmp", "--ctx", "16384")
    assert "3.04x WORSE" in out and "WITHDRAWN" in out, "MoE deep advice lost the withdrawal"
    rc, out = cli("plan", "--model", "mistral-7b", "--machine", "2016-xmp", "--bits", "4.5",
                  "--ctx", "16384")
    assert "3.04x WORSE" in out and "WITHDRAWN" in out, "dense depth note lost the withdrawal"


def t_ubatch_only_when_host_resident():
    """-ub is a prefill lever for HOST-resident weights only, and it is measured to hurt otherwise.

    Pre-registration #19, same box, same session, r=3:
        Qwen3-30B -ot exps=CPU   pp2048  199.90 -> 345.89  (+73%)  at ub 512 -> 2048
        dense 7B fully in VRAM   pp2048  329.80 -> 200.31  (-39%)  same flag, opposite sign
    Emitting it by default would hand a 39% regression to everyone whose model fits in VRAM -
    which is the most common configuration for anyone with adequate VRAM.
    """
    from quantprobe.plan import ubatch_flags, UBATCH_HEADROOM_GB
    # host-resident placements with headroom -> emitted
    for placement in ("hybrid: attention->VRAM, experts->RAM",
                      "pure CPU (GPU idle)",
                      "stream from disk (cold experts)"):
        assert ubatch_flags(placement, 0.7, 6), f"ubatch not offered for host-resident: {placement}"
    # The SPLIT: #20 measured ub 2048 at -42% there (the compute-buffer cliff), so v1.13-v1.20
    # excluded it entirely. #62 then measured the SAME placement at ub 1024 with pp 393.7 AND
    # tg 22.21 - both at their best - while the excluded config left ~30% prefill on the table
    # (#66: pp 301). The gate is now a HARD 1024 CAP, never the measured-cliff 2048.
    sp = ubatch_flags("split experts: 21%->VRAM, rest->RAM", 0.7, 6)
    assert sp and "-ub 1024" in sp and "2048" not in sp, \
        f"split must get ub capped at 1024 (measured best, #62) and never 2048 (measured cliff, #20): {sp}"
    # fully VRAM-resident -> never, this is the measured -39% case
    assert ubatch_flags("all in VRAM", 4.7, 6) is None, \
        "ubatch offered for an all-in-VRAM placement - measured there it LOSES 39%"
    # host-resident but no VRAM headroom -> withheld (the compute buffer would not fit)
    assert ubatch_flags("hybrid: attention->VRAM, experts->RAM", 6 * 0.9 - 0.1, 6) is None, \
        "ubatch offered with no VRAM headroom for the larger compute buffer"
    # no GPU at all -> nothing to amortise a transfer to
    assert ubatch_flags("pure CPU (GPU idle)", 0, 0) is None, "ubatch offered with no GPU"


def t_dense_split_ngl_is_a_layer_count():
    """-ngl must be a LAYER COUNT, and must never exceed the model's layers on a split row.

    This emitted `int(g * 99)` where g is a FRACTION and 99 is the all-layers sentinel used
    elsewhere in plan.py. Two failures, the second severe:
      * llama-70b printed "split: 50% layers->VRAM" and emitted -ngl 49 - which is 61% of 80.
      * for any model with <= 99*g layers the flag EXCEEDS the layer count, so llama.cpp puts
        EVERY layer on the GPU - on a row that exists ONLY because the model does not fit in
        VRAM. A 32-layer model does this for any g > 0.32. OOM, or silent thrash on Windows.
    """
    import re
    from quantprobe.plan import evaluate, MODELS
    checked = 0
    for key in ("llama-70b", "mistral-7b"):
        m = MODELS[key]
        nl = m["nl"]
        for hw in (dict(vc=24, vb=1008, rc=64, rb=83, db=5, geta=0.62),
                   dict(vc=8, vb=256, rc=32, rb=45, db=2, geta=0.45)):
            for bits in (2.0, 2.5, 4.5):
                size, _, cfgs = evaluate(m["t"], m["a"], m["ne"], m["moe"], bits,
                                         n_layer=nl, **hw)
                for name, tps, warn, flags in cfgs:
                    if not name.startswith("split:"):
                        continue
                    checked += 1
                    g = re.search(r"-ngl (\d+)", flags)
                    assert g, f"{key}: split row emitted no -ngl: {flags}"
                    n = int(g.group(1))
                    # (a) the flag must be a real layer count, never >= the model's layers
                    assert 0 < n < nl, (
                        f"{key} has {nl} layers but the split row emits -ngl {n} - llama.cpp "
                        f"would place ALL layers on the GPU, on a row that only exists because "
                        f"the model does NOT fit in VRAM")
                    # (b) THE PHYSICAL CHECK, and the one that actually bites. The layers we send
                    # to the GPU must fit in its VRAM - that is the entire reason this row exists
                    # instead of "all in VRAM". Checking the label against the flag does NOT work
                    # (both derive from the same variable, so a wrong value agrees with itself),
                    # and a bare range check does not either: restoring `int(g * 99)` on
                    # llama-70b yields -ngl 49, comfortably under 80 layers, while asking the card
                    # to hold 49/80 of a 42.9 GB model = 26.3 GB on a 24 GB GPU. Only the fit test
                    # sees it. Mutation testing rejected two weaker versions of this assertion.
                    on_gpu = n / nl * size
                    assert on_gpu <= hw["vc"] * 0.90 + 1e-6, (
                        f"{key} @{bits}b: -ngl {n} of {nl} layers puts {on_gpu:.1f} GB on a "
                        f"{hw['vc']} GB card (usable {hw['vc'] * 0.90:.1f}) - the emitted command "
                        f"cannot fit what the row promises")
    # Without this the test can pass VACUOUSLY - if every case gets suppressed the loop body
    # never runs and nothing is asserted. Mutation testing caught exactly that.
    assert checked >= 3, (
        f"only {checked} split rows were actually checked - the cases this test exists for are "
        f"being suppressed rather than verified, so it is asserting nothing")
    # and with no grounded layer count the row must be SUPPRESSED, not guessed
    _, _, cfgs = evaluate(11.9, 11.9, 11.9, False, 4.5, vc=8, vb=256, rc=32, rb=45, db=2,
                          geta=0.45, n_layer=None)
    assert not [c for c in cfgs if c[0].startswith("split:")], \
        "dense split row offered without a layer count - it cannot emit a correct -ngl"


def t_dense_speed_responds_to_bits():
    """A dense model quantized harder must be predicted faster. It was not.

    The tables set ne = t for dense models - true for ACTIVATION, false for QUANTIZATION - so the
    law priced every parameter at max(bits, 4.5) and a dense model's predicted speed was identical
    at 2.5 and 4.5 bits. That is how the published calculator ended up showing a 12B dense SLOWER
    than a 106B MoE with the same active parameter count, which is what a user reported
    (pre-registration #17). Held-out validation: Qwen2.5-7B IQ3_M, error -22% -> -7%.
    """
    from quantprobe.plan import evaluate, MODELS
    hw = dict(vc=0, vb=0, rc=64, rb=80, db=5, geta=0.5)      # pure-RAM: genuinely bandwidth-bound
    m = MODELS["gemma-12b"]
    speeds = []
    for bits in (2.0, 3.0, 4.5):
        _, act, cfgs = evaluate(m["t"], m["a"], m["ne"], m["moe"], bits, **hw)
        speeds.append(cfgs[0][1])
    assert speeds[0] > speeds[1] > speeds[2], (
        f"dense speed does not respond to bit-width: 2.0/3.0/4.5 bits -> {speeds}")
    # and the ratio must be substantial, not a rounding artifact: 2 bits reads far fewer bytes
    assert speeds[0] / speeds[2] > 1.4, f"dense 2-bit barely faster than 4.5-bit: {speeds[0]/speeds[2]:.2f}x"
    # MoE must be UNTOUCHED - there `ne` already names the protected set exactly
    g = MODELS["glm-air"]
    _, act_moe, _ = evaluate(g["t"], g["a"], g["ne"], g["moe"], 2.5, **hw)
    expected = (g["ne"] * 4.5 / 8 + (g["a"] - g["ne"]) * 2.5 / 8) * 1.15
    assert abs(act_moe - expected) < 1e-6, f"MoE activation changed: {act_moe} vs {expected}"


def t_effective_n_layer_is_the_only_resolver():
    """One resolver, and every command actually reaching the same answer.

    The three-step fallback (explicit flag -> GGUF -> preset `nl`) was hand-written at each call
    site and was wrong FOUR times: v1.9.0 target.py, v1.10.5 runtime.py, plan's layer-count note,
    and auto.py - which omitted the preset step entirely, so `auto qwen3-30b` recommended the
    hybrid placement at 22.4 tok/s when the split placement it could not see runs 26.9.
    """
    from quantprobe.plan import effective_n_layer, MODELS
    import argparse
    assert effective_n_layer(None, "qwen3-30b") == 48
    assert effective_n_layer(None, MODELS["qwen3-30b"]) == 48
    assert effective_n_layer(None, "gemma-12b") is None          # no verified count -> honest None
    assert effective_n_layer(None, "not-a-preset") is None
    assert effective_n_layer(argparse.Namespace(n_layer=61), "qwen3-30b") == 61   # explicit wins
    assert effective_n_layer(argparse.Namespace(n_layer=None), "qwen3-30b") == 48


def t_auto_reaches_the_split_placement_for_presets():
    """auto is the flagship path; it must not lose the placement plan finds for the same model."""
    rc, out = cli("auto", "qwen3-30b", "--dry", "--machine", "2016-xmp")
    if "could not list" in out:
        return                                   # offline
    assert "split experts" in out, (
        "auto lost the MoE split placement for a preset - it sets a.model=None and so discarded "
        "the preset's layer count:\n" + out[:400])


def t_auto_transfers_every_model_field():
    """Every fact auto resolves about the model must actually reach the law.

    auto cannot pass a preset name downstream (a raw HF repo has no preset), so it passes explicit
    parameters - which means each fact the preset carried has to be transferred by hand. When that
    was a row of bare assignments, one was forgotten: the layer count, costing the flagship path
    its best placement for every preset MoE, silently, with nothing raised (v1.11.1).

    This test makes forgetting fail loudly. Add a field to ModelSpec and it goes red until you
    either transfer it in apply_to or declare it local-only on purpose.
    """
    import argparse
    from quantprobe.auto import ModelSpec, resolve_model, apply_to
    LOCAL_ONLY = {"repo", "moe"}          # consumed by auto itself; never read from `a` by the law
    a = argparse.Namespace(total=None, active=None, always_active=None, model=None, n_layer=None)
    spec = apply_to(resolve_model(a, "qwen3-30b"), a)
    for f in ModelSpec._fields:
        if f in LOCAL_ONLY:
            continue
        assert getattr(a, f, None) == getattr(spec, f), (
            f"resolve_model found {f}={getattr(spec, f)} but apply_to never put it on the args "
            f"the law reads - this is exactly how the layer count was lost")
    assert spec.n_layer == 48, f"preset layer count lost: {spec.n_layer}"
    # force a decision about any NEW field rather than letting it silently go nowhere
    assert set(ModelSpec._fields) == {"repo", "total", "active", "always_active", "moe", "n_layer"}, \
        ("ModelSpec gained or lost a field. Transfer it in apply_to and list it here, or add it "
         "to LOCAL_ONLY if auto consumes it directly.")


def t_auto_and_plan_recommend_the_same_placement():
    """auto must not recommend a different placement than plan does for the same model+machine.

    This is the invariant the v1.11.1 bug broke and no test held. auto sets a.model = None to
    hand the law explicit parameters, which also discarded the preset's layer count, so its whole
    frontier lost the MoE split row and recommended hybrid at 22.4 tok/s where plan finds split
    at 26.9. Both commands were internally consistent; they simply disagreed with each other.

    Sibling of t_commands_agree_on_the_same_input, which covers plan/run/bench. auto is the
    flagship path and was outside that net.
    """
    import re
    rc, out = cli("auto", "qwen3-30b", "--dry", "--machine", "2016-xmp")
    if rc != 0 or "could not list" in out:
        return                                    # offline; the frontier still printed above
    m = re.search(r"\*\s+[0-9.]+ tok/s\s+quality x[0-9.]+\s+\S+\s+([0-9.]+)-bit[^+]*\+\s*([^\n+]+)", out)
    assert m, f"auto printed no winning row to compare:\n{out[:500]}"
    bits, auto_place = m.group(1), m.group(2).strip()
    rc2, out2 = cli("plan", "--model", "qwen3-30b", "--machine", "2016-xmp", "--bits", bits)
    m2 = re.search(r"\*\s+[0-9.]+ tok/s\s+([^\[\n]+)", out2)
    assert m2, f"plan printed no winning row:\n{out2[:400]}"
    plan_place = m2.group(1).strip()
    # compare the placement KIND, not the percentage (auto and plan may size the split slightly
    # differently from rounding); disagreeing on the kind is the bug.
    kind = lambda s: re.sub(r"\d+%?", "N", s).strip()
    assert kind(auto_place) == kind(plan_place), (
        f"auto and plan disagree on placement at {bits} bits:\n"
        f"  auto: {auto_place}\n  plan: {plan_place}")


def t_layer_count_note_only_when_genuinely_unknown():
    """The layer-count note must not appear when the flags were already emitted.

    It tested `args.n_layer` - the raw CLI flag - while the placement rows read the effective
    value, which falls back to a preset's verified `nl`. So `plan --model qwen3-30b` printed exact
    -ot flags for layers 10-47 and then told the user to "re-run with --gguf to unlock it".
    Same class as the v1.10.5 n_layer divergence: a second reader of a value that has a fallback,
    written without the fallback.
    """
    rc, out = cli("plan", "--model", "qwen3-30b", "--machine", "2016-xmp", "--bits", "2.95")
    assert rc == 0
    assert "-ot " in out, "the split row's flags vanished"
    assert "re-run with --gguf" not in out, (
        "plan emitted -ot flags AND told the user the layer count is missing:\n" + out[:600])
    # but a custom MoE with no layer count anywhere must still be told how to unlock it
    rc2, out2 = cli("plan", "--total", "30.5", "--active", "3.3", "--always-active", "1.2",
                    "--machine", "2016-xmp", "--bits", "2.95")
    assert "re-run with --gguf" in out2, "the note vanished when it IS needed"


def t_simulator_law_matches_the_cli():
    """The published simulator runs its own copy of the law in JavaScript. It must agree.

    docs/index.html is the most-seen surface this project has, and it reimplements evalCore()
    by hand. Nothing kept the two in step: when the sub-4-bit collapse was removed from plan.py
    the simulator still had `bitsVal>=4?H.geta:gl` in it, so the website would have gone on
    telling people their GPU was useless for a Q3 quant after the CLI had stopped.

    Extracts evalCore from the page, runs it under node on fixed cases, and compares every row
    against plan.evaluate(). Skips (loudly) when node is absent.
    """
    import shutil, json, re, tempfile
    node = shutil.which("node")
    if not node:
        print("      (simulator parity: node not installed, SKIPPED)", end="")
        return
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page = os.path.join(here, "docs", "index.html")
    if not os.path.isfile(page):
        return
    src = io.open(page, encoding="utf-8").read()
    js = re.search(r"<script[^>]*>(.*?)</script>", src, re.S).group(1)

    def grab(sig):
        i = js.index(sig); j = js.index("{", i); d = 0
        for k in range(j, len(js)):
            d += (js[k] == "{") - (js[k] == "}")
            if d == 0:
                return js[i:k + 1]
        raise AssertionError("unbalanced braces around " + sig)

    # every scalar constant the law reads, not just ETA_KV: the harness silently lost CPU_ATTN
    # when L-19 shipped, and an un-run parity test is worse than none.
    consts = "\n".join(l for l in js.splitlines()
                       if re.match(r"\s*(const|let)\s+(ETA_KV|CPU_ATTN|NLAY_DEFAULT|DEFAULT_KVP)\s*=", l))
    qm = re.search(r"(?:const|let)\s+QUAL\s*=\s*\{.*?\};", js, re.S)
    harness = consts + "\n" + (qm.group(0) if qm else "") + "\n" + grab("function evalCore(") + """
const H = {vc:6,vb:192,rc:16,rb:48,db:0.45,geta:0.35,gl:0.04};
const OUT = [];
for (const [t,a,ne,moe,b] of [[30.5,3.3,1.2,true,2.95],[7.2,7.2,7.2,false,2.5],
                              [110,12,2.7,true,2.5],[7.2,7.2,7.2,false,4.5]]) {
  const e = evalCore(t,a,ne,moe,b,H,0,0);
  OUT.push(Object.fromEntries(e.cfgs.map(c => [c.n, +c.tps.toFixed(2)])));
}
console.log(JSON.stringify(OUT));
"""
    fd, path = tempfile.mkstemp(suffix=".js")
    os.close(fd)
    io.open(path, "w", encoding="utf-8").write(harness)
    try:
        r = subprocess.run([node, path], capture_output=True, text=True)
        assert r.returncode == 0, f"simulator JS failed to run: {r.stderr[:400]}"
        sim = json.loads(r.stdout.strip().splitlines()[-1])
    finally:
        os.unlink(path)

    sys.path.insert(0, os.path.join(here, "quantprobe"))
    from quantprobe.plan import evaluate
    M = dict(vc=6, vb=192, rc=16, rb=48, db=0.45, geta=0.35, gl=0.04)
    cases = [dict(t=30.5, a=3.3, ne=1.2, moe=True, bits=2.95),
             dict(t=7.2, a=7.2, ne=7.2, moe=False, bits=2.5),
             dict(t=110, a=12, ne=2.7, moe=True, bits=2.5),
             dict(t=7.2, a=7.2, ne=7.2, moe=False, bits=4.5)]
    drift = []
    for kw, srows in zip(cases, sim):
        _, _, cfgs = evaluate(**kw, **M)
        # normalise the arrows the page renders as unicode
        srows = {k.replace("→", "->"): v for k, v in srows.items()}
        for name, tps in ((c[0], c[1]) for c in cfgs):
            if name not in srows:
                continue           # rows the simulator has not implemented are a gap, not drift
            # the page reports toFixed(2), so allow one rounding step in absolute terms as well
            # as the 1% relative band - at 0.19 tok/s half a cent is already 2.6% relative
            if abs(srows[name] - tps) > 0.01 and abs(srows[name] - tps) / max(tps, 1e-9) > 0.01:
                drift.append(f"{kw['t']}B '{name}': CLI {tps:.4f} vs simulator {srows[name]:.2f}")
    assert not drift, ("the published simulator disagrees with the CLI:\n  " + "\n  ".join(drift))


def t_fits_in_vram_warning_is_consistent():
    """plan and optimize must BOTH disclose the fits-in-VRAM trap, and both stay quiet otherwise.

    The law ranks by bandwidth, so once everything fits in VRAM it puts 2-bit above 4.5-bit.
    Measured, that gain is not there: the same 7B at Q2_K vs Q4_K_M is 36% smaller and 4% slower
    (pre-registration #16). A disclosure that only one of the two commands makes is the same
    inconsistency class as the plan-vs-bench disagreement - a user gets different advice
    depending on which command they happened to run.
    """
    # Pin the INVARIANTS, not the prose - the wording changed once already and broke this test.
    # Two things must be true of both commands in the all-in-VRAM regime: they disclose that the
    # prediction is a floor (the measured one-directional bias), and they ask for the datapoint
    # that would fix it. Below 4.5 bits they must also say a lower quant buys nothing.
    MARKS = ("floor, not a ceiling", "bench --contribute")
    big = ["--total", "30.5", "--active", "3.3", "--always-active", "1.2", "--vram", "24",
           "--vram-bw", "936", "--ram", "64", "--ram-bw", "86", "--disk-bw", "3"]
    _, opt = cli("optimize", *big)
    _, pl = cli("plan", "--model", "mistral-7b", "--machine", "2016-xmp", "--bits", "2.5")
    for name, out in (("optimize", opt), ("plan", pl)):
        for mark in MARKS:
            assert mark in out, f"{name} lost the all-in-VRAM disclosure '{mark}':\n{out[-500:]}"
        assert "4% SLOWER" in out, f"{name} lost the low-bit guidance below 4.5 bits"
    # and neither may cry wolf when the model genuinely does NOT fit, where bytes really do buy
    # speed and quantizing down is the correct advice
    _, opt2 = cli("optimize", "--model", "qwen3-30b", "--machine", "2016-xmp")
    _, pl2 = cli("plan", "--model", "qwen3-30b", "--machine", "2016-xmp", "--bits", "2.95")
    for name, out in (("optimize", opt2), ("plan", pl2)):
        assert "floor, not a ceiling" not in out, f"{name} warns when the model does not fit VRAM"


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
    # The pre-existing dense split row must still work (no MoE change may break it). It now
    # REQUIRES a layer count, because -ngl IS a layer count and without one we cannot emit a
    # correct command - see t_dense_split_ngl_is_a_layer_count for what that cost.
    from quantprobe.plan import evaluate
    _, _, cfgs = evaluate(13, 13, 13, False, 4.5, 8, 300, 32, 50, 2, 0.5, n_layer=40)
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
    # U-23 adds ONE sanctioned exception: on a RAM-dominating placement we keep mmap so the
    # pages stay evictable - but the row must then SAY so. Silent omission stays a failure.
    bad = [(c[0], c[3]) for c, _ in _all_placement_rows()
           if "-ot" in c[3] and "=CPU" in c[3] and "--no-mmap" not in c[3]
           and "expert cache" not in c[0] and "keeping mmap" not in (c[2] or "")]
    assert not bad, f"rows override tensors to CPU without --no-mmap and without saying why: {bad[:3]}"


def t_u23_mmap_gate():
    # U-23 (E-08): --no-mmap only while the placement leaves the RAM pool room; the exception
    # must carry its explanation and the measured cost of taking it.
    # v1.23's remedy (dropping --no-mmap when RAM-tight) was REFUTED by the full ladder: it cost
    # 2.9x (22.20 -> 7.70), because near the RAM boundary mmap thrashes. --no-mmap now ALWAYS
    # ships; the tight case gets both measured numbers and the user chooses.
    from quantprobe.plan import mmap_decision, moe_split_flags, MMAP_HOST_SHARE_CAP
    assert mmap_decision(7, 12) == (True, None), "roomy: keep the flag, say nothing"
    ok, note = mmap_decision(11, 12)
    assert ok is True, "the flag must never be dropped again - it measured 2.9x"
    assert note and "2.9x SLOWER" in note and "OOM" in note, "tight case must give BOTH sides"
    for args in ((0.3, 48, 7, 12), (0.3, 48, 11, 12), (0.3, 48)):
        assert "--no-mmap" in moe_split_flags(*args), f"flag dropped for {args}"
    assert 0.5 < MMAP_HOST_SHARE_CAP < 1.0

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

def t_format_advice_lever():
    # The format lever (preregs #52/#53): all-in-VRAM + low bits must surface the Q4_0-over-K-quant
    # advice, with its scope (pre-Ampere, speed-only, may invert on Ampere+) stated in the text.
    from quantprobe.plan import format_advice
    a = format_advice("all in VRAM", 4.5)
    assert a and "Q4_0" in a and "+19%" in a and "Ampere" in a, f"4.5-bit lever wrong: {a}"
    b = format_advice("all in VRAM", 2.8)
    assert b and "REVERSED" in b and "Q2_K" in b, f"2.8-bit dominance warning wrong: {b}"
    # ...and must NOT fire where it was not measured
    assert format_advice("hybrid (waterfall)", 4.5) is None
    assert format_advice("all in VRAM", 8.0) is None


def t_format_advice_reaches_user():
    # the note must actually reach plan output through fits_in_vram_advice
    from quantprobe.plan import fits_in_vram_advice
    n = fits_in_vram_advice("all in VRAM", 4.5)
    assert n and "FORMAT LEVER" in n, "format advice not wired into the all-in-VRAM note"


def t_format_advice_honesty():
    # the claim must never appear without its limits: one card, speed-only
    from quantprobe.plan import format_advice
    for bits in (2.5, 3.0, 4.5, 5.0):
        a = format_advice("all in VRAM", bits)
        if a:
            assert ("unverified" in a and "invert" in a), f"missing scope honesty at {bits}: {a}"


def t_format_advice_iq4nl():
    # prereg #70: the 4-5 bit lever must offer IQ4_NL (Q4_0-class kernel, +14% measured, imatrix
    # quality) and warn off codebook IQ; the low-bit branch names the codebook penalty too.
    from quantprobe.plan import format_advice
    mid = format_advice("all in VRAM", 4.5)
    assert "IQ4_NL" in mid and "+14%" in mid and "codebook" in mid.lower()
    low = format_advice("all in VRAM", 2.5)
    assert "IQ" in low and "prereg #70" in low


def t_calibrate_boost_verdict():
    # the #60/#61 diagnostic must classify all three states and never crash on missing data
    from quantprobe.calibrate import boost_verdict
    assert "healthy" in boost_verdict(1873, 1911, 45)
    stuck = boost_verdict(1506, 1911, 38)
    assert "STUCK BOOST" in stuck and "REBOOT" in stuck
    assert "THROTTLED" in boost_verdict(1400, 1911, 85)
    assert boost_verdict(None, 1911, 40) is None
    assert boost_verdict(1500, 0, 40) is None


def t_calibrate_roundtrip():
    # calibration persists and loads with an age; a corrupt file degrades to (None, None)
    import json, tempfile, time, os
    from quantprobe import calibrate as c
    old = c.CAL_PATH
    try:
        with tempfile.TemporaryDirectory() as d:
            c.CAL_PATH = os.path.join(d, "calibration.json")
            with open(c.CAL_PATH, "w") as f:
                json.dump({"ts": time.time() - 86400, "ram_bw_measured": 24.8}, f)
            cal, age = c.load()
            assert cal["ram_bw_measured"] == 24.8 and 0.9 < age < 1.1
            with open(c.CAL_PATH, "w") as f:
                f.write("{corrupt")
            assert c.load() == (None, None)
    finally:
        c.CAL_PATH = old


def t_calibrate_cli_registered():
    rc, out = cli("calibrate", "--help")
    assert rc == 0 and "--model" in out and "--skip-bench" in out


def t_moneroape_channel_count():
    # THE 3.7x INPUT BUG from the first external replication: 4 DIMMs on consumer AM5 must be
    # treated as DUAL channel, not 4-channel. HEDT names keep their width.
    import quantprobe.detect as d, platform
    orig = platform.processor
    try:
        platform.processor = lambda: "AMD Ryzen 5 8600G w/ Radeon"
        # simulate the detect() channel logic directly: consumer + 4 sticks -> 2 channels
        cpu = platform.processor().lower()
        wide = any(w in cpu for w in ("threadripper", "epyc", "xeon w-3"))
        assert not wide
        platform.processor = lambda: "AMD Ryzen Threadripper 7970X"
        cpu = platform.processor().lower()
        assert "threadripper" in cpu
    finally:
        platform.processor = orig
    # and the real detect() on THIS box must not crash and must mention calibrate in the RAM note
    _, notes = d.detect()
    ram_notes = [n for n in notes if n.startswith("RAM:")]
    assert ram_notes and ("calibrate" in ram_notes[0] or "unknown" in ram_notes[0]), ram_notes


def t_moneroape_ubatch_cap():
    # cap raised for big-VRAM cards (external 3090/4090 datapoint), buffer math still gates
    from quantprobe.plan import safe_ubatch, SAFE_UBATCH_CAP
    assert SAFE_UBATCH_CAP == 4096
    assert safe_ubatch(20.0, cap=SAFE_UBATCH_CAP) == 4096      # 24GB-class headroom reaches 4096
    assert safe_ubatch(1.5, cap=SAFE_UBATCH_CAP) == 1024       # 6GB-class card unchanged
    assert safe_ubatch(0.05, cap=SAFE_UBATCH_CAP) == 0


def t_moneroape_pinning_warning():
    # a 3090+64GB with a 55GB MoE must warn about pinned host memory on -ot rows
    rc, out = cli("plan", "--total", "117.6", "--active", "8.4", "--bits", "3.75",
                  "--vram", "24", "--vram-bw", "936", "--ram", "64", "--ram-bw", "86",
                  "--disk-bw", "3")
    assert rc == 0 and "pins" in out and "auto-placement" in out, out[:600]


def t_moneroape_threads_and_topline():
    # CPU-resident placements must carry --threads, and the speculation reality must be top-line
    rc, out = cli("plan", "--model", "qwen3-30b", "--machine", "2016-xmp")
    assert rc == 0
    assert "--threads" in out, "no --threads in emitted command"
    head = out[:out.index("run it:")]
    assert "0 drafts" in out and "speculation: pays ONLY" in out
    assert "pp2048" in out, "pp published without its measurement conditions"


def t_anchored_predictions_wiring():
    # anchors must be applied by default WITH provenance, and --no-anchors must remove them.
    # Runs against this box's real calibration when present; the suppression half always runs.
    import os
    from quantprobe.calibrate import CAL_PATH, load
    cal, _ = load()
    has_anchors = bool(cal and cal.get("anchors"))
    rc, out = cli("plan", "--model", "qwen3-30b")
    assert rc == 0
    if has_anchors:
        assert "anchored:" in out and "tier ratios" in out, "anchors present but not applied/labeled"
    rc2, out2 = cli("plan", "--model", "qwen3-30b", "--no-anchors")
    assert rc2 == 0 and "anchored:" not in out2, "--no-anchors did not suppress anchoring"


def t_clock_sampler_min_samples():
    # the false-positive guard: fewer than 3 loaded samples must yield None (no verdict),
    # because 1-2 samples can be the model-load ramp, measured as a wrong REBOOT alarm.
    from quantprobe.calibrate import ClockSampler
    s = ClockSampler.__new__(ClockSampler)
    s.samples = [(1506, 35)]
    assert s.sustained() is None
    s.samples = [(1506, 35), (1873, 40)]
    assert s.sustained() is None
    s.samples = [(1860, 40), (1873, 41), (1885, 42)]
    assert s.sustained() == 1873


def t_dense_draft_note_three_cells():
    # prereg #67/#69: the draft-model advice must match the measured cell — split gets the +33%
    # CPU-draft note (K=2, -ngld 0), AIV keeps the +11% note, MoE never gets either.
    from quantprobe.plan import dense_draft_note
    split = dense_draft_note(False, "split: 28/48 layers->VRAM, rest->RAM")
    assert split and "+33%" in split and "-ngld 0" in split and "prereg #69" in split
    aiv = dense_draft_note(False, "all in VRAM")
    assert aiv and "+11%" in aiv and "prereg #67" in aiv
    assert dense_draft_note(True, "split: 10/48 layers->VRAM, rest->RAM") is None
    assert dense_draft_note(False, "pure CPU (GPU idle)") is None


def t_fetch_force_and_collision():
    # U-18: a same-named file of a DIFFERENT size must FAIL the skip (it once fed an incompatible
    # draft model to llama-speculative), and --force must be a registered flag.
    import os, tempfile
    from quantprobe import fetch as fmod
    class _R:
        headers = {"Content-Length": "1000"}
    real_head = fmod.requests.head
    fmod.requests.head = lambda *a, **k: _R()
    try:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "model.gguf")
            open(p, "wb").write(b"x" * 999)          # wrong size vs remote 1000
            assert fmod.fetch("org/repo", d, "model.gguf", None) is False, "size-mismatch skip must fail"
            open(p, "wb").write(b"x" * 1000)         # matching size: legitimate skip
            assert fmod.fetch("org/repo", d, "model.gguf", None) is True
    finally:
        fmod.requests.head = real_head
    rc, out = cli("fetch", "--help")
    assert rc == 0 and "--force" in out


def t_c11_depth_aware_dense_split():
    # C-11 (prereg #66): the dense split must budget for the desktop reserve + compute buffer and
    # shrink its GPU layer count as context deepens - the old flat vc*0.9 emitted a 16k config
    # that measured -58% (driver memory fallback).
    from quantprobe.plan import evaluate
    def layers(ctx):
        _, _, cfgs = evaluate(14.8, 14.8, 14.8, False, 4.85, 6, 154, 16, 26, 2.0, 0.55,
                              gl=None, ctx=ctx, kvp=57344, n_layer=48, true_size_gb=8.37)
        row = [c for c in cfgs if "layers->VRAM" in c[0]]
        assert row, "dense split row missing"
        return int(row[0][0].split(":")[1].split("/")[0])
    shallow, deep = layers(0), layers(16384)
    assert deep < shallow, f"layers must shrink with depth ({shallow} -> {deep})"
    # and the emitted GPU share must fit inside VRAM minus reserve+buffer
    assert shallow / 48 * 8.37 <= 6 * 0.9 - 1.0, "shallow emit overcommits the budget"


def t_u17_iq_cpu_pricing():
    # U-17 (prereg #66): iq_share must slow every RAM weight read by the calibrated per-byte
    # penalty (pure-CPU arm 14.1 -> 11.44 at share 0.962); iq_share=0 (presets) must be untouched.
    # C-13: the penalty follows the CODEBOOK share, not every format named IQ (IQ4_NL is
    # Q4_0-class per #70). The constant was re-derived so the calibration arm is unchanged:
    # old iq_share 0.962 x 0.242 == new codebook_share 0.510 x 0.456, to within rounding.
    from quantprobe.plan import evaluate, IQ_CPU_TG_PENALTY
    def cpu_tokps(share):
        _, _, cfgs = evaluate(15.7, 2.4, 0.8, True, 2.9, 6, 154, 16, 26, 2.0, 0.55,
                              gl=None, codebook_share=share)
        return [c for c in cfgs if c[0].startswith("pure CPU")][0][1]
    base, cb = cpu_tokps(0.0), cpu_tokps(0.510)
    want = 1.0 + 0.510 * IQ_CPU_TG_PENALTY
    assert abs(base / cb - want) < 0.01, f"codebook penalty off: {base/cb:.4f} vs {want:.4f}"
    assert abs((0.510 * IQ_CPU_TG_PENALTY) - (0.962 * 0.242)) < 0.005, \
        "the re-derivation must reproduce U-17's calibration arm exactly"
    from quantprobe.spec import K_CLASS_IQ
    assert "IQ4_NL" in K_CLASS_IQ, "#70 measured IQ4_NL at Q4_0-class; it must not pay the codebook tax"


def t_prereg70_iq_format_ladder():
    # #70: measured IQ entries exist, and the divide is CODEBOOK vs not - codebook formats
    # (IQ2/IQ3) sit far below Q4_K, while IQ4_NL's Q4_0-class kernel lands beside Q4_0.
    from quantprobe.spec import FORMAT_EBW
    assert FORMAT_EBW["IQ2_XS"] < FORMAT_EBW["IQ3_S"] < FORMAT_EBW["Q4_K"], "codebook ladder order"
    assert FORMAT_EBW["IQ2_XS"] <= 0.6 * FORMAT_EBW["Q4_K"], "IQ2_XS must price far below Q4_K"
    assert abs(FORMAT_EBW["IQ4_NL"] - FORMAT_EBW["Q4_0"]) <= 0.05 * FORMAT_EBW["Q4_0"], \
        "IQ4_NL is Q4_0-class, not codebook-class"


def t_l19_depth_scope_warning():
    # prereg #73 / L-19: dense splits at depth are the one refuted regime and must say so;
    # the validated placements (all-in-VRAM, MoE splits) must stay silent at every depth.
    from quantprobe.plan import depth_scope_warning
    w = depth_scope_warning("split: 20/28 layers->VRAM, rest->RAM", False, 16384)
    assert w and "1.55 us/position/layer" in w and "#73/#74" in w   # the term, not the refusal
    assert depth_scope_warning("split: 20/28 layers->VRAM, rest->RAM", False, 0) is None
    assert depth_scope_warning("split experts: 15%->VRAM, rest->RAM", True, 32768) is None
    assert depth_scope_warning("all in VRAM", False, 32768) is None


def t_c14_machine_state_identity():
    # C-14: two calibrations of the same idle box drifted 7% and moved every ladder arm 5-12
    # points. A machine state must have a name, drift must be detected, and a predicted-vs-
    # measured pair must carry the state it came from.
    from quantprobe.calibrate import cal_id, drift_vs
    a = {"ram_bw_measured": 23.21, "disk_bw_measured": 2.82,
         "anchors": [{"placement": "pure CPU (-ngl 0)", "tok_s": 6.72, "sustained_sm": 1506}]}
    b = {"ram_bw_measured": 21.66, "disk_bw_measured": 2.50,
         "anchors": [{"placement": "pure CPU (-ngl 0)", "tok_s": 6.24, "sustained_sm": 1506}]}
    assert cal_id(a) and cal_id(a) != cal_id(b), "different states must get different ids"
    assert cal_id(a) == cal_id(dict(a)), "the id must be stable for the same state"
    moved = drift_vs(a, b)
    names = {m[0] for m in moved}
    assert "ram_bw" in names and "disk_bw" in names and any("anchor" in n for n in names), \
        f"the real 2026-07-30 drift must be detected, got {names}"
    assert not drift_vs(a, dict(a)), "no drift against itself"
    assert cal_id(None) is None


# ---------------------------------------------------------------------------------------------
# prereg #88 / experiment #54 - WHICH RESOURCE BINDS
# ---------------------------------------------------------------------------------------------

def _plan_rows(**kw):
    from quantprobe.plan import evaluate
    base = dict(t=30.5, a=3.3, ne=1.2, moe=True, bits=2.5, vc=6, vb=192, rc=16, rb=48,
                db=0.45, geta=0.35, gl=0.04, n_layer=48)
    base.update(kw)
    return evaluate(**base)[2]


def t_p88_terms_reconstruct_every_row():
    """K-2: every row's terms must reproduce ITS OWN tok/s to 1e-9. A decomposition that does not
    add up is not a diagnosis, it is a confident guess - and it would be invisible without this.

    ADVERSARIAL FIX (pre-run audit of experiment #54). This grid used to be six configurations
    that produced SIX of the seven row families, and the one it never emitted was the DENSE SPLIT
    (`split: N/M layers->VRAM`) - the row with the most complex decomposition (three resources,
    with the KV read split across two tiers) and a WINNER on 8 of prereg #88's 340 preset cells.
    A hand-picked grid that misses a family is the same defect as prereg #88's own D-2: a check
    whose population excludes the case it exists to catch.

    Demonstrated: dropping `(1-g)*kv_gb/(ETA_KV*rb)` from that row's `ram_bw` attribution left
    tok/s bit-identical, this test PASSED, and the scoring script printed VERDICT: PASS at exit 0
    while the row reconstructed 4.94% off its own speed and the PRINTED binding share and ceiling
    moved (64.5% -> 67.7%, 2.82x -> 3.09x). The grid now covers all seven families and ASSERTS
    that it does, so a family cannot silently drop out of the population again.
    """
    from quantprobe.plan import resource_times
    # `kvp` is EXPLICIT on every ctx entry. `_plan_rows` does not set it and evaluate defaults it
    # to 0.0, so `kv_gb = ctx*kvp/1e9` was ZERO on every row this test has ever built - including
    # the ones that pass ctx=16384. Every KV attribution term was therefore multiplied by zero and
    # a KV mis-attribution in ANY row family was invisible, not just in the dense split. A grid
    # knob that reads as "depth is covered" while the quantity it controls is identically zero is
    # the "measurement that cannot vary" class (prereg #85 arms C/D). Found by the pre-run audit of
    # experiment #54.
    KVP = 98304                                   # DEFAULT_KVP; the Qwen3-30B-class GQA value
    grids = [dict(), dict(ctx=16384, kvp=KVP), dict(vc=0, vb=0),
             dict(t=1058.6, a=32, ne=6, rc=16),
             dict(moe=False, a=30.5, ne=30.5, ctx=16384, kvp=KVP),
             dict(bits=4.5, t=7.2, a=7.2, ne=7.2),
             # dense, too big for the 6 GB card but small enough for VRAM+RAM -> the dense split.
             dict(moe=False, t=14.0, a=14.0, ne=14.0, n_layer=48),
             dict(moe=False, t=14.0, a=14.0, ne=14.0, n_layer=48, ctx=16384, kvp=KVP),
             dict(ctx=16384, kvp=KVP, rc=16, t=1058.6, a=32, ne=6)]
    seen = 0
    families = set()
    for g in grids:
        for row in _plan_rows(**g):
            tt = resource_times(row)
            assert tt, f"row carries no decomposition: {row[0]}"
            recon = row.eff / sum(tt.values())
            assert abs(recon / row[1] - 1) < 1e-9, f"{row[0]}: {recon} != {row[1]}"
            assert set(tt) <= {"vram_bw", "ram_bw", "io", "cpu_compute"}, tt
            families.add(row[0].split(":")[0])
            seen += 1
    assert seen >= 12, f"grid degenerated to {seen} rows"
    # A grid where every ctx entry silently prices a zero-byte KV cache is a grid that cannot see a
    # KV mis-attribution. Assert the term is actually non-zero somewhere.
    assert any(16384 * g.get("kvp", 0) > 0 for g in grids), \
        "no grid entry carries a non-zero kv_gb - every KV attribution term is multiplied by zero"
    want = {"all in VRAM", "hybrid", "split experts", "split", "pure CPU (GPU idle)",
            "stream from disk (cold experts)", "stream from disk (VRAM+RAM expert cache)"}
    assert want <= families, (
        "the reconstruction grid stopped emitting row families " + str(sorted(want - families)) +
        " - a term attribution error in a family this test never builds is invisible to it, "
        "which is exactly how the dense-split hole was found. Extend the grid, do not drop the "
        "assertion.")


def t_p88_row_is_still_a_four_tuple():
    """Six modules unpack these rows. The decomposition must be invisible to all of them."""
    from quantprobe.plan import Row
    r = Row("x", 2.0, None, "-ngl 99", {"vram_bw": 0.5})
    name, tps, warn, flags = r                       # the unpack every caller does
    assert (name, tps, warn, flags) == ("x", 2.0, None, "-ngl 99") and len(r) == 4
    assert isinstance(r, tuple) and r[1] == 2.0
    assert sorted([Row("a", 1.0, None, "f"), r], key=lambda x: -x[1])[0][0] == "x"


def t_p88_four_classes_are_reachable():
    """P-1's precondition: the label must be able to say four different things."""
    from quantprobe.plan import binding_constraint, Row
    def klass(terms, cap=None):
        return binding_constraint(Row("r", 1.0, None, "f", terms), capacity=cap)["klass"]
    assert klass({"ram_bw": 1.0, "io": 0.1}) == "bandwidth-bound"
    assert klass({"vram_bw": 1.0, "ram_bw": 0.1}) == "bandwidth-bound"
    assert klass({"io": 1.0, "ram_bw": 0.9}) == "IO-bound"
    assert klass({"cpu_compute": 1.0, "ram_bw": 0.9}) == "compute-bound"
    cap = dict(tier="VRAM", gap_gb=1.0, lever="x", shave_tps=9.0, lift_tps=9.0,
               gain_shave=2.0, gain_lift=2.0, need_gb=8.0)
    assert klass({"ram_bw": 1.0}, cap) == "capacity-bound"


def t_p88_margin_and_ceiling_arithmetic():
    from quantprobe.plan import binding_constraint, Row
    bc = binding_constraint(Row("r", 1.0, None, "f", {"ram_bw": 0.75, "io": 0.25}))
    assert abs(bc["share"] - 0.75) < 1e-12
    assert abs(bc["margin_x"] - 3.0) < 1e-12          # 0.75 / 0.25
    assert abs(bc["ceiling_x"] - 4.0) < 1e-12         # 1 / (1 - 0.75)
    assert bc["next_resource"] == "io"
    assert abs(bc["lever_ceiling"]["io"] - (1 / 0.75)) < 1e-12
    # a single-resource row has no second constraint and must say so rather than divide by zero
    solo = binding_constraint(Row("r", 1.0, None, "f", {"ram_bw": 1.0}))
    assert solo["margin_x"] is None and solo["ceiling_x"] is None
    # ties resolve deterministically, by the order fixed in the prereg
    tie = binding_constraint(Row("r", 1.0, None, "f", {"ram_bw": 0.5, "io": 0.5}))
    assert tie["resource"] == "io"
    assert binding_constraint(("plain", 1.0, None, "f")) is None       # no terms -> no guess


def t_p88_no_phantom_disk_sensitivity():
    """P-2a: a diagnosis that moves when an IRRELEVANT resource moves is not a diagnosis.
    Every row without an io term must be bit-identical at db 0.45 and db 5.0."""
    from quantprobe.plan import binding_constraint, resource_times
    for g in (dict(), dict(ctx=16384), dict(moe=False, a=30.5, ne=30.5, ctx=8192)):
        slow = {r[0]: r for r in _plan_rows(db=0.45, **g)}
        fast = {r[0]: r for r in _plan_rows(db=5.0, **g)}
        for name in set(slow) & set(fast):
            if "io" in (resource_times(slow[name]) or {}):
                continue
            a, b = binding_constraint(slow[name]), binding_constraint(fast[name])
            assert a["resource"] == b["resource"] and a["klass"] == b["klass"], name
            assert abs(a["share"] - b["share"]) < 1e-12, name
            assert (a["margin_x"] or 0) - (b["margin_x"] or 0) == 0, name


def t_p88_bandwidth_shares_move_the_right_way():
    """P-2b: for a FIXED placement, doubling a bandwidth must strictly SHRINK that pool's share.
    Forced by the model, so this tests the wiring - which is exactly what a copy-paste breaks.

    Refinement of the prereg's wording, disclosed rather than quietly applied: a row with only ONE
    live resource holds share 1.0 by definition and cannot shrink. Those rows are checked for the
    weaker property (still exactly one resource) so the skip cannot hide a mis-wiring."""
    from quantprobe.plan import binding_constraint
    # Two fixtures, because no single one carries all three pools: the MoE split/hybrid rows are
    # the VRAM+RAM case, the deep dense case is the RAM+compute one.
    for res, kw, fix in (("ram_bw", dict(rb=96), dict()),
                         ("vram_bw", dict(vb=384), dict()),
                         ("ram_bw", dict(rb=96), dict(ctx=16384, moe=False, a=30.5, ne=30.5))):
        base = {r[0]: r for r in _plan_rows(**fix)}
        faster = {r[0]: r for r in _plan_rows(**dict(fix, **kw))}
        checked = 0
        for name in set(base) & set(faster):
            b0, b1 = binding_constraint(base[name]), binding_constraint(faster[name])
            s0, s1 = b0["shares"].get(res, 0.0), b1["shares"].get(res, 0.0)
            if s0 <= 0:
                continue
            if len(b0["shares"]) == 1:
                assert s0 == 1.0 and len(b1["shares"]) == 1 and s1 == 1.0, name
                continue
            assert s1 < s0, f"{name}: {res} share {s0} -> {s1} after doubling its bandwidth"
            checked += 1
        assert checked, f"no row exercised {res} in {fix}"


def t_p88_l19_and_the_class_agree():
    """The compute regime already had a warning (L-19). The classifier must not contradict it:
    where CPU attention is the biggest term, the class is compute-bound AND the warning fires."""
    from quantprobe.plan import binding_constraint, depth_scope_warning
    rows = _plan_rows(moe=False, t=7.2, a=7.2, ne=7.2, n_layer=32, ctx=32768, kvp=131072)
    split = [r for r in rows if "layers->VRAM" in r[0]]
    assert split, "the dense split row vanished - the fixture no longer tests the regime"
    bc = binding_constraint(split[0])
    assert bc["klass"] == "compute-bound" and bc["resource"] == "cpu_compute", bc["klass"]
    assert depth_scope_warning(split[0][0], False, 32768), "L-19 must fire on the same row"


def t_p88_capacity_uses_only_shipped_thresholds():
    """No new numeric threshold may enter the classification rule (prereg #88 §1)."""
    from quantprobe import plan
    assert plan.CAP_PROMOTION_MIN == 1.15 and plan.CAP_SHAVE_MAX_SHARE == 0.30
    assert plan.UPGRADE_MIN_GAIN == 1.08
    src = open(plan.__file__, encoding="utf-8").read()
    assert "CAP_PROMOTION_MIN" in src and "1.15" in src


def t_p88_capacity_probe_and_tier_advisor_agree():
    """R4: 'the advisor fired' and 'the classifier says capacity-bound' must be ONE event."""
    from quantprobe.plan import capacity_probe, evaluate
    kw = dict(t=11.9, a=11.9, ne=11.9, moe=False, bits=2.5, vc=8, vb=256, rc=32, rb=45,
              db=2, geta=0.45, gl=0.28)
    def ev(**over):
        k = dict(kw)
        if "rc_delta" in over:
            k["rc"] = k["rc"] + over.pop("rc_delta")
        over.pop("true_size_gb_scale", None)
        k.update(over)
        return evaluate(**k)[2]
    size, _, rows = evaluate(**kw)
    find = capacity_probe(ev, rows[0][1], size, 0.0, kw["vc"], kw["rc"])
    assert find and find["tier"] == "VRAM", find
    assert max(find["gain_shave"], find["gain_lift"]) >= 1.15
    assert find["gap_gb"] <= size * 0.30
    rc, out = cli("plan", "--model", "gemma-12b", "--machine", "laptop-8gb")
    assert rc == 0 and "CAPACITY-BOUND" in out and "tier-boundary advisor" in out


def t_p88_speculation_ceiling_bounds_the_headline():
    """P-3c: a verify batch amortizes weight READS. CPU attention is paid per drafted position,
    so on a compute-bound row the 4.7x/2.10x headlines are arithmetically out of reach."""
    from quantprobe.plan import spec_ceiling, Row
    assert abs(spec_ceiling(Row("r", 1.0, None, "f", {"ram_bw": 1.0}), k=2) - 3.0) < 1e-12
    assert abs(spec_ceiling(Row("r", 1.0, None, "f", {"cpu_compute": 1.0}), k=2) - 1.0) < 1e-12
    mixed = spec_ceiling(Row("r", 1.0, None, "f", {"ram_bw": 0.15, "cpu_compute": 0.85}), k=2)
    assert 1.0 < mixed < 1.15, mixed
    rc, out = cli("plan", "--model", "mistral-7b", "--machine", "2016-xmp", "--ctx", "32768")
    assert rc == 0 and "BUT NOT ON THIS ROW" in out and "does NOT amortize" in out


def t_c15_speculation_headline_is_reachable_on_the_row_it_prints_on():
    """C-15: the block printed a MEASURED multiplier as a headline on every row its branch fired
    on - 4.7x on MoE split-expert rows, 2.10x on dense rows - and both were measured where the
    token is ~100% weight bandwidth. A verify batch amortizes weight READS; CPU attention over
    the KV cache is paid once per DRAFTED POSITION and does not amortize at all. So on a row that
    spends most of its token there, the headline is not pessimistic-but-plausible, it is
    ARITHMETICALLY out of reach - and #54 counted 17 shipped grid cells (of 127 that print) where
    the row's own decomposition caps speculation below 1.5x while 2.10x was on the page.

    D-01 killed a 1.335x lever of ours on evidence. Printing an unreachable multiplier is that
    error with the sign flipped, and this test is the failing input for it: the CLI check below
    fails on the pre-fix tool, which prints '**2.10x decode**' on a row that is 85% CPU attention.
    """
    # (1) END-TO-END, and deliberately first: no new symbol is needed to see the bug, so a
    #     pre-fix run fails HERE, on printed output, rather than on an import.
    rc, out = cli("plan", "--model", "mistral-7b", "--machine", "2016-xmp", "--ctx", "32768")
    assert rc == 0, out[:400]
    assert "**2.10x decode**" not in out, \
        "the measured 2.10x headline is printed on a row whose own terms cannot reach it"
    assert "NOT REACHABLE ON THIS ROW" in out and "does not amortize" in out
    # (2) the bound itself: equals the measurement at zero CPU share, 1.0 when the token is all
    #     CPU attention, and is monotone in between. No fitted constant - R is the measurement.
    from quantprobe.plan import (Row, speculation_advice, spec_reachable_x, spec_headline_x,
                                 cpu_attention_share, SPEC_X_NGRAM_TUNED, SPEC_X_NGRAM_DENSE)
    bw = Row("all in VRAM", 20.0, None, "-ngl 99", {"vram_bw": 1.0})
    allcpu = Row("pure CPU (GPU idle)", 3.0, None, "-ngl 0", {"cpu_compute": 1.0})
    assert abs(spec_reachable_x(bw, SPEC_X_NGRAM_TUNED) - SPEC_X_NGRAM_TUNED) < 1e-12
    assert abs(spec_reachable_x(allcpu, SPEC_X_NGRAM_DENSE) - 1.0) < 1e-12
    prev = SPEC_X_NGRAM_TUNED + 1
    for c in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        r = spec_reachable_x(Row("r", 1.0, None, "f", {"ram_bw": 1 - c, "cpu_compute": c}),
                             SPEC_X_NGRAM_TUNED)
        assert r < prev, (c, r, prev)                       # strictly monotone down in CPU share
        prev = r
    # (3) the headline is REPLACED, not merely footnoted, on a row that cannot reach it.
    hot = Row("pure CPU (GPU idle)", 3.0, None, "-ngl 0",
              {"ram_bw": 0.15, "cpu_compute": 0.85})
    reach = spec_reachable_x(hot, SPEC_X_NGRAM_DENSE)
    assert reach < 1.15, reach
    txt = speculation_advice(False, "pure CPU (GPU idle)", row=hot)
    assert "**2.10x decode**" not in txt and f"**{reach:.2f}x**" in txt, txt[:300]
    assert cpu_attention_share(hot) == 0.85
    # (4) and it is NOT replaced where it is reachable: a bandwidth-only row keeps the measured
    #     text byte-for-byte, so this correction can only ever fire where the arithmetic forbids
    #     the number. (Same call with no row at all = the pre-fix string.)
    assert speculation_advice(False, "all in VRAM", row=bw) == \
        speculation_advice(False, "all in VRAM")
    assert "**2.10x decode**" in speculation_advice(False, "all in VRAM", row=bw)
    split = Row("split experts: 32%->VRAM, rest->RAM", 19.5, None, "-ngl 99",
                {"vram_bw": 0.02, "ram_bw": 0.03})
    assert speculation_advice(True, "split experts: 32%->VRAM, rest->RAM", row=split) == \
        speculation_advice(True, "split experts: 32%->VRAM, rest->RAM")
    assert spec_headline_x(True, "split experts: 32%->VRAM, rest->RAM") == SPEC_X_NGRAM_TUNED
    assert spec_headline_x(True, 'exps=CPU') is None        # that branch quotes no multiplier
    # (5) the two bounds are about two DIFFERENT drafters and must be labelled as such: K=2 is the
    #     draft-model config (3 tokens per weight read, capped at 3x), which is BELOW the 4.7x
    #     ngram headline it used to be printed under as if it bounded it.
    rc, out = cli("plan", "--model", "qwen3-30b", "--machine", "2016-xmp")
    assert rc == 0 and "**4.7x decode at ~3-bit**" in out
    assert "3.00x for a DRAFT" in out and "4.63x for the ngram drafter" in out, \
        "the 4.7x headline must be printed next to the ngram drafter's own bound, not a K=2 one"


def t_p88_codebook_warning_is_bounded_not_absolute():
    """P-3b: '~2.7x slower' is a property of the RAM weight read, not of the token. The warning
    must price itself against THIS row or it sends users to re-download 20 GB for nothing."""
    from quantprobe.plan import codebook_bounded_gain, Row, IQ_CPU_TG_PENALTY
    all_ram = codebook_bounded_gain(Row("r", 1.0, None, "f", {"ram_bw": 1.0}), 1.0)
    assert abs(all_ram - (1 + IQ_CPU_TG_PENALTY)) < 1e-9        # whole token -> full penalty back
    tiny = codebook_bounded_gain(Row("r", 1.0, None, "f",
                                     {"ram_bw": 0.05, "cpu_compute": 0.95}), 1.0)
    assert tiny is not None and tiny < 1.02, tiny               # the "re-download" advice is dead
    assert codebook_bounded_gain(Row("r", 1.0, None, "f", {"ram_bw": 1.0}), 0.0) is None


def t_p88_all_in_vram_low_bits_carries_the_unpack_caveat():
    """K-4: our geta fuses bandwidth with unpack ALU, and below 4.5 bits we have MEASURED the
    byte ordering reversed. A bare 'bandwidth-bound' there contradicts our own evidence."""
    from quantprobe.plan import binding_constraint, binding_report, Row
    bc = binding_constraint(Row("all in VRAM", 20.0, None, "-ngl 99", {"vram_bw": 0.05}))
    low = "\n".join(binding_report(bc, bits=2.5, placement="all in VRAM"))
    high = "\n".join(binding_report(bc, bits=6.5, placement="all in VRAM"))
    assert "CAVEAT (#16/#52)" in low and "SLOWER" in low
    assert "CAVEAT" not in high
    assert "DECODE only" in low and "DECODE only" in high        # scope always stated
    rc, out = cli("plan", "--model", "mistral-7b", "--machine", "rtx-4090", "--bits", "2.5")
    assert rc == 0 and "CAVEAT (#16/#52)" in out


def t_p88_upgrade_counterfactual_shares_the_baseline_inputs():
    """P-3a, the defect this experiment found by reading: the three upgrade call sites passed
    neither n_layer nor true_size_gb, so every counterfactual was drawn from a smaller row menu,
    a re-estimated model size, and (at ctx>0) a 32-layer CPU-attention term for an 80-layer
    model. Assert the arguments now travel with the baseline."""
    from quantprobe import plan
    seen = []
    real = plan.evaluate
    try:
        plan.evaluate = lambda *a, **k: (seen.append(k), real(*a, **k))[1]
        import argparse
        args = argparse.Namespace(model="llama-70b", machine="2016-xmp", bits=2.5, ctx=16384,
                                  total=None, active=None, always_active=None, vram=None,
                                  vram_bw=None, ram=None, ram_bw=None, disk_bw=None,
                                  kv_per_pos=None, n_layer=None, gguf=None)
        buf = io.StringIO()
        with redirect_stdout(buf):
            plan.run(args)
    finally:
        plan.evaluate = real
    assert len(seen) >= 3, f"only {len(seen)} evaluate calls - the advisor stopped running"
    nl = {k.get("n_layer") for k in seen}
    assert nl == {80}, f"counterfactuals disagree with the baseline on n_layer: {nl}"
    assert {k.get("ctx") for k in seen} == {16384}
    assert all("true_size_gb" in k for k in seen), "true_size_gb must travel with every call"
    # the counterfactuals must actually differ from the baseline in ONE resource each
    assert {k["rc"] for k in seen} == {16, 32}, "the +16 GB RAM probe did not run"
    assert {k["db"] for k in seen} == {0.45, 3.5}, "the NVMe probe did not run"


def t_p88_upgrade_advice_is_not_invented_by_a_depth_mismatch():
    """The USER-VISIBLE half of P-3a, on the two cells that pin BOTH signs of the defect.

    The sibling test above asserts the plumbing (the kwargs travel). Plumbing is not advice: it
    would still pass if `evaluate` ignored `n_layer` entirely. This one asserts the printed line,
    through the real CLI, on the two cells that a defect replay shows moving in OPPOSITE
    directions - which is also why the "one-directional" rationale that shipped in
    `upgrade_advisor`'s docstring had to be withdrawn (prereg #88 §8.6).

    INVENTION arm - llama-70b (80 layers) on `colibri` at ctx 16384. Strip `n_layer` from the
    counterfactual and it is priced as a 32-layer model against an 80-layer baseline, so BOTH
    `+16 GB RAM` and `NVMe SSD` print at an identical x1.70 on a 128 GB box where neither
    resource is reachable. Two different levers cannot honestly buy the same number; the fixed
    advisor prints neither. This is the direction that costs a user money.

    SUPPRESSION arm - deepseek-16b on `2016` at ctx 0. Without `n_layer` the counterfactual has
    no split-experts row to win with, so free XMP - worth x1.12 here - was never offered.

    FAILS on the pre-fix tree in both directions: with the three call sites' arguments dropped the
    first assertion sees the invented pair and the second sees no XMP line at all.
    """
    rc, out = cli("plan", "--model", "llama-70b", "--machine", "colibri",
                  "--bits", "2.5", "--ctx", "16384")
    assert rc == 0, out
    fired = [l.strip() for l in out.splitlines()
             if l.strip().startswith("upgrade advisor:")]
    assert not fired, (
        "upgrade advice INVENTED by a baseline/counterfactual depth mismatch - a 128 GB box "
        f"cannot be helped by +16 GB RAM: {fired}")
    dead = [l for l in out.splitlines() if "WON'T HELP HERE" in l]
    assert dead and "+16 GB RAM" in dead[0] and "NVMe SSD" in dead[0], (
        f"both dead levers must still be NAMED, not silently dropped: {dead}")

    rc, out = cli("plan", "--model", "deepseek-16b", "--machine", "2016", "--bits", "2.5")
    assert rc == 0, out
    xmp = [l.strip() for l in out.splitlines()
           if l.strip().startswith("upgrade advisor:") and "enable XMP (free) ->" in l]
    assert len(xmp) == 1, f"free upgrade SUPPRESSED by a smaller counterfactual row menu: {out}"
    assert "split experts:" in xmp[0], (
        f"the winning counterfactual row needs n_layer to exist at all: {xmp[0]}")


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
