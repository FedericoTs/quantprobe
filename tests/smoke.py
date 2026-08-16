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

FAIL, SKIP = [], []


def check(name, fn):
    """A SKIP IS NOT A PASS - verify.py layer 3 has enforced that since v1.12, and this harness
    did not. A test that returned a "skipped (...)" string printed `ok` and was counted in the
    "N tests green" line, which is how t_c17_disk_probe_is_not_page_cache_contaminated - the
    regression test for a disk number that shipped 6.8x too fast - read as green while measuring
    nothing. Skips are now printed as skips and listed at the end. Return the string SKIP:<why>."""
    try:
        r = fn()
        if isinstance(r, str) and r.startswith("SKIP:"):
            SKIP.append((name, r[5:].strip()))
            print(f"  SKIP  {name}: {r[5:].strip()}")
        else:
            print(f"  ok    {name}")
    except Exception as e:
        FAIL.append((name, e))
        print(f"  FAIL  {name}: {e}")


def cli(*args):
    r = subprocess.run([sys.executable, "-m", "quantprobe.cli"] + list(args),
                       capture_output=True, text=True, errors="replace")
    return r.returncode, r.stdout + r.stderr


def hf_unreachable(out):
    """True when the Hugging Face listing failed for reasons that are not our code.

    Five `auto` tests need a live HF tree API. When it is rate-limited or down they were
    failing HARD, reporting quantprobe as broken because someone else's server said 429 -
    which is how commit c3c1982 came back green and red six seconds apart on 2026-08-03.

    DELIBERATELY NARROW. It matches ONLY the message `auto` raises when the listing itself
    failed ("could not list <repo>: ..."), plus explicit transport errors. A traceback, a wrong
    pick, a bad prediction or any other non-zero exit is still a FAILURE - this must never
    become a way for real breakage to go quiet, which is the obvious way a helper like this
    rots. Skips are surfaced by the runner and by verify.py layer 1, so a permanently
    unreachable HF shows up as a suite that stopped checking, not as a green one.
    """
    if "could not list" in out:
        return True
    return any(sig in out for sig in ("HTTP Error 429", "HTTP Error 5", "URLError",
                                      "Temporary failure in name resolution",
                                      "Connection reset", "timed out"))


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


def t_l26_cpu_expert_moe_prefill_gets_ub4096():
    """L-26 (prereg #92): the CPU-expert MoE placement must emit -b 4096 -ub 4096.

    The prefill law sec/token = A + X/C was validated OUT OF SAMPLE on this exact placement
    (Qwen3-30B-A3B, exps=CPU, this box) with predictions staked before running: C=256 -0.27%,
    C=4096 -8.2%, both inside the 10% kill band - and ub 4096 is a BANKED number, 360.76 pp2048
    vs 345.89 at ub 2048 (+4.3% for one flag). Before this wiring the tool sized the same
    placement to ub 2048 on the reference card: the half-budget margin (calibrated on the SPLIT,
    #23, where resident experts compete with the compute buffer) rejected by 15 MiB a buffer
    L-26 measured running clean at 51.5% of headroom.

    Also pins the cap: the fit's asymptote bends below 1/A, so extrapolation past C=4096 is not
    licensed and no headroom, however large, may raise the batch further.

    FAILS on the pre-fix tree (verified by reverting quantprobe/plan.py in a scratch copy):
    the reference-box hybrid geometry returned '-b 2048 -ub 2048' and the plan output quoted
    the pre-L-26 hardcoded '-b 2048 -ub 2048' for the long-prompt alternative.
    """
    from quantprobe.plan import ubatch_flags, MODELS
    # the reference-box geometry L-26 was measured on: qwen3-30b hybrid, attention resident
    res = MODELS["qwen3-30b"]["ne"] * 4.5 / 8 * 1.08
    fl = ubatch_flags("hybrid: attention->VRAM, experts->RAM", res, 6)
    assert fl == "-b 4096 -ub 4096", (
        f"CPU-expert MoE placement must carry the measured -b 4096 -ub 4096 (L-26): {fl}")
    # the cap is a law boundary, not a budget artifact: a 48 GB card gets the SAME 4096
    big = ubatch_flags("hybrid: attention->VRAM, experts->RAM", res, 48)
    assert big == "-b 4096 -ub 4096", (
        f"extrapolation past C=4096 is not licensed (asymptote bends below 1/A): {big}")
    # a tight card still steps DOWN - the flag is sized, never pinned
    tight = ubatch_flags("hybrid: attention->VRAM, experts->RAM", 6 * 0.9 - 1.6, 6)
    assert tight and "4096" not in tight, f"tight card must not be handed the 4096 buffer: {tight}"
    # and the flag reaches the USER: the long-prompt alternative in plan's own output
    rc, out = cli("plan", "--model", "qwen3-30b", "--bits", "2.95", "--machine", "2016-xmp")
    assert rc == 0, out
    assert "-b 4096 -ub 4096" in out, (
        "plan output no longer offers -b 4096 -ub 4096 for the CPU-expert MoE placement")
    # The flags must be quoted WITH their measurement AND an unambiguous citation. "#92" is not
    # one: two different pre-registrations claimed that number, and this run is neither of them -
    # its predictions were staked in the header of its own log, which is what we cite.
    assert "360.76" in out, "the L-26 flags lost their out-of-sample measurement"
    assert "prereg92b_ub_oos.log" in out, "the staked-before-the-run evidence is not cited"
    # ...and the METRIC must be the one the log records. 360.76 is pp4096; the 345.89 baseline is
    # pp2048 at ub2048 (prereg #19). The tool said "360.76 pp2048" for a run that never measured
    # pp2048, and called the gap between two prompt lengths "+4.3% for one flag".
    assert "pp4096" in out, "360.76 is a pp4096 measurement and must be labelled as one"
    assert "360.76 pp2048" not in out, "the ub-4096 arm is mislabelled as a pp2048 measurement"
    assert "control was never run" in out, \
        "the missing like-for-like pp4096-at-ub2048 control must be disclosed"


def t_l26_dense_rows_never_get_ub4096():
    """L-26 scope: dense rows must NOT get -b 4096 -ub 4096 - the dense control VIOLATES the form.

    Prereg #19 P-2 measured the dense-in-VRAM control COLLAPSING at ub 2048 (-39%), so the
    prefill law that licenses 4096 is scoped to the CPU-expert MoE tier only. Before this gate a
    dense layer-split with room to spare was quietly handed -ub 4096 on the authority of a MoE
    datapoint (the 3090/117B external replication) - the locked ladder's Qwen2.5-14B row shipped
    with it. Dense host-resident rows keep the sized ubatch at the 2048 cap the half-budget rule
    was measured with; the fully-in-VRAM row keeps getting nothing at all.

    FAILS on the pre-fix tree (verified by reverting quantprobe/plan.py in a scratch copy): the
    dense layer-split at 6 GB and 24 GB headroom both returned '-b 4096 -ub 4096', and the deep
    dense-split plan command carried -ub 4096.
    """
    from quantprobe.plan import ubatch_flags
    # the measured -39% control: fully in VRAM gets NO batch flag, ever
    assert ubatch_flags("all in VRAM", 4.36, 6) is None, \
        "ubatch offered for an all-in-VRAM placement - measured there it LOSES 39% (prereg #19 P-2)"
    # dense layer-splits are host-resident and keep a SIZED ubatch - but never the MoE-only 4096
    for vc in (6, 24):
        fl = ubatch_flags("split: 21/48 layers->VRAM, rest->RAM", 0.0, vc)
        assert fl and "4096" not in fl, (
            f"dense split at vc={vc} got a batch the dense control measured collapsing: {fl}")
    # pure CPU and disk rows: same scope boundary
    for placement in ("pure CPU (GPU idle)", "stream from disk (cold experts)"):
        fl = ubatch_flags(placement, 0.0, 24)
        assert fl and "4096" not in fl, f"{placement} exceeded the L-26 scope: {fl}"
    # and the command a dense-split user is actually handed is clean of it
    rc, out = cli("plan", "--model", "mistral-7b", "--machine", "2016-xmp", "--bits", "4.5",
                  "--ctx", "16384")
    assert rc == 0, out
    runline = next(l for l in out.splitlines() if "run it:" in l)
    assert "-ub 4096" not in runline, f"dense split run command carries the MoE-only flag: {runline}"
    assert "-b 2048 -ub 2048" in runline, f"dense split lost its sized ubatch entirely: {runline}"


def t_l26_ub_prose_claims_track_the_measured_tier():
    """The +73% prefill claim may be printed as a property of THIS placement only on the
    experts->RAM tier it was measured on (-ot exps=CPU, prereg #19). Caught on the v1.23
    validation pass, live in the tool's own output: the winning split-experts row claimed
    '+73% on this placement' for its sized command two paragraphs after the tool itself printed
    that placement's measured 161.9 pp2048, and a dense split (and a disk-stream row) were told
    'your experts sit in RAM' - one about a model with no experts, the other about experts it
    streams from disk. Same boundary as the emission gate (L-26): a measured claim travels with
    the tier that measured it; every other row gets the sized-safe-batch wording that names
    where the numbers come from and why they do not transfer."""
    # split-experts winner (reference box): sized flags, no borrowed measured claim
    rc, out = cli("plan", "--model", "qwen3-30b", "--bits", "2.95", "--machine", "2016-xmp")
    assert rc == 0, out
    assert "sized safe batch" in out, "split-experts row lost its honest sized-batch wording"
    assert "worth **+73% prefill** on this placement" not in out, \
        "the hybrid tier's +73% claim leaked onto the split-experts row again"
    # dense split: no experts, so no expert-mechanism story at all
    rc, out = cli("plan", "--model", "mistral-7b", "--machine", "2016-xmp", "--bits", "4.5",
                  "--ctx", "16384")
    assert rc == 0, out
    assert "experts sit in RAM" not in out, \
        "dense split told 'your experts sit in RAM' about a model with no experts"
    assert "sized safe batch" in out, "dense split lost the sized-batch wording"
    # the measured tier KEEPS its measured claim - but the CELL must match the emitted ubatch.
    # prereg #19 swept ub 512/1024/2048 = 199.90/277.17/345.89 pp2048, and "+73%" is the 2048
    # cell. A 3 GB card is sized to -ub 1024, where the same sweep measured +38.7%; the tool
    # printed +73% there anyway - the placement-level leak repeated between cells of one sweep.
    from quantprobe.plan import UB_SWEEP_PP2048
    rc, out = cli("plan", "--total", "30.5", "--active", "3.3", "--always-active", "1.2",
                  "--bits", "2.95", "--vram", "3", "--vram-bw", "192", "--ram", "32",
                  "--ram-bw", "48", "--disk-bw", "2")
    assert rc == 0, out
    line = next(l for l in out.splitlines() if "prompt speed:" in l)
    assert "-ub 1024" in line, f"fixture drifted, expected the ub-1024 cell: {line}"
    assert "+39% prefill" in line and "277.2" in line, \
        f"the ub-1024 command quotes a percentage measured at a different ubatch: {line}"
    assert "+73%" not in line, "the ub-2048 cell's headline is printed on a ub-1024 command"
    # ...and the 2048 cell keeps +73% where 2048 is what we emit
    rc, out = cli("plan", "--total", "30.5", "--active", "3.3", "--always-active", "1.2",
                  "--bits", "2.95", "--vram", "5", "--vram-bw", "192", "--ram", "32",
                  "--ram-bw", "48", "--disk-bw", "2")
    assert rc == 0, out
    line = next(l for l in out.splitlines() if "prompt speed:" in l)
    assert "-ub 2048" in line and "+73% prefill" in line, line
    assert round(UB_SWEEP_PP2048[2048] / UB_SWEEP_PP2048[512] - 1, 2) == 0.73


def t_disk_tier_kv_deficit_disclosed_not_clamped():
    """When KV alone crowds RAM below the 1 GB expert-cache floor, the disk-stream row must SAY so.

    The row prices every KV read at RAM bandwidth - it ASSUMES host-RAM KV residency. Before this
    fix, a config whose KV cache did not fit (ra_eff clamped at the 1 GB floor) kept that pricing
    silently: the deficit was absorbed into a number that looked like a prediction. The fix is
    DISCLOSURE, not refusal and not repricing - no disk-tier anchor was ever measured with paging
    KV, so a repriced number would be invented. The row stays, its arithmetic stays (the eff/terms
    reconstruction check in t_binding_* keeps guaranteeing that), and the warn must name the
    shortfall in GB and the consequence (the printed tok/s is an upper bound).
    """
    from quantprobe.plan import evaluate
    # oversized on purpose: rc=8 -> ra=4 GB budget; ctx 65536 x kvp 98304 -> kv_gb ~ 6.44 GB.
    # KV + the 1 GB floor overshoot RAM by ~3.4 GB while the 100B file streams from disk.
    _, _, rows = evaluate(100, 10, 2, True, 2.5, 0, 0, 8, 40, 2, 0.5,
                          ctx=65536, kvp=98304.0, n_layer=48)
    disk = [r for r in rows if r[0] == "stream from disk (cold experts)"]
    assert disk, "oversized config lost its disk-stream row - the fix must disclose, never refuse"
    warn = disk[0][2] or ""
    assert "KV DEFICIT" in warn and "UPPER BOUND" in warn, \
        f"KV residency deficit silently absorbed again (the 1 GB ra_eff clamp): {warn!r}"
    kv_gb = 65536 * 98304.0 / 1e9
    assert f"{kv_gb + 1.0 - 4.0:.1f} GB" in warn, \
        f"disclosure must print the actual shortfall ({kv_gb + 1.0 - 4.0:.1f} GB): {warn!r}"
    # control: same machine, shallow ctx -> KV fits beside the floor, no deficit text
    _, _, rows2 = evaluate(100, 10, 2, True, 2.5, 0, 0, 8, 40, 2, 0.5,
                           ctx=2048, kvp=98304.0, n_layer=48)
    disk2 = [r for r in rows2 if r[0] == "stream from disk (cold experts)"]
    assert disk2 and "KV DEFICIT" not in (disk2[0][2] or ""), \
        f"deficit disclosed where there is none: {disk2[0][2]!r}"
    # and the disclosure must survive to the PRINTED row, not just the Row object
    rc, out = cli("plan", "--total", "100", "--active", "10", "--always-active", "2",
                  "--bits", "2.5", "--vram", "0", "--ram", "8", "--ram-bw", "40",
                  "--disk-bw", "2", "--ctx", "65536")
    assert rc == 0, out
    row_line = next((l for l in out.splitlines()
                     if "stream from disk (cold experts)" in l), "")
    assert "KV DEFICIT" in row_line and "UPPER BOUND" in row_line, \
        f"deficit disclosure did not reach the printed row: {row_line!r}"


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

def t_auto_gated_repo_sends_token_and_hints():
    """A gated HF repo must not dead-end `auto`. Two halves, both from a live failure (unsloth
    gated their Mistral repo and `auto mistral-7b` died on the LISTING with a bare 401): the
    tree listing now sends the same token `fetch` sends one step later (it was anonymous-only),
    and a tokenless 401/403 carries the gated-repo hint instead of a bare 'Unauthorized'."""
    import io, json as _json, urllib.request
    from unittest import mock
    from quantprobe import auto as automod
    seen = {}
    def fake_urlopen(req, timeout=None):
        seen["auth"] = req.get_header("Authorization")
        return io.BytesIO(_json.dumps([]).encode())
    with mock.patch.object(urllib.request, "urlopen", fake_urlopen), \
         mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}):
        files = automod.list_ggufs("some/gated-repo")
    assert files == [] and seen["auth"] == "Bearer hf_test_token", seen
    # the hint half: 401/403 get the actionable suffix, other errors stay bare
    assert "gated or private" in automod.gated_hint(Exception("HTTP Error 401: Unauthorized"))
    assert "HF_TOKEN" in automod.gated_hint(Exception("HTTP Error 403: Forbidden"))
    assert automod.gated_hint(Exception("HTTP Error 404: Not Found")) == ""


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
    if hf_unreachable(out):
        return "SKIP: Hugging Face listing unavailable (rate limit or outage) - not a quantprobe failure"

    assert rc == 0 and "doesn't need the surgery" in out and "closest file" in out, \
        f"custom gate broken: rc={rc} {out[:300]}"

def t_auto_force_custom():
    # --force-custom overrides the gate: the source pick must happen
    rc, out = cli("auto", "qwen3-30b", "--custom", "--force-custom", "--dry", "--vram", "24",
                  "--vram-bw", "936", "--ram", "64", "--ram-bw", "86", "--disk-bw", "3")
    if hf_unreachable(out):
        return "SKIP: Hugging Face listing unavailable (rate limit or outage) - not a quantprobe failure"

    assert rc == 0 and "source:" in out and "surgery" not in out, \
        f"force-custom broken: rc={rc} {out[:300]}"

def t_auto_wizard_dry():
    # no model argument -> interactive wizard: answers piped, --dry keeps it offline-light
    rc, out = cli_in("qwen3-30b\n1\nn\n", "auto", "--dry", "--machine", "2016-xmp")
    if hf_unreachable(out):
        return "SKIP: Hugging Face listing unavailable (rate limit or outage) - not a quantprobe failure"

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
    if hf_unreachable(out):
        return "SKIP: Hugging Face listing unavailable (rate limit or outage) - not a quantprobe failure"

    assert "Traceback" not in out and "GLM-5.2-GGUF" in out, f"744b preset broken: rc={rc} {out[:300]}"
    assert (rc == 0 and "closest file" in out) or "no ready-to-run quant" in out,         f"neither pick nor clean explanation: rc={rc} {out[:300]}"

def t_auto_bf16_only_graceful():
    # kimi-k2.6: today BF16-only upstream -> the >9-bit filter must yield the honest
    # explanation (with the --custom tip), not a bare error; when quants land, it just works
    rc, out = cli("auto", "kimi-k2.6", "--dry", "--vram", "96", "--vram-bw", "1800",
                  "--ram", "768", "--ram-bw", "300", "--disk-bw", "7")
    if hf_unreachable(out):
        return "SKIP: Hugging Face listing unavailable (rate limit or outage) - not a quantprobe failure"

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
    """THE 2x INPUT BUG from the first external replication: 4 DIMMs on consumer AM5 must be
    treated as DUAL channel, not 4-channel (173 GB/s quoted where the platform delivers ~86).

    This test used to RE-IMPLEMENT the wide-CPU check in its own body and assert on its own
    copy, so it never touched detect.py at all - mutation-verified: restoring the shipped bug
    `channels = max(1, min(sticks or 2, 8))` left it green. It now calls the shipping function.
    """
    import quantprobe.detect as d
    # consumer platform: stick count must NOT become channel count, at any stick count
    for sticks in (2, 4, 8):
        n, src = d.ram_channels(sticks, "AMD Ryzen 5 8600G w/ Radeon Graphics")
        assert n == 2, f"consumer AM5 with {sticks} sticks priced as {n}-channel: {src}"
    assert d.ram_channels(1, "AMD Ryzen 5 8600G")[0] == 1        # one stick is one channel
    assert d.ram_channels(None, "AMD Ryzen 5 8600G")[0] == 2     # unknown -> conservative 2
    assert "does NOT mean" in d.ram_channels(4, "AMD Ryzen 5 8600G")[1], "the trap is not named"
    # HEDT/server names keep their width
    for cpu in ("AMD Ryzen Threadripper 7970X", "AMD EPYC 9554", "Intel(R) Xeon(R) w9-3495X"):
        n, src = d.ram_channels(8, cpu)
        assert n == 8 and "HEDT" in src, (cpu, n, src)
    assert d.ram_channels(None, "AMD EPYC 9554")[0] == 4         # HEDT default, not 2
    # the bandwidth this feeds must move with it, not with the stick count
    assert round(d.ram_channels(4, "Ryzen 5 8600G")[0] * 5200 * 8 / 1000) == 83
    # and the real detect() on THIS box must not crash; the RAM note must disclose that the
    # DELIVERED stream is below peak and point at calibrate (no bare "typically ~55%" claim)
    _, notes = d.detect()
    ram_notes = [n for n in notes if n.startswith("RAM:")]
    assert ram_notes, notes
    assert "calibrate" in ram_notes[0] or "speed unknown" in ram_notes[0], ram_notes
    assert "typically" not in ram_notes[0], (
        "the stream-realism fraction is n=1 machine - it may not be quoted as a typical value")


def t_p88_binding_scope_line_names_the_placement():
    """The binding-constraint block's DECODE-only footnote used to illustrate itself with a bare
    '-ub 2048 can be worth +73% on prefill', printed on EVERY row it fires on - including the
    all-in-VRAM rows where prereg #19 P-2 measured the SAME flag at -39%. Same leak the v1.23
    validation pass caught in the 'prompt speed:' paragraph, one block higher and on every
    invocation. A measured number keeps its placement even when it is only making a point."""
    from quantprobe.plan import binding_report, binding_constraint, Row
    txt = "\n".join(binding_report(binding_constraint(
        Row("all in VRAM", 20.0, None, "-ngl 99", {"vram_bw": 1.0})), bits=4.5,
        placement="all in VRAM"))
    assert "+73%" in txt, "the illustration was deleted rather than scoped"
    assert "CPU-expert MoE" in txt, "the +73% is printed without the placement that measured it"
    assert "-39%" in txt, "the opposite-sign control on THIS row's own placement is not shown"
    assert "Neither number is a prediction for this row" in txt, txt
    # and on the CLI, where an all-in-VRAM user actually reads it
    rc, out = cli("plan", "--model", "mistral-7b", "--machine", "rtx-4090", "--bits", "4.5")
    assert rc == 0, out[:300]
    scope = next(l for l in out.splitlines() if "scope" in l and "DECODE only" in l)
    assert "CPU-expert MoE" in scope, scope


def t_c17_mixed_calibration_is_disclosed():
    """C-17 defect (2): partial calibration measured WORSE than none for the components it
    skipped (RAM-only 12.5% median |err| vs 8.8% fully-preset), yet plan printed the identical
    'calibration applied' line for a one-component state and a complete one. The gap must be
    named at the same prominence as what WAS measured."""
    from quantprobe.plan import calibration_gap_warning, CAL_COMPONENTS
    full = {k: 1 for _, k in CAL_COMPONENTS}
    assert calibration_gap_warning(full) == [], "a complete calibration must print no warning"
    for name, key in CAL_COMPONENTS:
        partial = {k: 1 for _, k in CAL_COMPONENTS if k != key}
        w = calibration_gap_warning(partial)
        assert len(w) == 1 and name in w[0], (key, w)
        assert "MIXED" in w[0] and "12.5%" in w[0] and "C-17" in w[0], w
    w = calibration_gap_warning({"ram_bw_measured": 24.4})
    assert "disk" in w[0] and "decode anchors" in w[0], w


def t_no_test_is_defined_after_the_runner():
    """The runner reads globals() at the point `if __name__ == '__main__'` executes, so anything
    defined BELOW it does not exist yet and never runs. That is not hypothetical: the C-17 disk
    regression test sat below the runner from the commit that introduced it and was never
    executed once. Nothing catches this - the suite just quietly gets smaller."""
    import re
    src = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke.py"),
                  encoding="utf-8").read()
    i = src.rindex('if __name__ == "__main__":')      # rindex: this test quotes the string too
    after = re.findall(r"^def (t_\w+)", src[i:], re.M)
    assert not after, (f"{len(after)} test(s) defined after the runner and therefore never "
                       f"collected: {after}")


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


def t_math500_scorer_reads_the_boxed_answer():
    # EV-1 protocol v3. The stock hendrycks_math500 slices the candidate answer as "everything
    # between the FIRST $ and the LAST $" of the response, so a model that reasons in LaTeX
    # scores 0 no matter what it answered (measured: 89.4% emitted a well-formed boxed answer,
    # task reported 0.00%). These cases pin the fix AND its direction - case 3 is a response
    # whose body is full of $...$ spans, exactly the shape the stock extractor mangles.
    import sys as _sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, os.path.join(root, "weights", "lm_eval_tasks"))
    try:
        import math500_utils as M
    except ImportError:
        # lm-evaluation-harness is a RESEARCH dependency, not a package dependency - CI installs
        # `pip install .` and has no reason to carry it. Skipped, never silently passed: this
        # guard is the reason our MATH rows are not 0.00%, so a green tick without lm_eval
        # present would be a lie about what was checked.
        return "SKIP: lm_eval not installed (research dep, not a package dep - runs locally)"

    def score(resp, gold):
        return M.process_results({"answer": gold}, [resp])

    r = score(r"The answer is $\boxed{42}$.", "42")
    assert r == {"exact_match": 1, "emitted_boxed": 1}, f"plain correct answer: {r}"

    r = score(r"So $\boxed{126}$.", "42")
    assert r == {"exact_match": 0, "emitted_boxed": 1}, f"wrong answer must be graded, not skipped: {r}"

    latex_heavy = (r"We have $r = \sqrt{0^2 + 3^2} = 3$ and $\theta = \arctan(3/0)$, "
                   r"so $\theta = \frac{\pi}{2}$. Final: $\boxed{(3, \frac{\pi}{2})}$")
    r = score(latex_heavy, r"\left( 3, \frac{\pi}{2} \right)")
    assert r["exact_match"] == 1, \
        "the stock 'first $ to last $' slice is back - LaTeX-heavy solutions score 0 again"

    r = score("The answer is 49.", "49")
    assert r == {"exact_match": 0, "emitted_boxed": 0}, \
        f"no boxed answer must read as UNGRADEABLE, not merely wrong: {r}"


def t_ev1_flags_route_to_the_right_tasks():
    # EV-1 v3. Every cell of the night, asserted without a GPU. The failures this catches are
    # SILENT ones - a stray "put your answer in \boxed{}" on IFEval does not crash, it just
    # competes with the instructions IFEval is grading and quietly lowers the score, which
    # then reads as a capability finding. Three rows were already spent on that class of bug.
    import sys as _sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, os.path.join(root, "weights"))
    import ev1_run as E

    TASKS = ("math500_boxed", "aime24_boxed", "aime25_boxed", "ifeval", "gsm8k_cot_zeroshot")
    for model in ("0.6B", "4B", "7B", "30B"):
        for task in TASKS:
            cmd = E.build_cmd(model, task)
            joined = " ".join(cmd)
            assert task in cmd, f"{model}/{task}: task missing from argv"
            assert "--apply_chat_template" in cmd, f"{model}/{task}: chat template not applied"
            assert E.TASK_PATH in cmd, f"{model}/{task}: --include_path missing (our task won't resolve)"
            assert "max_gen_toks" in joined, f"{model}/{task}: no generation budget"
            boxed = "--system_instruction" in cmd
            if task in ("math500_boxed", "aime24_boxed", "aime25_boxed"):
                assert boxed, f"{model}/{task}: boxed-graded task lost its answer-format instruction"
            else:
                assert not boxed, \
                    f"{model}/{task}: system instruction leaked onto a task that is NOT graded " \
                    f"by boxed extraction - on IFEval this silently competes with the rubric"

    # server flags: thinking off only for the family that has thinking to turn off
    assert E.server_extra("0.6B") == ("--reasoning", "off")
    assert E.server_extra("4B") == ("--reasoning", "off")
    assert E.server_extra("7B") == (), "7B is not a thinking model - flag would be rejected"
    assert E.server_extra("30B") == (), "30B is not a thinking model - flag would be rejected"

    # a probe must never write into the scored tree
    assert "_probe" in " ".join(E.build_cmd("0.6B", "aime24_boxed", tag="_probe"))
    assert E.out_dir("0.6B", "aime24_boxed") != E.out_dir("0.6B", "aime24_boxed", "_probe")

    # THE REAL length limit is the server slot, not max_gen_toks: one AIME item measured
    # 11,386 chars at max_gen_toks=4096 and 10,970 at 8192 - unchanged, because both were
    # stopped by ctx_per_slot=4096. Assert the knob that actually binds, and that widening it
    # keeps slots x ctx constant so VRAM pressure (and therefore the placement) does not move.
    base_ctx, base_conc = E.slot_plan("ifeval")
    for task in ("aime24_boxed", "aime25_boxed", "math500_boxed"):
        ctx, conc = E.slot_plan(task)
        assert ctx >= 8192, f"{task}: ctx_per_slot {ctx} truncates long reasoning"
        assert ctx * conc == base_ctx * base_conc, \
            f"{task}: total KV moved ({ctx}x{conc} vs {base_ctx}x{base_conc}) - changes the " \
            f"placement mid-suite and breaks one-machine-state comparability"
        assert f"num_concurrent={conc}" in " ".join(E.build_cmd("0.6B", task)), \
            f"{task}: harness concurrency must match the server's slot count"

    # A slot's context holds PROMPT + GENERATION. max_gen_toks == ctx_per_slot leaves zero room
    # for the prompt and the row WEDGES - 30B AIME24 ran 90 minutes, stopped advancing, and the
    # watchdog killed it at 102m. MATH-500 survived the identical slot plan only because its
    # budget left 5,120 tokens of headroom. Budgets must be DERIVED from the slot, never
    # declared beside it where the two can drift apart.
    for task in TASKS:
        ctx, _ = E.slot_plan(task)
        budget = int(E.GEN[task].split("=")[1])
        assert ctx - budget >= E.PROMPT_RESERVE, \
            f"{task}: budget {budget} of ctx {ctx} leaves {ctx-budget} tokens for the prompt " \
            f"(need >= {E.PROMPT_RESERVE}) - this is the wedge that killed 30B AIME24"

    # The HTTP timeout must cover the LONGEST budget at a 3 t/s floor. 600s bought ~4,900
    # tokens at the 30B's measured 8.2 t/s against AIME's 8,192-token budget - every long item
    # timed out, retried 3x, and a 166-minute row produced rc=1 and nothing else.
    import re as _re3
    for task in TASKS:
        joined = " ".join(E.build_cmd("30B", task))
        budget = int(E.GEN[task].split("=")[1])
        mt = _re3.search(r"timeout=(\d+)", joined)
        assert mt and int(mt.group(1)) * 3 >= budget, \
            f"{task}: HTTP timeout {mt.group(1) if mt else '?'}s cannot cover budget {budget} at 3 t/s"


def t_every_runner_guards_against_every_other_lock():
    # Surveyed 2026-08-08, mid-EV-1-night: the copy-pasted lock lists had DRIFTED.
    # autotune_sweep checked 2 of 5 and phaseb_gen checked 4 - so with EV-1 holding the box
    # and a 30B row in flight, autotune_sweep would have started and contended for the GPU.
    # That is the overlap that voided the 2026-07-31 ladder. One list, shared, or this fails.
    import sys as _sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, os.path.join(root, "weights"))
    import runner as R

    for f in ("autotune_sweep.py", "ev1_run.py", "phaseb_gen.py"):
        p = os.path.join(root, "weights", f)
        if not os.path.exists(p):
            continue
        src = open(p, encoding="utf-8").read()
        assert "runner" in src and "owns_the_box" in src, \
            f"{f} still hand-rolls its lock discipline - use runner.owns_the_box so the " \
            f"guarded set cannot drift again"

    # ANYTHING THAT READS THE LOCKS, not just runners that take them. The test above covered
    # the three takers and missed verify.py, which CHECKS locks before benchmarking and had
    # hand-rolled three of the five - so the release gate ran a benchmark while an EV-1 30B
    # row held the GPU and reported the contaminated 68% spread as a model problem. A file
    # that names any lock literal must source the set from runner.
    import re as _re
    offenders = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".venv", "build")]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root).replace("\\", "/")
            if rel.endswith("weights/runner.py") or rel.startswith("tests/"):
                continue                      # the definition itself, and this test
            src = open(path, encoding="utf-8", errors="replace").read()
            names = _re.findall(r'["\'](\.\w*_lock)["\']', src)
            if not names:
                continue
            if "LOCK_NAMES" not in src and "owns_the_box" not in src:
                offenders.append(f"{rel} hand-rolls {sorted(set(names))}")
            elif set(names) - set(R.LOCK_NAMES):
                offenders.append(f"{rel} names a lock outside LOCK_NAMES: "
                                 f"{sorted(set(names) - set(R.LOCK_NAMES))}")
    assert not offenders, \
        "lock lists must come from runner.LOCK_NAMES, or they drift:\n  " + "\n  ".join(offenders)

    # and the shared guard must actually refuse on somebody else's lock
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        os.mkdir(os.path.join(td, ".p0_lock"))            # someone else owns the box
        try:
            with R.owns_the_box(".ev1_lock", td, kill=False):
                raise AssertionError("started while another runner held a lock")
        except R.BoxBusy:
            pass
        assert not os.path.isdir(os.path.join(td, ".ev1_lock")), "left a lock behind on refusal"

    # own lock is released even when the body raises - a stale lock blocks every future run
    with tempfile.TemporaryDirectory() as td:
        try:
            with R.owns_the_box(".ev1_lock", td, kill=False):
                raise ValueError("boom")
        except ValueError:
            pass
        assert not os.path.isdir(os.path.join(td, ".ev1_lock")), \
            "lock survived an exception - the box would look busy forever"


def t_a_failing_row_cannot_cancel_the_night():
    # 2026-08-08: subprocess.TimeoutExpired was uncaught in run_row. Uncaught, it walks out of
    # the row loop, past the finally that drops the lock, and TERMINATES THE RUN - so one slow
    # row cancels every row queued behind it, silently, with no failure tail (that write sits
    # after the line that raises). Caught 25 minutes before it would have killed eight rows.
    # The contract this pins: a row may fail; the night may not die with it.
    import subprocess as _sp
    import sys as _sys
    import tempfile
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, os.path.join(root, "weights"))
    import ev1_run as E

    saved = (E.start_server, E.stop_server, E.gpu_state, E.done, E.DATA, _sp.run)
    saved_watch = (E.STALL_MIN, E._progress)
    saved_popen = _sp.Popen
    calls = []
    try:
        with tempfile.TemporaryDirectory() as td:
            E.DATA = td
            E.start_server = lambda *a, **k: (object(), None)
            E.stop_server = lambda *a, **k: None
            E.gpu_state = lambda *a, **k: None
            E.done = lambda *a, **k: False          # never "already complete"
            # a child that runs forever and never advances the progress counter = wedged
            E.STALL_MIN = 0.02                      # ~1.2s, so the test is not a sleep
            E._progress = lambda: 7                 # frozen: no forward progress, ever
            E.run_watched.__globals__["subprocess"] = _sp

            class _Wedged:
                returncode = None
                def poll(self): return None
                def kill(self): type(self).returncode = -9
                def wait(self): return type(self).returncode
                def communicate(self): return ("out", "err")
            _sp.Popen = lambda *a, **k: _Wedged()

            # must RETURN, not raise, and must not hang - that is the whole contract
            E.run_row("0.6B", "ifeval", lambda s: calls.append(s))

            tails = [f for f in os.listdir(td) if f.startswith("ev1_fail_")]
            assert tails, "a killed row must leave a failure tail on disk"
            assert any("STALLED" in c for c in calls), \
                f"the stall must be logged, got: {calls}"
            assert any("Nothing was written" in c for c in calls), \
                "the log must say no partial results exist - a killed row has none"
    finally:
        (E.start_server, E.stop_server, E.gpu_state, E.done, E.DATA, _sp.run) = saved
        E.STALL_MIN, E._progress = saved_watch
        _sp.Popen = saved_popen

    # NO WALL-CLOCK CAP may come back. A cap is wrong in both directions - it killed a healthy
    # 30B row 103 minutes from the end, and would have let a wedged row burn six hours doing
    # nothing. Rows are watched for PROGRESS instead: never interrupt work that is flowing,
    # catch work that has stopped in ~25 minutes.
    import re as _re
    src = open(os.path.join(root, "weights", "ev1_run.py"), encoding="utf-8").read()
    assert not _re.search(r"subprocess\.run\([^)]*timeout=\d+\s*\*\s*3600", src, _re.S), \
        "a wall-clock cap on a row is back - watch progress, do not cap duration"
    assert "def run_watched" in src and "STALL_MIN" in src, \
        "the progress watchdog is gone; a row could now hang forever unnoticed"


def t_scoring_never_depends_on_math_verify():
    # math_verify returns False for EVERYTHING on this box - verify(42, 42) is False, because
    # parse() returns [] when its timeout wrapper cannot spawn a subprocess (WinError 87). A
    # scorer that silently answers "not equivalent" to every question would zero every math row
    # while looking healthy, so our scoring path must never reach for it.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "weights", "lm_eval_tasks", "math500_utils.py"),
               encoding="utf-8").read()
    assert "math_verify" not in src, \
        "math500_utils must not import math_verify - it returns False for every pair here"


def t_partial_hw_flags_still_yield_a_vram_rate():
    # Passing ANY hardware flag skips auto-detect, and VRAM bandwidth fell through to 0 while
    # RAM/disk carried fallbacks - so `plan --vram 24` (sizing a card you do not own yet, the
    # single most natural use of the flag) raised ZeroDivisionError inside evaluate(). The
    # contract: capacity given without a rate must produce a rate (borrowed and announced) or
    # an actionable refusal - never a traceback, and never a silent zero.
    from quantprobe.plan import resolve_hw

    class A:
        machine = None; vram = 24; vram_bw = None
        ram = None; ram_bw = None; disk_bw = None
    # Two legal outcomes, and a traceback is not one of them. On a box with a GPU the rate is
    # borrowed and announced; on a GPU-less runner there is nothing to borrow from, so the
    # contract is a clean actionable refusal naming the flag to pass. Either is fine. What must
    # never happen again is the original ZeroDivisionError three frames down, or a silent 0.
    try:
        vc, vb, rc, rb, db, _geta, _gl, _hw = resolve_hw(A(), announce=False)
    except SystemExit as e:
        assert "--vram-bw" in str(e), \
            f"refusal must name the flag that fixes it, got: {e}"
        return "SKIP: no GPU here to borrow a VRAM bandwidth from - refusal path checked instead"
    assert vc == 24, f"explicit --vram must survive resolution, got {vc}"
    assert vb > 0, "VRAM capacity without a bandwidth is not a machine - it divided by zero"
    assert rc > 0 and rb > 0 and db > 0, "RAM/disk fallbacks must still hold"


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


def _kv_ev(kw):
    """run()'s ev closure for the KV-lever tests, INCLUDING the kvp_scale override the L-24
    capacity probe emits - a mirror that silently dropped it would measure f16 KV and call it
    q8_0."""
    from quantprobe.plan import evaluate
    def ev(**over):
        k = dict(kw)
        if "rc_delta" in over:
            k["rc"] = k["rc"] + over.pop("rc_delta")
        if "kvp_scale" in over:
            k["kvp"] = k["kvp"] * over.pop("kvp_scale")
        over.pop("true_size_gb_scale", None)
        k.update(over)
        return evaluate(**k)[2]
    return ev


def t_l24_capacity_probe_offers_q8_kv_lever_at_depth():
    """L-24/L-25 wiring, direction 1: when a deep-context config misses the boundary by less
    than the f16-vs-q8_0 KV difference, the capacity finding must OFFER the KV lever and the
    report must PREFER it over dropping a weight quant tier - the quality half is MEASURED
    (PPL ratio 1.00031 +/- 0.0188 at d7168, prereg #91) where a quant-tier drop below ~3 bits
    is not free (Laws 1-2). Before this wiring, capacity_probe shaved weights only - the exact
    gap L-25's audit reported."""
    from quantprobe.plan import (capacity_probe, binding_constraint, binding_report, evaluate,
                                 KV_Q8_BYTES_PER_F16_BYTE)
    kw = dict(t=11.9, a=11.9, ne=11.9, moe=False, bits=2.5, vc=9, vb=300, rc=32, rb=45,
              db=2, geta=0.45, gl=0.28, ctx=16384, kvp=73728)
    size, _, rows = evaluate(**kw)
    kv_gb = kw["ctx"] * kw["kvp"] / 1e9
    cap = kw["vc"] * 0.90
    assert size + kv_gb > cap, "fixture drifted: config no longer misses the VRAM boundary"
    assert size + kv_gb * KV_Q8_BYTES_PER_F16_BYTE <= cap, \
        "fixture drifted: q8_0 KV no longer closes the gap alone"
    find = capacity_probe(_kv_ev(kw), rows[0][1], size, kv_gb, kw["vc"], kw["rc"])
    assert find and find["tier"] == "VRAM", find
    assert find.get("kv_closes") and find["kv_tps"] > 0 and find["kv_gain"] > 1.15, find
    assert abs(find["kv_saved_gb"] - kv_gb * (1 - KV_Q8_BYTES_PER_F16_BYTE)) < 1e-9, find
    text = "\n".join(binding_report(binding_constraint(rows[0], capacity=find), bits=2.5,
                                    placement=rows[0][0]))
    assert "-ctk q8_0 -ctv q8_0" in text and "PREFER" in text, text
    assert "1.00031" in text and "d7168" in text, text     # quality half cited (prereg #91)
    assert "+37%" in text and "#25" in text, text          # speed half cited too (prereg #25)
    assert "E-10" in text, text                            # niche-domain caveat is permanent
    # the same event on the CLI surface: the tier-boundary advisor offers the same lever
    rc, out = cli("plan", "--total", "11.9", "--active", "11.9", "--vram", "9",
                  "--vram-bw", "300", "--ram", "32", "--ram-bw", "45", "--disk-bw", "2",
                  "--ctx", "16384", "--kv-per-pos", "72")
    assert rc == 0 and "CAPACITY-BOUND" in out, out[:400]
    assert "-ctk q8_0 -ctv q8_0" in out and "instead of dropping a quant tier" in out, out
    head, _, tail = out.partition("tier-boundary advisor")
    assert "+37%" in head and "+37%" in tail, \
        "the measured speed half (prereg #25) must reach BOTH CLI surfaces (report + advisor)"


def t_l24_capacity_kv_lever_is_additive_and_silent_without_kv():
    """Direction 2: at ctx 0 the finding must carry NO KV fields and the report must print no
    -ctk advice - the lever may never fire on a config with nothing to quantize. And the KV
    wiring must be ADDITIVE: every weight-lever number in the finding is byte-identical to a
    probe that never saw KV fields (prereg #88's thresholds untouched). When KV exists but
    cannot close the gap alone, the report says 'partial' and never PREFER."""
    from quantprobe.plan import capacity_probe, binding_constraint, binding_report, evaluate
    kw = dict(t=11.9, a=11.9, ne=11.9, moe=False, bits=2.5, vc=8, vb=256, rc=32, rb=45,
              db=2, geta=0.45, gl=0.28)
    size, _, rows = evaluate(**kw)
    find = capacity_probe(_kv_ev(kw), rows[0][1], size, 0.0, kw["vc"], kw["rc"])
    assert find and not any(k.startswith("kv") for k in find), find
    text = "\n".join(binding_report(binding_constraint(rows[0], capacity=find), bits=2.5,
                                    placement=rows[0][0]))
    assert "q8_0" not in text and "KV lever" not in text, text
    # partial direction: shallow KV cannot close a mostly-weights gap; offered as a pairing,
    # never as the preferred lever
    kw2 = dict(kw, ctx=16384, kvp=8192)
    size2, _, rows2 = evaluate(**kw2)
    kv2 = kw2["ctx"] * kw2["kvp"] / 1e9
    find2 = capacity_probe(_kv_ev(kw2), rows2[0][1], size2, kv2, kw2["vc"], kw2["rc"])
    assert find2 and find2.get("kv_closes") is False, find2
    assert "kv_tps" not in find2, "no counterfactual speed may be quoted for a lever that does not fit"
    # additive check: weight levers identical whether or not the probe saw the KV fields
    for key in ("tier", "gap_gb", "lever", "shave_tps", "lift_tps", "gain_shave", "gain_lift",
                "need_gb"):
        assert key in find2, find2
    text2 = "\n".join(binding_report(binding_constraint(rows2[0], capacity=find2), bits=2.5,
                                     placement=rows2[0][0]))
    assert "partial:" in text2 and "-ctk q8_0 -ctv q8_0" in text2 and "PREFER" not in text2, text2


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
    # since X-1 the reachable dense branch quotes the kernel-rule numbers (measured all-in-VRAM,
    # so they are legitimate exactly here)...
    assert "up to 5.8x" in speculation_advice(False, "all in VRAM", row=bw)
    # ...and those numbers must NEVER leak onto a row whose own decomposition cannot reach them -
    # that is C-15's whole point, re-asserted against the new text.
    assert "5.8x" not in txt and "88.5" not in txt, \
        "X-1 numbers printed on a row that cannot reach them"
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


def t_c17_disk_probe_reads_the_whole_file_not_a_warm_tail():
    """C-17: measure_disk read a fixed 512MB TAIL jittered by <=7MB, so ~98.6% of the span
    overlapped between calls and buffering=0 does not bypass the OS page cache. Measured:
    cold 0.44, then 2.99 / 2.99 GB/s - the warm number is RAM and it shipped as a disk-tier
    input 6.8x too fast.

    THREE DEFECTS IN THE FIRST VERSION OF THIS TEST, all fixed here:
      1. it was defined BELOW the `if __name__ == "__main__"` runner, so the function did not
         exist when the loop read globals() - it never ran once, on any commit;
      2. it returned a bare "skipped" string without a >2GB fixture, and the harness printed a
         returned value as `ok` - so even collected it would have counted as green;
      3. its only assertion was that repeated TIMINGS AGREE, which a fully page-cached file
         satisfies perfectly. It could not tell the fix from the failure it guards.

    So the property is checked where it lives instead: the offset distribution. `probe_offset`
    must be uniform over the WHOLE file. No fixture, no timing, no cache to contaminate - and it
    fails by construction on the tail-jitter code (verified by mutation)."""
    from quantprobe.detect import probe_offset
    # 20 GB: a real disk-tier GGUF, and past 4 GiB - which is where a 32-bit random draw silently
    # stops. The first version of this fix used os.urandom(4) and could not reach 80% of a file
    # this size; the reachable prefix is exactly the part a partial download already warmed.
    size, span = 20 * 1024**3, 64 * 1024**2
    offs = [probe_offset(size, span) for _ in range(400)]
    room = size - span
    assert min(offs) < room * 0.05, (
        f"probe never reads near the START of the file (min offset {min(offs)/room:.3f} of the "
        f"range) - this is the fixed-tail probe C-17 measured 6.8x too fast")
    assert max(offs) > room * 0.95, f"probe never reaches the END of the file: {max(offs)/room:.3f}"
    # coverage, not just extremes: a 7 MB jitter would put every draw in one bucket
    buckets = {int(o / room * 10) for o in offs}
    assert len(buckets) >= 9, (
        f"probe offsets cluster in {len(buckets)}/10 deciles - the span between calls overlaps, "
        f"which is exactly how the second read measured RAM instead of the disk")
    assert probe_offset(span, span) == 0 and probe_offset(10, 1 << 30) == 0   # degenerate sizes
    # determinism seam: with the draw pinned, the offset is the arithmetic one
    assert probe_offset(size, span, rnd=lambda: 0) == 0
    assert probe_offset(size, span, rnd=lambda: room) == room


def t_c17_disk_probe_timings_agree_on_a_real_file():
    """The end-to-end half, when a real fixture is available. Kept SEPARATE from the offset test
    above so that its absence cannot make the offset property look checked - and it now returns
    a SKIP the harness prints as a skip rather than as `ok`."""
    import os
    from quantprobe.detect import measure_disk
    p = os.environ.get("QP_DISK_TEST_FILE")
    if not p or not os.path.exists(p):
        return "SKIP: set QP_DISK_TEST_FILE to a file larger than free page cache"
    runs = [measure_disk(p, mb=64, samples=3) for _ in range(3)]
    lo, hi = min(runs), max(runs)
    assert hi / lo < 2.5, (
        f"disk probe drifts {hi/lo:.1f}x across repeats {runs} - page-cache contamination "
        "is back; the minimum-of-N estimator is not holding (#97)")
    return None


def t_p97_disk_probe_returns_the_cold_draw_not_the_warm_one():
    """prereg #97: a single warm draw must not become the reported disk bandwidth.

    Fixture-free and deterministic - `_one_read` is replaced by a scripted sequence, so this
    tests the ESTIMATOR rather than the weather. Against the pre-#97 single-sample code this
    fails by construction: that version returned whatever one draw it happened to take, and
    measured reality supplied the failing input (6 of 8 draws on a warmed 13.7 GB file came
    back >1.5 GB/s, max 2.854 - RAM reported as disk, a 6.3x error).
    """
    from quantprobe import detect
    real = detect._one_read
    try:
        # one genuinely cold region, the rest served from page cache
        seq = [2.85, 0.45, 2.78, 2.83, 2.44]
        it = iter(seq)
        detect._one_read = lambda path, mb: next(it)
        bw, info = detect.measure_disk("ignored", samples=len(seq), detail=True)
        assert abs(bw - 0.45) < 1e-9, (
            f"reported {bw} GB/s from draws {seq}: a cached read is being shipped as disk "
            f"bandwidth. The minimum is the only draw that can be the device.")
        assert info["warm_draws"] == 4, f"expected 4 warm draws flagged, got {info['warm_draws']}"
        # and the all-cold case must not invent a warning
        it2 = iter([0.45, 0.46, 0.44])
        detect._one_read = lambda path, mb: next(it2)
        _, info2 = detect.measure_disk("ignored", samples=3, detail=True)
        assert info2["warm_draws"] == 0, "warning fires on a consistent set - it would be noise"
    finally:
        detect._one_read = real
    return None


def t_p98_kld_parses_and_never_falls_back_to_perplexity():
    """prereg #98: the KL block parses, and its ABSENCE returns UNMEASURED rather than
    quietly becoming a perplexity number. Fixture-free - parses a captured real output.

    The second half is the load-bearing one. `--kl-divergence` REPLACES the perplexity report
    rather than adding to it, which is exactly how the first run of #98 produced a half-empty
    result; the kill rule caught it. A silent fallback here would answer a different question
    than the caller asked while looking like an answer.
    """
    from quantprobe.probe import parse_kld
    real = (
        "Maximum KLD:   1.461354\n"
        "99.9%   KLD:   2.900000\n"
        "99.0%   KLD:   1.461354\n"
        "95.0%   KLD:   0.700000\n"
        "90.0%   KLD:   0.450000\n"
        "Median  KLD:   0.182194\n"
        "Same top p: 72.614 +/- 0.500 %\n")
    r = parse_kld(real)
    assert r["99.0%"] == 1.461354 and r["Median"] == 0.182194, f"KLD percentiles mis-parsed: {r}"
    assert r["same_top_p"] == 72.614, f"same-top-p mis-parsed: {r}"
    # a perplexity-only run carries NO KLD block: must come back empty, not fabricated
    ppl_only = "Final estimate: PPL = 17.7733 +/- 1.20000\n"
    assert parse_kld(ppl_only) == {}, (
        "a perplexity-only output produced a KLD reading - the metric is being invented from "
        "a run that never computed it")
    return None


def t_e11_layer_by_layer_reads_the_whole_model_not_just_active():
    """E-11: the layer-by-layer row must price ALL weights per token, not the active set.

    This is the ONE thing that makes the placement different from every other row in the menu,
    and it is the thing a reader is most likely to get wrong: airllm visits every layer, so a
    235B MoE with 22B active moves all 235B. If this row ever prices `act` it will outrank the
    expert-offload rows on MoE and recommend a placement that is an order of magnitude slower.
    """
    from quantprobe.plan import evaluate
    size, act, rows = evaluate(t=235, a=22, ne=22, moe=True, bits=4, vc=4, vb=200, rc=64,
                               rb=25, db=3.5, geta=0.5, n_layer=94)
    ll = [r for r in rows if "layer-by-layer" in r[0]]
    assert ll, "layer-by-layer row not emitted for a 128 GB MoE on a 4 GB card"
    row = ll[0]
    # terms must reconstruct size/bw, not act/bw
    moved = row.terms["io"] * 3.5 + row.terms["ram_bw"] * (25 * 0.55)   # approx eta_r*rb
    assert moved > act * 2, (
        f"layer-by-layer moves only ~{moved:.0f} GB of weights per token against an active set "
        f"of {act:.1f} GB and a model of {size:.1f} GB - it is pricing the ACTIVE set, which is "
        f"the whole point of this placement being different")
    best = rows[0]
    assert "layer-by-layer" not in best[0], (
        f"layer-by-layer won on a MoE at {row[1]:.4f} vs {best[1]:.4f} - it reads every expert "
        f"every token and must never beat expert offload here")
    assert "MoE PENALTY" in (row[2] or ""), "MoE row must name the all-experts penalty"
    return None


def t_e11_layer_by_layer_fits_a_card_the_model_cannot():
    """E-11: emitted exactly when the MODEL does not fit VRAM but one LAYER does - that is the
    claim ("70B on a 4 GB card") and the only reason the placement exists. Also asserts the two
    unpriced costs are disclosed on the row, since the printed number is an upper bound."""
    from quantprobe.plan import evaluate
    k = dict(t=70, a=70, ne=70, moe=False, bits=16, vb=200, rc=128, rb=50, db=3.5,
             geta=0.5, n_layer=80)
    _, _, small = evaluate(vc=4, **k)                 # 151 GB model, 1.9 GB layer -> emit
    ll = [r for r in small if "layer-by-layer" in r[0]]
    assert ll, "70B fp16 on a 4 GB card must emit the layer-by-layer row"
    w = ll[0][2] or ""
    assert "PCIe" in w and "C-23" in w, (
        f"row must disclose BOTH unpriced costs (PCIe transfer, C-23 streaming gap); got: {w}")
    assert "UPPER BOUND" in w, "row must say the printed speed is an upper bound"
    _, _, tiny = evaluate(vc=1, **k)                  # 1 GB card cannot hold a 1.9 GB layer
    assert not [r for r in tiny if "layer-by-layer" in r[0]], (
        "a 1 GB card cannot hold a 1.9 GB layer - the row must not be emitted")
    _, _, huge = evaluate(vc=200, **k)                # model fits entirely: row is pointless
    assert not [r for r in huge if "layer-by-layer" in r[0]], (
        "the model fits in VRAM - streaming layer by layer must not be offered")
    return None



def t_auto_never_trusts_an_incomplete_local_gguf():
    """`auto` must not read a partial download's header, and must not die on one.

    BEHAVIOURAL, and fixture-free: the size check short-circuits before from_gguf, so a tiny
    dummy file exercises it without a real multi-GB GGUF. The FIRST version of this guard
    inspected auto.py SOURCE for the strings "try:" and "getsize"; disabling the size check
    left it GREEN. That is the decorative-test pattern this suite exists to catch, written
    here by the same person who spent the week removing it.

    Two failure modes, the quiet one worse:
      LOUD  from_gguf raises on a truncated file (reproduced: 200 KB of a 531 MB GGUF gives
            ValueError from the tensor-shape reader) and killed the whole `auto` command.
      QUIET from_gguf succeeds on a truncated file and returns a plausible-but-wrong spec, so
            `auto` prints a confident number for a model that is not there. Only the size
            check catches this one, and it is the reason the check exists.
    """
    import os, tempfile
    from quantprobe.auto import local_spec_or_none
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "model.gguf")
        with open(f, "wb") as h:
            h.write(b"GGUF" + bytes(4096))

        spec, note = local_spec_or_none(f, expected_size=531068480, n_parts=1)
        assert spec is None, "a 4 KB file was accepted as a 531 MB model"
        assert note and "INCOMPLETE" in note, f"incomplete file not reported: {note}"

        spec, note = local_spec_or_none(f, expected_size=os.path.getsize(f), n_parts=1)
        assert spec is None, "garbage header produced a spec"
        assert note and "could not be read" in note, f"unreadable header not reported: {note}"

        spec, note = local_spec_or_none(f, expected_size=531068480, n_parts=8)
        assert note is None or "INCOMPLETE" not in note, (
            "size check applied to a multi-part download, where it is meaningless")

        assert local_spec_or_none(os.path.join(d, "nope.gguf"), 123, 1) == (None, None)
    return None

def t_ollama_eval_rate_is_generation_not_prompt():
    """audit-ollama must never read the PROMPT rate as the generation rate.

    ollama --verbose prints "prompt eval rate:" BEFORE "eval rate:", so a bare re.search for
    "eval rate" takes the wrong one. Measured on real output: 186.59 (prompt) against 19.92
    (generation) - a 9x error that does not look wrong, because it is still a plausible tok/s.
    It made audit-ollama report ollama at 205.6 tok/s and conclude nothing was worth
    recommending. Fixture is verbatim ollama output, so this fails on the old regex by
    construction.
    """
    from quantprobe.ollama import _parse_rate
    real = ("total duration:       1.4859158s" + "\n"
            "load duration:        214.1493ms" + "\n"
            "prompt eval count:    31 token(s)" + "\n"
            "prompt eval rate:     186.59 tokens/s" + "\n"
            "eval count:           11 token(s)" + "\n"
            "eval duration:        552.199ms" + "\n"
            "eval rate:            19.92 tokens/s" + "\n")
    got = _parse_rate(real)
    assert got is not None and abs(got - 19.92) < 0.01, (
        f"read {got} tok/s - 186.59 is the PROMPT rate and must never be returned")
    fb = _parse_rate("eval count:           40 token(s)" + "\n" +
                     "eval duration:        2 s" + "\n")
    assert fb is not None and abs(fb - 20.0) < 0.01, f"count/duration fallback wrong: {fb}"
    assert _parse_rate("total duration: 1s" + "\n") is None, "invented a rate from nothing"
    return None


def t_ollama_store_reader_survives_a_broken_store():
    """audit-ollama reads a directory it does not own, so it must degrade rather than crash.

    Every case here is one a real machine produces: a half-deleted model leaves a manifest
    with no blob, ollama drops non-JSON files into the tree, and the store may not exist at
    all on a box where ollama was never installed. A traceback out of an audit command is a
    bad outcome - it looks like quantprobe is broken when the user's store simply is.

    Fixture-free: builds the layouts in a tempdir, no ollama required.
    """
    import json, os, tempfile
    from quantprobe.ollama import installed, store_root, MODEL_MEDIA

    assert installed("Z:/definitely/not/here") == [], "nonexistent store should be empty"
    with tempfile.TemporaryDirectory() as d:
        assert installed(d) == [], "empty dir should be empty"

        # manifest pointing at a blob that is gone (half-deleted model)
        mp = os.path.join(d, "manifests", "registry.ollama.ai", "library", "ghost")
        os.makedirs(mp)
        json.dump({"layers": [{"mediaType": MODEL_MEDIA, "digest": "sha256:dead", "size": 1}]},
                  open(os.path.join(mp, "latest"), "w"))
        assert installed(d) == [], "a manifest whose blob is missing must not be listed"

        # non-JSON junk in the manifest tree must be skipped, not parsed
        open(os.path.join(d, "manifests", "README.txt"), "w").write("not json")
        assert installed(d) == [], "junk in manifests/ must be skipped without raising"

    # resolution order: explicit --store beats env, env beats the default
    old = os.environ.get("OLLAMA_MODELS")
    try:
        os.environ["OLLAMA_MODELS"] = "Z:/from-env"
        assert store_root() == "Z:/from-env", "OLLAMA_MODELS ignored"
        assert store_root("Z:/explicit") == "Z:/explicit", "--store did not win over env"
    finally:
        if old is None:
            os.environ.pop("OLLAMA_MODELS", None)
        else:
            os.environ["OLLAMA_MODELS"] = old
    return None


def _business_tasks_mod():
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(root, "weights", "business_tasks.py")
    if not os.path.exists(p):
        return None
    spec = importlib.util.spec_from_file_location("business_tasks", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def t_business_checks_reject_the_answer_that_sounds_right():
    """The whole point of executable checks is that a confident wrong answer scores zero.

    The first version of this task set graded against prose like "sendable with at most a name
    edit", which cannot compare two models. These predicates must reject fluent nonsense.
    """
    bt = _business_tasks_mod()
    if bt is None:
        return "weights/business_tasks.py absent"
    # a plausible-sounding answer that invents a number the source never contained
    name, fn = bt.nonums(4.2, 12, 8)
    assert not fn("Revenue reached 9.9M this quarter"), "nonums accepted an invented number"
    assert fn("Revenue reached 4.2M, up 12%"), "nonums rejected numbers that were in the source"
    # JSON checks must not be satisfied by prose that merely mentions the keys
    name, fn = bt.js("company", "amount_usd", "due_days")
    assert not fn("The company is Acme, amount_usd is 47500, due_days is 30"), \
        "js() accepted prose instead of a JSON object"
    assert fn('{"company":"Acme","amount_usd":47500,"due_days":30}'), "js() rejected valid JSON"
    # a label check must reject an answer that hedges across two labels
    name, fn = bt.label("BILLING", "TECHNICAL", "SALES")
    assert not fn("This is either BILLING or TECHNICAL"), "label() accepted a hedge"
    assert fn("BILLING"), "label() rejected a clean single label"
    # and the whole suite must survive its own self-test
    assert bt.selftest() == 0, "business task self-test failed"
    return None


def t_business_hallucination_check_does_not_invent_the_hallucination():
    """The number extractor must not manufacture numbers that are not in the text.

    A false accusation is the worst failure mode a scoring harness has: it publishes a confident
    verdict against a model that did the task correctly. Both strings below are real outputs that
    the naive regex failed. "Q3 revenue rose" was charged with containing the number 3, and a
    byte-perfect CSV was charged with containing 31200000 - the regex had matched across the
    field separator and stripped the comma.
    """
    bt = _business_tasks_mod()
    if bt is None:
        return "weights/business_tasks.py absent"
    # a quarter label is not a number
    assert bt._nums_in("Q3 revenue rose, up YoY but below plan.") == set(), \
        f"quarter label read as a number: {bt._nums_in('Q3 revenue rose')}"
    # a version string is not three numbers
    assert bt._nums_in("running v1.24.0 here") == set(), \
        f"version string read as numbers: {bt._nums_in('running v1.24.0 here')}"
    # a comma between fields does not join two numbers into one
    csv = "region,quarter,revenue\nEMEA,Q3,1200000\nAMER,Q3,3000000"
    assert bt._nums_in(csv) == {"1200000", "3000000"}, \
        f"CSV field separator merged numbers: {bt._nums_in(csv)}"
    # ...but a thousands separator still does
    assert bt._nums_in("we will refund 47,500 USD") == {"47500"}, "thousands separator lost"
    assert bt._nums_in("ARR of 3,000,000 total") == {"3000000"}, "multi-group thousands lost"
    # units attached to a number do not hide it
    assert bt._nums_in("revenue was 4.2M, up 12%") == {"4.2", "12"}, \
        f"unit-suffixed numbers lost: {bt._nums_in('revenue was 4.2M, up 12%')}"
    # and the check itself must still catch a real invention
    _, fn = bt.nonums(4.2, 12)
    assert not fn("revenue was 9.9M"), "a genuinely invented number was let through"
    assert fn("revenue was 4.2M, up 12%"), "sourced numbers were rejected"
    assert fn("Q3 revenue rose with no figures"), "a number-free summary was rejected"
    return None


def t_business_reasoning_never_reaches_the_scorer():
    """A thinking model's scratchpad must not be graded as its answer.

    Qwen3 spends most of its tokens reasoning. If <think> content leaked into the graded text,
    a model that reasons "the answer is 17839.92" and then states the wrong final figure would
    score CORRECT - the checks look for the number anywhere in the string.
    """
    bt = _business_tasks_mod()
    if bt is None:
        return "weights/business_tasks.py absent"
    blob = "<think>Let me compute. 49*37*12*0.82 = 17839.92 so that is the answer.</think>19999"
    answer, think = bt.strip_reasoning(blob)
    assert answer == "19999", f"reasoning leaked into the answer: {answer!r}"
    assert "17839.92" in think, "reasoning was discarded instead of captured"
    # the scorer must now FAIL this task, because the stated answer is wrong
    _, fn = bt.num(17839.92)
    assert not fn(answer), "a wrong final answer scored correct because reasoning leaked"
    assert fn(blob), "control: the number really is present in the unstripped blob"
    # an unterminated <think> (hit the token cap mid-thought) must not pass everything through
    answer2, _ = bt.strip_reasoning("<think>still thinking and never closed")
    assert answer2 == "", f"unterminated reasoning leaked: {answer2!r}"
    return None


def t_business_a_truncated_answer_is_not_a_wrong_answer():
    """Running out of token budget mid-thought is a harness limit, not a model failure.

    On a reasoning model the budget covers thinking too. At 1024 tokens five arithmetic tasks
    burned the whole budget before emitting anything and scored as five confident failures -
    which would have published "2.5-bit cannot do arithmetic" when the truth was "we cut it off".
    Truncated tasks must be quarantined, AND the shrunken denominator must be disclosed, because
    quietly dropping hard tasks is how a headline flatters itself.
    """
    bt = _business_tasks_mod()
    if bt is None:
        return "weights/business_tasks.py absent"
    import io, json, contextlib, tempfile as tf
    rows = [{"cluster": "arithmetic", "id": "a1", "kind": "auto", "prompt": "", "output": "",
             "seconds": 59.0, "think_words": 393, "gen_tokens": 1024,
             "finish_reason": "length", "truncated": True, "passed": None,
             "checks": [["answer is 17839.92", False]], "rubric": None},
            {"cluster": "arithmetic", "id": "a6", "kind": "auto", "prompt": "", "output": "29.4",
             "seconds": 52.0, "think_words": 467, "gen_tokens": 908,
             "finish_reason": "stop", "truncated": False, "passed": True,
             "checks": [["answer is 29.4", True]], "rubric": None}]
    path = os.path.join(tf.gettempdir(), "qp_bt_trunc_guard.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"model": "m", "args": "", "results": rows}, fh)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            bt.score(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    out = buf.getvalue()
    # the truncated task must NOT be counted as a failure: 1 of 1 scorable passed
    assert "1/1 pass = 100.0%" in out, f"truncated task was scored as a failure:\n{out}"
    # and the exclusion must be stated, with the worst case spelled out
    assert "TRUNCATED" in out, f"quarantine was silent:\n{out}"
    assert "50.0%" in out, f"worst-case (all truncations as failures) not disclosed:\n{out}"
    assert "arithmetic/a1" in out, f"did not name which tasks were excluded:\n{out}"
    return None


def t_probe_creates_workdir_and_refuses_to_bless_an_incomplete_curve():
    """Two defects from the A2A pipeline (2026-08-04), one harness. A --workdir that did not
    exist made llama-quantize fail at stream-open on step 1, every later step failed on the
    missing files - and the probe still exited 0 with '(incomplete: curve unavailable)', so
    the caller read success over ten failures. The probe must (a) create the workdir it was
    given, (b) exit nonzero when any band produced no perplexity."""
    import types, tempfile as tf, io, contextlib
    import quantprobe.probe as P
    calls = []
    real = (P.find_llama, P.n_layers, P.sh, P.ppl)
    wd = os.path.join(tf.gettempdir(), "qp_probe_wd_guard", "nested", "deep")
    import shutil
    shutil.rmtree(os.path.join(tf.gettempdir(), "qp_probe_wd_guard"), ignore_errors=True)
    gguf = os.path.join(tf.gettempdir(), "qp_probe_fake.gguf")
    open(gguf, "wb").write(b"\0" * 1024)
    a = types.SimpleNamespace(gguf=gguf, workdir=wd, bands=4, chunks=2, eval="e.raw",
                              ngl=0, llama_dir=None, dry_run=False, apply=False,
                              out=None, imatrix=None)
    try:
        P.find_llama = lambda d: tf.gettempdir()
        P.n_layers = lambda p: 8
        P.sh = lambda cmd, dry: calls.append(cmd)
        P.ppl = lambda *x, **k: None                      # every band fails to score
        buf = io.StringIO()
        rc = None
        try:
            with contextlib.redirect_stdout(buf):
                P.run(a)
        except SystemExit as e:
            rc = e.code
    finally:
        P.find_llama, P.n_layers, P.sh, P.ppl = real
        try:
            os.unlink(gguf)
        except OSError:
            pass
    assert os.path.isdir(wd), "probe did not create the workdir it was given"
    assert rc == 1, f"incomplete curve must exit 1, got {rc!r}"
    assert "INCOMPLETE" in buf.getvalue(), "the failure is not stated in the output"
    return None


def t_serving_advisory_stays_on_the_placement_it_was_measured_on():
    """U-38/U-39's multi-user numbers are placement-specific: 9.5x aggregate scaling was
    measured dense-in-VRAM, the 2.0x cap was measured experts-in-RAM, and printing either on
    the other family would be the C-15 leak all over again. Pure-CPU and other unmeasured
    placements must print NOTHING."""
    from quantprobe.plan import serving_advisory
    dense = "\n".join(serving_advisory("all in VRAM"))
    moe = "\n".join(serving_advisory("split experts: 39%->VRAM, rest->RAM"))
    hyb = "\n".join(serving_advisory("hybrid: attention->VRAM, experts->RAM"))
    cpu = "\n".join(serving_advisory("pure CPU (GPU idle)"))
    assert "219.4 at 32" in dense and "widths 2-8 are STRICTLY DOMINATED" in dense
    assert "Ampere+" in dense, "the one-box caveat is missing from the dense block"
    assert "2.0x cap" in moe and "inverts the choice" in moe
    assert "2.0x cap" in hyb, "hybrid expert-offload must carry the same measured cap"
    assert "STRICTLY DOMINATED" not in moe, "the dense kernel rule leaked onto the MoE row"
    assert "219.4 at 32" not in moe.split("inverts")[0], \
        "dense aggregate presented as this row's own number"
    assert cpu == "", "an unmeasured placement printed serving advice"
    return None


def t_x1_draft_length_rule_reaches_the_user():
    """X-1 measured that drafts of 4-7 sit in the slow kernel (48-51 tok/s) while m>=8 rides the
    fast one (88-132) - a 2.5x gap the old advice silently forfeited by omitting draft length.
    The dense speculation advisory must now state the rule, recommend a concrete m past the
    boundary, and keep the honest limits (Ampere unverified; prose gains nothing)."""
    from quantprobe.plan import speculation_advice
    # dense all-in-VRAM, not oversold - the branch every 7B-on-GPU user hits
    note = speculation_advice(moe=False, placement="all in VRAM", row=None)
    assert note is not None, "dense advisory vanished"
    assert "size-m 12" in note, f"no concrete draft length recommended: {note[:120]}"
    assert "m>=8" in note or "m >= 8" in note, "the kernel boundary rule is missing"
    assert "5.8x" in note, "the measured ceiling is not stated"
    assert "Ampere" in note, "the unverified-on-Ampere caveat is missing"
    assert "1.01x" in note, "the prose-gains-nothing honesty line is missing"
    # the old text that parked users in the slow kernel must be gone from this branch
    assert "measured **2.10x decode**" not in note, "the underselling 2.10x default is back"
    return None


def t_docs_are_strict_utf8_or_pages_dies():
    """GitHub Pages builds docs/ with kramdown, which hard-fails on invalid UTF-8.

    On Windows, `python script > file` under a bash shell encodes stdout in cp1252, so an
    em-dash becomes byte 0x97 and the whole Pages deployment goes red - which is exactly how
    docs/MATRIX.md broke the site for two pushes on 2026-08-04. Regenerate docs with
    PYTHONUTF8=1. This guard fails on the first non-UTF-8 byte in any markdown we ship.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bad = []
    for base in ("docs", "."):
        d = os.path.join(root, base)
        for name in sorted(os.listdir(d)):
            if not name.endswith(".md"):
                continue
            raw = open(os.path.join(d, name), "rb").read()
            try:
                raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as e:
                bad.append(f"{base}/{name} byte {e.start}: {raw[e.start:e.start+3]!r}")
    assert not bad, "invalid UTF-8 (Pages will fail to build): " + "; ".join(bad)
    return None


def t_shipped_markdown_carries_no_invisible_control_characters():
    """Valid UTF-8 is not the same as readable text, and the difference is invisible in review.

    A backspace (0x08) reached findings/REGISTER.json and FINDINGS.md on 2026-08-09, because a
    text fragment containing a LaTeX command crossed a shell heredoc and its backslash was
    consumed as an escape: the register said the extractor "never inspects \\boxed{}" and the
    published file said it never inspects "oxed{}". Perfectly valid UTF-8, renders as garbage,
    and survives every check that only asks whether the bytes decode.

    Tab and newline are the only control characters markdown has any use for.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bad = []
    # media/ is in the list because the third occurrence put a backspace inside an SVG, where
    # it is not merely ugly: the XML parser rejects char 8 outright, so the chart rendered as
    # "PCDATA invalid Char value 8" and everything after the offending note was dropped. A
    # generated file is exactly where this is hardest to notice, and easiest to check.
    for base in ("docs", ".", "findings", "media"):
        d = os.path.join(root, base)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith((".md", ".json", ".svg")):
                continue
            text = open(os.path.join(d, name), encoding="utf-8", errors="replace").read()
            for i, ch in enumerate(text):
                if ord(ch) < 32 and ch not in "\n\t\r":
                    bad.append(f"{base}/{name} offset {i}: {hex(ord(ch))} "
                               f"near {text[max(0, i - 25):i + 12]!r}")
                    break
    assert not bad, "control characters in shipped text: " + "; ".join(bad)
    return None


def t_a_correct_answer_before_a_truncated_tail_still_scores():
    """A model that loops into the token cap must not be scored as if it never answered.

    The 4B reaches the right answer on AIME, then repeats "\\boxed{116}" 683 times until
    generation is cut off mid-token. lm-eval's last_boxed_only_string takes the LAST \\boxed,
    finds it unbalanced, returns None - and a correct answer scored zero. Measured across the
    banked rows: 9 answers rescued, 0 lost, and 8 of 10 rows unchanged.

    Both directions are pinned. A scorer that gets more generous is exactly the change that
    needs a guard against generosity, so this also asserts the extractor still refuses a
    response whose ONLY box is truncated - there is no correct answer to rescue there, and
    inventing one would be the thumb on the scale this whole protocol exists to prevent.
    """
    import sys as _sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, os.path.join(root, "weights", "lm_eval_tasks"))
    try:
        import math500_utils as M
    except ImportError:
        return "SKIP: lm_eval not installed (research dep, not a package dep - runs locally)"

    bs = chr(92)
    box = bs + "boxed"

    # 1. THE REAL FAILING INPUT: correct answer, then a repetition loop cut off mid-token.
    looped = ("The answer is " + box + "{116}." + ("\n\n" + box + "{116}") * 40 + "\n\n" + box
    )
    r = M.process_results({"answer": "116"}, [looped])
    assert r == {"exact_match": 1, "emitted_boxed": 1}, \
        f"correct answer before a truncated tail must still score: {r}"

    # 2. Cut off after the brace opens - same shape, one character further along.
    r = M.process_results({"answer": "540"}, [box + "{540}\n\n" + box + "{540"])
    assert r["exact_match"] == 1, "an unbalanced trailing box must not discard the good one"

    # 3. NOTHING TO RESCUE: the only box is truncated. Must still refuse.
    r = M.process_results({"answer": "7"}, ["I think the answer is " + box + "{7"])
    assert r == {"exact_match": 0, "emitted_boxed": 0}, \
        f"a response with no well-formed box must not be graded: {r}"

    # 4. The generosity guard - a rescued box that is WRONG is still wrong.
    r = M.process_results({"answer": "116"}, [box + "{999}\n\n" + box])
    assert r == {"exact_match": 0, "emitted_boxed": 1}, \
        f"wrong answer must be graded, not rescued: {r}"

    # 5. Ordinary responses are untouched - the last box still wins when it is well-formed.
    r = M.process_results({"answer": "42"}, ["First " + box + "{7}, on reflection " + box + "{42}."])
    assert r["exact_match"] == 1, "the LAST well-formed box must win, not the first"
    return None


def t_no_runner_polls_a_pipe_it_never_drains():
    """Repo-wide: never wait on a child's exit while its output pipe goes unread.

    The shape is Popen(stdout=PIPE) -> a loop on .poll() -> communicate() AFTER the loop. The
    pipe fills, the child blocks in write(), .poll() never changes, and communicate() is never
    reached. It cost four 30B AIME rows and two misdiagnoses (C-27), and an audit found the
    same pattern in two more files, so a comment in one docstring is not enough.

    The check is positional rather than a keyword count, because every offender DOES call
    communicate - just too late. What matters is whether a .poll() sits between the Popen and
    the drain. Files listed in ALLOWED have been read and fixed; the list is here so that
    adding to it is a deliberate act with a reason attached, not a silent widening.
    """
    import ast as _ast
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ALLOWED = {}          # empty: every site drains before it waits

    def is_popen_with_pipe(node):
        if not isinstance(node, _ast.Call):
            return False
        fn = node.func
        nm = fn.attr if isinstance(fn, _ast.Attribute) else getattr(fn, "id", "")
        if nm != "Popen":
            return False
        # PIPE may arrive as subprocess.PIPE, a bare PIPE, or the -1 it equals
        return any("PIPE" in _ast.dump(kw.value) for kw in node.keywords)

    offenders = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".venv", "build")]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), root).replace("\\", "/")
            src = open(os.path.join(dirpath, name), encoding="utf-8", errors="replace").read()
            try:
                tree = _ast.parse(src)
            except SyntaxError:
                continue
            # AST, not regex: this guard's OWN docstring contains the words "Popen(stdout=PIPE)"
            # and ".poll()", and a text scan duly reported the guard as an offender. Prose that
            # describes a defect is not the defect.
            lines = src.splitlines()
            for node in _ast.walk(tree):
                if not is_popen_with_pipe(node):
                    continue
                after = "\n".join(lines[node.lineno:])
                drain = min((p for p in (after.find(".communicate("),
                                         after.find(".stdout.read("),
                                         after.find("Thread(")) if p != -1), default=len(after))
                poll = after.find(".poll()")
                if poll != -1 and poll < drain and rel not in ALLOWED:
                    offenders.append(f"{rel}:{node.lineno} polls before draining its PIPE")
    assert not offenders, \
        "a child's exit is awaited while its pipe goes unread (C-27):\n  " + "\n  ".join(offenders)
    return None


def t_a_chatty_child_cannot_deadlock_the_watchdog():
    """A child that writes more than a pipe buffer holds must still finish.

    This is the bug that cost four 30B AIME rows. run_watched took stdout=PIPE and stderr=PIPE
    and then polled in a sleep loop, reading nothing until communicate() after exit - so the OS
    pipe filled, lm-eval blocked in write(), stopped issuing HTTP requests, and sat at 0.00s
    CPU with zero sockets while llama-server idled. It was misdiagnosed twice, once as an
    lm-eval concurrency bug and once as a slot-plan problem; an entire attempt was spent
    running ctx 16384 x 1 slot, which wedged anyway at 24 of 30 items.

    Windows pipe buffers are a few KB to 64 KB, so 2 MB is far past any of them. Against the
    pre-fix code this test hangs rather than fails, which is the honest shape of the defect -
    hence the timeout, so a regression shows up as a red test instead of a hung suite.
    """
    import sys as _sys
    import threading
    import time as _time
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, os.path.join(root, "weights"))
    import ev1_run as E

    saved = (E.DATA, E._progress, E.stop_server, E.gpu_state, E.STALL_MIN)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        E.DATA = td
        E._progress = lambda: 0                 # frozen: never counts as forward progress
        E.stop_server = lambda *a, **k: None
        E.gpu_state = lambda *a, **k: None
        E.STALL_MIN = 90                        # long enough that the stall path cannot fire
        chatty = ("import sys\n"
                  "sys.stdout.write('x' * 2_000_000)\n"
                  "sys.stdout.flush()\n")
        result = {}

        def go():
            # capture the exception too: a daemon thread that dies silently would report as a
            # deadlock, which is a different bug and would send the next reader down the wrong
            # path exactly as the first misdiagnosis did.
            try:
                result["r"] = E.run_watched([_sys.executable, "-c", chatty], os.environ.copy(),
                                            None, "M", "T", lambda *a: None, _time.time())
            except BaseException as exc:                     # noqa: BLE001
                result["exc"] = f"{type(exc).__name__}: {exc}"

        th = threading.Thread(target=go, daemon=True)
        th.start()
        th.join(timeout=120)
        assert not th.is_alive(), \
            "run_watched did not return - a 2 MB child deadlocked it, which is the pipe-buffer " \
            "bug that killed four 30B AIME rows"
        assert "exc" not in result, f"run_watched raised: {result['exc']}"
        r = result.get("r")
        assert r is not None and r.returncode == 0, f"chatty child did not succeed: {r}"
        assert len(r.stdout) > 0, "the child's output was not captured at all"
    (E.DATA, E._progress, E.stop_server, E.gpu_state, E.STALL_MIN) = saved
    return None


def t_boxed_rows_are_regraded_with_the_current_extractor():
    """Rows on disk carry the verdict of whatever extractor was loaded when they ran.

    That is not an abstraction: the extractor was fixed mid-campaign, the rows already banked
    still held the old verdict, and the row running that night was being graded by the old code
    held inside its own process. Uniform treatment therefore has to happen offline, from the
    logged samples, or the suite compares models that were graded by different rules.

    Built on a synthetic row in a temp directory rather than the local corpus, so it runs the
    same on CI as it does here - and asserts the delta, not just that the code path executes.
    """
    import json as _json
    import sys as _sys
    import tempfile
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, os.path.join(root, "weights"))
    _sys.path.insert(0, os.path.join(root, "weights", "lm_eval_tasks"))
    try:
        import math500_utils  # noqa: F401
    except ImportError:
        return "SKIP: lm_eval not installed (research dep, not a package dep - runs locally)"
    import ev1_report as R

    bs = chr(92)
    box = bs + "boxed"
    looped = box + "{116}" + ("\n\n" + box + "{116}") * 30 + "\n\n" + box   # cut off mid-token

    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "4B", "aime24_boxed", "run")
        os.makedirs(d)
        # what lm-eval WROTE at run time: the old extractor scored this zero
        with open(os.path.join(d, "results_x.json"), "w", encoding="utf-8") as fh:
            _json.dump({"results": {"aime24_boxed": {"exact_match,none": 0.0,
                                                     "emitted_boxed,none": 0.0}}}, fh)
        with open(os.path.join(d, "samples_x.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(_json.dumps({"doc_id": 0, "doc": {"Answer": "116"}, "target": "116",
                                  "resps": [[looped]], "exact_match": 0,
                                  "emitted_boxed": 0}) + "\n")

        raw = R.load_rows(root=td, rescore=False)[("4B", "aime24_boxed")]
        assert raw["exact_match,none"] == 0.0, "rescore=False must return the untouched verdict"

        fixed = R.load_rows(root=td, rescore=True)[("4B", "aime24_boxed")]
        assert fixed["exact_match,none"] == 1.0, \
            f"the correct answer before the truncated tail was not rescued: {fixed}"
        assert fixed["_rescued"] == 1 and fixed["_lost"] == 0, \
            f"provenance must record what the correction did: {fixed}"

        # Provenance must never be mistaken for a score by the uniform-zero guard - `_lost`
        # is 0 on every row when the correction is working, which is the opposite of suspect.
        assert R.check_publishable({("a", "t"): {"_lost": 0.0}, ("b", "t"): {"_lost": 0.0},
                                    ("c", "t"): {"_lost": 0.0}}) == [], \
            "underscore-prefixed provenance tripped the uniform-zero guard"
    return None


def t_no_shipped_chart_has_text_running_off_its_canvas():
    """A clipped sentence is a silent defect: the asset looks finished and says less than it says.

    Twice on 2026-08-09 an asset shipped with its last words past the right edge - and the
    second time was AFTER the first had been fixed by hand, which is the argument for a rule
    instead of remembering. brand.wrap now does the wrapping and this checks the output, both
    using brand.CHAR_W so authoring and checking cannot drift apart.

    The estimate is deliberately conservative rather than exact: without a font engine there is
    no true advance width, so this catches gross overruns (the failure that actually happened)
    and stays quiet on tight-but-fine lines. Tolerance is generous for the same reason - a
    guard that cries wolf on every chart gets switched off.
    """
    import re as _re
    import sys as _sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, os.path.join(root, "weights"))
    import brand as B

    media = os.path.join(root, "media")
    bad = []
    for name in sorted(os.listdir(media)):
        if not name.endswith(".svg"):
            continue
        src = open(os.path.join(media, name), encoding="utf-8").read()
        vb = _re.search(r'viewBox="0 0 (\d+) (\d+)"', src)
        if not vb:
            continue
        cw = int(vb.group(1))
        for tag in _re.finditer(r'<text ([^>]*)>(.*?)</text>', src, _re.S):
            attrs, inner = tag.group(1), tag.group(2)
            if 'text-anchor' in attrs:            # right/centre anchored: grows leftward
                continue
            mx = _re.search(r'\bx="(-?[\d.]+)"', attrs)
            ms = _re.search(r'font-size="(\d+)"', attrs)
            if not mx or not ms:
                continue
            txt = _re.sub(r'<[^>]+>', '', inner)
            width = len(txt) * int(ms.group(1)) * B.CHAR_W
            # 1.18x slack absorbs the per-glyph error in a character-count estimate.
            if float(mx.group(1)) + width > cw * 1.18:
                bad.append(f"{name}: {txt[:56]!r} runs to "
                           f"~{float(mx.group(1)) + width:.0f}px on a {cw}px canvas")
    assert not bad, "chart text runs off the canvas:\n  " + "\n  ".join(bad[:6])
    return None


def t_a_server_log_is_never_opened_in_a_mode_that_destroys_the_last_one():
    """The server logs are the only record of in-session decode throughput. Do not truncate.

    start_server() names the log by (model, slots), so re-running a row with the same slot plan
    reopens the same path - and `open(logp, "w")` erased the previous session. The 30B's
    MATH-500 log was 12 lines by the time anyone read it, overwritten by a later short-lived
    start, so its accuracy survived in results_*.json while the speed at which it produced
    those answers did not. Evidence for the score and evidence for the number must not have
    different lifetimes.

    Appending is only safe if session boundaries are explicit - otherwise a reader medians two
    machine states together, which is the C-14 violation the lock discipline exists to prevent.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "weights", "p0_lanes.py"), encoding="utf-8").read()
    import re as _re
    i = src.find("def start_server(")
    assert i > 0, "start_server not found - this guard is pointed at the wrong file"
    body = src[i:i + 2500]
    truncating = _re.findall(r"open\(\s*logp\s*,\s*[\"']w[\"']", body)
    assert not truncating, \
        "start_server opens the server log in truncating mode - a re-run destroys the previous " \
        "session's throughput evidence"
    assert _re.search(r"open\(\s*logp\s*,\s*[\"']a[\"']", body), \
        "start_server no longer appends to the server log"
    assert "SESSION START" in body, \
        "appended sessions need an explicit banner, or a reader will median two machine states"
    return None


def t_findings_md_is_regenerable_and_matches_the_register():
    """FINDINGS.md is GENERATED, so a generator that cannot run means docs silently go stale.

    That is not hypothetical. `priority` is written as an int by early entries and as a
    descriptive sentence by every recent one; render() sorted on the raw value, so the moment
    both forms coexisted in one section the sort raised TypeError. Nothing noticed, because the
    drift check runs FIRST and exits non-zero on any uncited pre-registration - so the crash
    downstream of it was never reached. FINDINGS.md sat three days stale (last written
    2026-08-06) while entries kept landing in the register.

    Two failures compounding: a guard that exits early hides everything behind it, and a
    generated file that nobody diffs is indistinguishable from a current one. This asserts the
    generator runs AND that what it produces is what is committed.
    """
    import sys as _sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, root)
    import findings as F

    reg = F.load()
    problems = F.validate(reg)
    assert not problems, "register has drifted: " + "; ".join(problems[:4])

    # Mixed priority forms must not crash the sort - the actual regression, pinned.
    ranks = [F.priority_rank(e) for e in reg["untried"]]
    assert all(isinstance(r, float) for r in ranks), "priority_rank returned a non-sortable value"
    assert F.priority_rank({"priority": 1}) < F.priority_rank({"priority": "low - someday"}), \
        "int and string priorities do not order against each other"
    sorted(reg["untried"], key=F.priority_rank)      # raises TypeError pre-fix

    produced = F.render(reg)
    on_disk = open(os.path.join(root, "FINDINGS.md"), encoding="utf-8").read()
    norm = lambda s: s.replace("\r\n", "\n").strip()
    assert norm(produced) == norm(on_disk), \
        "FINDINGS.md does not match the register - run `python findings.py` and commit the result"
    return None


def t_business_no_verdict_from_an_empty_staked_set():
    """0/0 is not 0%. A tier-only results file has no staked tasks; scoring one printed
    "KILL RULE FIRED (0.0%)" over an empty denominator - a confident verdict about evidence
    that does not exist. And a task that got no answer for a HARNESS reason (HTTP timeout,
    dead server) must be excluded, not scored as a wrong answer."""
    bt = _business_tasks_mod()
    if bt is None:
        return "weights/business_tasks.py absent"
    import io, json, contextlib, tempfile as tf
    rows = [{"cluster": "tier4", "id": "t4a1", "kind": "auto", "prompt": "", "output": "",
             "seconds": 900.0, "think_words": 0, "gen_tokens": 0, "finish_reason": "",
             "truncated": False, "error": "timed out", "passed": None,
             "checks": [], "rubric": None},
            {"cluster": "tier3", "id": "t3a1", "kind": "auto", "prompt": "", "output": "1646.35",
             "seconds": 100.0, "think_words": 10, "gen_tokens": 200, "finish_reason": "stop",
             "truncated": False, "error": None, "passed": True,
             "checks": [["answer is 1646.35", True]], "rubric": None}]
    path = os.path.join(tf.gettempdir(), "qp_bt_tieronly_guard.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"model": "m", "args": "", "results": rows}, fh)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            bt.score(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    out = buf.getvalue()
    assert "KILL RULE" not in out, f"kill rule fired on an empty staked set:\n{out}"
    assert "not applicable" in out, f"missing the explicit not-applicable statement:\n{out}"
    assert "HARNESS reason" in out, f"harness-timeout exclusion not disclosed:\n{out}"
    assert "t4a1" in out, f"the errored task is not named:\n{out}"
    return None


def t_business_never_scores_a_run_that_did_not_happen():
    """A verdict from a run that produced nothing is worse than no verdict.

    score() reads the results FILE, so an aborted run would silently grade whatever a previous
    run left on disk. That actually happened: a failed preflight printed "KILL RULE FIRED
    (33.3%)" from stale data.
    """
    bt = _business_tasks_mod()
    if bt is None:
        return "weights/business_tasks.py absent"
    called = []
    real_run, real_score = bt.run, bt.score
    try:
        bt.run = lambda *a, **k: []                     # the run produces nothing
        bt.score = lambda *a, **k: (called.append(1), 0)[1]
        old = sys.argv
        try:
            sys.argv = ["business_tasks.py", "--run", "M", "--server", "http://127.0.0.1:1"]
            rc = bt.main()
        finally:
            sys.argv = old
    finally:
        bt.run, bt.score = real_run, real_score
    assert not called, "an empty run was still scored - a stale results file would be graded"
    assert rc == 1, f"an empty run must exit non-zero, got {rc}"
    return None


def t_contribute_payload_carries_the_resolved_machine_not_none():
    """A contributed datapoint's entire purpose is the machine it was measured on. Under
    auto-detect (the default, i.e. nearly every contributor) the raw args are all None, and
    v1.26.1 shipped printing 'hardware: vram=None vram_bw=None ram=None...' in both the body
    and the issue title. Caught by the pre-launch gauntlet. The payload must carry the SAME
    resolved values the prediction used."""
    import types, io, contextlib
    from quantprobe.runtime import _emit_contribution
    a = types.SimpleNamespace(machine=None, vram=None, vram_bw=None, ram=None, ram_bw=None,
                              disk_bw=None, model=None, total=7.6, active=7.1, bits=4.9)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _emit_contribution(a, ("all in VRAM", 25.0), 30.0, 0.5, 20.0)
    out = buf.getvalue()
    assert "vram=None" not in out and "ram_bw=None" not in out, \
        f"contribute payload still carries None hardware:\n{out[:300]}"
    import re as _re
    m = _re.search(r"hardware: vram=([0-9.]+) vram_bw=([0-9.]+) ram=([0-9.]+)", out)
    assert m, f"no numeric resolved hardware line in payload:\n{out[:300]}"
    assert "issues/new" in out, "the pre-filled issue link vanished"
    return None


def t_contribute_payload_carries_model_spec_not_none():
    """Issue #1 (the tool's first external datapoint, RX 5700 XT): the title read 'total=None
    active=None @ 2.5-bit' for a 7.6B Q4_0 - the model side of the payload read raw args that
    no resolution path had written back (split-GGUF autospec failure -> every fallback fired).
    Same bug class as the v1.26.1 hardware fix, other operand. The payload must carry the spec
    THE PREDICTION USED (stashed by best_flags at its resolution moment), plus the GGUF
    filename when there is one - the field a human reader actually recognises."""
    import types, io, contextlib
    from quantprobe.runtime import _emit_contribution
    base = dict(machine=None, vram=8.0, vram_bw=448.0, ram=32.0, ram_bw=22.0, disk_bw=3.4,
                model=None, total=None, active=None, bits=4.65)
    a = types.SimpleNamespace(**base, _resolved_spec=(7.6, 7.6),
                              gguf="D:/x/qwen2.5-7b-q4_0-00001-of-00002.gguf")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _emit_contribution(a, ("all in VRAM", 73.1), 73.18, 0.16, 0.1)
    out = buf.getvalue()
    assert "total=None" not in out and "active=None" not in out, \
        f"payload still ships a None model spec (issue #1):\n{out[:300]}"
    assert "total=7.6" in out and "qwen2.5-7b-q4_0-00001-of-00002.gguf" in out, \
        f"resolved spec / filename missing from payload:\n{out[:300]}"
    # Without the stash the legacy fallback prints None/None - the exact shipped bug. Assert
    # that path still exists as the LAST resort so this test pins the mutation direction: if
    # someone deletes the stash in best_flags, the first assert above is what fails.
    b = types.SimpleNamespace(**base)
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        _emit_contribution(b, ("all in VRAM", 73.1), 73.18, 0.16, 0.1)
    assert "total=None" in buf2.getvalue(), "legacy fallback changed; update this test's premise"
    return None


def t_decon_screen_mutation_directions_pinned():
    """The Phase B decontamination screen is a kill rule (program law 2026-08-05): a verbatim
    protected-bench text MUST flag, an 8-gram-sharing paraphrase MUST flag, a clean sample
    MUST pass. If evalplus data is unavailable offline this test SKIPS LOUDLY rather than
    passing vacuously - a skip is not a pass."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "weights"))
    try:
        try:
            import decon
            from evalplus.data import get_mbpp_plus
        except Exception as e:
            print(f"      (decon screen: evalplus data unavailable, SKIPPED: {e})", end="")
            return
        hashes, grams, meta = decon.load_protected()
        assert meta["n_grams"] > 10000, "protected gram set implausibly small"
        t = next(iter(get_mbpp_plus().values()))
        ok, _ = decon.screen_one(t["prompt"] + t["canonical_solution"], hashes, grams)
        assert not ok, "verbatim bench text passed the screen"
        toks = decon._tokens(t["canonical_solution"])[:decon.NGRAM]
        ok2, _ = decon.screen_one("training filler " + " ".join(toks) + " more filler", hashes, grams)
        assert not ok2, "8-gram paraphrase passed the screen"
        ok3, why = decon.screen_one("a wholly original sample about clamping kelvin sensor "
                                    "glitches while logging their indices", hashes, grams)
        assert ok3, f"clean sample flagged: {why}"
    finally:
        sys.path.pop(0)
    return None


def t_sandbox_side_effects_stay_in_temp():
    """Candidate code writes files (2026-08-06: 44 junk artifacts - test.db, pickles, a
    Dockerfile - appeared in the repo ROOT because the sandbox inherited our cwd). Both
    executors must confine side-effects to a temp dir that dies with the run."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "weights"))
    try:
        from p0_lanes import run_candidate, expected_outputs
        canary = os.path.join(os.getcwd(), "smoke_sandbox_canary.txt")
        if os.path.exists(canary):
            os.remove(canary)
        code = ("def f(x):\n"
                "    open('smoke_sandbox_canary.txt', 'w').write('leaked')\n"
                "    return x + 1\n")
        n, tot = run_candidate(code, "f", [[1]], [2])
        assert n == 1, "sandbox no longer executes correct code"
        assert not os.path.exists(canary), \
            "SANDBOX LEAK: candidate code wrote into the working directory"
    finally:
        sys.path.pop(0)
    return None


def t_media_svgs_have_png_twins():
    """Standard process (2026-08-06): every media asset ships SVG + PNG - X and Reddit take
    PNGs. An orphan SVG in media/ is an asset that cannot be posted, and this test refuses it."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    media = os.path.join(root, "media")
    if not os.path.isdir(media):
        return
    orphans = [f for f in os.listdir(media) if f.endswith(".svg")
               and not os.path.isfile(os.path.join(media, f[:-4] + ".png"))]
    assert not orphans, f"media SVGs missing PNG twins (run their generator or brand.render_png): {orphans}"
    return None


def t_hardware_doc_matches_the_code():
    """docs/HARDWARE.md is GENERATED from detect.py's tables; if someone edits either side
    alone, the doc lies about the tool. Regenerate in memory and compare. (Absent doc = fail:
    the README links it.)"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "weights"))
    try:
        from make_hardware_table import render, rows
    finally:
        sys.path.pop(0)
    p = os.path.join(root, "docs", "HARDWARE_TABLE.md")
    assert os.path.isfile(p), "docs/HARDWARE_TABLE.md missing - run weights/make_hardware_table.py"
    on_disk = open(p, encoding="utf-8").read()
    assert on_disk == render(rows()), \
        "docs/HARDWARE_TABLE.md drifted from detect.py - re-run weights/make_hardware_table.py"
    return None


def t_amd_gpu_detection_prices_the_field_case():
    """Issue #1's contributor ran an RX 5700 XT and detect printed 'GPU: none detected' - the
    tool was nvidia-smi-only, and they had to hand-pass the exact 448 GB/s the table now
    carries. v1.27: non-NVIDIA adapters come from the driver registry (qwMemorySize;
    Win32_VideoController.AdapterRAM is a uint32 that caps at 4 GB), a table-known card is
    priced, an unknown card is NAMED with its VRAM and asked for flags, and virtual adapters
    never leak through."""
    from quantprobe.detect import _parse_win_adapters, gpu_lookup
    txt = ("Radeon RX 5700 XT|8589934592\n"
           "Microsoft Basic Display Adapter|0\n"
           "DameWare Development Mirror Driver 64-bit|\n"
           "Radeon RX 5700 XT|4294967296\n")          # CIM duplicate with capped AdapterRAM
    ads = _parse_win_adapters(txt)
    assert ads == [("Radeon RX 5700 XT", 8.0)], f"parse/dedup/filter broken: {ads}"
    bw, _geta, _gl, src = gpu_lookup("Radeon RX 5700 XT")
    assert bw == 448 and "table" in src, f"field case must price at spec 448: {bw} {src}"
    bw2, _, _, src2 = gpu_lookup("Banana Graphics 9000")
    assert "default" in src2 and bw2 == 300, "unknown cards must fall to the explicit default"
    assert gpu_lookup("Intel Arc B580 Graphics")[0] == 456, "Arc entries missing"
    # an NVIDIA name must never match an AMD fragment ('rtx 5070' vs 'rx 5700')
    assert gpu_lookup("NVIDIA GeForce RTX 5070")[0] == 672, "cross-vendor fragment collision"
    return None


def t_split_gguf_siblings_and_size():
    """Issue #1 root cause: a 2-part GGUF specced from part 1 alone halves total params and
    every size-derived quantity. split_siblings must map ANY part to the full ordered set,
    refuse a partial set outright (a spec from half a model is wrong, not approximate), and
    gguf_size must sum the parts while staying byte-identical to getsize on plain files."""
    import tempfile
    from quantprobe.spec import split_siblings, gguf_size
    d = tempfile.mkdtemp()
    def _w(name, nbytes):
        p = os.path.join(d, name)
        with open(p, "wb") as fh:
            fh.write(b"x" * nbytes)
        return p
    p1 = _w("m-00001-of-00002.gguf", 100)
    p2 = _w("m-00002-of-00002.gguf", 50)
    sib = split_siblings(p2)                  # part 2 in, full ordered set out
    assert [os.path.basename(x) for x in sib] == \
        ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"], sib
    assert gguf_size(p1) == 150, gguf_size(p1)
    plain = _w("plain.gguf", 7)
    assert split_siblings(plain) == [plain] and gguf_size(plain) == 7
    os.remove(p2)                             # simulate a half-downloaded split
    try:
        split_siblings(p1)
        assert False, "missing split part must raise, not spec half a model"
    except FileNotFoundError as e:
        assert "of 2" in str(e), str(e)
    return None


def t_version_string_has_one_source_of_truth():
    """The 1.26.0 wheel shipped self-reporting 1.25.0: pyproject was bumped, the __version__
    literal was not, and the release verification printed the version without asserting it.
    A package whose --version lies breaks the provable-headline rule. This guard fails the
    gate whenever the two version declarations disagree."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py = open(os.path.join(root, "pyproject.toml"), encoding="utf-8").read()
    import re as _re
    m = _re.search(r'^version = "([^"]+)"', py, _re.M)
    assert m, "no version in pyproject.toml"
    init = open(os.path.join(root, "quantprobe", "__init__.py"), encoding="utf-8").read()
    m2 = _re.search(r'^__version__ = "([^"]+)"', init, _re.M)
    assert m2, "no __version__ literal in quantprobe/__init__.py"
    assert m.group(1) == m2.group(1), \
        f"version desync: pyproject {m.group(1)} vs __init__ {m2.group(1)}"
    return None


def t_a_metric_that_is_zero_for_every_model_is_never_published():
    """A uniform 0.0 across unrelated model sizes is a scorer artifact, and must not ship.

    Three-for-three so far: hendrycks MATH-500 (0.00% while 89.4% of answers carried a boxed
    value), zero-shot AIME (no format requested, so the 4B wrote a bare "Answer: 49" in 30 of
    30), and GSM8K cot_zeroshot strict-match, which demands the literal sentence "The answer
    is N." that its own prompt never asks for - 0 of 3,957 responses across 0.6B/4B/7B matched.

    Each was found by hand, after a row had been spent. This makes it mechanical. The guard
    REFUSES rather than corrects: it cannot know whether a uniform zero is an artifact or a
    genuine wall, and quietly picking the friendlier filter would be the same error pointed
    the other way. Mutation-checked in both directions - a guard that cannot fail proves
    nothing, and a guard that fires on real scores would push us to publish the wrong metric.
    """
    import sys as _sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, os.path.join(root, "weights"))
    import ev1_report as R

    # 1. FIRES on an undiagnosed uniform zero - the failing input, constructed.
    bad = {(m, "newbench"): {"exact_match,strict-match": 0.0, "sample_len": 30}
           for m in ("0.6B", "4B", "30B")}
    try:
        R.check_publishable(bad)
        raise AssertionError("undiagnosed uniform zero was allowed through - the guard is dead")
    except R.SuspectMetric as e:
        assert "newbench" in str(e) and "0.0 on all 3 models" in str(e), \
            f"guard fired but does not say what to look at: {e}"

    # 2. SILENT on ragged real scores - no false positive that would relabel a genuine result.
    good = {("0.6B", "b"): {"exact_match,none": 0.0},          # one model CAN score zero
            ("4B", "b"):   {"exact_match,none": 0.333},
            ("30B", "b"):  {"exact_match,none": 0.772}}
    assert R.check_publishable(good) == [], "fired on ragged scores - would suppress real data"

    # 3. Two models are not enough. Coincidence is possible at n=2; at n=3 across a 0.6B and a
    #    30B it is not, and demanding three is what keeps the rule from crying wolf on a pair.
    pair = {(m, "c"): {"acc,none": 0.0} for m in ("0.6B", "4B")}
    assert R.check_publishable(pair) == [], "fired on only two models"

    # 4. A DIAGNOSED zero passes, and is still reported as flagged rather than forgotten.
    diagnosed = {(m, "gsm8k_cot_zeroshot"): {"exact_match,strict-match": 0.0}
                 for m in ("0.6B", "4B", "7B")}
    flagged = R.check_publishable(diagnosed)
    assert ("gsm8k_cot_zeroshot", "exact_match,strict-match", 3) in flagged, \
        "a diagnosed artifact must still be surfaced, not silently dropped"

    # 5. The headline metric for GSM8K is the one that measures the model.
    metric, why = R.REPORTED["gsm8k_cot_zeroshot"]
    assert metric == "exact_match,flexible-extract", \
        f"GSM8K would publish {metric}, which is 0.0 for every model by construction"
    assert "artifact" in why.lower(), "the reason for the choice must travel with it"
    return None


def t_a_prereg_verdict_is_computed_by_code_not_by_eye():
    """Prereg #98's P1/P2/P3 must fall out of the thresholds, including when none of them fit.

    The failure this guards is not arithmetic, it is judgement: a human looking at "+1.8 on
    MATH-500" after staking "+2.0" is very likely to call it a win, and a human looking at a
    result that matches NONE of the three staked outcomes is very likely to round it to the
    nearest one. Both are unfalsifiable moves that leave no trace. So the thresholds live in
    code, the outcome space is checked for holes, and KR-5 is a refusal rather than a habit.

    Written and committed while exactly one of the six verdict-bearing rows existed on disk and
    before any of them had been read.
    """
    import sys as _sys
    import contextlib
    import io as _io
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, os.path.join(root, "weights"))
    import prereg98_score as P

    M = {"math500_boxed": "exact_match,none",
         "gsm8k_cot_zeroshot": "exact_match,flexible-extract",
         "ifeval": "prompt_level_strict_acc,none"}

    def rows(**pairs):
        """rows(math500_boxed=(naive_pct, ours_pct), ...) -> the shape load_rows returns."""
        out = {}
        for task, (a, b) in pairs.items():
            out[(P.NAIVE, task)] = {M[task]: a / 100.0}
            out[(P.OURS, task)] = {M[task]: b / 100.0}
        return out

    healthy = lambda arm, task: (500, 0, 0)                              # noqa: E731

    def run(r, health=healthy):
        with contextlib.redirect_stdout(_io.StringIO()):
            return P.score(r, health)

    ALL = dict(math500_boxed=(50.0, 50.0), gsm8k_cot_zeroshot=(80.0, 80.0), ifeval=(70.0, 70.0))

    # 1. KR-5 REFUSES on a partial pair. This is the one that matters most while a run is live:
    #    the MATH-500 pair completing is not the verdict, because P1 also constrains GSM8K and
    #    IFEval, and a scorer that answered early would make peeking free.
    half = rows(**ALL)
    del half[(P.OURS, "ifeval")]
    out = run(half)
    assert out["verdict"] is None, "printed a verdict with a benchmark still pending"
    assert (P.OURS, "ifeval") in out["missing"], f"did not name the missing row: {out['missing']}"

    # 2. P1 confirmed: primary clears +2.0 and nothing else falls more than 1.0 behind.
    out = run(rows(math500_boxed=(50.0, 52.5), gsm8k_cot_zeroshot=(80.0, 79.5),
                   ifeval=(70.0, 70.2)))
    assert out["p1"] and out["verdict"] == ["P1"], f"P1 should be the sole verdict: {out}"

    # 3. MUTATION: the same shape with the primary at +1.8 must NOT confirm. A threshold that
    #    cannot fail is decoration.
    out = run(rows(math500_boxed=(50.0, 51.8), gsm8k_cot_zeroshot=(80.0, 79.5),
                   ifeval=(70.0, 70.2)))
    assert not out["p1"], "P1 confirmed at +1.8 against a staked bar of +2.0"

    # 4. MUTATION the other way: primary clears, but a secondary drops 1.5. P1's guard must bite.
    out = run(rows(math500_boxed=(50.0, 53.0), gsm8k_cot_zeroshot=(80.0, 78.5),
                   ifeval=(70.0, 70.0)))
    assert not out["p1"], "P1 ignored its own -1.0 guard on the secondaries"

    # 5. P2 via the primary clause, and P2 via the two-losses clause with the primary flat.
    assert run(rows(math500_boxed=(50.0, 47.5), gsm8k_cot_zeroshot=(80.0, 80.0),
                    ifeval=(70.0, 70.0)))["p2"], "P2 missed a -2.5 primary"
    two = run(rows(math500_boxed=(50.0, 49.5), gsm8k_cot_zeroshot=(80.0, 78.5),
                   ifeval=(70.0, 68.5)))
    assert two["p2"], "P2 missed two secondaries each losing more than 1.0 pt"
    assert two["p3"], "P2 and P3 overlap by construction here and both must be reported"
    assert set(two["verdict"]) == {"P2", "P3"}, f"overlap collapsed to one outcome: {two}"

    # 6. P3: everything inside the null band, and the recipe demonstrably bought no accuracy.
    out = run(rows(math500_boxed=(50.0, 51.0), gsm8k_cot_zeroshot=(80.0, 79.5),
                   ifeval=(70.0, 70.8)))
    assert out["verdict"] == ["P3"], f"a flat result must read as the staked null: {out}"

    # 7. THE HOLE. +2.5 on the primary with a 1.5 secondary loss satisfies none of the three.
    #    The prereg's outcome space is not exhaustive, and the scorer must say "unclassified"
    #    rather than pick whichever prediction is closest.
    out = run(rows(math500_boxed=(50.0, 52.5), gsm8k_cot_zeroshot=(80.0, 78.5),
                   ifeval=(70.0, 70.0)))
    assert out["verdict"] == [], f"an unanticipated outcome was rounded to a prediction: {out}"

    # 8. KR-4 drops the benchmark for BOTH arms when EITHER is degraded - dropping it only for
    #    the arm that tripped would keep the healthy half and quietly select on quality.
    sick = lambda arm, task: (500, 0, 0) if task != "ifeval" or arm == P.NAIVE else (500, 200, 0)
    out = run(rows(**ALL), sick)                                          # noqa: E731
    assert "ifeval" in out["dropped"], "a 40%-empty arm left IFEval in the verdict"
    assert "ifeval" not in out["deltas"], "the degraded benchmark still contributed a delta"

    # 9. KR-3: an arm whose samples cannot be re-graded is carrying a different scorer than its
    #    partner. That is an abort, not a caveat with a comparison printed under it.
    stale = rows(**ALL)
    stale[(P.OURS, "math500_boxed")]["_rescore_unavailable"] = 1
    out = run(stale)
    assert out["verdict"] is None, "compared a re-graded arm against an un-re-graded one"
    assert out["unregraded"], "the KR-3 abort did not name the row"
    return None


def t_the_ceiling_chain_verdict_is_mechanical_and_bands_do_not_round():
    """Prereg #100's four outcomes fall out of thresholds; the region between bands stays empty.

    #99 proved the failure mode is not arithmetic but rounding: an outcome near a staked band
    gets read as inside it. #100 has a genuine gap by construction - BF16-OURS in (12, 25] with
    a healthy OURS-NAIVE margin matches none of P-C1..P-C4 - so the scorer must say BETWEEN
    STAKED BANDS there, and every threshold edge must bite in both directions.

    Committed while the children were still building and no chain arm had run one item.
    """
    import sys as _sys
    import contextlib
    import io as _io
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, os.path.join(root, "weights"))
    import prereg100_score as P

    M = {"math500_boxed": "exact_match,none",
         "gsm8k_cot_zeroshot": "exact_match,flexible-extract",
         "ifeval": "prompt_level_strict_acc,none"}

    def rows(bf16, ours, naive):
        """Three dicts task->pct into load_rows shape."""
        out = {}
        for arm, scores in ((P.BF16, bf16), (P.OURS, ours), (P.NAIVE, naive)):
            for task, v in scores.items():
                out[(arm, task)] = {M[task]: v / 100.0}
        return out

    FLAT = {"gsm8k_cot_zeroshot": 80.0, "ifeval": 75.0}
    healthy = lambda arm, task: (500, 0, 0)                               # noqa: E731

    def run(r, health=healthy):
        with contextlib.redirect_stdout(_io.StringIO()):
            return P.score(r, health)

    def chain(bf16_math, ours_math, naive_math, **over):
        b = dict(FLAT, math500_boxed=bf16_math); o = dict(FLAT, math500_boxed=ours_math)
        n = dict(FLAT, math500_boxed=naive_math)
        for k, (vb, vo, vn) in over.items():
            b[k], o[k], n[k] = vb, vo, vn
        return rows(b, o, n)

    # 1. KR-C5: a single missing row means no verdict, and the row is named.
    half = chain(85.0, 78.0, 60.0)
    del half[(P.NAIVE, "ifeval")]
    out = run(half)
    assert out["verdict"] is None and (P.NAIVE, "ifeval") in out["missing"]

    # 2. P-C1: deficit +8 inside (0,12], margin +18 >= 5.
    out = run(chain(86.0, 78.0, 60.0))
    assert out["verdict"] == ["P-C1"], f"expected clean P-C1: {out}"

    # 3. MUTATION: deficit +13 must NOT be P-C1 - and with margin healthy and nothing else
    #    firing, it is the between-bands hole, reported as such.
    out = run(chain(91.0, 78.0, 60.0))
    assert not out["verdict"], f"+13 deficit rounded into a band: {out}"

    # 4. MUTATION the other way: margin +4.9 fails P-C1's second clause.
    out = run(chain(86.0, 78.0, 73.1))
    assert "P-C1" not in out["verdict"], "margin below 5 still confirmed P-C1"

    # 5. P-C2 overlaps P-C1 when the deficit sits in (0, 2): both print, neither is chosen.
    out = run(chain(79.5, 78.0, 60.0, gsm8k_cot_zeroshot=(80.5, 80.0, 70.0),
                    ifeval=(75.5, 75.0, 65.0)))
    assert set(out["verdict"]) == {"P-C1", "P-C2"}, f"overlap collapsed: {out}"

    # 6. P-C3: a +30 deficit is the size-class wall.
    assert run(chain(90.0, 60.0, 40.0))["verdict"] == ["P-C3"]

    # 7. P-C4 both ways: naive matching ours, and ours beating BF16 beyond noise.
    assert "P-C4" in run(chain(86.0, 60.0, 60.0))["verdict"], "NAIVE == OURS missed"
    assert "P-C4" in run(chain(75.0, 78.5, 60.0))["verdict"], "OURS > BF16+2 missed"

    # 8. KR-C4 pools health across ALL arms: one sick arm drops the benchmark for the chain.
    sick = lambda arm, task: (500, 200, 0) if task == "ifeval" and arm == P.NAIVE \
        else (500, 0, 0)                                                  # noqa: E731
    out = run(chain(86.0, 78.0, 60.0), sick)
    assert "ifeval" in out["dropped"] and "ifeval" not in out["d_bf16_ours"]

    # 9. KR-C3: an un-regradable row aborts rather than comparing two scorers.
    stale = chain(86.0, 78.0, 60.0)
    stale[(P.OURS, "math500_boxed")]["_rescore_unavailable"] = 1
    out = run(stale)
    assert out["verdict"] is None and out["unregraded"]
    return None


def t_the_http_timeout_covers_an_unquantized_model_at_one_token_per_second():
    """A row must not die on a per-request timeout that assumes a rate the model can't hit.

    The 3 t/s floor (budget/3) was calibrated on quantized models and shredded a 10.5h BF16
    gsm8k row: rc=1, results=MISSING. BF16 at 4 concurrent slots runs below 3 t/s per stream, so
    a single full-budget item could not finish inside budget/3 = 683s. The floor is now 1 t/s,
    and this pins it so the faster floor cannot creep back unnoticed - mutation-checked, because
    a timeout test that passes at both 1 and 3 t/s tests nothing.

    The real backstop against a wedged row is run_watched's STALL detector, not this timeout, so
    a generous per-request window has no downside - which is the whole argument for the change.
    """
    import re as _re
    import sys as _sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, os.path.join(root, "weights"))
    import ev1_run as E

    def timeout_of(task):
        cmd = E.build_cmd("4B-BF16", task)
        args = cmd[cmd.index("--model_args") + 1]
        return int(_re.search(r"timeout=(\d+)", args).group(1))

    # 1. Every powered task's timeout must cover its own budget at 1 t/s. Under the old budget/3
    #    floor gsm8k got 683s for a 2048-token budget; the row it killed proves 683 was short.
    for task in ("gsm8k_cot_zeroshot", "ifeval", "math500_boxed"):
        budget = int(E.GEN[task].split("=")[1])
        got = timeout_of(task)
        assert got >= budget, (f"{task}: timeout {got}s < budget {budget} - a full-budget item at "
                               f"1 t/s per stream cannot finish, which is exactly what failed")

    # 2. MUTATION: the old floor must be rejected. gsm8k's 2048 budget at 3 t/s is 683s; the
    #    timeout must be strictly larger than that, or we have silently reverted.
    assert timeout_of("gsm8k_cot_zeroshot") > 2048 // 3, "the 3 t/s floor is back"

    # 3. The 600s minimum still holds for a hypothetical tiny budget (never regress the floor).
    assert E.build_cmd is not None and max(600, 100) == 600
    return None


def t_kv_is_priced_on_full_attention_layers_only():
    """U-51 (prereg #101 P-5): hybrid models must not pay KV for linear-attention layers.

    Qwen3.8-27B has 48 of 64 layers as linear attention (fixed state, no positional KV), and the
    old formula priced all of them: 260 KB/pos read vs ~64 real, a 4x over-estimate that would
    mis-advise memory pressure at long context. The count comes from the FILE - a block with an
    attn_k projection caches K+V, a linear block has none - so no per-arch table can go stale.

    Mutation-checked in both directions: the hybrid must come out BELOW the all-layers figure by
    the block ratio, and a full-attention model must be BYTE-IDENTICAL to the old formula
    (kv_layers == n_layer), which is the regression guard that lets this ship inside v1.28.
    Uses the real GGUFs when present; skips honestly otherwise (a skip is not a pass).
    """
    import sys as _sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, root)
    from quantprobe.spec import from_gguf

    hybrid = "D:/evo-compress-data/gguf/qwen38/Qwen3.8-27B-Q4_K_M.gguf"
    dense = "D:/evo-compress-data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    if not (os.path.isfile(hybrid) and os.path.isfile(dense)):
        return "SKIP: needs the Qwen3.8 + Qwen2.5-7B GGUFs on D: (this box only)"

    h = from_gguf(hybrid)
    # 1. The hybrid counts KV blocks from the file: 17 of 65 (16 full-attn + the MTP block,
    #    which carries attn_k too - the pre-existing n_layer convention, unchanged here).
    assert h["kv_layers"] == 17 and h["n_layer"] == 65, (h["kv_layers"], h["n_layer"])
    # 2. And prices ONLY those: 17 x 4 KV-heads x (256+256) x 2B = 69,632 B/pos. The old
    #    formula gave 65/17 = 3.8x more; if kvp comes back near 260 KB the fix has regressed.
    assert h["kvp"] == 17 * 4 * 512 * 2, h["kvp"]

    d = from_gguf(dense)
    # 3. REGRESSION GUARD: on a full-attention model every block has attn_k, so the new count
    #    equals n_layer and kvp is byte-identical to the old n_layer formula.
    assert d["kv_layers"] == d["n_layer"], (d["kv_layers"], d["n_layer"])
    assert d["kvp"] == d["n_layer"] * 4 * 256 * 2, d["kvp"]
    return None


if __name__ == "__main__":
    print("quantprobe smoke suite")
    for n, f in list(globals().items()):
        if n.startswith("t_"):
            check(n, f)
    if SKIP:
        print(f"\n{len(SKIP)} SKIPPED (a skip is not a pass):")
        for n, why in SKIP:
            print(f"  - {n}: {why}")
    if FAIL:
        sys.exit(f"\n{len(FAIL)} FAILURES")
    print("\nall green")
