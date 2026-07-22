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
