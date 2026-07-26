"""quantprobe verify — the pre-release gate. One command, four layers.

Every bug that reached users this project has shipped was caught by a DIFFERENT layer, and
never by the one before it:

  layer 1  unit + invariant tests    caught: prose-in-flags, missing --no-mmap (retroactively)
  layer 2  installed-artifact check  caught: PATH/permission installs that pass in a repo cwd
  layer 3  END-TO-END with llama.cpp caught: an 82%-below-prediction config that 54 unit tests
                                             sat green through
  layer 4  anchors vs measured       catches: the law quietly ceasing to retrodict reality

Layer 3 needs a real GGUF and a real llama.cpp; it SKIPS cleanly when they are absent, and says
so rather than passing silently. A skip is not a pass.

    python verify.py [--gguf FILE --llama-dir DIR]
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys

FAIL, SKIP = [], []


def step(name, fn):
    print(f"\n=== {name} ===", flush=True)
    try:
        r = fn()
        print(f"  PASS  {name}" if r is not False else f"  SKIP  {name}")
    except AssertionError as e:
        FAIL.append((name, str(e))); print(f"  FAIL  {name}: {e}")
    except Exception as e:
        FAIL.append((name, repr(e))); print(f"  FAIL  {name}: {e!r}")


def layer1_tests():
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, "tests", "smoke.py")],
                       capture_output=True, text=True, errors="replace")
    out = r.stdout + r.stderr
    assert "all green" in out, "smoke suite not green:\n" + "\n".join(
        l for l in out.splitlines() if "FAIL" in l)
    print(f"  {sum(1 for l in out.splitlines() if l.strip().startswith('ok'))} tests green")


def layer2_installed_artifact():
    """The package must work from an INSTALLED location, not just a repo checkout - a repo cwd
    shadows site-packages and has produced false confidence here before."""
    r = subprocess.run([sys.executable, "-m", "quantprobe", "--help"],
                       capture_output=True, text=True, errors="replace", cwd=os.path.expanduser("~"))
    assert r.returncode == 0 and "plan" in r.stdout, \
        f"`python -m quantprobe` broken outside the repo: {(r.stdout+r.stderr)[:200]}"
    # report what the SUBPROCESS resolved, not our own import - ours comes from the repo cwd,
    # which is precisely the false confidence this layer exists to prevent.
    q = subprocess.run([sys.executable, "-c",
                        "import quantprobe,os;print(quantprobe.__version__, os.path.dirname(quantprobe.__file__))"],
                       capture_output=True, text=True, cwd=os.path.expanduser("~"))
    assert q.returncode == 0, f"quantprobe does not import outside the repo: {q.stderr[:200]}"
    ver, loc = q.stdout.strip().split(" ", 1)
    print(f"  quantprobe {ver} resolves from {loc}")
    assert "site-packages" in loc, f"resolved from a source tree, not an install: {loc}"
    # verifying a stale install is verifying the wrong code - this run caught exactly that
    # (installed 1.10.0 while the repo was 1.10.2) and passed anyway. Not any more.
    here = os.path.dirname(os.path.abspath(__file__))
    repo_ver = None
    for line in open(os.path.join(here, "quantprobe", "__init__.py"), encoding="utf-8"):
        if line.startswith("__version__"):
            repo_ver = line.split('"')[1]; break
    assert repo_ver == ver, (f"installed {ver} but repo is {repo_ver} - you are verifying stale "
                             f"code. Re-install first: python -m pip install --user .")


def layer4_anchors():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "tests"))
    import smoke
    smoke.t_measured_anchors_still_retrodicted()
    print(f"  {len(smoke.MEASURED_ANCHORS)} measured anchors still retrodicted")


def layer3_e2e(gguf, llama_dir):
    """The layer that actually caught the worst bug. Runs the tool's OWN recommendation against
    real llama.cpp and compares predicted vs measured."""
    if not gguf or not os.path.isfile(gguf):
        SKIP.append("E2E: no --gguf given"); return False
    env = dict(os.environ)
    if llama_dir:
        env["QUANTPROBE_LLAMA_DIR"] = llama_dir
    r = subprocess.run([sys.executable, "-m", "quantprobe", "bench", "--gguf", gguf, "--reps", "3"],
                       capture_output=True, text=True, errors="replace", env=env)
    out = r.stdout + r.stderr
    if "not found" in out and "llama" in out.lower():
        SKIP.append("E2E: llama.cpp not available"); return False
    # CROSS-COMMAND CONSISTENCY on a real file. The offline suite cannot catch this: the
    # file-size calibration path only runs when a GGUF exists, and a double correction there
    # made bench 11% optimistic while plan was 1.4% accurate on identical input.
    pr = subprocess.run([sys.executable, "-m", "quantprobe", "plan", "--gguf", gguf],
                        capture_output=True, text=True, errors="replace", env=env)
    mp = re.search(r"\*\s+([0-9.]+) tok/s", pr.stdout + pr.stderr)
    assert mp, "plan printed no winning row for the E2E file"
    m = re.search(r"measured: ([0-9.]+) \+/- ([0-9.]+) tok/s \(predicted ([0-9.]+), ([+-][0-9]+)%\)", out)
    assert m, "E2E produced no predicted-vs-measured line:\n" + out[-400:]
    meas, err, pred, delta = float(m[1]), float(m[2]), float(m[3]), int(m[4])
    plan_tps = float(mp.group(1))
    assert abs(pred - plan_tps) / plan_tps < 0.01, (
        f"plan and bench disagree on the same file: plan {plan_tps} vs bench {pred}")
    print(f"  plan and bench agree at {plan_tps} tok/s")
    print(f"  predicted {pred}, measured {meas} +/- {err}  ({delta:+d}%)")
    assert err <= meas * 0.15, f"measurement too noisy to trust ({err/meas*100:.0f}% spread) - re-run warm"
    assert abs(delta) <= 25, f"prediction outside the stated +/-25% band: {delta:+d}%"


def main():
    ap = argparse.ArgumentParser(description="pre-release verification gate")
    ap.add_argument("--gguf", default=os.environ.get("QUANTPROBE_VERIFY_GGUF"))
    ap.add_argument("--llama-dir", default=os.environ.get("QUANTPROBE_LLAMA_DIR"))
    a = ap.parse_args()

    step("layer 1: unit + invariant tests", layer1_tests)
    step("layer 2: installed artifact", layer2_installed_artifact)
    step("layer 3: end-to-end vs real llama.cpp", lambda: layer3_e2e(a.gguf, a.llama_dir))
    step("layer 4: measured anchors", layer4_anchors)

    print("\n" + "=" * 60)
    if SKIP:
        print("SKIPPED (a skip is not a pass):")
        for s in SKIP:
            print("  - " + s)
    if FAIL:
        print(f"\n{len(FAIL)} LAYER(S) FAILED — do not release:")
        for n, e in FAIL:
            print(f"  {n}: {e}")
        sys.exit(1)
    print("all layers passed" + (" (with skips above)" if SKIP else ""))


if __name__ == "__main__":
    main()
