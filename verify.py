"""quantprobe verify — the pre-release gate. One command, five layers.

Every bug that reached users this project has shipped was caught by a DIFFERENT layer, and
never by the one before it:

  layer 1  unit + invariant tests    caught: prose-in-flags, missing --no-mmap (retroactively)
  layer 2  installed-artifact check  caught: PATH/permission installs that pass in a repo cwd
  layer 3  END-TO-END with llama.cpp caught: an 82%-below-prediction config that 54 unit tests
                                             sat green through
  layer 4  anchors vs measured       catches: the law quietly ceasing to retrodict reality
  layer 5  findings reach the code   caught: a result we measured, published, and then kept
                                             contradicting in code for a full day

Layer 3 needs a real GGUF and a real llama.cpp. It now FINDS BOTH ITSELF — walking up to the
enclosing checkout for a build (preferring a CUDA one) and picking the smallest real model on
disk. That matters more than it sounds: this gate had been reporting "all layers passed" while
layer 3 silently skipped on every single run, because nobody remembered to pass --llama-dir.
A gate you have to remember to arm is not a gate.

If the end-to-end layer still cannot run, the gate exits 2 rather than 0. A skip is not a pass.
Waive it deliberately with --allow-skip-e2e (a CI box with no GPU); never by accident.

    python verify.py [--gguf FILE --llama-dir DIR] [--allow-skip-e2e]
"""
from __future__ import annotations
import argparse, glob, os, re, subprocess, sys

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
    # THIRD source of truth: pyproject.toml carries the version pip and PyPI actually publish,
    # and this layer never looked at it. Bumping __init__.py alone built a "1.11.0" release as
    # 1.10.5 - caught only because the build printed the filename. Two version strings in one
    # repo will drift; the gate has to compare all of them.
    import re as _re
    pyproj = os.path.join(here, "pyproject.toml")
    pv = _re.search(r'^version\s*=\s*"([^"]+)"', open(pyproj, encoding="utf-8").read(), _re.M)
    assert pv, "pyproject.toml has no version field"
    assert pv.group(1) == repo_ver, (
        f"pyproject.toml says {pv.group(1)} but quantprobe/__init__.py says {repo_ver} - the "
        f"published artifact would carry the wrong version")
    print(f"  version agrees across __init__.py and pyproject.toml: {repo_ver}")


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


def layer5_audit():
    """Does what we MEASURED actually reach what we SHIP?

    Layers 1-4 all check the code against itself or against anchors we chose. None of them can
    catch code that is perfectly self-consistent with a belief we have already disproved - which
    is exactly how the sub-4-bit decode collapse survived: measured false on 2026-07-25,
    published in LAWS.md, and still gating the planner a day later. See audit.py.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import audit
    n, pa = audit.audit_findings_reach_code()
    surface, pb = audit.audit_decision_surface_is_evidenced()
    # The findings register is the third leg of this layer. audit.py checks that scored findings
    # reach the code; findings.py checks that every STAKED pre-registration reaches the register,
    # that no claim is recorded without the scope it was measured in, and that no `wired_into`
    # points at a symbol that no longer exists. Together they close the loop: measured -> recorded
    # -> shipped, with a failure at any hop breaking the release rather than being noticed weeks
    # later by a user.
    import findings
    pc = findings.validate(findings.load())
    problems = pa + pb + pc
    assert not problems, ("what we know and what we ship have drifted apart:\n  "
                          + "\n  ".join(problems))
    reg = findings.load()
    total = sum(len(reg.get(s, [])) for s, _, _ in findings.SECTIONS)
    print(f"  {n} scored findings all declare where they landed; "
          f"{len(surface)} placements all evidenced or declared")
    print(f"  findings register: {total} entries, every pre-registration cited, every scope stated")


def autodiscover_llama():
    """Find a llama.cpp build without being told where it is.

    Layer 3 is the layer that caught the worst bug this project has had, and it was silently
    skipping on every run because nobody passed --llama-dir. A gate you have to remember to arm
    is not a gate. Prefers a CUDA-capable build: a CPU-only binary cannot exercise the GPU
    placements, which is where the errors have actually been.
    """
    # Walk up to the filesystem root: a git worktree lives several levels below the checkout
    # that actually holds tools/, so a fixed two-level walk found nothing here.
    root = os.path.dirname(os.path.abspath(__file__))
    bases, d = [], root
    while True:
        bases.append(d)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    cands = []
    for base in bases:
        for pat in ("tools/llamacpp-*", "tools/llama.cpp*/build*/bin", "llama.cpp/build/bin*",
                    "tools/llama.cpp*/build*/bin/*"):
            cands += glob.glob(os.path.join(base, pat))
    exe = "llama-bench.exe" if os.name == "nt" else "llama-bench"
    found = [d for d in cands if os.path.isfile(os.path.join(d, exe))]
    if not found:
        return None
    # a build shipping a CUDA runtime can measure the GPU rows; prefer it
    cuda = [d for d in found if glob.glob(os.path.join(d, "*cudart*"))
            or glob.glob(os.path.join(d, "*ggml-cuda*"))]
    return (cuda or found)[0]


def main():
    ap = argparse.ArgumentParser(description="pre-release verification gate")
    ap.add_argument("--gguf", default=os.environ.get("QUANTPROBE_VERIFY_GGUF"))
    ap.add_argument("--llama-dir", default=os.environ.get("QUANTPROBE_LLAMA_DIR"))
    ap.add_argument("--allow-skip-e2e", action="store_true",
                    help="exit 0 even if the end-to-end layer could not run (CI without a GPU)")
    a = ap.parse_args()
    if not a.llama_dir:
        a.llama_dir = autodiscover_llama()
        if a.llama_dir:
            print(f"[verify] found llama.cpp: {a.llama_dir}")
    if not a.gguf:
        for d in (os.environ.get("QUANTPROBE_GGUF_DIR"), "D:/evo-compress-data/gguf"):
            hits = sorted(glob.glob(os.path.join(d, "*.gguf"))) if d and os.path.isdir(d) else []
            # An imatrix is calibration data in a .gguf container, not a model - picking one by
            # size gave "using GGUF: qwen35-35b.imatrix.gguf" and a layer that could never run.
            hits = [h for h in hits if "imatrix" not in os.path.basename(h).lower()]
            # smallest real model = fastest honest end-to-end check
            if hits:
                a.gguf = min(hits, key=os.path.getsize)
                print(f"[verify] using GGUF: {os.path.basename(a.gguf)}")
                break

    step("layer 1: unit + invariant tests", layer1_tests)
    step("layer 2: installed artifact", layer2_installed_artifact)
    step("layer 3: end-to-end vs real llama.cpp", lambda: layer3_e2e(a.gguf, a.llama_dir))
    step("layer 4: measured anchors", layer4_anchors)
    step("layer 5: findings reach the code", layer5_audit)

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
    # A skipped end-to-end layer must not read as a pass. This gate reported "all layers passed
    # (with skips above)" and exited 0 on every run where layer 3 never executed - which was
    # every run, because it could not find llama.cpp. Green now means green.
    if any(s.startswith("E2E") for s in SKIP) and not a.allow_skip_e2e:
        print("\nNOT RELEASEABLE: the end-to-end layer never ran, so nothing here has been")
        print("checked against real llama.cpp. Point it at a build with --llama-dir and a model")
        print("with --gguf, or pass --allow-skip-e2e to accept a weaker gate deliberately.")
        sys.exit(2)
    print("all layers passed" + (" (with skips above)" if SKIP else ""))


if __name__ == "__main__":
    main()
