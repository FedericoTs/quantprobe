"""`quantprobe bench` must not turn a FAILED llama-bench run into a measurement.

`bench` ran llama-bench with `check=False` and then never looked at `.returncode`. llama-bench
prints its result table as it goes, so a process that dies AFTER emitting one `tg32 | x +/- y`
row - a teardown segfault, a CUDA OOM part-way through the sweep, an external kill - still left a
parseable number in the captured output. That number was parsed, printed as `measured: ... tok/s`,
compared against the prediction, and under `--contribute` packaged into a pre-filled eta data
point. A run that did not complete is not a measurement, and the one thing this project cannot
ship is a number whose provenance is a crash.

The caller is `quantprobe/cli.py` -> `runtime.bench(a)` (the `bench` subcommand), so the failure
has to reach the user as a non-zero exit, not as a printed note the shell cannot see.

These tests drive the real `runtime.bench` (real `best_flags`, real `find_llama`, real
`calibrate.load`); only llama-bench itself is substituted - either by a fake `subprocess.run`, or
by a tiny synthetic executable that prints a parseable table and then exits non-zero.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import stat
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantprobe import runtime

# One llama-bench sweep that DID print a number. Every case below feeds exactly this text, so the
# only thing separating a scored run from a refused one is the exit status.
PARSEABLE_OUTPUT = (
    "| model                | size     |  params | backend | ngl |  test |            t/s |\n"
    "| -------------------- | -------: | ------: | ------- | --: | ----: | -------------: |\n"
    "| llama 13B Q4_K - M   | 7.33 GiB | 13.02 B | CUDA    |  99 |  tg32 |   41.23 ± 0.35 |\n"
)
MEASURED_TPS, MEASURED_ERR = 41.23, 0.35
FATAL_TAIL = "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4096.00 MiB failed\n"
# Where the failure report stops explaining and starts quoting llama-bench verbatim. Everything
# before it is quantprobe speaking; the raw tail after it may of course contain the dead run's
# own numbers.
TAIL_MARKER = "last output"


_TEMP_DIRS = []


def _temp_dir(prefix):
    directory = tempfile.TemporaryDirectory(prefix=prefix)
    _TEMP_DIRS.append(directory)
    return directory.name


def _run_check(fn):
    try:
        return fn()
    finally:
        while _TEMP_DIRS:
            _TEMP_DIRS.pop().cleanup()


class _FakeSubprocess:
    """Stands in for the `subprocess` module inside runtime - for `run` only."""

    def __init__(self, returncode, stdout="", stderr=""):
        self._rc, self._out, self._err = returncode, stdout, stderr
        self.calls = []

    def run(self, cmd, **kw):
        self.calls.append((list(cmd), kw))
        return subprocess.CompletedProcess(
            args=list(cmd), returncode=self._rc, stdout=self._out, stderr=self._err
        )

    def __getattr__(self, name):  # anything else stays the real module
        return getattr(subprocess, name)


def _llama_dir(rc=None, body=PARSEABLE_OUTPUT):
    """A --llama-dir for the real `find_llama` to resolve.

    With `rc` None it holds an inert placeholder (the process is faked, so the file is only ever
    stat'd). With an `rc` it holds a real executable that prints `body` and exits `rc`.
    """
    d = _temp_dir(prefix="qp-bench-exit-bin-")
    p = os.path.join(d, runtime.exe("llama-bench"))
    if rc is None:
        open(p, "w").close()
        return d
    # "+/-" not "±": the stub's bytes go through the child's locale decoding, and the regex
    # accepts both spellings. The in-process fake keeps the unicode form.
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"#!/bin/sh\ncat <<'QPEOF'\n{body.replace('±', '+/-')}QPEOF\nexit {rc}\n")
    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return d


def _args(**over):
    """A `quantprobe bench` namespace exactly as cli.py's parser builds one.

    Explicit hardware keeps `best_flags` off the auto-detect path, so the placement is a pure
    function of these numbers and the test cannot depend on the machine running it. The GGUF path
    is inside a fresh temp dir and does not exist: autospec declines, and llama-bench - the thing
    that would read the file - is what is being substituted.
    """
    a = argparse.Namespace(
        cmd="bench",
        gguf=os.path.join(_temp_dir(prefix="qp-bench-exit-"), "synthetic-13b-q4_k_m.gguf"),
        model=None,
        machine=None,
        bits=4.0,
        total=13.0,
        active=13.0,
        always_active=None,
        vram=24.0,
        vram_bw=900.0,
        ram=64.0,
        ram_bw=80.0,
        disk_bw=3.0,
        ctx=0,
        kv_per_pos=None,
        llama_dir=None,
        dry=False,
        no_anchors=True,
        reps=3,
        depth=None,
        contribute=False,
    )
    for k, v in over.items():
        setattr(a, k, v)
    if a.llama_dir is None and not a.dry:
        a.llama_dir = _llama_dir()
    assert not os.path.exists(a.gguf)
    return a


class _Result:
    def __init__(self, out, emitted, exc):
        self.out, self.emitted, self.exc = out, emitted, exc

    @property
    def failed(self):
        return self.exc is not None

    @property
    def scored(self):
        """Did this run present itself to the user as a completed measurement?"""
        return "measured:" in self.out


def _bench(a, fake=None):
    """Drive the real runtime.bench, recording stdout and every _emit_contribution call.

    _emit_contribution is spied on, not stubbed: it still runs, so the positive cases exercise the
    real payload path.
    """
    emitted = []
    real_emit, real_sub = runtime._emit_contribution, runtime.subprocess

    def spy(*args, **kw):
        emitted.append(args)
        return real_emit(*args, **kw)

    runtime._emit_contribution = spy
    if fake is not None:
        runtime.subprocess = fake
    buf, exc = io.StringIO(), None
    try:
        with contextlib.redirect_stdout(buf):
            try:
                runtime.bench(a)
            except SystemExit as e:
                exc = e
    finally:
        runtime._emit_contribution = real_emit
        runtime.subprocess = real_sub
    result = _Result(buf.getvalue(), emitted, exc)
    result.calls = None if fake is None else fake.calls
    return result


def _assert_refused(r, rc_shown):
    """A failed llama-bench must exit non-zero and must never look like a measurement."""
    assert r.failed, f"exit {rc_shown} was not reported at all; bench returned normally:\n{r.out}"
    code = r.exc.code
    assert code not in (0, None), f"exit {rc_shown} reported with a success exit code {code!r}"
    msg = r.out + ("" if isinstance(code, int) else str(code))
    assert "llama-bench FAILED:" in msg and "No data point was taken" in msg, msg
    if r.calls is not None:
        assert r.calls, "failure occurred before the synthetic benchmark was reached"
    assert not r.scored, f"exit {rc_shown} still printed a measurement:\n{r.out}"
    assert not r.emitted, f"exit {rc_shown} still reached _emit_contribution: {r.emitted}"
    assert str(MEASURED_TPS) not in msg.split(TAIL_MARKER, 1)[0], (
        f"the tok/s parsed out of a failed run is presented as a result:\n{msg}"
    )
    return msg


# --- the defect ------------------------------------------------------------------------------


def t_bench_refuses_a_failed_run_that_printed_a_number():
    """exit 1 after a parseable tg row: refused, not scored."""
    r = _bench(_args(), _FakeSubprocess(1, PARSEABLE_OUTPUT, FATAL_TAIL))
    msg = _assert_refused(r, 1)
    assert "llama-bench" in msg, f"the failure message does not name the tool:\n{msg}"
    assert "1" in msg, f"the failure message does not carry the exit code:\n{msg}"


def t_a_failed_run_never_reaches_the_contribution_payload():
    """--contribute is the sharp end: a crashed run must not become a published eta point."""
    r = _bench(_args(contribute=True), _FakeSubprocess(1, PARSEABLE_OUTPUT, FATAL_TAIL))
    _assert_refused(r, 1)
    assert "issues/new" not in r.out, f"a failed run was offered for contribution:\n{r.out}"


def t_signal_style_returncodes_are_reported():
    """POSIX reports a killed child as a NEGATIVE returncode. `!= 0` is the only correct test - a
    `> 0` check would wave through every segfault and SIGKILL, which is exactly the class of
    llama-bench failure that prints a table on its way out."""
    for rc in (-11, -9):
        r = _bench(_args(contribute=True), _FakeSubprocess(rc, PARSEABLE_OUTPUT))
        msg = _assert_refused(r, rc)
        assert str(rc) in msg or "signal" in msg.lower(), (
            f"returncode {rc} is not identifiable from the failure message:\n{msg}"
        )


def t_a_failed_run_keeps_bounded_raw_output():
    """The tail is the whole diagnostic value of a failed run - and the only thing between the
    user and a multi-megabyte backend dump if it is unbounded."""
    noise = "".join(f"ggml debug line {i}\n" for i in range(4000))
    r = _bench(_args(), _FakeSubprocess(1, noise + PARSEABLE_OUTPUT, FATAL_TAIL))
    msg = _assert_refused(r, 1)
    assert "allocating 4096.00 MiB failed" in msg, f"the fatal tail was dropped:\n{msg[-800:]}"
    assert "ggml debug line 0\n" not in msg, "the whole 4000-line log was replayed"
    assert len(msg) < 4000, f"the failure report is unbounded ({len(msg)} chars)"


def t_the_failure_is_raised_before_the_output_is_interpreted():
    """Order matters: the guard belongs before parsing, calibration and residency, not after. A
    failed run must not be stamped with a machine state on its way to being discarded."""
    r = _bench(_args(), _FakeSubprocess(1, PARSEABLE_OUTPUT))
    _assert_refused(r, 1)
    assert "machine state" not in r.out and "uncalibrated" not in r.out, (
        f"a failed run was still stamped with a machine state:\n{r.out}"
    )


# --- what must NOT change --------------------------------------------------------------------


def t_a_clean_run_is_still_measured_and_still_contributes():
    """The positive control. rc 0 keeps every line of the shipped behaviour."""
    r = _bench(_args(contribute=True), _FakeSubprocess(0, PARSEABLE_OUTPUT))
    assert not r.failed, f"a clean run must not raise: {r.exc}"
    assert r.scored and str(MEASURED_TPS) in r.out, f"rc 0 stopped reporting:\n{r.out}"
    assert r.emitted, "rc 0 with --contribute no longer emits the data point"
    assert "issues/new" in r.out, "the pre-filled issue link vanished from a clean run"
    _a, best, meas, err, _delta = r.emitted[0]
    assert (meas, err) == (MEASURED_TPS, MEASURED_ERR), f"contributed the wrong pair: {meas} {err}"
    assert best[1] > 0, "the contributed prediction is empty"


def t_a_clean_run_without_contribute_still_invites_one():
    r = _bench(_args(), _FakeSubprocess(0, PARSEABLE_OUTPUT))
    assert not r.failed and r.scored, f"rc 0 regressed:\n{r.out}"
    assert not r.emitted, "--contribute was not asked for, but a payload was emitted"
    assert "--contribute" in r.out, "the invitation to contribute disappeared"


def t_a_clean_run_with_unparseable_output_still_prints_the_tail():
    """rc 0 with nothing to parse is a DIFFERENT condition (llama-bench changed its table), and it
    keeps its own, non-fatal message."""
    r = _bench(_args(), _FakeSubprocess(0, "no table here\nnothing to see\n"))
    assert not r.failed, f"an unparseable but successful run must not become fatal: {r.exc}"
    assert "could not parse" in r.out, f"the unparseable-output path changed:\n{r.out}"


def t_dry_mode_still_runs_nothing():
    fake = _FakeSubprocess(1, PARSEABLE_OUTPUT)
    r = _bench(_args(dry=True, contribute=True), fake)
    assert not r.failed, f"--dry must not fail: {r.exc}"
    assert not fake.calls, "--dry spawned a process"
    assert not r.emitted and not r.scored, "--dry produced a measurement"
    assert "[quantprobe] bench:" in r.out, "--dry stopped previewing the command"


def t_the_llama_bench_command_is_unchanged():
    """The guard must not touch what is executed: a real run's argv has to stay identical to the
    one --dry advertises - flags, -n/-p/-r and --mmap included."""
    a = _args()  # ONE namespace, previewed and then executed - same model path, same flags
    a.dry = True
    dry = _bench(a)
    preview = [ln for ln in dry.out.splitlines() if ln.startswith("[quantprobe] bench:")]
    assert preview, f"no previewed command:\n{dry.out}"
    a.dry = False
    fake = _FakeSubprocess(0, PARSEABLE_OUTPUT)
    _bench(a, fake)
    assert len(fake.calls) == 1, f"expected exactly one llama-bench spawn, got {len(fake.calls)}"
    cmd, kw = fake.calls[0]
    shown = preview[0].split(": ", 1)[1].split()
    assert cmd[1:] == shown[1:], f"executed argv drifted from the preview:\n{cmd}\n{shown}"
    assert cmd[1] == "-m" and "-n" in cmd and "-r" in cmd and "--mmap" in cmd, f"argv shape: {cmd}"
    assert kw.get("capture_output") and kw.get("text"), f"capture flags changed: {kw}"


# --- the same thing end to end, with a real process -------------------------------------------


def t_a_real_failing_executable_is_refused_and_a_passing_one_is_not():
    """Real find_llama, real child process, real exit status - no subprocess substitution."""
    if os.name == "nt":
        return "SKIP: the /bin/sh stub is POSIX-only"
    r = _bench(_args(llama_dir=_llama_dir(rc=1), contribute=True))
    msg = _assert_refused(r, 1)
    assert "llama-bench" in msg, f"the failure message does not name the tool:\n{msg}"
    ok = _bench(_args(llama_dir=_llama_dir(rc=0), contribute=True))
    assert not ok.failed and ok.scored and ok.emitted, (
        f"the same stub exiting 0 must still be measured:\n{ok.out}"
    )


def t_the_cli_bench_subcommand_exits_nonzero():
    """The actual caller: cli.py -> runtime.bench. The shell has to see the failure."""
    if os.name == "nt":
        return "SKIP: the /bin/sh stub is POSIX-only"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    gone = os.path.join(_temp_dir(prefix="qp-bench-exit-cli-"), "synthetic-13b.gguf")
    argv = [
        sys.executable,
        "-m",
        "quantprobe.cli",
        "bench",
        "--gguf",
        gone,
        "--total",
        "13",
        "--active",
        "13",
        "--bits",
        "4",
        "--vram",
        "24",
        "--vram-bw",
        "900",
        "--ram",
        "64",
        "--ram-bw",
        "80",
        "--disk-bw",
        "3",
        "--contribute",
        "--llama-dir",
        _llama_dir(rc=1),
    ]
    p = subprocess.run(
        argv, capture_output=True, text=True, errors="replace", env=env, timeout=120, check=False
    )
    out = p.stdout + p.stderr
    assert p.returncode != 0, f"`quantprobe bench` exited 0 on a failed llama-bench:\n{out}"
    assert "measured:" not in out, f"a crashed run was reported as measured:\n{out}"
    assert "issues/new" not in out, f"a crashed run was offered for contribution:\n{out}"
    assert "llama-bench" in out, f"the failure does not name the tool:\n{out}"


def t_failed_unparseable_output_is_not_a_successful_format_change():
    r = _bench(_args(), _FakeSubprocess(1, "no result table here\n"))
    _assert_refused(r, 1)
    assert "could not parse" not in r.out


# --- runners ----------------------------------------------------------------------------------

CHECKS = [v for k, v in sorted(globals().items()) if k.startswith("t_")]


def run_smoke():
    """Entry point for tests/smoke.py - plain asserts, no pytest."""
    assert CHECKS, "benchmark regressions must not be empty"
    results = [_run_check(c) for c in CHECKS]
    skips = [r for r in results if isinstance(r, str) and r.startswith("SKIP:")]
    return f"SKIP: {len(skips)}/{len(CHECKS)} subchecks: " + "; ".join(skips) if skips else None


def _pytest_case(fn):
    def test():
        r = _run_check(fn)
        if isinstance(r, str) and r.startswith("SKIP:"):
            import pytest

            pytest.skip(r[5:].strip())

    test.__name__ = "test_" + fn.__name__[2:]
    test.__doc__ = fn.__doc__
    return test


for _check in CHECKS:
    _case = _pytest_case(_check)
    globals()[_case.__name__] = _case
del _check, _case
