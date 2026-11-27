"""`ollama ps` row selection for audit-ollama.

loaded_placement() answers "what placement did ollama ACTUALLY pick for THIS model", and
audit-ollama --measure turns that answer into `-ngl` for a llama-bench run and into the
"Nx AVAILABLE" recommendation it prints. Reading the wrong row is therefore not a display
bug: it benches one model's layer split at another model's context depth and reports the
difference as a placement finding.

The fixtures are `ollama ps` output verbatim in shape (NAME / ID / SIZE / PROCESSOR /
CONTEXT / UNTIL, and the older layout without CONTEXT). Nothing runs ollama: the owning
boundary - the `subprocess.run([bin, "ps"])` inside quantprobe.ollama - is stubbed.

Every case here also runs from tests/smoke.py, through run_smoke() at the bottom of this
file. That harness is why pytest is imported optionally: smoke.py is pytest-free by
contract (its first line says so), and the repo asks for `python tests/smoke.py` to stay
green, so the cases have to be runnable with pytest absent.
"""

from __future__ import annotations

import inspect
import sys
import traceback

from quantprobe import ollama as om

try:
    import pytest
except ImportError:  # running under the pytest-free smoke harness; fixtures are bound there
    pytest = None

FAKE_BIN = "/usr/local/bin/ollama"

# Two tags of one model, both resident - an everyday state, and 14b's split is not 7b's.
TWO_TAGS = """\
NAME             ID              SIZE      PROCESSOR          CONTEXT    UNTIL
qwen2.5:14b      7cdf5a0187d5    11 GB     47%/53% CPU/GPU    8192       4 minutes from now
qwen2.5:7b       845dbda0ea48    6.0 GB    16%/84% CPU/GPU    4096       4 minutes from now
"""

# `qwen2.5-coder:7b` begins with `qwen2.5` and is an entirely different set of weights.
PREFIXED_SIBLING = """\
NAME                ID              SIZE      PROCESSOR          CONTEXT    UNTIL
qwen2.5-coder:7b    2b0496514337    6.0 GB    38%/62% CPU/GPU    16384      4 minutes from now
qwen2.5:7b          845dbda0ea48    6.0 GB    100% GPU           4096       4 minutes from now
"""

# A registry host carries its own colon, so a tag-stripped prefix is just the bare host.
CUSTOM_HOST = """\
NAME  ID  SIZE  PROCESSOR  CONTEXT  UNTIL
hub.acme:5000/ml/mistral-small:v3  aa11  14 GB  22%/78% CPU/GPU  8192  4 minutes from now
hub.acme:5000/ml/mistral:v3  bb22  4.4 GB  100% GPU  2048  4 minutes from now
"""

# The same default tag, on a host that carries a port. `:5000` is an address, not a tag.
HOST_PORT_DEFAULT_TAG = """\
NAME  ID  SIZE  PROCESSOR  CONTEXT  UNTIL
hub.example:5000/team/model-mini:latest  aa11  3.1 GB  30%/70% CPU/GPU  2048  4 minutes from now
hub.example:5000/team/model:latest  bb22  8.0 GB  100% GPU  16384  4 minutes from now
"""

# Same host and port, but the resident tag is NOT the default one.
HOST_PORT_OTHER_TAG = """\
NAME  ID  SIZE  PROCESSOR  CONTEXT  UNTIL
hub.example:5000/team/model:v3  cc33  8.0 GB  12%/88% CPU/GPU  4096  4 minutes from now
"""

# `ollama run qwen2.5` loads qwen2.5:latest, and that resolved tag is what ps prints.
DEFAULT_TAG = """\
NAME                ID              SIZE      PROCESSOR          CONTEXT    UNTIL
qwen2.5-coder:7b    2b0496514337    6.0 GB    38%/62% CPU/GPU    16384      4 minutes from now
qwen2.5:latest      845dbda0ea48    4.7 GB    9%/91% CPU/GPU     4096       4 minutes from now
"""

# The three PROCESSOR shapes ollama prints.
EVERY_PROCESSOR_SHAPE = """\
NAME         ID    SIZE      PROCESSOR          CONTEXT    UNTIL
mixed:7b     aa    6.0 GB    16%/84% CPU/GPU    4096       4 minutes from now
allgpu:7b    bb    6.0 GB    100% GPU           32768      2 hours from now
allcpu:7b    cc    6.0 GB    100% CPU           1024       30 seconds from now
"""

# Pre-CONTEXT ollama: same row, one column fewer.
OLD_LAYOUT = """\
NAME            ID              SIZE      PROCESSOR          UNTIL
qwen2.5:7b      845dbda0ea48    6.0 GB    16%/84% CPU/GPU    4 minutes from now
"""

HEADER_ONLY = "NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL\n"

SOMEONE_ELSE = """\
NAME           ID    SIZE      PROCESSOR    CONTEXT    UNTIL


llama3.1:8b    dd    4.9 GB    100% GPU     8192       4 minutes from now
"""


class _Completed:
    """The two attributes loaded_placement touches on a CompletedProcess."""

    def __init__(self, stdout):
        self.stdout = stdout
        self.returncode = 0


def _ps_installer(patch):
    """Build the `ps` fixture on anything exposing monkeypatch's setattr (see _Patch)."""
    calls = []

    def _install(out):
        def fake_run(cmd, **kwargs):
            calls.append((list(cmd), kwargs))
            return _Completed(out)

        patch.setattr(om, "ollama_bin", lambda: FAKE_BIN)
        patch.setattr(om.subprocess, "run", fake_run)
        return calls

    return _install


if pytest is not None:

    @pytest.fixture
    def ps(monkeypatch):
        """Serve a canned `ollama ps` stdout; return the list recording how it was called."""
        return _ps_installer(monkeypatch)


def test_reads_its_own_tag_not_another_tag_of_the_same_model(ps):
    ps(TWO_TAGS)
    assert om.loaded_placement("qwen2.5:7b") == (84, 4096)
    assert om.loaded_placement("qwen2.5:14b") == (53, 8192)


def test_a_prefixed_sibling_is_a_different_model(ps):
    ps(PREFIXED_SIBLING)
    # 100% GPU means audit-ollama recommends nothing; 62% sends it off to bench a split.
    assert om.loaded_placement("qwen2.5:7b") == (100, 4096)
    assert om.loaded_placement("qwen2.5-coder:7b") == (62, 16384)


def test_custom_namespace_and_host_are_matched_whole(ps):
    ps(CUSTOM_HOST)
    assert om.loaded_placement("hub.acme:5000/ml/mistral:v3") == (100, 2048)
    assert om.loaded_placement("hub.acme:5000/ml/mistral-small:v3") == (78, 8192)


def test_an_untagged_request_resolves_to_the_latest_row(ps):
    ps(DEFAULT_TAG)
    assert om.loaded_placement("qwen2.5") == (91, 4096)
    # :latest is the default tag, not a wildcard over the other tags of the same model.
    assert om.loaded_placement("qwen2.5:3b") == (None, None)


def test_an_untagged_request_on_a_host_with_a_port_resolves_to_the_latest_row(ps):
    """A colon in the registry host is an address, not a tag.

    `hub.example:5000/team/model` names a model with no tag - ollama resolves it to
    :latest exactly as it does for a bare `qwen2.5`, and that is the row ps prints. Deciding
    "is this name tagged?" on any colon in the whole string calls the port a tag, so the
    untagged form never normalizes and the placement reads as unknown even while the model
    is resident. The tag, if there is one, lives in the last slash-separated component.
    """
    ps(HOST_PORT_DEFAULT_TAG)
    assert om.loaded_placement("hub.example:5000/team/model") == (100, 16384)
    # The neighbour sharing that host, that namespace and that prefix is a different model:
    # it must neither answer for the request above nor lose its own row.
    assert om.loaded_placement("hub.example:5000/team/model-mini") == (70, 2048)
    assert om.loaded_placement("hub.example:5000/team/model:latest") == (100, 16384)


def test_on_a_host_with_a_port_latest_is_still_a_default_and_not_a_wildcard(ps):
    """Only :latest is filled in. An untagged request does not adopt some other resident tag."""
    ps(HOST_PORT_OTHER_TAG)
    assert om.loaded_placement("hub.example:5000/team/model:v3") == (88, 4096)
    assert om.loaded_placement("hub.example:5000/team/model") == (None, None)


def test_ordinary_rows_still_parse(ps):
    """The CPU/GPU, all-GPU and CPU-only readings are unchanged by row selection."""
    ps(EVERY_PROCESSOR_SHAPE)
    assert om.loaded_placement("mixed:7b") == (84, 4096)
    assert om.loaded_placement("allgpu:7b") == (100, 32768)
    # A CPU-only row reports no GPU percentage (existing semantics): the caller reads a None
    # gpu_pct as "do not compare", which is the right refusal for a model ollama put on CPU.
    assert om.loaded_placement("allcpu:7b") == (None, 1024)


def test_older_ps_layout_without_a_context_column(ps):
    """Pre-CONTEXT ollama prints no depth; placement is still readable, ctx is None."""
    ps(OLD_LAYOUT)
    assert om.loaded_placement("qwen2.5:7b") == (84, None)


def test_header_blank_lines_and_a_model_that_is_not_loaded(ps):
    ps(SOMEONE_ELSE)
    assert om.loaded_placement("qwen2.5:7b") == (None, None)
    # ...and the row that IS there is still reached from behind the header and the blanks.
    assert om.loaded_placement("llama3.1:8b") == (100, 8192)


def test_nothing_loaded_at_all(ps):
    ps(HEADER_ONLY)
    assert om.loaded_placement("qwen2.5:7b") == (None, None)


def test_the_subprocess_boundary_is_ollama_ps_and_nothing_else(ps):
    calls = ps(TWO_TAGS)
    om.loaded_placement("qwen2.5:7b")
    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd == [FAKE_BIN, "ps"]
    assert kwargs.get("timeout")  # an unresponsive daemon must not wedge the audit


def test_no_ollama_on_path_and_a_failing_daemon_are_both_none(monkeypatch):
    monkeypatch.setattr(om, "ollama_bin", lambda: None)
    assert om.loaded_placement("qwen2.5:7b") == (None, None)

    def _boom(*a, **k):
        raise OSError("daemon not running")

    monkeypatch.setattr(om, "ollama_bin", lambda: FAKE_BIN)
    monkeypatch.setattr(om.subprocess, "run", _boom)
    assert om.loaded_placement("qwen2.5:7b") == (None, None)


# --- the pytest-free harness tests/smoke.py runs these cases through --------------------


class _Patch:
    """The two monkeypatch methods these cases use: setattr, and undo on the way out.

    The patched attribute is the real `subprocess.run` (quantprobe.ollama imports the
    module, not the function), so leaving one installed would hand the next smoke test a
    stubbed subprocess. undo() therefore runs in a finally, failure or not.
    """

    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        while self._undo:
            obj, name, old = self._undo.pop()
            setattr(obj, name, old)


_SMOKE_FIXTURES = {"ps": _ps_installer, "monkeypatch": lambda patch: patch}


def smoke_cases():
    """Every test_ case in this module, in definition order. Discovered, never listed."""
    return [(n, f) for n, f in globals().items() if n.startswith("test_") and inspect.isfunction(f)]


def run_smoke(cases=None):
    """Run these cases without pytest. Returns the failure lines - empty is the only pass.

    smoke.py is the harness CONTRIBUTING.md asks to keep green, and it has no pytest. The
    tempting shortcut is to hand-copy a few assertions into it; that is a second copy of
    the intent, and every case added here afterwards stays invisible to it - the failure
    mode this repo already paid for once, when a test sat below the runner and never ran.
    So the smoke hook runs ALL of them, through the same fixtures bound to _Patch.

    A case that raises is recorded and the run continues; nothing is skipped and no
    exception is swallowed. Collecting nothing is itself a failure, because a suite that
    silently shrank to zero would otherwise report green.
    """
    cases = smoke_cases() if cases is None else list(cases)
    if not cases:
        return ["run_smoke: no case was collected - an empty run is not a pass"]
    failures = []
    for name, fn in cases:
        patch = _Patch()
        try:
            kwargs = {}
            for p in inspect.signature(fn).parameters:
                if p not in _SMOKE_FIXTURES:
                    raise TypeError(f"no pytest-free binding for fixture {p!r}")
                kwargs[p] = _SMOKE_FIXTURES[p](patch)
            fn(**kwargs)
        except Exception as e:
            frame = traceback.extract_tb(e.__traceback__)[-1]
            # a bare `assert a == b` carries no message, so quote the line that failed
            detail = str(e) or (frame.line or "").strip()
            failures.append(f"{name}: {type(e).__name__}: {detail} ({frame.lineno})")
        finally:
            patch.undo()
    return failures


def test_every_case_in_this_file_also_runs_under_the_pytest_free_harness():
    """Discovery covers this file, every fixture it asks for has a binding, and they pass.

    Everything except this case, which is excluded only to avoid recursing into itself.
    """
    mine = test_every_case_in_this_file_also_runs_under_the_pytest_free_harness.__name__
    cases = smoke_cases()
    assert mine in [n for n, _ in cases], "discovery does not see the module it lives in"
    for name, fn in cases:
        unbound = [p for p in inspect.signature(fn).parameters if p not in _SMOKE_FIXTURES]
        assert not unbound, f"{name} asks for {unbound}, which smoke.py cannot supply"
    failures = run_smoke([(n, f) for n, f in cases if n != mine])
    assert not failures, failures


def test_the_pytest_free_harness_reports_failures_instead_of_swallowing_them():
    """A harness that cannot go red is worse than no harness: it would report this fix green
    after a regression. Deliberate breakage of each shape, plus the empty-collection guard
    and the restoration of the patched subprocess boundary."""

    def _passes():
        return None

    def _asserts():
        assert False, "deliberate"

    def _raises(ps):
        ps(TWO_TAGS)  # patches the real subprocess.run, then dies before any undo of its own
        raise RuntimeError("boom")

    def _unbindable(nonexistent_fixture):
        return None

    before = om.subprocess.run
    failures = run_smoke(
        [
            ("_passes", _passes),
            ("_asserts", _asserts),
            ("_raises", _raises),
            ("_unbindable", _unbindable),
        ]
    )
    assert len(failures) == 3, failures
    assert "_asserts: AssertionError: deliberate" in failures[0], failures
    assert "_raises: RuntimeError: boom" in failures[1], failures
    assert "_unbindable: TypeError" in failures[2], failures
    assert om.subprocess.run is before, "a failing case left the subprocess boundary stubbed"
    assert run_smoke([]), "collecting no case must not read as a pass"


if __name__ == "__main__":  # runnable on its own; exit code is the verdict
    _failures = run_smoke()
    for _f in _failures:
        print(f"  FAIL  {_f}")
    print(f"{len(smoke_cases())} case(s), {len(_failures)} failure(s)")
    sys.exit(1 if _failures else 0)
