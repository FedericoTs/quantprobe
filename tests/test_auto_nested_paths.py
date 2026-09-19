"""`auto` must keep the remote subdirectory of every file it downloads.

`fetch(repo, dest, fname)` writes `os.path.join(dest, fname)` - so a repo that keeps its quants
in subfolders (`Q2_K/Model-Q2_K.gguf`, the layout every large GGUF repo uses) lands on disk at
`dest/Q2_K/Model-Q2_K.gguf`. `auto` then rebuilt that path as `dest/basename(path)`, which is a
file that does not exist: the run handoff, the `--run` launch, the `--custom` probe input, the
pasteable `quantize` command and the "already on disk" header read all pointed one directory too
high. The download succeeded and every path printed after it was wrong.

These are behavioural, not source-text checks. The fake download writes a sentinel byte string to
the REAL destination `fetch` would write to, and the assertions ask whether the file `auto` hands
downstream actually exists. Only the network (`list_ggufs`, `fetch`), the GGUF header reader and
the probe/runtime subprocesses are faked; every path expression under test runs for real.

Deliberately NOT covered here: `fetch` creating the missing parent directory. That is a separate
defect with its own tests - these cases pre-create the subdirectory, so they fail on the path
bookkeeping alone and stay red whether or not `fetch` learned to mkdir.

Each scenario is a plain `_case_*(tmpdir)` function so `tests/smoke.py` can run the same
regressions without pytest.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from types import SimpleNamespace
from unittest import mock

REPO = "acme/Nested-GGUF"
TOTAL_B = 0.01  # 10M params: keeps the exact-size fixture file kilobytes, not gigabytes
HW = ["--vram", "24", "--vram-bw", "900", "--ram", "64", "--ram-bw", "80", "--disk-bw", "3"]

NESTED_Q2 = "Q2_K/Model-Q2_K.gguf"
NESTED_Q8 = "Q8_0/Model-Q8_0.gguf"
NESTED_SPLIT_1 = "Q2_K/Model-Q2_K-00001-of-00002.gguf"
NESTED_SPLIT_2 = "Q2_K/Model-Q2_K-00002-of-00002.gguf"
FLAT_Q4 = "Model-Q4_K_M.gguf"

#: What the faked header reader returns, shaped like `spec.from_gguf`.
SPEC = {
    "t": TOTAL_B,
    "a": TOTAL_B,
    "ne": TOTAL_B,
    "moe": False,
    "bits": 2.5,
    "kvp": 98304.0,
    "n_layer": 24,
    "codebook_share": 0.0,
}


def _size(bits, total_b=TOTAL_B):
    """Bytes a file must declare for `auto` to read it back as `bits` effective bits."""
    return round(bits * total_b * 1e9 / 8)


def _drive(tmp, target, extra, files, *, local=None):
    """Run `quantprobe auto ...` through cli.main with every external boundary faked.

    Returns what the real path expressions produced: which files were fetched, what `--run` and
    the probe were handed, and whether those paths existed when they were handed over.
    """
    from quantprobe import auto as automod
    from quantprobe import cli as climod
    from quantprobe import fetch as fetchmod
    from quantprobe import probe as probemod
    from quantprobe import runtime as rtmod
    from quantprobe import spec as specmod

    dest = os.path.join(tmp, "models")
    os.makedirs(dest, exist_ok=True)
    # The subdirectories a previous download already left behind, so nothing here depends on
    # fetch() creating them (that is the sibling defect, tested separately).
    for path, _ in files:
        sub = os.path.dirname(path)
        if sub:
            os.makedirs(os.path.join(dest, sub), exist_ok=True)
    if local:
        name, size = local
        with open(os.path.join(dest, name), "wb") as fh:
            fh.write(b"GGUF")
            fh.truncate(size)

    rec = SimpleNamespace(
        dest=dest,
        fetched=[],
        run_gguf=None,
        run_existed=None,
        probe_args=None,
        probe_src_existed=None,
        out="",
    )

    def fake_fetch(repo, dst, fname, tok, tries=100, force=False):
        rec.fetched.append((repo, dst, fname))
        with open(os.path.join(dst, fname), "wb") as fh:  # exactly where fetch() writes
            fh.write(b"GGUF sentinel")
        return True

    def fake_runtime_run(a):
        rec.run_gguf = a.gguf
        rec.run_existed = bool(a.gguf) and os.path.isfile(a.gguf)

    def fake_probe_run(pa):
        rec.probe_args = pa
        rec.probe_src_existed = os.path.isfile(pa.gguf)
        with open(pa.out, "wb") as fh:
            fh.write(b"GGUF depth-aware sentinel")

    def fake_ensure_eval(d):
        p = os.path.join(d, "wiki.test.raw")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("held-out text\n")
        return p

    argv = ["quantprobe", "auto", target, "--dir", dest] + HW + list(extra)
    buf = io.StringIO()
    with contextlib.ExitStack() as st:
        st.enter_context(mock.patch.object(automod, "list_ggufs", lambda repo: list(files)))
        st.enter_context(mock.patch.object(automod, "ensure_eval", fake_ensure_eval))
        st.enter_context(mock.patch.object(fetchmod, "fetch", fake_fetch))
        st.enter_context(mock.patch.object(fetchmod, "token", lambda: None))
        st.enter_context(mock.patch.object(probemod, "run", fake_probe_run))
        st.enter_context(mock.patch.object(rtmod, "run", fake_runtime_run))
        st.enter_context(mock.patch.object(specmod, "from_gguf", lambda p: dict(SPEC)))
        st.enter_context(mock.patch.object(sys, "argv", argv))
        st.enter_context(contextlib.redirect_stdout(buf))
        climod.main()
    rec.out = buf.getvalue()
    return rec


def _case_standard_route_hands_off_the_downloaded_file(tmp):
    """`auto <repo> --run`: the file handed to the runtime must be the one fetch wrote."""
    rec = _drive(tmp, REPO, ["--total", str(TOTAL_B), "--run"], [(NESTED_Q2, _size(2.5))])
    want = os.path.join(rec.dest, "Q2_K", "Model-Q2_K.gguf")

    assert rec.fetched == [(REPO, rec.dest, NESTED_Q2)], rec.fetched
    assert os.path.isfile(want), "the fake download did not land where fetch() writes"
    assert rec.run_gguf == want, f"--run was handed {rec.run_gguf!r}, fetch wrote {want!r}"
    assert rec.run_existed, f"--run was handed a path that does not exist: {rec.run_gguf!r}"
    assert f"quantprobe run --gguf {want}" in rec.out, (
        "the printed ready command names a file that was never written:\n" + rec.out[-500:]
    )


def _case_split_first_part_handoff(tmp):
    """A split download: every part lands under the subfolder, and the handoff is part 1."""
    rec = _drive(tmp, REPO, ["--total", str(TOTAL_B), "--run"], [(NESTED_SPLIT_1, _size(2.5))])
    want = os.path.join(rec.dest, "Q2_K", "Model-Q2_K-00001-of-00002.gguf")

    assert rec.fetched == [
        (REPO, rec.dest, NESTED_SPLIT_1),
        (REPO, rec.dest, NESTED_SPLIT_2),
    ], rec.fetched
    assert os.path.isfile(want)
    assert rec.run_gguf == want, f"--run was handed {rec.run_gguf!r}, fetch wrote {want!r}"
    assert rec.run_existed, f"split handoff points at a missing file: {rec.run_gguf!r}"
    assert f"quantprobe run --gguf {want}" in rec.out, rec.out[-500:]


def _case_flat_remote_path_is_unchanged(tmp):
    """Control: a repo-root file has no subdirectory to lose, and must behave exactly as before."""
    rec = _drive(tmp, REPO, ["--total", str(TOTAL_B), "--run"], [(FLAT_Q4, _size(4.5))])
    want = os.path.join(rec.dest, FLAT_Q4)

    assert rec.fetched == [(REPO, rec.dest, FLAT_Q4)], rec.fetched
    assert rec.run_gguf == want and rec.run_existed
    assert f"quantprobe run --gguf {want}" in rec.out, rec.out[-500:]


def _case_custom_route_probes_the_downloaded_source(tmp):
    """`--custom`: the probe input is the fetched high-precision source, not a phantom sibling."""
    files = [(NESTED_Q8, _size(8.0)), (NESTED_Q2, _size(2.5))]
    rec = _drive(tmp, REPO, ["--total", str(TOTAL_B), "--custom", "--force-custom"], files)
    src = os.path.join(rec.dest, "Q8_0", "Model-Q8_0.gguf")

    assert rec.fetched == [(REPO, rec.dest, NESTED_Q8)], rec.fetched
    assert rec.probe_args is not None, "the probe never ran"
    assert rec.probe_args.gguf == src, f"probe input {rec.probe_args.gguf!r}, fetch wrote {src!r}"
    assert rec.probe_src_existed, f"probe was handed a missing file: {rec.probe_args.gguf!r}"
    # Derived output naming for the NEW artifact stays flat in --dir: it is built here, not
    # fetched, so it has no remote directory to preserve.
    out = os.path.join(rec.dest, "Model-Q8_0-depthaware.gguf")
    assert rec.probe_args.out == out, rec.probe_args.out
    assert f"quantprobe run --gguf {out}" in rec.out, rec.out[-500:]


def _case_custom_atlas_hint_quotes_a_pasteable_path(tmp):
    """The atlas shortcut prints the path the fetch below WILL produce - so it must be that one."""
    rec = _drive(
        tmp, "qwen3-30b", ["--custom", "--force-custom", "--dry"], [(NESTED_Q8, _size(8.0, 30.5))]
    )
    want = os.path.join(rec.dest, "Q8_0", "Model-Q8_0.gguf")

    assert rec.fetched == [], "--dry downloaded something"
    assert f"quantprobe quantize --gguf {want} --recipe qwen3-30b" in rec.out, (
        "the atlas shortcut prints a command the user cannot paste:\n" + rec.out[-800:]
    )


def _case_existing_local_file_is_found_under_its_remote_path(tmp):
    """A completed download already on disk must be read, not missed and re-estimated."""
    size = _size(2.5)
    rec = _drive(
        tmp, REPO, ["--total", str(TOTAL_B), "--dry"], [(NESTED_Q2, size)], local=(NESTED_Q2, size)
    )

    assert rec.fetched == [], "--dry downloaded something"
    assert "[from the file's own header - already on disk]" in rec.out, (
        "the file on disk was not found, so auto quoted a pre-download estimate:\n" + rec.out[-800:]
    )
    assert "NOTE: pre-download estimate from preset params" not in rec.out, rec.out[-800:]
    assert "INCOMPLETE" not in rec.out, rec.out[-800:]


def _case_dry_downloads_nothing(tmp):
    """`--dry` still names the remote path in full, and touches the network for no bytes."""
    rec = _drive(tmp, REPO, ["--total", str(TOTAL_B), "--dry"], [(NESTED_Q2, _size(2.5))])

    assert rec.fetched == [], "--dry downloaded something"
    assert rec.run_gguf is None, "--dry launched the runtime"
    assert NESTED_Q2 in rec.out, rec.out[-500:]
    assert "(--dry: nothing downloaded)" in rec.out, rec.out[-500:]
    assert not os.path.exists(os.path.join(rec.dest, NESTED_Q2))


#: Shared with tests/smoke.py so the suite runs these regressions without pytest.
CASES = [
    _case_standard_route_hands_off_the_downloaded_file,
    _case_split_first_part_handoff,
    _case_flat_remote_path_is_unchanged,
    _case_custom_route_probes_the_downloaded_source,
    _case_custom_atlas_hint_quotes_a_pasteable_path,
    _case_existing_local_file_is_found_under_its_remote_path,
    _case_dry_downloads_nothing,
]


def test_standard_route_hands_off_the_downloaded_file(tmp_path):
    _case_standard_route_hands_off_the_downloaded_file(str(tmp_path))


def test_split_first_part_handoff(tmp_path):
    _case_split_first_part_handoff(str(tmp_path))


def test_flat_remote_path_is_unchanged(tmp_path):
    _case_flat_remote_path_is_unchanged(str(tmp_path))


def test_custom_route_probes_the_downloaded_source(tmp_path):
    _case_custom_route_probes_the_downloaded_source(str(tmp_path))


def test_custom_atlas_hint_quotes_a_pasteable_path(tmp_path):
    _case_custom_atlas_hint_quotes_a_pasteable_path(str(tmp_path))


def test_existing_local_file_is_found_under_its_remote_path(tmp_path):
    _case_existing_local_file_is_found_under_its_remote_path(str(tmp_path))


def test_dry_downloads_nothing(tmp_path):
    _case_dry_downloads_nothing(str(tmp_path))
