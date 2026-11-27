"""`fetch` must be able to write to the destination it was handed.

`quantprobe fetch <repo> <dest> <file>` dispatches through ``fetch.run``, which passes
``--dest`` straight to ``fetch()`` - unlike the ``python -m quantprobe.fetch`` entry point,
which has always done ``os.makedirs(dest)`` first. So the CLI died on ``open(part, mode)``
with FileNotFoundError whenever dest did not exist yet, and also whenever the *remote*
filename carried its own subdirectory (repos nest split shards under a quant folder), which
no dest-level mkdir would have covered either.

Synthetic HTTP throughout: no network, no real repo, no credential read.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import tempfile
from unittest import mock

from quantprobe import fetch as fmod

BODY = b"GGUF" + b"\x00" * 60


class _Response:
    """Only the surface `fetch` touches: status, Content-Length, chunked body."""

    def __init__(self, body, status=200):
        self._body, self.status_code = body, status
        self.headers = {"Content-Length": str(len(body))}

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i : i + n]


@contextlib.contextmanager
def synthetic_http(body=BODY, forbid_get=False):
    """Stub requests.head/get for the duration; count the calls so a "skip" that quietly
    re-downloads cannot pass."""
    calls = {"head": 0, "get": 0}

    def _head(url, **kw):
        calls["head"] += 1
        return _Response(body)

    def _get(url, **kw):
        calls["get"] += 1
        if forbid_get:
            raise AssertionError("fetch re-downloaded a file it reported as already complete")
        return _Response(body)

    with (
        mock.patch.object(fmod.requests, "head", _head),
        mock.patch.object(fmod.requests, "get", _get),
    ):
        yield calls


def test_fetch_creates_a_missing_destination_directory():
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "weights", "gguf")
        with synthetic_http() as calls:
            assert fmod.fetch("org/repo", dest, "model.gguf", None) is True
        out = os.path.join(dest, "model.gguf")
        with open(out, "rb") as f:
            assert f.read() == BODY
        assert not os.path.exists(out + ".part"), ".part must be renamed away on completion"
        assert calls["get"] == 1


def test_fetch_creates_the_subdirectory_named_by_the_remote_filename():
    # The filename is a remote path, not a bare basename - `UD-Q2_K_XL/...` is how the split
    # shards of a published build are addressed, and it needs a directory under dest.
    fname = "UD-Q2_K_XL/model-00001-of-00002.gguf"
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "weights")
        with synthetic_http():
            assert fmod.fetch("org/repo", dest, fname, None) is True
        with open(os.path.join(dest, "UD-Q2_K_XL", "model-00001-of-00002.gguf"), "rb") as f:
            assert f.read() == BODY


def test_fetch_still_writes_into_a_destination_that_already_exists():
    with tempfile.TemporaryDirectory() as dest:
        with synthetic_http():
            assert fmod.fetch("org/repo", dest, "model.gguf", None) is True
        assert os.path.getsize(os.path.join(dest, "model.gguf")) == len(BODY)


def test_an_already_complete_output_is_still_skipped_untouched():
    # U-18's skip must survive: a matching-size file on disk is not re-downloaded and not
    # rewritten, and creating parents must not become a reason to touch it.
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "weights")
        os.makedirs(dest)
        out = os.path.join(dest, "model.gguf")
        with open(out, "wb") as f:
            f.write(BODY)
        before = os.stat(out)
        with synthetic_http(forbid_get=True) as calls:
            assert fmod.fetch("org/repo", dest, "model.gguf", None) is True
        assert calls["get"] == 0
        after = os.stat(out)
        assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)


def test_a_parent_path_blocked_by_a_file_still_raises():
    # A regular file sitting where a directory must go is a real error. It must stay an error -
    # not be swallowed, and not clobber the file that is in the way.
    with tempfile.TemporaryDirectory() as tmp:
        blocker = os.path.join(tmp, "weights")
        with open(blocker, "wb") as f:
            f.write(b"not a directory")
        with synthetic_http() as calls:
            try:
                fmod.fetch("org/repo", blocker, "model.gguf", None)
            except OSError:
                pass
            else:
                raise AssertionError("a file blocking the parent path must not be worked around")
        assert calls == {"head": 0, "get": 0}, "blocked parents must fail before HTTP"
        assert os.path.isfile(blocker)
        with open(blocker, "rb") as f:
            assert f.read() == b"not a directory"


def test_run_writes_to_the_destination_the_cli_handed_it():
    # The actual dispatch path: `quantprobe fetch org/repo <missing dest> <sub/file.gguf>`.
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "weights")
        a = argparse.Namespace(
            repo="org/repo", dest=dest, files=["UD-Q2_K_XL/model.gguf"], force=False
        )
        with synthetic_http(), mock.patch.object(fmod, "token", lambda: None):
            try:
                fmod.run(a)
            except SystemExit as e:
                assert e.code == 0, f"fetch.run exited {e.code}, expected 0"
            else:
                raise AssertionError("fetch.run must exit")
        assert os.path.isfile(os.path.join(dest, "UD-Q2_K_XL", "model.gguf"))


def run_smoke():
    """Run every destination regression without requiring pytest."""
    ran = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
            except Exception as exc:
                raise AssertionError(f"{name}: {exc}") from exc
            ran += 1
    assert ran == 6, f"expected 6 destination cases, ran {ran}"
