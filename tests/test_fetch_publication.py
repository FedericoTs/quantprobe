"""Regression tests for `quantprobe fetch --force` publication ordering.

A forced refresh deleted the published file BEFORE the first network call, so any failure
after that point - HEAD, a non-2xx status, a broken stream, or the rename itself - left the
user with neither the new model nor the one that was working. Publication must happen at the
atomic ``os.replace`` and nowhere earlier.

These exercise the real ``quantprobe.fetch`` module against temporary directories, with only
``requests`` and the retry ``sleep`` mocked. Runs under pytest and, via ``run_smoke()``, under
``python tests/smoke.py``.
"""

from __future__ import annotations

import io
import os
import stat
import tempfile
import unittest
from argparse import Namespace
from contextlib import ExitStack, redirect_stdout
from unittest import mock

import requests

from quantprobe import fetch as fmod

OLD = b"OLD-PUBLISHED-MODEL-BYTES"
NEW = b"N" * 1000
OLD_MODE = 0o640


class _Resp:
    """Stand-in for a requests HEAD/GET response."""

    def __init__(self, status=200, length=None, chunks=(), boom=None):
        self.status_code = status
        self.headers = {} if length is None else {"Content-Length": str(length)}
        self._chunks = list(chunks)
        self._boom = boom

    def iter_content(self, _size):
        yield from self._chunks
        if self._boom is not None:
            raise self._boom


class FetchPublicationTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dest = tmp.name
        self.out = os.path.join(self.dest, "model.gguf")
        self.part = self.out + ".part"
        self.gets = []
        # the retry backoff is real seconds; tests must not pay for it
        self._patch(mock.patch.object(fmod.time, "sleep", lambda *_a: None))

    def _patch(self, patcher):
        patcher.start()
        self.addCleanup(patcher.stop)

    # -- fixtures -------------------------------------------------------------------------

    def publish_old(self):
        """A model already on disk and in use, with a mode the user set."""
        with open(self.out, "wb") as f:
            f.write(OLD)
        os.chmod(self.out, OLD_MODE)

    def net(self, head, gets):
        """Mock requests.head/get. `head` and each entry of `gets` is a _Resp or an exception
        to raise; the last GET entry repeats for every further attempt."""

        def _head(*_a, **_k):
            if isinstance(head, BaseException):
                raise head
            return head

        def _get(*_a, **kw):
            self.gets.append(dict(kw.get("headers") or {}))
            r = gets[min(len(self.gets) - 1, len(gets) - 1)]
            if isinstance(r, BaseException):
                raise r
            return r

        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(fmod.requests, "head", _head))
        stack.enter_context(mock.patch.object(fmod.requests, "get", _get))

    def fetch(self, **kw):
        buf = io.StringIO()
        with redirect_stdout(buf):
            ok = fmod.fetch("org/repo", self.dest, "model.gguf", None, **kw)
        return ok, buf.getvalue()

    def assertOldIntact(self):
        """The previously published bytes AND their mode are still the ones on disk."""
        self.assertTrue(os.path.exists(self.out), "the published model was destroyed")
        with open(self.out, "rb") as f:
            self.assertEqual(f.read(), OLD, "the published model was overwritten")
        self.assertEqual(stat.S_IMODE(os.stat(self.out).st_mode), OLD_MODE)

    def read_out(self):
        with open(self.out, "rb") as f:
            return f.read()

    # -- the happy paths, which the fix must not break ------------------------------------

    def test_normal_download_publishes(self):
        self.net(_Resp(length=len(NEW)), [_Resp(200, chunks=[NEW])])
        ok, log = self.fetch()
        self.assertTrue(ok)
        self.assertEqual(self.read_out(), NEW)
        self.assertFalse(os.path.exists(self.part), ".part must be consumed by the rename")
        self.assertIn("DONE", log)

    def test_successful_force_refresh_replaces_the_old_file(self):
        self.publish_old()
        self.net(_Resp(length=len(NEW)), [_Resp(200, chunks=[NEW])])
        ok, _ = self.fetch(force=True)
        self.assertTrue(ok)
        self.assertEqual(self.read_out(), NEW, "a successful force refresh must publish")
        self.assertFalse(os.path.exists(self.part))

    def test_force_resets_a_prior_partial(self):
        """When a published output exists, force starts a fresh replacement."""
        self.publish_old()
        with open(self.part, "wb") as f:
            f.write(b"S" * 500)
        self.net(_Resp(length=len(NEW)), [_Resp(200, chunks=[NEW])])
        ok, _ = self.fetch(force=True)
        self.assertTrue(ok)
        self.assertNotIn("Range", self.gets[0], "forced attempt resumed a stale partial")
        self.assertEqual(self.read_out(), NEW)

    # -- the failure paths: the old bytes are the user's working model --------------------

    def test_force_http_failure_keeps_old_bytes(self):
        self.net(_Resp(length=len(NEW)), [_Resp(503)])
        self.publish_old()
        ok, log = self.fetch(force=True, tries=3)
        self.assertFalse(ok)
        self.assertIn("INCOMPLETE", log)
        self.assertOldIntact()

    def test_force_stream_break_leaves_partial_unpublished(self):
        """Short download: .part exists but its size disagrees with Content-Length, so there is
        nothing to publish and the old file must survive."""
        self.publish_old()
        broken = _Resp(200, chunks=[b"P" * 400], boom=requests.exceptions.ChunkedEncodingError())
        self.net(_Resp(length=len(NEW)), [broken])
        ok, log = self.fetch(force=True, tries=2)
        self.assertFalse(ok)
        self.assertIn("INCOMPLETE", log)
        self.assertEqual(os.path.getsize(self.part), 400, "the short partial should be kept")
        self.assertOldIntact()

    def test_force_head_exception_keeps_old_bytes(self):
        self.publish_old()
        self.net(requests.exceptions.ConnectionError("dns"), [])
        with self.assertRaises(requests.exceptions.ConnectionError):
            self.fetch(force=True)
        self.assertOldIntact()

    def test_publication_failure_keeps_old_bytes(self):
        """Even a complete download must not cost the old file if the rename itself fails."""
        self.publish_old()
        self.net(_Resp(length=len(NEW)), [_Resp(200, chunks=[NEW])])
        self._patch(
            mock.patch.object(
                fmod.os, "replace", mock.Mock(side_effect=OSError("cross-device link"))
            )
        )
        with self.assertRaises(OSError):
            self.fetch(force=True)
        self.assertOldIntact()
        self.assertEqual(os.path.getsize(self.part), len(NEW), "the new bytes should be kept too")

    def test_repeated_force_restarts_replacement_and_preserves_old_until_success(self):
        self.publish_old()
        broken = _Resp(200, chunks=[b"P" * 400], boom=requests.exceptions.ReadTimeout())
        self.net(_Resp(length=len(NEW)), [broken, _Resp(200, chunks=[NEW])])
        ok, _ = self.fetch(force=True, tries=1)
        self.assertFalse(ok)
        self.assertOldIntact()
        self.assertEqual(os.path.getsize(self.part), 400)
        ok, _ = self.fetch(force=True, tries=1)
        self.assertTrue(ok)
        self.assertNotIn("Range", self.gets[1])
        self.assertEqual(self.read_out(), NEW)
        self.assertFalse(os.path.exists(self.part))

    def test_force_without_published_output_resumes_existing_partial(self):
        with open(self.part, "wb") as f:
            f.write(NEW[:400])
        self.net(_Resp(length=len(NEW)), [_Resp(206, chunks=[NEW[400:]])])
        ok, _ = self.fetch(force=True, tries=1)
        self.assertTrue(ok)
        self.assertEqual(self.gets[0].get("Range"), "bytes=400-")
        self.assertEqual(self.read_out(), NEW)
        self.assertFalse(os.path.exists(self.part))

    # -- the CLI contract ------------------------------------------------------------------

    def test_cli_dispatch_still_exits_nonzero_on_failure(self):
        self.publish_old()
        self.net(_Resp(length=len(NEW)), [_Resp(503)])
        self._patch(mock.patch.object(fmod, "token", lambda: None))
        with self.assertRaises(SystemExit) as cm, redirect_stdout(io.StringIO()):
            fmod.run(Namespace(repo="org/repo", dest=self.dest, files=["model.gguf"], force=True))
        self.assertEqual(cm.exception.code, 1)
        self.assertOldIntact()


def run_smoke():
    """Entry point for tests/smoke.py: returns None when green, raises otherwise."""
    suite = unittest.TestLoader().loadTestsFromTestCase(FetchPublicationTest)
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    if result.testsRun == 0 or result.skipped:
        raise AssertionError("publication regressions must run without skips")
    bad = result.failures + result.errors
    if bad:
        raise AssertionError(
            f"{len(bad)}/{result.testsRun} fetch publication test(s) failed: "
            + "; ".join(
                f"{t.id().rsplit('.', 1)[-1]} ({m.strip().splitlines()[-1]})" for t, m in bad
            )
        )


if __name__ == "__main__":
    unittest.main()
