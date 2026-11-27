"""`fetch(..., tries=N)` must bound the number of GET requests actually issued.

A clean-but-incomplete 200/206 - the server answers, the body is short or empty, nothing
raises - never charged the retry counter, so `tries` could not stop the loop: a remote that
keeps replying "here are zero more bytes" kept `fetch` requesting forever, and `quantprobe
auto` (auto.py:531, auto.py:638, both on the default tries=100) hung behind it with no
ceiling. Unknown remote size is the same trap from the other side: `total == 0` makes the
completeness break unreachable, so even a perfectly served body looped.

These tests pin one attempt per issued GET, while keeping partial progress and the ordinary
break-then-resume success. The fake remote raises AttemptCeilingReached instead of looping,
so the pre-fix behaviour fails fast here rather than running to the harness timeout.

Run: python -m pytest tests/test_fetch_attempt_budget.py
Also hooked into the plain-assert suite via tests/smoke.py:t_fetch_attempt_budget.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import time
import unittest

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantprobe import fetch as fmod

TOTAL = 4096
PAYLOAD = bytes(range(256)) * (TOTAL // 256)


class AttemptCeilingReached(Exception):
    """Stand-in for "this would have run forever".

    Deliberately NOT a requests exception: `fetch` catches those and folds them into its own
    retry path, which is exactly the loop under test. This one escapes the function and turns
    an unbounded run into a fast, readable failure.
    """


class FakeResponse:
    def __init__(self, status_code, body=b"", headers=None, break_after=False):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body
        self._break_after = break_after

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]
        if self._break_after:
            raise requests.exceptions.ChunkedEncodingError("connection broken: truncated response")


class FakeRemote:
    """A scripted HF endpoint. Entry i of `script` serves GET i; the last entry repeats.

    Actions: ("bytes", n) serve n bytes from the requested offset and end cleanly;
    ("bytes_then_break", n) serve n then drop the connection; ("rest",) serve to the end of
    the file; ("status", code) answer with an error status; ("timeout",) / ("connerror",)
    fail in transport.
    """

    exceptions = requests.exceptions

    def __init__(self, script, total=TOTAL, ceiling=8):
        self.script = list(script)
        self.total = total
        self.ceiling = ceiling
        self.gets = []  # Range header of each issued GET, None when unranged
        self.heads = 0

    def head(self, url, **kw):
        self.heads += 1
        return FakeResponse(200, headers={"Content-Length": str(self.total)} if self.total else {})

    def get(self, url, headers=None, **kw):
        if len(self.gets) >= self.ceiling:
            raise AttemptCeilingReached(
                f"fetch() issued GET #{len(self.gets) + 1} with a budget of {self.ceiling}: "
                f"the retry counter is not bounding requests"
            )
        rng = (headers or {}).get("Range")
        self.gets.append(rng)
        action = self.script[min(len(self.gets) - 1, len(self.script) - 1)]
        kind = action[0]
        if kind == "timeout":
            raise requests.exceptions.ReadTimeout("read timed out")
        if kind == "connerror":
            raise requests.exceptions.ConnectionError("connection reset by peer")
        if kind == "status":
            return FakeResponse(action[1])
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        end = len(PAYLOAD) if kind == "rest" else min(len(PAYLOAD), start + action[1])
        return FakeResponse(
            206 if rng else 200,
            PAYLOAD[start:end],
            break_after=(kind == "bytes_then_break"),
        )


class NoSleepClock:
    """Real clock, no real waiting: the code under test sleeps 3-5s between retries and these
    tests issue a budget's worth. time() still has to move for the progress printer."""

    def __init__(self):
        self.slept = []

    def time(self):
        return time.time()

    def sleep(self, seconds):
        self.slept.append(seconds)


@contextlib.contextmanager
def wired(remote, clock):
    """Swap the module's transport and clock. `fetch` itself is untouched."""
    real_requests, real_time = fmod.requests, fmod.time
    fmod.requests, fmod.time = remote, clock
    try:
        yield
    finally:
        fmod.requests, fmod.time = real_requests, real_time


class Outcome:
    def __init__(self, ok, gets, part_size, out_bytes, printed, slept):
        self.ok = ok
        self.gets = gets
        self.part_size = part_size
        self.out_bytes = out_bytes
        self.printed = printed
        self.slept = slept


class FetchAttemptBudgetTests(unittest.TestCase):
    FNAME = "model.gguf"

    def run_fetch(self, script, tries, *, total=TOTAL, part=None, ceiling=None):
        if ceiling is None:
            ceiling = max(tries, 0) + 3
        remote = FakeRemote(script, total=total, ceiling=ceiling)
        clock = NoSleepClock()
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            part_path = os.path.join(d, self.FNAME + ".part")
            out_path = os.path.join(d, self.FNAME)
            if part is not None:
                with open(part_path, "wb") as f:
                    f.write(part)
            with wired(remote, clock), contextlib.redirect_stdout(buf):
                ok = fmod.fetch("org/repo", d, self.FNAME, None, tries=tries)
            return Outcome(
                ok,
                list(remote.gets),
                os.path.getsize(part_path) if os.path.exists(part_path) else None,
                open(out_path, "rb").read() if os.path.exists(out_path) else None,
                buf.getvalue(),
                list(clock.slept),
            )

    # --- a clean but incomplete 200/206 must cost an attempt -------------------------------

    def test_short_success_consumes_the_single_attempt(self):
        r = self.run_fetch([("bytes", 512)], tries=1)
        self.assertEqual(len(r.gets), 1, "tries=1 must issue exactly one GET")
        self.assertFalse(r.ok)
        self.assertEqual(r.part_size, 512, "the bytes that did arrive must be kept")
        self.assertIsNone(r.out_bytes, "an incomplete download must not be published")

    def test_short_success_consumes_one_attempt_each(self):
        r = self.run_fetch([("bytes", 512)], tries=2)
        self.assertEqual(len(r.gets), 2, "tries=2 must issue exactly two GETs")
        self.assertFalse(r.ok)
        self.assertEqual(r.gets[1], "bytes=512-", "the second GET must resume, not restart")
        self.assertEqual(r.part_size, 1024, "partial progress accumulates across attempts")

    def test_empty_success_response_consumes_an_attempt(self):
        r = self.run_fetch([("bytes", 0)], tries=3)
        self.assertEqual(len(r.gets), 3, "a zero-byte 200 is still a request against the budget")
        self.assertFalse(r.ok)
        self.assertEqual(r.part_size, 0)

    def test_mixed_failures_and_short_successes_share_one_budget(self):
        r = self.run_fetch(
            [("bytes", 256), ("bytes_then_break", 256), ("status", 503), ("timeout",)], tries=4
        )
        self.assertEqual(len(r.gets), 4, "every issued GET spends one attempt, whatever it does")
        self.assertFalse(r.ok)
        self.assertEqual(r.part_size, 512, "bytes from the surviving attempts are kept")

    def test_unknown_remote_size_fails_finitely(self):
        """No Content-Length means the completeness test can never fire, so the budget is the
        only thing that can end the loop - and the result must be a failure, not a silent
        "complete". Content-Range/HEAD validation is out of scope on this branch."""
        r = self.run_fetch([("bytes", 1024)], tries=3, total=0)
        self.assertEqual(len(r.gets), 3)
        self.assertFalse(r.ok, "an unverifiable transfer must never be accepted as complete")
        self.assertIsNone(r.out_bytes)
        self.assertIn("INCOMPLETE", r.printed)

    # --- what must keep working ------------------------------------------------------------

    def test_resume_after_break_succeeds_on_the_last_allowed_attempt(self):
        r = self.run_fetch([("bytes_then_break", 1024), ("rest",)], tries=2)
        self.assertTrue(r.ok, "a resume that completes within the budget must still succeed")
        self.assertEqual(len(r.gets), 2)
        self.assertEqual(r.gets, [None, "bytes=1024-"])
        self.assertEqual(r.out_bytes, PAYLOAD)
        self.assertIsNone(r.part_size, "the .part is renamed into place on success")

    def test_single_attempt_that_completes_is_not_charged_twice(self):
        r = self.run_fetch([("rest",)], tries=1)
        self.assertTrue(r.ok)
        self.assertEqual(len(r.gets), 1)
        self.assertEqual(r.out_bytes, PAYLOAD)

    def test_complete_part_needs_no_attempt(self):
        r = self.run_fetch([("rest",)], tries=1, part=PAYLOAD)
        self.assertTrue(r.ok)
        self.assertEqual(r.gets, [], "an already-complete .part must not be re-requested")
        self.assertEqual(r.out_bytes, PAYLOAD)

    def test_complete_part_with_zero_tries_still_completes(self):
        """The final publication check still accepts a complete partial without entering the loop."""
        r = self.run_fetch([("rest",)], tries=0, part=PAYLOAD)
        self.assertTrue(r.ok)
        self.assertEqual(r.gets, [])
        self.assertEqual(r.out_bytes, PAYLOAD)

    # --- error paths keep costing exactly one attempt each ---------------------------------

    def test_http_error_status_consumes_exactly_one_attempt(self):
        r = self.run_fetch([("status", 500)], tries=2)
        self.assertEqual(len(r.gets), 2, "an error status must cost one attempt, not two")
        self.assertFalse(r.ok)
        self.assertEqual(r.slept, [5, 5], "the existing backoff per failed status is unchanged")

    def test_transport_failures_consume_exactly_one_attempt(self):
        for action, label in (("timeout", "read timeout"), ("connerror", "connection error")):
            with self.subTest(label):
                r = self.run_fetch([(action,)], tries=3)
                self.assertEqual(len(r.gets), 3, f"{label} must cost one attempt per GET")
                self.assertFalse(r.ok)
                self.assertEqual(r.slept, [3, 3, 3])

    # --- invalid budgets, under the existing contract --------------------------------------

    def test_zero_tries_issues_no_request(self):
        r = self.run_fetch([("rest",)], tries=0)
        self.assertEqual(r.gets, [])
        self.assertFalse(r.ok)

    def test_negative_tries_issues_no_request(self):
        r = self.run_fetch([("rest",)], tries=-3)
        self.assertEqual(r.gets, [])
        self.assertFalse(r.ok)


def run_smoke():
    """Entry point for tests/smoke.py, which runs plain asserts and has no pytest."""
    buf = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FetchAttemptBudgetTests)
    result = unittest.TextTestRunner(stream=buf, verbosity=0).run(suite)
    if not result.wasSuccessful():
        bad = result.failures + result.errors
        first = bad[0][1].strip().splitlines()[-1] if bad else "?"
        raise AssertionError(f"{len(bad)}/{result.testsRun} fetch budget tests failed: {first}")


if __name__ == "__main__":
    unittest.main()
