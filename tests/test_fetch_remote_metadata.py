"""`quantprobe fetch` must not publish bytes it did not check, or rename what it did not measure.

Two halves of one defect, because both end at the same place - a size comparison standing in for
"this is the right file":

  * the skip path read Content-Length off a HEAD without looking at the status, so an ERROR
    PAGE's body length decided, by byte count, whether the file on disk was announced complete
    or denounced as "NOT THIS FILE" (and the user sent to `--force`, which deletes it);
  * the resume path appended a 206's body without reading its Content-Range, so bytes delivered
    at the wrong offset landed on top of a good prefix and produced a file of exactly the right
    length and the wrong contents, which the final size check then promoted.

What is deliberately NOT changed: an already-present file whose size cannot be confirmed is
still reused. That is the module's long-standing offline behaviour and `auto` depends on it -
it is just no longer allowed to call itself "complete" or claim the size "matches". Presence is
reported as presence. There is no content authentication here, and these tests pin the wording
as much as the return value, because the wording is the whole difference.

Every fixture is finite: the fake transport answers an exhausted queue with a retryable 503, and
`tries` is small, so a regression cannot turn these into a hang. Nothing touches the network, a
model file, or a real inference run.
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

import requests

from quantprobe import fetch as fmod

REPO, NAME = "org/repo", "model.gguf"
BODY = b"0123456789"  # 10 B "model"; short enough that a wrong offset is visible by eye


class FakeResponse:
    """The part of ``requests.Response`` that `fetch` touches: status, headers, body, close.

    ``iter_content`` ignores the caller's chunk size deliberately. `fetch` asks for 4 MiB and
    these bodies are ten bytes, so a fixed small step is the only way to model a connection
    that breaks PART WAY through a response.
    """

    def __init__(self, status_code=200, headers=None, body=b"", break_after=None, step=3):
        self.status_code = status_code
        self.headers = requests.structures.CaseInsensitiveDict(headers or {})
        self.body = body
        self.break_after = break_after
        self.step = step
        self.closed = False

    def iter_content(self, chunk_size=1):
        sent = 0
        while sent < len(self.body):
            if self.break_after is not None and sent >= self.break_after:
                raise requests.exceptions.ChunkedEncodingError("connection broken")
            chunk = self.body[sent : sent + self.step]
            sent += len(chunk)
            yield chunk

    def close(self):
        self.closed = True


class FakeRequests:
    """Stands in for the ``requests`` module inside quantprobe.fetch."""

    exceptions = requests.exceptions

    def __init__(self, head, gets=()):
        self._head = head
        self._gets = list(gets)
        self.head_calls = 0
        self.get_calls = 0
        self.ranges = []

    def head(self, url, headers=None, allow_redirects=False, timeout=None):
        self.head_calls += 1
        if isinstance(self._head, Exception):
            raise self._head
        return self._head

    def get(self, url, headers=None, stream=False, timeout=None, allow_redirects=False):
        self.get_calls += 1
        self.ranges.append((headers or {}).get("Range"))
        if self._gets:
            return self._gets.pop(0)
        # Queue exhausted: a retryable status, so the bound on the loop stays `tries` and no
        # fixture can spin.
        return FakeResponse(status_code=503)


class FakeClock:
    """Real ``time()`` (the progress printer does arithmetic on it), recorded ``sleep()``."""

    def __init__(self):
        self.slept = []

    def time(self):
        return time.time()

    def sleep(self, seconds):
        self.slept.append(seconds)


class FetchCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="qp-fetch-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.out = os.path.join(self.dir, NAME)
        self.part = self.out + ".part"
        real_requests, real_time = fmod.requests, fmod.time
        self.clock = FakeClock()
        fmod.time = self.clock

        def restore():
            fmod.requests, fmod.time = real_requests, real_time

        self.addCleanup(restore)

    def net(self, head, gets=()):
        self.fake = FakeRequests(head, gets)
        fmod.requests = self.fake
        return self.fake

    def run_fetch(self, tries=2, force=False):
        buf = io.StringIO()
        with redirect_stdout(buf):
            ok = fmod.fetch(REPO, self.dir, NAME, None, tries=tries, force=force)
        self.printed = buf.getvalue()
        return ok

    def write(self, path, data):
        with open(path, "wb") as f:
            f.write(data)

    def read(self, path):
        with open(path, "rb") as f:
            return f.read()

    def assert_reported_as_present_not_verified(self):
        """The reuse is allowed to say presence and nothing more."""
        low = self.printed.lower()
        self.assertIn("already present", low)
        self.assertIn("not verified", low)
        self.assertNotIn("already complete", low)
        self.assertNotIn("size matches", low)


class TestUnverifiableMetadataNeverDecidesByByteCount(FetchCase):
    """An error page's Content-Length must not certify a file, and must not condemn one either.

    The refusal these cases pin is narrow and specific: a length that did not come from a
    successful response is not a length at all, so it is neither compared nor believed. What
    happens next for an already-present file is the unchanged offline behaviour - reuse it, and
    say plainly that this is presence, not verification.
    """

    def test_error_response_length_is_neither_completion_nor_an_accusation(self):
        # The sharp case: a 404's JSON body is 27 B and so is the half-written file on disk.
        # Either verdict reached through that number is reached through an error page.
        self.write(self.out, b"x" * 27)
        fake = self.net(FakeResponse(404, {"Content-Length": "27"}, b"x" * 27))
        self.assertIs(self.run_fetch(), True)
        self.assert_reported_as_present_not_verified()
        self.assertNotIn("NOT THIS FILE", self.printed)
        self.assertEqual(fake.get_calls, 0, "an unverified skip must not open a download")
        self.assertEqual(self.read(self.out), b"x" * 27, "the local file must be left alone")

    def test_non_200_head_is_not_reported_as_a_foreign_file(self):
        # Accusing the user's file of being "not this file" on the strength of an error page's
        # Content-Length is a wrong diagnosis that sends them to --force, which deletes it.
        self.write(self.out, BODY)
        self.net(FakeResponse(403, {"Content-Length": "12"}, b"forbidden!!!"))
        self.assertIs(self.run_fetch(), True)
        self.assertNotIn("NOT THIS FILE", self.printed)
        self.assert_reported_as_present_not_verified()

    def test_missing_content_length_is_presence_only(self):
        self.write(self.out, BODY)
        self.net(FakeResponse(200, {}))
        self.assertIs(self.run_fetch(), True)
        self.assert_reported_as_present_not_verified()

    def test_malformed_content_length_is_refused_not_raised(self):
        # ValueError is not a RequestException: unguarded, "1,024" left fetch() as a traceback.
        self.write(self.out, BODY)
        self.net(FakeResponse(200, {"Content-Length": "not-a-number"}))
        self.assertIs(self.run_fetch(), True)
        self.assert_reported_as_present_not_verified()

    def test_negative_content_length_is_refused(self):
        self.write(self.out, BODY)
        self.net(FakeResponse(200, {"Content-Length": "-1"}))
        self.assertIs(self.run_fetch(), True)
        self.assertNotIn("NOT THIS FILE", self.printed)
        self.assert_reported_as_present_not_verified()

    def test_zero_content_length_is_refused(self):
        # A 0 B "model" is not a size this module can complete against, so it is not one it
        # will quote either - but the file on disk keeps its long-standing reuse.
        self.write(self.out, BODY)
        self.net(FakeResponse(200, {"Content-Length": "0"}))
        self.assertIs(self.run_fetch(), True)
        self.assert_reported_as_present_not_verified()

    def test_failed_head_keeps_the_cached_file_usable(self):
        # The offline case, and the reason the fallback survives: a user with the right file and
        # no network must still be able to use it.
        self.write(self.out, BODY)
        fake = self.net(requests.exceptions.ConnectionError("no route to host"))
        self.assertIs(self.run_fetch(), True)
        self.assert_reported_as_present_not_verified()
        self.assertEqual(fake.get_calls, 0)
        self.assertEqual(self.read(self.out), BODY, "an unreachable remote must not cost the file")

    def test_unverified_reuse_never_promotes_a_partial(self):
        # Presence of `out` is what is being reused. A `.part` next to it is someone else's
        # interrupted download and must stay exactly where it is, under its own name.
        self.write(self.out, BODY)
        self.write(self.part, b"ZZZZZZZZZZZZ")
        self.net(requests.exceptions.ConnectionError("no route to host"))
        self.assertIs(self.run_fetch(), True)
        self.assert_reported_as_present_not_verified()
        self.assertEqual(self.read(self.out), BODY)
        self.assertEqual(self.read(self.part), b"ZZZZZZZZZZZZ", "the partial must not be renamed")

    def test_matching_size_on_a_good_head_is_completion(self):
        self.write(self.out, BODY)
        self.net(FakeResponse(200, {"Content-Length": str(len(BODY))}))
        self.assertIs(self.run_fetch(), True)
        self.assertIn("already complete", self.printed)
        self.assertIn("size matches remote", self.printed)

    def test_size_mismatch_on_a_good_head_still_refuses(self):
        # U-18 stays intact: a same-named file of a different size is not this file.
        self.write(self.out, BODY[:9])
        self.net(FakeResponse(200, {"Content-Length": str(len(BODY))}))
        self.assertIs(self.run_fetch(), False)
        self.assertIn("NOT THIS FILE", self.printed)

    def test_untrusted_head_does_not_start_a_download(self):
        # Nothing is published here yet, so there is no presence to fall back on. Completion for
        # a NEW download IS the byte count, so a download against a bogus total could never be
        # certified - and a `.part` sized to an error page must never be promoted.
        fake = self.net(FakeResponse(500, {"Content-Length": "31"}, b"x" * 31))
        self.assertIs(self.run_fetch(), False)
        self.assertEqual(fake.get_calls, 0, "no bytes should be requested against a bogus size")
        self.assertFalse(os.path.exists(self.part))
        self.assertFalse(os.path.exists(self.out))


class TestRangeIntegrity(FetchCase):
    """What a 206 claims is checked against what we asked for, before anything is written."""

    def head_ok(self, gets):
        return self.net(FakeResponse(200, {"Content-Length": str(len(BODY))}), gets)

    def test_plain_200_download_completes(self):
        self.head_ok([FakeResponse(200, {"Content-Length": "10"}, BODY)])
        self.assertIs(self.run_fetch(), True)
        self.assertEqual(self.read(self.out), BODY)

    def test_valid_206_resume_appends_exactly(self):
        self.write(self.part, BODY[:4])
        fake = self.head_ok([FakeResponse(206, {"Content-Range": "bytes 4-9/10"}, BODY[4:])])
        self.assertIs(self.run_fetch(), True)
        self.assertEqual(self.read(self.out), BODY)
        self.assertEqual(fake.ranges[0], "bytes=4-")

    def test_short_but_consistent_206_keeps_its_progress_and_asks_again(self):
        # A server may answer an open-ended range with LESS than the remainder. That is legal,
        # the bytes are at the offset we asked for, and rolling them back would make some CDNs
        # unresumable. It must be accepted and the loop must re-ask from the new offset.
        self.write(self.part, BODY[:4])
        fake = self.head_ok(
            [
                FakeResponse(206, {"Content-Range": "bytes 4-6/10"}, BODY[4:7]),
                FakeResponse(206, {"Content-Range": "bytes 7-9/10"}, BODY[7:]),
            ]
        )
        self.assertIs(self.run_fetch(tries=3), True)
        self.assertEqual(self.read(self.out), BODY)
        self.assertEqual(fake.ranges, ["bytes=4-", "bytes=7-"], "must resume from 7, not 4")
        self.assertNotIn("rolled back", self.printed, "a legal short range is not a rollback")

    def test_206_at_the_wrong_offset_is_rejected_and_the_prefix_survives(self):
        # Server restarts from 0 while we hold 4 B. Appending gives 10 B - the exact size the
        # completion check wants - of a file that is four bytes of nonsense at the front.
        self.write(self.part, BODY[:4])
        self.head_ok([FakeResponse(206, {"Content-Range": "bytes 0-5/10"}, BODY[:6])])
        self.assertIs(self.run_fetch(), False)
        self.assertFalse(os.path.exists(self.out), "a mis-offset resume must never be promoted")
        self.assertEqual(self.read(self.part), BODY[:4], "the good prefix must be kept as-is")

    def test_206_for_a_different_total_is_rejected(self):
        # The upload changed under us: the remainder is from a file we never measured.
        self.write(self.part, BODY[:4])
        self.head_ok([FakeResponse(206, {"Content-Range": "bytes 4-9/99"}, BODY[4:])])
        self.assertIs(self.run_fetch(), False)
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual(self.read(self.part), BODY[:4])

    def test_206_ending_past_the_end_of_the_file_is_rejected(self):
        # `bytes 4-10/10` addresses an eleventh byte of a ten-byte file. The range is
        # self-contradictory, so nothing it carries is trustworthy - and believing `end` would
        # size the write past the total the completion check uses.
        self.write(self.part, BODY[:4])
        self.head_ok([FakeResponse(206, {"Content-Range": "bytes 4-10/10"}, BODY[4:] + b"X")])
        self.assertIs(self.run_fetch(), False)
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual(self.read(self.part), BODY[:4])

    def test_206_ending_before_its_own_start_is_rejected(self):
        # `bytes 4-3/10` is a negative-length range: it would make the allowed payload -1 B.
        self.write(self.part, BODY[:4])
        self.head_ok([FakeResponse(206, {"Content-Range": "bytes 4-3/10"}, BODY[4:])])
        self.assertIs(self.run_fetch(), False)
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual(self.read(self.part), BODY[:4])

    def test_206_without_a_content_range_is_rejected(self):
        self.write(self.part, BODY[:4])
        self.head_ok([FakeResponse(206, {}, BODY[4:])])
        self.assertIs(self.run_fetch(), False)
        self.assertEqual(self.read(self.part), BODY[:4])

    def test_206_longer_than_its_declared_range_rolls_back(self):
        self.write(self.part, BODY[:4])
        self.head_ok([FakeResponse(206, {"Content-Range": "bytes 4-9/10"}, BODY[4:] + b"XXX")])
        self.assertIs(self.run_fetch(), False)
        self.assertEqual(self.read(self.part), BODY[:4], "excess must be rolled back, not kept")

    def test_206_shorter_than_its_declared_range_rolls_back(self):
        # Ended cleanly at the wrong length: the server contradicting itself, not a broken
        # connection, so the bytes are suspect and the known-good prefix is restored.
        self.write(self.part, BODY[:4])
        self.head_ok([FakeResponse(206, {"Content-Range": "bytes 4-9/10"}, BODY[4:7])])
        self.assertIs(self.run_fetch(), False)
        self.assertEqual(self.read(self.part), BODY[:4])

    def test_interrupted_206_keeps_its_progress_and_resumes(self):
        # The opposite case, and the reason rollback is scoped to clean responses: a transfer
        # that dies mid-body delivered real bytes at the right offset. Keep them.
        self.write(self.part, BODY[:4])
        fake = self.head_ok(
            [
                FakeResponse(206, {"Content-Range": "bytes 4-9/10"}, BODY[4:], break_after=3),
                FakeResponse(206, {"Content-Range": "bytes 7-9/10"}, BODY[7:]),
            ]
        )
        self.assertIs(self.run_fetch(tries=3), True)
        self.assertEqual(self.read(self.out), BODY)
        self.assertEqual(fake.ranges, ["bytes=4-", "bytes=7-"], "must resume from 7, not 4")

    def test_200_answer_to_a_range_request_restarts_instead_of_appending(self):
        # A server that ignores Range sends the whole file; treating it as a continuation
        # doubles the prefix.
        self.write(self.part, b"ZZZZ")
        self.head_ok([FakeResponse(200, {"Content-Length": "10"}, BODY)])
        self.assertIs(self.run_fetch(), True)
        self.assertEqual(self.read(self.out), BODY)


class TestResponseBoundaries(FetchCase):
    def test_excess_bytes_before_transport_failure_cannot_be_published(self):
        class ExcessThenFailure(FakeResponse):
            def iter_content(self, chunk_size=1):
                yield BODY[4:]
                raise requests.exceptions.ReadTimeout("after excess bytes")

        self.write(self.part, BODY[:4])
        response = ExcessThenFailure(206, {"Content-Range": "bytes 4-5/10"})
        self.net(FakeResponse(200, {"Content-Length": "10"}), [response])
        self.assertFalse(self.run_fetch(tries=1))
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual(self.read(self.part), BODY[:4])

    def test_unrepresentable_range_numbers_reject_without_escaping(self):
        self.write(self.part, BODY[:4])
        response = FakeResponse(206, {"Content-Range": "bytes 4-" + "9" * 5000 + "/10"})
        self.net(FakeResponse(200, {"Content-Length": "10"}), [response])
        self.assertFalse(self.run_fetch(tries=1))
        self.assertEqual(self.read(self.part), BODY[:4])

    def test_oversized_partial_is_named_as_such_and_never_promoted(self):
        # `have >= total` ends the loop with no request at all, so this is fully deterministic.
        # The old report was a bare "INCOMPLETE", which is the one thing a 15 B partial of a
        # 10 B file is not - and it named no file and no way out. `--force` is no help either:
        # it only clears a `.part` when a completed file sits beside it, and none does here.
        self.write(self.part, BODY + b"XXXXX")
        fake = self.net(FakeResponse(200, {"Content-Length": "10"}))
        self.assertIs(self.run_fetch(tries=2), False)
        self.assertEqual(fake.get_calls, 0, "an oversized partial needs no request to diagnose")
        self.assertFalse(os.path.exists(self.out), "nothing may be published from it")
        self.assertEqual(self.read(self.part), BODY + b"XXXXX", "its bytes are the user's")
        low = self.printed.lower()
        self.assertIn("oversized", low)
        self.assertNotIn("incomplete", low, "15 B of a 10 B file is not an incomplete download")
        self.assertIn(self.part, self.printed, "the message must name THIS partial, by path")
        self.assertIn("move or delete", low)
        self.assertIn("`--force` does not clear it", self.printed)


class TestMetadataReviewControls(FetchCase):
    def test_force_rejects_untrusted_head_before_deleting_published_file(self):
        # Ordering matters more than the verdict: --force deletes `out` to make room for the
        # download, so the size must be known to be fetchable BEFORE the file is gone.
        self.write(self.out, BODY)
        self.net(FakeResponse(404, {"Content-Length": "17"}))
        self.assertFalse(self.run_fetch(tries=1, force=True))
        self.assertEqual(self.read(self.out), BODY)

    def test_range_unit_and_header_names_are_case_insensitive(self):
        self.write(self.part, BODY[:4])
        self.net(
            FakeResponse(200, {"cOnTeNt-LeNgTh": "10"}),
            [FakeResponse(206, {"content-range": "ByTeS 4-9/10"}, BODY[4:])],
        )
        self.assertTrue(self.run_fetch(tries=1))
        self.assertEqual(self.read(self.out), BODY)

    def test_error_head_cannot_promote_matching_partial(self):
        # The complement of the presence fallback: there is no published file to reuse, only a
        # `.part` whose length happens to equal an error page's. It stays a `.part`.
        self.write(self.part, BODY)
        self.net(FakeResponse(500, {"Content-Length": "10"}))
        self.assertFalse(self.run_fetch(tries=1))
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual(self.read(self.part), BODY)


class TestUpstreamSmokeGuardStillHolds(unittest.TestCase):
    """The one pre-existing test this work touched, run here so the edit is checkable.

    `tests/smoke.py:t_fetch_force_and_collision` (U-18) mocks a HEAD response with nothing but
    `headers`, which no real ``requests.Response`` ever is. Reading a length off that object
    means reading one off something indistinguishable from an error page, so the mock gained a
    `status_code = 200` - the state it was always standing in for. Its two assertions were not
    touched, and this pins that: if the mock edit had relaxed the U-18 refusal instead of
    describing a successful response, the upstream test would fail right here.
    """

    def test_u18_force_and_collision_smoke_test_passes_unchanged(self):
        from tests.smoke import t_fetch_force_and_collision

        buf = io.StringIO()
        with redirect_stdout(buf):
            t_fetch_force_and_collision()


class TestAutoReusesAnAlreadyPresentModel(unittest.TestCase):
    """`quantprobe auto` is the caller that pays for this, so it is the caller that is tested.

    `auto.run` treats a falsy `fetch` as fatal (`SystemExit("download failed...")`), so a skip
    path that refuses an unverifiable HEAD does not degrade gracefully - it ends the command for
    a user who already has the model on disk and no working network. This drives the real
    `quantprobe.fetch.fetch` through the real `auto.run`; only the three external-I/O edges are
    replaced (the HF tree listing, the GGUF header parse, and hardware detection), and no
    provider, model download or inference is involved.
    """

    LISTED = ("model-Q4_K_M.gguf", 4_030_000_000)

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="qp-auto-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.local = os.path.join(self.dir, self.LISTED[0])
        with open(self.local, "wb") as f:
            f.write(b"GGUF-already-on-disk")
        real_requests, real_time = fmod.requests, fmod.time
        fmod.time = FakeClock()
        self.net = FakeRequests(requests.exceptions.ConnectionError("no route to host"))
        fmod.requests = self.net

        def restore():
            fmod.requests, fmod.time = real_requests, real_time

        self.addCleanup(restore)

    def run_auto(self):
        from quantprobe import auto as automod
        from quantprobe import cli as climod
        from quantprobe import detect as detmod

        argv = [
            "quantprobe",
            "auto",
            REPO,
            "--total",
            "7.2",
            "--dir",
            self.dir,
            # explicit hardware: `optimize.resolve` only probes the host when every one of
            # these is None, so detection is never reached. Stubbed as well, so a change to
            # that rule shows up as a wrong number here rather than as a real hardware read.
            "--vram",
            "24",
            "--vram-bw",
            "936",
            "--ram",
            "64",
            "--ram-bw",
            "86",
            "--disk-bw",
            "3",
        ]
        buf = io.StringIO()
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(automod, "list_ggufs", lambda repo: [self.LISTED]),
            mock.patch.object(automod, "local_spec_or_none", lambda *a: (None, None)),
            mock.patch.object(
                detmod,
                "detect",
                lambda *a, **k: (_ for _ in ()).throw(AssertionError("host hardware probed")),
            ),
            redirect_stdout(buf),
        ):
            climod.main()
        return buf.getvalue()

    def test_auto_still_runs_a_present_model_when_the_head_fails(self):
        printed = self.run_auto()
        self.assertIn("ready. Run it:", printed)
        self.assertIn(self.local, printed)
        low = printed.lower()
        self.assertIn("already present", low)
        self.assertIn("not verified", low)
        self.assertNotIn("already complete", low)
        self.assertNotIn("size matches", low)
        self.assertEqual(self.net.head_calls, 1, "the real fetch path must have been taken")
        self.assertEqual(self.net.get_calls, 0, "a present model must not be re-downloaded")
        with open(self.local, "rb") as f:
            self.assertEqual(f.read(), b"GGUF-already-on-disk", "the cached file is untouched")


def run_smoke():
    """Hook for tests/smoke.py, which is pytest-free and collects plain callables."""
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    buf = io.StringIO()
    result = unittest.TextTestRunner(stream=buf, verbosity=0).run(suite)
    if not result.wasSuccessful():
        raise AssertionError(
            f"{len(result.failures)} failed, {len(result.errors)} errored:\n{buf.getvalue()}"
        )
