"""`auto --custom --yes` must carry the consent it was given into the probe.

`auto` accepts `--yes` ("skip time-commitment confirmations") and `probe.run` is the step that
asks for one: above ~2h estimated it prompts, and with no tty it exits. `auto._custom` builds the
probe's argument Namespace by hand, so a flag it forgets to copy is simply not there - and the
user who answered the question up front is asked it again AFTER the multi-GB source download,
under an error that tells them to "re-run with --yes", which is what they did.

Hermetic: the HF listing, the download, the eval-corpus fetch and the probe itself are the four
external boundaries, and all four are replaced. No GGUF is read, nothing is fetched, no llama.cpp
runs. What is exercised is the real `cli.main` -> `auto.run` custom pipeline, including the real
optimizer and planner, on a preset machine.
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantprobe import auto as automod

PRESET = "qwen3-30b"
TOTAL_B = automod.MODEL_REPOS[PRESET][1]
REPO = automod.MODEL_REPOS[PRESET][0]
SOURCE = "Qwen3-30B-A3B-Q8_0.gguf"


def _bytes_at(bits):
    """Size a synthetic listing entry so `auto`'s size->bits arithmetic lands on `bits`."""
    return int(bits * TOTAL_B * 1e9 / 8)


# One high-precision source (what --custom requantizes) plus two ordinary quants, so the
# pre-custom ranking loop has something real to score.
LISTING = [
    ("Qwen3-30B-A3B-Q2_K.gguf", _bytes_at(2.7)),
    ("Qwen3-30B-A3B-Q4_K_M.gguf", _bytes_at(4.6)),
    (SOURCE, _bytes_at(8.2)),
]


class _Recorder:
    """The four external boundaries, recorded in call order."""

    def __init__(self, dest):
        self.dest = dest
        self.fetched = []
        self.eval_dirs = []
        self.probe_args = []
        self.order = []

    def fetch(self, repo, dest, fname, tok, tries=100, force=False):
        self.fetched.append((repo, dest, fname))
        self.order.append("fetch")
        return True

    def ensure_eval(self, dest_dir):
        self.eval_dirs.append(dest_dir)
        self.order.append("ensure_eval")
        path = os.path.join(dest_dir, "wiki.test.raw")
        with open(path, "w") as f:
            f.write("synthetic held-out text\n")
        return path

    def probe_run(self, a):
        self.probe_args.append(a)
        self.order.append("probe.run")


def run_auto(*extra):
    """Drive the real CLI so the Namespace `auto` reads is the one argparse actually builds."""
    from quantprobe import cli
    from quantprobe import fetch as fetchmod
    from quantprobe import probe as probemod

    dest = tempfile.mkdtemp(prefix="qp-auto-confirm-")
    rec = _Recorder(dest)
    argv = [
        "quantprobe",
        "auto",
        PRESET,
        "--machine",
        "2016-xmp",
        "--custom",
        "--dir",
        dest,
        *extra,
    ]
    buf = io.StringIO()
    with (
        mock.patch.object(automod, "list_ggufs", lambda _repo: list(LISTING)),
        mock.patch.object(automod, "ensure_eval", rec.ensure_eval),
        mock.patch.object(fetchmod, "fetch", rec.fetch),
        mock.patch.object(fetchmod, "token", lambda: None),
        mock.patch.object(probemod, "run", rec.probe_run),
        mock.patch.object(sys, "argv", argv),
        redirect_stdout(buf),
    ):
        cli.main()
    rec.out = buf.getvalue()
    return rec


class AutoCustomConfirmationTest(unittest.TestCase):
    def run_auto(self, *extra):
        rec = run_auto(*extra)
        self.addCleanup(shutil.rmtree, rec.dest, ignore_errors=True)
        return rec

    def assert_took_custom_path(self, rec):
        # A machine that declined the surgery would skip the custom branch entirely and leave
        # every assertion below vacuously true, so every test checks the branch actually ran.
        self.assertIn(
            "[quantprobe auto --custom] source: " + SOURCE,
            rec.out,
            "the custom branch did not run - the machine gate declined the surgery",
        )

    def test_yes_reaches_probe_run(self):
        rec = self.run_auto("--yes")
        self.assert_took_custom_path(rec)
        self.assertEqual(len(rec.probe_args), 1, "probe.run must be called exactly once")
        pa = rec.probe_args[0]
        self.assertIs(
            getattr(pa, "yes", None),
            True,
            "auto --custom --yes must forward the consent to probe.run, or the probe re-asks "
            "for it after the source download and aborts telling the user to pass --yes",
        )

    def test_without_yes_the_probe_still_confirms(self):
        rec = self.run_auto()
        self.assert_took_custom_path(rec)
        pa = rec.probe_args[0]
        self.assertFalse(
            getattr(pa, "yes", False),
            "no --yes must leave probe.run's confirmation in place - this fix forwards a flag, "
            "it does not grant approval nobody asked for",
        )

    def test_the_flag_changes_nothing_but_the_flag(self):
        """The download steps, the probe's build parameters and the ordering stay identical."""
        plain, consented = self.run_auto(), self.run_auto("--yes")
        for rec in (plain, consented):
            self.assert_took_custom_path(rec)

        self.assertEqual(
            [(r, f) for r, _, f in plain.fetched],
            [(REPO, SOURCE)],
            "the source download must still be the one fetch --custom does",
        )
        self.assertEqual(
            [(r, f) for r, _, f in plain.fetched],
            [(r, f) for r, _, f in consented.fetched],
            "--yes must not change what is downloaded or from where",
        )
        self.assertEqual(
            plain.order,
            ["fetch", "ensure_eval", "probe.run"],
            "--custom must download the source and the eval corpus before probing",
        )
        self.assertEqual(plain.order, consented.order, "--yes must not reorder the pipeline")
        self.assertEqual(
            len(plain.eval_dirs), 1, "the eval corpus is resolved once, in the download directory"
        )

        a_plain, a_yes = vars(plain.probe_args[0]).copy(), vars(consented.probe_args[0]).copy()
        # the two runs use separate temp dirs, so path-valued fields differ by construction
        for k in ("gguf", "workdir", "out", "eval"):
            a_plain.pop(k, None)
            a_yes.pop(k, None)
        differing = {k for k in set(a_plain) | set(a_yes) if a_plain.get(k) != a_yes.get(k)}
        self.assertEqual(
            differing,
            {"yes"},
            "--yes must alter the confirmation flag and nothing else in the probe's arguments",
        )
        self.assertTrue(a_yes["apply"], "the custom build still applies the depth-aware recipe")
        self.assertFalse(a_yes["dry_run"], "a real --custom run is not a dry run")

    def test_dry_still_executes_nothing(self):
        rec = self.run_auto("--yes", "--dry")
        self.assert_took_custom_path(rec)
        self.assertIn("(--dry: nothing downloaded)", rec.out)
        self.assertEqual(rec.order, [], "--dry must not fetch, resolve an eval corpus, or probe")
        self.assertEqual(os.listdir(rec.dest), [], "--dry must leave the download directory empty")


if __name__ == "__main__":
    unittest.main()
