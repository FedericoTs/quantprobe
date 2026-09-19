"""The name audit-ollama reads off a manifest path must be the name ollama answers to.

`installed()` walks manifests/<registry>/<namespace>/<model>/<tag> and hands that name
straight to `ollama run` and `ollama stop`. Keeping only the last two path parts prints a
model ollama cannot resolve for anything that is not on the default registry under
`library` - `hf.co/bartowski/x:Q4_K_M` is reported and run as `x:Q4_K_M`, which errors or,
worse, resolves to a DIFFERENT model of that basename that happens to be pulled. The
audit then prices one file and measures another.

Synthetic manifest+blob trees only: no ollama, no daemon, no network.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from quantprobe.ollama import MODEL_MEDIA, installed

BLOB = b"not a real gguf, only a file the reader must find"


def _store(root, entries):
    """Lay out manifests/<parts...> + the blobs they point at. entries: {path: digest}."""
    for relpath, digest in entries.items():
        mp = os.path.join(root, "manifests", *relpath.split("/"))
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        with open(mp, "w", encoding="utf-8") as fh:
            json.dump(
                {"layers": [{"mediaType": MODEL_MEDIA, "digest": f"sha256:{digest}", "size": 1}]},
                fh,
            )
        blob = os.path.join(root, "blobs", f"sha256-{digest}")
        os.makedirs(os.path.dirname(blob), exist_ok=True)
        with open(blob, "wb") as fh:
            fh.write(BLOB)
    return root


def _names(root):
    return [n for n, _blob, _size in installed(root)]


class TestManifestPathToOllamaName(unittest.TestCase):
    def test_default_registry_and_library_keep_the_shorthand(self):
        """`ollama pull qwen2.5:7b` lands under registry.ollama.ai/library and is addressed
        by the bare name - expanding it here would print an identity no user recognises."""
        with tempfile.TemporaryDirectory() as d:
            _store(d, {"registry.ollama.ai/library/qwen2.5/7b": "aaa"})
            self.assertEqual(_names(d), ["qwen2.5:7b"])

    def test_custom_namespace_on_the_default_registry_is_kept(self):
        """A namespace that is not `library` is part of the name: `ollama run mymodel:latest`
        does not find `myuser/mymodel:latest`."""
        with tempfile.TemporaryDirectory() as d:
            _store(d, {"registry.ollama.ai/myuser/mymodel/latest": "bbb"})
            self.assertEqual(_names(d), ["myuser/mymodel:latest"])

    def test_non_default_registry_host_is_kept(self):
        """hf.co models are addressed by their full path, including the host."""
        with tempfile.TemporaryDirectory() as d:
            _store(d, {"hf.co/bartowski/Qwen2.5-7B-GGUF/Q4_K_M": "ccc"})
            self.assertEqual(_names(d), ["hf.co/bartowski/Qwen2.5-7B-GGUF:Q4_K_M"])

    def test_library_under_a_foreign_registry_is_not_shortened(self):
        """`library` is only implicit on the default registry; elsewhere it is a real
        namespace and dropping it would rename someone else's model."""
        with tempfile.TemporaryDirectory() as d:
            _store(d, {"registry.example.com/library/mistral/latest": "ddd"})
            self.assertEqual(_names(d), ["registry.example.com/library/mistral:latest"])

    def test_same_basename_in_two_namespaces_stays_two_distinct_models(self):
        """The collision that makes the audit lie: two different files, one printed name.

        Whichever row the user acts on, `ollama run llama3:latest` reaches at most one of
        them, while the header numbers above it came from the other blob.
        """
        with tempfile.TemporaryDirectory() as d:
            _store(
                d,
                {
                    "registry.ollama.ai/library/llama3/latest": "1111",
                    "hf.co/someone/llama3/latest": "2222",
                },
            )
            got = {name: os.path.basename(blob) for name, blob, _size in installed(d)}
            self.assertEqual(
                got,
                {
                    "llama3:latest": "sha256-1111",
                    "hf.co/someone/llama3:latest": "sha256-2222",
                },
            )

    def test_tag_defaults_are_not_invented(self):
        """The tag comes from the file name on disk, never from a guess."""
        with tempfile.TemporaryDirectory() as d:
            _store(d, {"registry.ollama.ai/library/phi3/mini-4k-instruct-q4_0": "eee"})
            self.assertEqual(_names(d), ["phi3:mini-4k-instruct-q4_0"])


class TestBrokenStoresStillDegrade(unittest.TestCase):
    """Naming must not cost the reader its tolerance for a store it does not own."""

    def test_missing_store(self):
        self.assertEqual(installed(os.path.join(tempfile.gettempdir(), "no-such-store-qp")), [])

    def test_manifest_without_its_blob_is_not_listed(self):
        with tempfile.TemporaryDirectory() as d:
            mp = os.path.join(d, "manifests", "hf.co", "someone", "ghost")
            os.makedirs(mp)
            with open(os.path.join(mp, "latest"), "w", encoding="utf-8") as fh:
                json.dump(
                    {"layers": [{"mediaType": MODEL_MEDIA, "digest": "sha256:dead", "size": 1}]}, fh
                )
            self.assertEqual(installed(d), [])

    def test_junk_and_layerless_manifests_are_skipped_without_raising(self):
        with tempfile.TemporaryDirectory() as d:
            _store(d, {"registry.ollama.ai/library/real/latest": "fff"})
            with open(os.path.join(d, "manifests", "README.txt"), "w", encoding="utf-8") as fh:
                fh.write("not json")
            noml = os.path.join(d, "manifests", "hf.co", "someone", "config")
            os.makedirs(noml)
            with open(os.path.join(noml, "latest"), "w", encoding="utf-8") as fh:
                json.dump({"layers": [{"mediaType": "application/vnd.ollama.image.license"}]}, fh)
            self.assertEqual(_names(d), ["real:latest"])

    def test_a_manifest_directly_under_manifests_does_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "manifests"))
            os.makedirs(os.path.join(d, "blobs"))
            with open(os.path.join(d, "manifests", "stray"), "w", encoding="utf-8") as fh:
                json.dump(
                    {"layers": [{"mediaType": MODEL_MEDIA, "digest": "sha256:999", "size": 1}]}, fh
                )
            with open(os.path.join(d, "blobs", "sha256-999"), "wb") as fh:
                fh.write(BLOB)
            self.assertEqual(_names(d), ["stray"])


class TestDownstreamUsesTheFullIdentity(unittest.TestCase):
    """The name is not cosmetic: measure() and unload() hand it to the ollama CLI."""

    def _run_with_fake_cli(self, fn):
        """Record the argv a call would have used. No ollama binary is ever invoked."""

        class FakeCompleted:
            stdout = "eval rate: 19.92 tokens/s\n"
            stderr = ""

        calls = []

        def fake_run(cmd, **_kw):
            calls.append(list(cmd))
            return FakeCompleted()

        with (
            mock.patch("quantprobe.ollama.ollama_bin", return_value="/fake/ollama"),
            mock.patch("quantprobe.ollama.subprocess.run", side_effect=fake_run),
        ):
            fn()
        return calls

    def test_measure_runs_the_name_the_store_reader_returned(self):
        from quantprobe import ollama as om

        with tempfile.TemporaryDirectory() as d:
            _store(d, {"hf.co/bartowski/Qwen2.5-7B-GGUF/Q4_K_M": "ccc"})
            (name, _blob, _size) = installed(d)[0]
            calls = self._run_with_fake_cli(lambda: om.measure(name, prompt="hi"))
        self.assertEqual(
            calls[0][:3], ["/fake/ollama", "run", "hf.co/bartowski/Qwen2.5-7B-GGUF:Q4_K_M"]
        )

    def test_unload_stops_the_name_the_store_reader_returned(self):
        from quantprobe import ollama as om

        with tempfile.TemporaryDirectory() as d:
            _store(d, {"registry.ollama.ai/myuser/mymodel/latest": "bbb"})
            (name, _blob, _size) = installed(d)[0]
            calls = self._run_with_fake_cli(lambda: om.unload(name, tries=0))
        self.assertEqual(calls[0], ["/fake/ollama", "stop", "myuser/mymodel:latest"])


def run_smoke():
    """Hook for tests/smoke.py, which has no pytest. Raises on the first failure."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__, fromlist=["*"]))
    with open(os.devnull, "w", encoding="utf-8") as quiet:
        result = unittest.TextTestRunner(stream=quiet, verbosity=0).run(suite)
    if not result.wasSuccessful():
        bad = result.failures + result.errors
        raise AssertionError(f"{len(bad)} ollama identity test(s) failed: {bad[0][1].strip()}")


if __name__ == "__main__":
    unittest.main()
