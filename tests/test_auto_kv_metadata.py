"""`auto` must carry a model's known KV-per-position into the law, not drop it.

`auto.apply_to` clears `a.model` on purpose - a raw HF repo has no preset, so the law is handed
explicit parameters instead of a preset lookup. Every fact the preset carried therefore has to be
transferred by hand, and kv-per-position was not. `optimize.resolve` reads

    a.kv_per_pos * 1024   if the user passed --kv-per-pos
    plan.MODELS[a.model]["kvp"]   otherwise

and with `a.model` cleared and `a.kv_per_pos` never set, the second branch collapsed to
`plan.DEFAULT_KVP` - 96 KB/pos, the Qwen3-30B class - for EVERY model `auto` resolves. For
llama-70b the existing table records 320 KB/pos - architecture-derived (80L x 8KV x 128d), not a
benchmark anyone here ran - so at `--ctx 32768` the command priced 3.2 GB of KV cache against the
table's own 10.7 GB. That 3.3x sits in a term entering both the per-token byte budget and the tier
capacity check, so what is asserted below is that the COMPUTED FRONTIER changes. Which placement
wins on any given box is neither claimed nor tested here: that would be a new measurement.

Same shape as the v1.11.1 layer-count loss the ModelSpec record was introduced to stop, one field
later. These tests pin the transfer itself, the units it crosses (MODELS/recipes store BYTES, the
args carry KB), the explicit-override precedence including what a 0 means, and the fallback for a
model no row covers.

No network, no GGUF, no inference: every input is a preset row or a shipped recipe already in the
repo, and the only thing asserted is which number reaches `plan.evaluate`.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

# Test the code in THIS repo, not whatever happens to be installed - same reason tests/smoke.py
# does it, and the same bug if it is omitted (two copies validated at once).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantprobe import auto as automod
from quantprobe import optimize as optmod
from quantprobe import plan as planmod
from quantprobe import recipes as recmod
from quantprobe.auto import ModelSpec, apply_to, resolve_model

MACHINE = "rtx-4090"  # a preset machine, so nothing auto-detects or reads calibration state
DEEP_CTX = 32768


def auto_args(**over):
    """The args namespace `quantprobe auto <target> --machine ...` builds, before resolution."""
    base = {
        "target": None,
        "model": None,
        "machine": MACHINE,
        "bits": None,
        "total": None,
        "active": None,
        "always_active": None,
        "vram": None,
        "vram_bw": None,
        "ram": None,
        "ram_bw": None,
        "disk_bw": None,
        "ctx": 0,
        "kv_per_pos": None,
        "n_layer": None,
        "gguf": None,
        "no_anchors": True,
    }
    base.update(over)
    return argparse.Namespace(**base)


def resolved_kvp(a, target):
    """The kvp (BYTES) the law actually receives, after auto resolves `target` onto `a`."""
    with redirect_stdout(io.StringIO()):
        apply_to(resolve_model(a, target), a)
        return optmod.resolve(a)[-1]


def ranked_tps(a):
    with redirect_stdout(io.StringIO()):
        return [round(r["tps"], 6) for r in optmod.run(a)]


class AutoKvMetadata(unittest.TestCase):
    def test_nondefault_preset_kv_per_pos_reaches_the_law(self):
        """llama-70b: 320 KB/pos in the table, and it must not arrive as the 96 KB/pos default.

        The 320 is the table's architecture-derived figure, transported faithfully; this change
        neither re-derives nor measures it.
        """
        known = planmod.MODELS["llama-70b"]["kvp"]
        self.assertNotEqual(
            known, planmod.DEFAULT_KVP, "pick a preset whose kvp differs from the default"
        )
        a = auto_args(ctx=DEEP_CTX)
        with redirect_stdout(io.StringIO()):
            spec = apply_to(resolve_model(a, "llama-70b"), a)
        self.assertEqual(spec.kv_per_pos, known / 1024, "ModelSpec lost the preset's KV fact")
        self.assertEqual(a.kv_per_pos, known / 1024, "apply_to never put KV on the args")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(optmod.resolve(a)[-1], known)

    def test_recipe_kv_per_pos_reaches_the_law(self):
        """An atlas record drives `auto` too, and carries its own kv_per_pos.

        Unlike the preset table this one IS file-derived: the params block records the GGUF it
        was read out of (`measured_from`).
        """
        key = "qwen3.5-35b"
        p = recmod.params(recmod.find(key=key))
        self.assertTrue(p and p.get("kv_per_pos"), f"{key} lost its params block")
        self.assertNotEqual(p["kv_per_pos"], planmod.DEFAULT_KVP)
        a = auto_args(ctx=DEEP_CTX)
        self.assertEqual(resolved_kvp(a, key), p["kv_per_pos"])
        self.assertEqual(a.kv_per_pos, p["kv_per_pos"] / 1024)

    def test_explicit_kv_per_pos_beats_the_preset(self):
        """--kv-per-pos is the user measuring their own build; nothing may overwrite it.

        A NONZERO override wins. The gate is truthiness, not `is not None`, deliberately: both
        consumers (`optimize.py`, `plan.py`) read `args.kv_per_pos * 1024 if getattr(args,
        "kv_per_pos", None) else <fallback>`, so `--kv-per-pos 0` already meant "not given" to
        the law before this change. `auto` uses the identical test so it cannot disagree with
        its own consumers about what a 0 means - pinned below, not left to be inferred.
        """
        a = auto_args(ctx=DEEP_CTX, kv_per_pos=72)
        self.assertEqual(resolved_kvp(a, "llama-70b"), 72 * 1024)
        self.assertEqual(a.kv_per_pos, 72)

        known = planmod.MODELS["llama-70b"]["kvp"]
        zero = auto_args(ctx=DEEP_CTX, kv_per_pos=0)
        self.assertEqual(
            resolved_kvp(zero, "llama-70b"),
            known,
            "0 is not an override: it means unset here exactly as it does in optimize/plan",
        )
        self.assertEqual(zero.kv_per_pos, known / 1024)

    def test_unknown_model_still_falls_back_to_the_default(self):
        """A raw HF repo described by --total has no KV fact - the documented guess stands."""
        a = auto_args(ctx=DEEP_CTX, total=13.0, active=13.0)
        with redirect_stdout(io.StringIO()):
            spec = apply_to(resolve_model(a, "some-org/Some-Model-GGUF"), a)
        self.assertIsNone(spec.kv_per_pos, "invented a KV number for a model no row covers")
        self.assertIsNone(a.kv_per_pos)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(optmod.resolve(a)[-1], planmod.DEFAULT_KVP)

    def test_preset_without_a_kv_row_falls_back_to_the_default(self):
        """An `auto` preset plan.MODELS has no row for keeps the fallback, not a borrowed number.

        The absent-row state is CONSTRUCTED, not borrowed from whichever real preset happens to
        lack a row today: a synthetic preset is added to the `auto` fixture table for the
        duration of the test and removed after (mock.patch.dict restores the global). Adding a
        real `kvp` row for any shipped preset is an improvement and must not turn this red -
        an earlier version asserted `qwen3-coder not in plan.MODELS` and would have done exactly
        that. Only the fixture table is patched: resolve_model, optimize and plan all run real.
        """
        key = "zz-fixture-no-kv-row"
        self.assertNotIn(key, planmod.MODELS, "the fixture key must not name a real model")
        # same shape as a real MODEL_REPOS row: (repo, total, active, always_active, moe)
        with mock.patch.dict(
            automod.MODEL_REPOS, {key: ("some-org/Fixture-GGUF", 30.5, 3.3, 1.2, True)}
        ):
            a = auto_args(ctx=DEEP_CTX)
            self.assertEqual(resolved_kvp(a, key), planmod.DEFAULT_KVP)
            self.assertIsNone(a.kv_per_pos)
        self.assertNotIn(key, automod.MODEL_REPOS, "the fixture preset leaked out of the test")

    def test_kv_per_pos_is_propagated_into_every_evaluate_call(self):
        """The argument has to reach plan.evaluate, not merely sit on the namespace.

        optimize.run also prices a q8 KV counterfactual at x0.75 where the GPU allows it, so both
        the f16 value and its scaled twin are legitimate - DEFAULT_KVP is not.
        """
        known = planmod.MODELS["llama-70b"]["kvp"]
        a = auto_args(ctx=DEEP_CTX)
        seen = []
        real = planmod.evaluate

        def spy(*args, **kw):
            seen.append(kw.get("kvp"))
            return real(*args, **kw)

        with redirect_stdout(io.StringIO()):
            apply_to(resolve_model(a, "llama-70b"), a)
            planmod.evaluate = spy
            try:
                optmod.run(a)
            finally:
                planmod.evaluate = real
        self.assertTrue(seen, "optimize.run evaluated no placement at all")
        self.assertIn(known, seen, f"the law never saw {known} B/pos; it saw {sorted(set(seen))}")
        self.assertTrue(
            all(v in (known, known * 0.75) for v in seen),
            f"an evaluate call used an unexplained kvp: {sorted(set(seen))}",
        )

    def test_zero_context_is_unaffected_but_still_records_the_fact(self):
        """kvp only prices a token once --ctx > 0 - which is why this loss stayed invisible.

        At depth the assertion is only that the COMPUTED frontier differs. No direction, no
        magnitude, and no claim about which placement is recommended: nothing measured here
        would support one.
        """
        known = planmod.MODELS["llama-70b"]["kvp"]
        flat = auto_args(ctx=0)
        with redirect_stdout(io.StringIO()):
            apply_to(resolve_model(flat, "llama-70b"), flat)
        self.assertEqual(
            flat.kv_per_pos, known / 1024, "the KV fact is a model property, not a ctx one"
        )
        wrong = auto_args(ctx=0, kv_per_pos=planmod.DEFAULT_KVP / 1024)
        with redirect_stdout(io.StringIO()):
            apply_to(resolve_model(wrong, "llama-70b"), wrong)
        self.assertEqual(ranked_tps(flat), ranked_tps(wrong), "ctx 0 must not depend on kvp")

        deep = auto_args(ctx=DEEP_CTX)
        deep_wrong = auto_args(ctx=DEEP_CTX, kv_per_pos=planmod.DEFAULT_KVP / 1024)
        with redirect_stdout(io.StringIO()):
            apply_to(resolve_model(deep, "llama-70b"), deep)
            apply_to(resolve_model(deep_wrong, "llama-70b"), deep_wrong)
        self.assertNotEqual(
            ranked_tps(deep),
            ranked_tps(deep_wrong),
            f"at ctx {DEEP_CTX} a 3.3x KV error changed nothing - the frontier stopped "
            "responding to context and this whole fix would be unobservable",
        )

    def test_the_kv_field_crosses_onto_the_args_in_kb(self):
        """The record carries the field, and it crosses in KB - plan.MODELS stores BYTES.

        Scope is deliberately just this one field. Whole-record transfer policy (which fields
        are local-only, what a NEW field obliges you to do) lives once, in smoke's
        t_auto_transfers_every_model_field; a second copy of that policy here would be another
        thing to keep in sync and would go stale silently.
        """
        known = planmod.MODELS["llama-70b"]["kvp"]
        self.assertIn("kv_per_pos", ModelSpec._fields)
        a = auto_args()
        with redirect_stdout(io.StringIO()):
            spec = apply_to(resolve_model(a, "llama-70b"), a)
        self.assertEqual(spec.kv_per_pos, known / 1024, "the record must hold KB, not bytes")
        self.assertEqual(a.kv_per_pos, spec.kv_per_pos, "apply_to dropped ModelSpec.kv_per_pos")
        self.assertNotEqual(a.kv_per_pos, known, "the args carry KB; a raw byte count is 1024x off")


class AutoRuntimeKvHandoff(unittest.TestCase):
    def run_pipeline(self, custom, explicit):
        from quantprobe import cli, fetch, probe, runtime

        total = automod.MODEL_REPOS["qwen3-30b"][1]
        listing = [
            ("source-Q8_0.gguf", int(8.2 * total * 1e9 / 8)),
            ("model-Q2_K.gguf", int(2.7 * total * 1e9 / 8)),
        ]
        handed = []
        with tempfile.TemporaryDirectory() as dest:
            argv = [
                "quantprobe",
                "auto",
                "qwen3-30b",
                "--machine",
                "2016-xmp",
                "--dir",
                dest,
                "--run",
            ]
            if custom:
                argv.extend(["--custom", "--force-custom", "--yes"])
            if explicit is not None:
                argv.extend(["--kv-per-pos", str(explicit)])
            with (
                mock.patch.object(automod, "list_ggufs", return_value=listing),
                mock.patch.object(
                    automod, "ensure_eval", return_value=os.path.join(dest, "eval.txt")
                ),
                mock.patch.object(fetch, "fetch", return_value=True),
                mock.patch.object(fetch, "token", return_value=None),
                mock.patch.object(probe, "run"),
                mock.patch.object(
                    runtime, "run", side_effect=lambda a: handed.append(a.kv_per_pos)
                ),
                mock.patch.object(sys, "argv", argv),
                redirect_stdout(io.StringIO()),
            ):
                cli.main()
        self.assertEqual(handed, [explicit])

    def test_standard_handoff_does_not_shadow_the_file_header(self):
        self.run_pipeline(False, None)

    def test_custom_handoff_does_not_shadow_the_file_header(self):
        self.run_pipeline(True, None)

    def test_standard_handoff_preserves_explicit_override(self):
        self.run_pipeline(False, 72)

    def test_custom_handoff_preserves_explicit_override(self):
        self.run_pipeline(True, 72)


def run_smoke():
    """Hook for tests/smoke.py, which collects `t_*` callables rather than unittest cases."""
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    )
    if not result.wasSuccessful():
        raise AssertionError(
            "; ".join(
                f"{case}: {tb.strip().splitlines()[-1]}"
                for case, tb in result.failures + result.errors
            )
        )


if __name__ == "__main__":
    unittest.main()
