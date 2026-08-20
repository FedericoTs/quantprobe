# Product gaps — 2026-08-20 sweep

A product-gap pass over the CLI (not the physics): where a real user hits friction. Ranked by
(user impact × cheapness). Status tracked here so deferred gaps are a roadmap, not a memory.

## Fixed in v1.32.0

- **[HIGH] Silent wrong-machine prediction.** `run`/`bench`/`report`/`dashboard`/`audit-ollama`
  accepted a typo'd `--machine` (e.g. `rtx4090` for `rtx-4090`) and silently predicted for the
  auto-detected box instead — while the banner claimed no hardware flags were passed. The tool's
  whole value is a trustworthy number, and quietly changing the target is the worst failure it can
  have. All five now route through `plan.check_presets`, the loud refusal `plan`/`optimize`/`target`
  already used. Pinned by `t_no_command_predicts_for_a_silently_wrong_machine_or_model`.
  *(quantprobe/runtime.py best_flags, report.py, ollama.py; plan.py check_presets.)*

- **[MED] Bare `plan` fabricated a 13B.** `quantprobe plan` with no `--model`/`--gguf`/`--total`
  silently assumed a generic 13B dense model and printed a confident prediction for it. Now
  `plan.warn_if_no_model` labels the assumption unmissably — the same "label every default" rule the
  rest of the tool follows. Kept out of `check_presets` so `audit-ollama` (which prices real stored
  models, never a 13B) does not false-fire it.

- **[LOW] No `--version`.** `quantprobe --version` errored with `unrecognized arguments`; the number
  was only in the no-args banner. Packagers and bug reports expect the flag. Added.

- **[HIGH claim] README advertised a flag that errors.** The "free speed" section presented
  `--spec-type ngram-simple` as "one flag," but it is a **llama.cpp** flag — copy-pasting
  `quantprobe run --spec-type …` gave `unrecognized arguments`. README now names it as a passthrough
  and shows the working `quantprobe run --extra "…"` invocation.

## Deferred (roadmap, with the reason)

- **[MED] Three preset vocabularies drift.** `fetch.py` knows 4 models, `auto.py` knows 13,
  `plan.py MODELS` a different set, and the measured `recipes/*.json` atlas (6, incl. the flagship
  qwen3.6-35b) is a **fourth** list none of the others cross-reference. So `quantprobe auto
  qwen3.6-35b` — the model our own HF build and artifact are about — returns "not a preset," and
  `fetch laguna-s` fails though `auto laguna-s` works. *Fix (M): one shared preset table, and have
  `auto`/`target` fall back to the recipe atlas so any recipe we add becomes reachable by name — and
  surface the prebuilt HF file when one exists.* Deferred because it is a structural refactor across
  four modules, and a full `auto` preset also needs a fetchable high-precision source repo we do not
  yet have committed for every recipe model.

- **[MED] `probe --eval` forces a manual wikitext download.** `--eval` is required, but `auto.py`
  already implements the wikitext download+unzip (`WIKI_URL`). The README pitches `probe --gguf …
  --eval wiki.test.raw` without saying where that file comes from, so "measure YOUR model" stalls at
  step one. *Fix (S): `--eval auto` reusing auto.py's downloader.*

- **[MED] `--machine`/`--model` help is bare.** `hwargs()` adds them with no help text and `plan`'s
  `--machine` help lists 5 of 17 presets, so `run --help`/`bench --help` give no hint the presets
  exist. *Fix (S): help strings, or a `--list-machines`.*

- **[MED] Recently shipped output is undocumented.** `RESIDENCY:` (L-29/C-32), the expert-ceiling
  line (L-30), and the interleave-contamination warning (L-31) print in real runs but appear in no
  README or `--help`, so a user who sees "L-30" mid-run has nowhere to look it up. *Fix (M): a short
  "what the tool tells you" section keyed to the register IDs.*

## Not gaps (checked)

- No-args behaviour is correct (banner + full help). Explicit `raise SystemExit` messages are a
  strength — almost all are actionable. `grep TODO|FIXME|XXX|HACK|NotImplemented` across
  `quantprobe/*.py`: **zero** user-facing stubs. `quantize` and `probe --apply` overlap by design.
