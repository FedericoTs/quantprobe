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

- **[LOW] Preset vocabularies are bridged, not yet merged.** *Mostly closed in v1.35.0.* The four
  lists (`fetch.PRESETS`, `auto.MODEL_REPOS`, `plan.MODELS`, `recipes/*.json`) still exist
  separately, but they no longer strand a user: `auto <recipe-key>` cites the measured band and the
  published build instead of refusing, `fetch <recipe-key>` downloads that build, and `fetch
  <auto-preset>` redirects to `auto` rather than claiming the name is unknown. A property test over
  the whole atlas keeps every future recipe reachable. What remains is cosmetic rather than a funnel
  leak — one table instead of four — plus two schema gaps that block the rest, both found by
  trying to write a true sentence about the tool and discovering it wasn't:

  - **`plan --model <recipe-key>` cannot work, and that is the one users most want.** A recipe
    carries `arch/n_layer/moe` but no parameter counts, so nothing can build a plan spec from it.
    "Will this run on my machine, and how fast?" is the README's opening question and the reason
    someone on a model card hesitates before 14 GB — and for a model in our own atlas we can only
    answer it *after* they download (`plan --gguf`). *Fix (S): optional `params: {total_b,
    active_b, always_active_b}` on the recipe schema, measured from the GGUF at recipe-harvest
    time — never hand-entered, or the atlas starts carrying numbers with no evidence behind them.*
  - **`auto` still cannot build from a key** for want of a fetchable high-precision `source_repo`
    field. *Fix (S) once the above lands, since both are recorded at harvest time.*

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
