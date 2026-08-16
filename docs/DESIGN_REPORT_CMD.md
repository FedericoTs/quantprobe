# Design: `quantprobe report` v1 - the forwardable artifact

Status: design, pre-implementation. Owner: quantprobe. Target: v1.29.

This is gap 2 of docs/NOTES_GAP_2026-08.md: the tool has 14 subcommands and no artifact a
user can hand to someone who was not at the terminal. plan's output, pasted into issue #1,
arrived as ~40 line-mangled rows full of register IDs (#16/#52, C-14, D-10) that mean nothing
outside FINDINGS.md. The reader of THIS feature is new: an IT manager deciding a hardware
buy, a consultant's client, an ISV writing hardware requirements. They will not run the tool;
they will read a file someone forwarded. Everything below follows from that one fact - the
report must carry its own provenance, its own error bars, and its own scope lines, because
the person who could explain them is not in the room.

Two design invariants, both inherited from the repo's existing discipline:

1. No second physics. Every number in the report is produced by the same functions plan
   already runs. The report is a RENDERER over plan's engine, exactly as binding_report()
   is a renderer over binding_constraint(). If report and plan can disagree about the same
   file on the same box, the feature is wrong (the C-15 auto-vs-plan divergence, 26.2 vs
   19.5 on one model, is the failure class - it took a measurement to notice).
2. Every number is labeled. plan already tags its inputs ([measured], [est],
   [est, unvalidated]); the report extends the same palette to every printed figure:

   - `[measured]`   an instrument, benchmark, or file read on record (with the source named)
   - `[derived]`    exact arithmetic from measured/estimated inputs (the law's outputs, the
                    binding shares - implicit in plan today, explicit here)
   - `[est]`        a fitted band or preset (the QUAL table, non-reference machine presets)
   - `UNVALIDATED`  stated, and no measurement covers this exact case (Mac presets,
                    2-bit capability at sizes between the measured points)

---

## 1. CLI surface

```
quantprobe report --gguf model.gguf [--machine PRESET | --vram/--vram-bw/--ram/--ram-bw/--disk-bw]
                  [--ctx N] [--kv-per-pos KB] [--no-anchors]
                  [--out report.md] [--bench-log FILE]
```

Wiring in cli.py, using the existing `hwargs(sp)` helper so report can never grow a private
hardware-flag dialect:

```python
rp = sub.add_parser("report", help="one-page forwardable Markdown report: verdict, placement, "
                                   "binding constraint, quality - every number labeled")
rp.add_argument("--gguf", required=True, help="the file the report is about (autospec grounds every number)")
hwargs(rp)                       # --model/--machine/--bits/--ctx/... identical to run/bench
rp.add_argument("--out", default=None,
                help="output path (default: quantprobe-report-<model>.md in the current directory)")
rp.add_argument("--bench-log", default=None,
                help="a llama-bench log to quote as the [measured] column next to the prediction")
```

Dispatch: `elif a.cmd == "report": from . import report; report.run(a)`.

Decisions, with reasons:

- `--gguf` is REQUIRED in v1. plan accepts presets; report does not, because the quality
  section needs (arch, n_layer) for recipe matching and the capacity math needs the real
  file size (true_size_gb - the dense-size inflation bug at plan.py:1112 is exactly what
  happens without it). A preset report is v2.
- Default out path: `quantprobe-report-<gguf basename without extension>.md` in the CURRENT
  directory, not next to the GGUF (model dirs are often read-only NAS) - the tool's name in
  the filename tells the recipient of a forwarded file what generated it. Re-running
  overwrites the same file - regenerate-in-place is the workflow. The absolute path is
  printed after writing.
- `--bench-log` is the only channel for a measured speed in v1. report NEVER runs
  llama-bench itself (plan's contract: downloads nothing, takes a second, needs only
  Python). The parser is the same llama-bench markdown-row shape calibrate._bench_anchor
  already parses. Guard: the bench row's param count must match autospec's `t` within 2%,
  else the report prints the mismatch instead of a predicted-vs-measured ratio - quoting a
  ratio between two different models is worse than no ratio.
- Output format: Markdown only (one file, plain ASCII, renders on GitHub/Slack/email,
  survives forwarding). No HTML/PDF in v1 - see section 6.
- Runtime: pure reads (GGUF header via spec.from_gguf, ~/.quantprobe/calibration.json
  via calibrate.load, quantprobe/recipes/*.json via recipes.find, optional bench log).
  Measured ~10 s on the 17 GB flagship GGUF - the header walk scales with tensor count and
  disk speed, so "~1 s" (this doc's original estimate) only holds for small files; README
  says "~10 s on this 17 GB file" for that reason.

## 2. Getting structured data out of plan without duplicating its physics

### What run() is today

plan.run() (plan.py:1879) is a compute prefix followed by a print monolith:

- lines ~1880-1927: spec.apply, check_presets, preset/model resolution, resolve_hw
  (calibration + anchors), resolve_gpu_eta/resolve_cpu_bw, the `ev_kw` dict, the `ev()`
  counterfactual closure, `evaluate(**ev_kw)` -> (size, act, cfgs), `qual_of` -> q.
- line ~1943: `capacity_probe(ev, ...)` -> cap_find; `binding_constraint(best, capacity=...)` -> bc.
- everything else: print statements reading those values, plus calls to advice functions
  that RETURN strings/lines (binding_report, speculation_advice, dense_draft_note,
  fits_in_vram_advice, depth_scope_warning, serving_advisory, phase_advice,
  upgrade_advisor) - the "returned rather than printed so the smoke suite can assert"
  discipline that already exists is what makes this feature cheap.

Everything the report needs is either already returned by a pure function or lives in
run()'s local variables. The internal structures:

- the placement rows: `cfgs`, a list of `Row` (plan.py:891) - a 4-tuple subclass carrying
  `terms` ({resource: seconds}, reconstructing tok_s to 1e-9), `eff`, and `runnable`.
- the binding verdict: `bc` from binding_constraint (plan.py:936) - resource, klass, share,
  shares, margin_x, ceiling_x, lever_ceiling, capacity.
- the capacity finding: `cap_find` from capacity_probe (plan.py:989) - tier, gap_gb, lever,
  shave_tps, lift_tps, gains, kv lever fields.
- the advice: the pure functions above.

### The refactor: extract `build_rows(args)`

Move the compute prefix (through cap_find/bc) into one function; run() keeps every print.

```python
def build_rows(args):
    """Everything plan.run COMPUTES, none of what it prints.

    Returns a dict:
      inputs  dict(t, a, ne, moe, bits, ctx, kvp, nlay, true_size_gb, gguf, arch,
                   iq_share, codebook_share, model_hint, machine_hint)
      hw      dict(vc, vb, rc, rb, db, geta, gl, cal_active, gpu_ratio, cpu_ratio)
      size, act, q                      # file GB, active GB/token, QUAL multiplier
      rows    [Row]                     # terms/eff/runnable intact, already sorted
      best    rows[0]
      cap     capacity_probe finding or None
      bc      binding_constraint dict or None
      ev      the counterfactual closure (shares the baseline's own argument dict,
              prereg #88 P-3a - report's advisors MUST use this, never a re-spelled kwarg set)
    """
```

run() becomes `px = build_rows(args)` followed by the unchanged print stream reading from
px. report.py calls build_rows(args) plus the same pure advice functions and renders
Markdown.

Why this is stdout-safe by construction: every print that happens during the compute
prefix happens inside NESTED calls that move with it (spec.apply's autospec banner,
resolve_hw's auto-detect banner, apply_calibration_overrides' calibration/anchored lines,
resolve_gpu_eta/resolve_cpu_bw's size-class notes), and evaluate/capacity_probe/
binding_constraint print nothing. The moved block is a strict prefix of run() and executes
first in both versions, so the byte order of stdout cannot change. In report mode those
same nested banners still print to the terminal (they are the live provenance trail; the
FILE gets its provenance from the returned structure and calibrate.load()).

One additive line outside plan.py: spec.apply() currently sets iq_share/codebook_share/
fmt_bw on args but NOT `arch`; report's recipe matching needs (arch, n_layer) for
recipes.find(). Add `a.arch = s.get("arch")` next to the iq_share line (spec.py:255) -
read-only attribute, same pattern, no consumer changes.

Explicitly OUT of this refactor: runtime.best_flags (runtime.py:29) re-implements the same
preamble and should eventually consume build_rows too - that is a follow-up, not this
change. One change at a time; the byte-guard below covers plan only.

### The regression guard (byte-identical stdout)

- One-time migration guard, RUN 2026-08-16 (result recorded here instead of as a script):
  `python -m quantprobe plan --model M --machine X --ctx C` over the same 340-cell preset
  grid prereg #88 used (10 models x 17 machines x ctx {0, 16384}), stdout sha256 per cell,
  pre-refactor HEAD (git archive) vs the worktree: **340/340 cells byte-identical**, plus
  stderr and return codes identical on a 14-cell spot grid including custom-flag and
  autospec cells. The planned weights/exp_report_stdout_guard.py script is NOT kept: once
  the refactor merges, HEAD contains it and the script would diff the tree against itself -
  migration evidence that can never fail again is not evidence. Preset machines skip
  calibration (resolve_hw only applies it on the auto-detect path), so the grid was
  deterministic across boxes.
- Permanent guard in tests/smoke.py: for 3 preset cells, parse plan's printed row lines
  (name + tok/s) and assert they equal `build_rows(args)["rows"]` under the same "%6.1f"
  formatting run() uses. This pins "run prints what build_rows returns" forever, the same
  shape as the existing binding_report line-list tests.

## 3. Report sections, with every number's producer and label

Section order is the decision-maker's reading order: verdict first, evidence behind it,
scope last-but-mandatory. All content plain ASCII.

### 3.1 Verdict

| number | produced by | label |
|---|---|---|
| headline predicted tok/s | build_rows -> rows[0][1] | [derived] (Law 4) |
| the +/-25% band, printed inline as a range | rows[0][1] * 0.75 / 1.25 (the footer claim plan already prints) | [derived] band |
| all-in-VRAM case instead: ">= 0.90x floor, 13/13" | fits_in_vram_advice() | [measured] one-sided claim |
| measured tok/s (only with --bench-log) | report._parse_bench_log | [measured] |
| predicted-vs-measured ratio | measured / predicted | [derived] |

The band is printed NEXT TO the number, not in a footnote - NOTES_GAP gap 3 names the
defect ("plan prints a bare 36.8 tok/s with the band buried in a caveat paragraph"); the
report is where it gets fixed first.

### 3.2 What was asked (inputs + provenance)

| number | produced by | label |
|---|---|---|
| file size GB | spec.gguf_size (split-aware) | [measured] (bytes on disk) |
| total/active params, moe flag | spec.from_gguf tensor walk | [measured] (read from file) |
| effective bits/weight | spec.from_gguf: bytes*8/params | [measured] |
| KV KB/pos (+ hybrid "N of L layers cache KV" when kv_layers < n_layer) | spec.from_gguf (U-51) | [derived] (exact arithmetic from header fields) |
| machine constants vc/vb/rc/rb/db | MACHINES preset, detect.detect(), or calibrate.load via resolve_hw | preset: [measured] for 2016/2016-xmp, [est] for other GPUs, [est] UNVALIDATED for Macs (the table's own hint tags travel verbatim); auto-detect: [os]; calibrated: [measured] + cal_id + date + staleness |
| calibration state line | calibrate.load + calibration_gap_warning | [measured] or the MIXED-state warning verbatim (C-17: partial calibration measured WORSE than none) |
| anchor ratios if active | hw dict `_gpu_ratio`/`_cpu_ratio` from apply_calibration_overrides | [measured] (user's own anchor runs) |

### 3.3 Predicted placements (the table)

| number | produced by | label |
|---|---|---|
| each row's tok/s + name + warning | evaluate() via build_rows | [derived] |
| unrunnable rows (runnable=False) | Row.runnable | printed BELOW a separator with "needs a runtime llama.cpp does not have; upper bound with named unpriced costs - not comparable to the rows above" (the row's own warn text travels with it) |

Row warnings (pinned-memory, RAM-boundary, KV-deficit, mmap trade) are quoted verbatim -
they are the disclosure-on-the-row discipline (D-10) and must not be summarized away.

### 3.4 What limits this machine

| number | produced by | label |
|---|---|---|
| binding class + share ("RAM bandwidth is 88% of the token") | binding_constraint via build_rows | [derived] (terms reconstruct the row's speed to 1e-9 - the smoke suite's own check) |
| capacity gap GB, shave/lift/KV-lever tok/s | capacity_probe via build_rows | [derived] counterfactuals of the same law |
| KV-lever measured support (+37% at d16384; PPL ratio 1.00031 +/- 0.0188) | binding_report text (preregs #25/#91) | [measured], with the E-10 niche-domain caveat attached |
| per-lever Amdahl ceilings ("NVMe capped at 1.05x") | bc["lever_ceiling"] / binding_report | [derived] |
| upgrade advisor rows (gain >= 1.08x) and the WON'T-HELP list | upgrade_advisor(px["ev"], ...) - the closure, never re-spelled kwargs | [derived] |
| scope line "DECODE only; prefill binds differently" | binding_report | fixed wording, mandatory |

Plain-words rule: register IDs (preregs, C-/L-/U- codes) do NOT appear in the report body.
Where plan cites "prereg #25", the report says "measured, 2026-07 reference-box run" and
puts the register pointer in a final small-print Sources block. The reader this file is for
has no FINDINGS.md.

### 3.5 The command

| item | produced by | label |
|---|---|---|
| llama.cpp command line | best[3] + ubatch_flags + append_threads_flag | verbatim what plan emits (the two can never disagree: same functions) |
| --threads note | append_threads_flag | fixed wording |

### 3.6 Measured check (present only when data exists)

With --bench-log: the bench rows quoted verbatim ([measured], build hash named), the ratio
against the prediction, and an explicit in/out-of-band verdict. A pp row whose error bar
exceeds ~50% of its value is quoted but marked "not usable as a number" (the qwen38 pp512
case below: 44.22 +/- 39.26). Without --bench-log and without anchors, the section is
REPLACED by the M1 wording of section 5 - absence of measurement is stated, never implied.

### 3.7 Quality: what this quant costs

See section 4.

### 3.8 Scope - read before forwarding

Fixed, mandatory, never conditional. Contains the M1 and M2 wordings (section 5), the
single-stream statement, the decode-only statement, and "prediction, not guarantee".

### 3.9 Reproduce

The exact `quantprobe report ...` command that generated the file, tool version, date,
and the `quantprobe bench --gguf ...` line that replaces predictions with measurements.

## 4. The quality section

Inputs: `q` from qual_of(moe, bits) (the QUAL fitted band - the same "est. quality cost"
plan prints), `arch`/`n_layer` from autospec, recipes.find(arch=..., n_layer=...), and the
size-dependence law from docs/QUANT_QUALITY.md.

Logic, in order:

1. This file's tier: "X.XX bits/weight -> est. quality cost xQ.QQ" - qual_of - [est],
   always with the proxy disclaimer: "a fitted perplexity band, not a task score."
2. Recipe atlas: if recipes.find matches, print the measured fragile band with its
   provenance block (band, ratio, date, eval corpus, hardware, base_quant) - [measured] -
   with the raw_log PATH deferred to the Sources block ("raw log in Sources below"): a
   weights/ path is a repo pointer, and repo pointers mean nothing to the forwarded reader
   (the plain-words rule wins the tension with putting it inline)
   - INCLUDING the recipe's own scope note verbatim where one exists (the
   qwen3.8-27b entry's "band LOCATION is trustworthy, absolute BF16 loss is not measured"
   line). No match: "no measured recipe for this architecture (5 families in the atlas);
   the probe still works on your model: quantprobe probe --gguf ..." - and nothing invented.
3. The size-dependence law (QUANT_QUALITY.md sections 1-3), applied to THIS model's size,
   with its measured anchors and its honest gap:
   - large (35B-class, measured): recipe + aggressive 2-bit pays - MATH-500 81.0 vs 57.0
     naive [measured, Qwen3.5-35B].
   - small (4B-class, measured): 2-bit is a false economy even with the recipe (50.2 vs
     81.0 BF16; naive 2.6) - "the sensible quant for a small model is Q4, near-lossless
     there (77.6 vs 81.0)" [measured, Qwen3.5-4B].
   - between the measured points: UNVALIDATED, said in exactly those words. A 27B gets:
     "2-bit capability at this size is unmeasured - the measured points are 4B (fails)
     and 35B (holds); this model sits between them."
   - always: "capability numbers were measured on OTHER models (Qwen3.5-35B/4B), on
     MATH-500/GSM8K/IFEval at temp 0 - they scope the method, they do not score this file."
4. If the capacity advisor recommended a shave (cap_find fired), price it: the shave's
   implied bits (bits x fit_scale), that tier's qual_of value [est], and - if the shave
   crosses below ~3 bits - the UNVALIDATED line plus the depth-aware build command with
   the matched recipe. This is the one place speed and quality advice meet, and the report
   must show both sides of the trade on one line (the tier-boundary advisor already knows
   the speed side; the quality side comes from qual_of + the law).
5. Closing line, fixed: "None of the above is a test of your workload. quantprobe cannot
   yet run your tasks (planned - see Scope)."

## 5. The two most likely ways this report misleads a decision-maker

Chosen for the forwarded-reader failure mode specifically; each gets a MANDATORY verbatim
block in the template (a smoke test asserts both strings are present in every generated
report).

MISLEAD 1: a predicted number is read as a measured one. An IT manager procures hardware
on "1.8 tok/s" as if someone had run the box. Terminal users see the caveat trail; a
forwarded file must carry it inline.

Prevention (exact wording, and the layout rule as shipped: every number-bearing line or
wrapped block closes with its honesty label, and the Verdict speeds - the numbers a
decision-maker actually quotes - additionally carry the word "PREDICTED" or "[measured]".
Placement rows and counterfactual speeds are [derived], which is their true provenance;
labeling them PREDICTED would claim a band they do not carry):

> Every speed in this report marked PREDICTED was computed from this machine's memory
> bandwidths - it was not run. Predictions of this kind have a stated error band of
> +/-25% (printed next to each number), validated at 8.4% median error on the reference
> machine ladder. [When no measurement exists:] No number in this report was measured on
> this machine. One command replaces the predictions with measurements:
> `quantprobe bench --gguf <file>`.

MISLEAD 2: a single-user speed is read as service capacity. An ISV writes hardware
requirements for N concurrent users from a single-stream number - and the measured
inversion (U-38/U-39) means the error is not just scale, it can flip WHICH model/hardware
to buy: 219 aggregate tok/s (dense 7B in VRAM) vs 40 (30B MoE, experts in RAM) at 32
streams, on the same box, while at 1 stream the MoE is the better choice.

Prevention (exact wording, mandatory in Scope even when no serving advice applies):

> Every number here is ONE user's speed. Aggregate throughput under concurrent users is a
> different quantity, and on the one machine where we measured it, concurrency INVERTED
> the right choice of model: a dense 7B in VRAM served 219 tokens/s aggregate at 32
> streams while a RAM-offloaded 30B MoE capped near 40 - yet the MoE wins at 1 stream.
> Do not size a multi-user service from this page; measured multi-user notes, where they
> exist for this placement, appear above (one Pascal-class box - treat as indicative, not
> a prediction for your hardware).

(The quality-as-clearance misread is third on the list; it is prevented structurally by
section 4's mandatory proxy disclaimer and closing line rather than called out here.)

## 6. What report v1 explicitly does NOT do (named v2 items)

- No browser rendering, no HTML, no PDF. Markdown file only. (v2: an HTML render of the
  same structure; PDF only if a real reader asks.)
- No bring-your-own-tasks. The quality section quotes proxies and other-model capability
  measurements with their scope; it cannot score YOUR workload. (v2: the task-spec format
  + runner of NOTES_GAP gap 8, reusing weights/business_tasks.py's predicate discipline.)
- No benchmark execution. report never launches llama-bench/llama-server; --bench-log
  ingests a run the user already made. (v2: `report --bench` chaining the existing bench.)
- No preset-only reports (--gguf required; the "sizing a model I haven't downloaded" story
  stays with plan/target/optimize until report can label a preset-grounded quality section
  honestly).
- No multi-machine comparison matrix in one file (docs/MATRIX.md exists; v2 candidate).
- No file fingerprint (sha256 of a 17 GB GGUF is not a 1-second operation; v2 flag).
- No auto-upload, no telemetry, nothing sent anywhere - same contract as calibrate.

## 7. Full mock - Qwen3.8-27B on 2016-xmp

Source of every number: weights/data/qwen38_plan.log (plan, 2016-xmp preset, ctx 0) and
weights/data/qwen38_bench.log (llama-bench, same flags, build 0278d8362). Two notes for
reviewers of this design (not part of the report): (a) the plan log predates the v1.28
hybrid-KV correction, so its autospec line reads "KV 260 KB/pos"; a fresh run prints
~68 KB/pos (= 260/65 x 17 layers that carry attn_k) - at ctx 0 no downstream number
moves, and the mock quotes the log faithfully; (b) the bench file size 15.92 GiB equals
17.1 GB - the same file in two units, which the report says out loud so the reader does
not see a contradiction.

```markdown
# quantprobe report - Qwen3.8-27B-Q4_K_M.gguf on 2016-xmp

Generated 2026-08-16 by quantprobe v1.29.0 (report v1).
Labels: [measured] instrument/benchmark on record - [derived] arithmetic from measured
inputs - [est] fitted band or preset - UNVALIDATED: stated, no measurement covers it.

## Verdict

PREDICTED decode speed, one user:   1.8 tok/s   (band 1.4 - 2.3)          [derived]
MEASURED on this machine:           2.04 +/- 0.02 tok/s  (llama-bench)    [measured]
The measurement is 1.13x the prediction - inside the +/-25% band, on the side our
misses err (low).

Fit: the 17.1 GB file does not fit this machine's 6 GB VRAM or 16 GB RAM whole. It
runs SPLIT: 15 of 65 layers on the GPU, the rest in system RAM. At ~2 tok/s this is
usable for short answers and background jobs, not for interactive work.

Every speed in this report marked PREDICTED was computed from this machine's memory
bandwidths - it was not run. Predictions of this kind have a stated error band of
+/-25% (printed next to each number), validated at 8.4% median error on the reference
machine ladder.

## What was asked

file       Qwen3.8-27B-Q4_K_M.gguf - 17.1 GB on disk                     [measured]
model      27.3B params, 26.0B active/token, dense hybrid,
           5.01 effective bits/weight - read from the file itself        [measured]
KV cache   260 KB/pos f16 (ctx 0 in this report: no KV term is priced)   [derived]
machine    "2016-xmp" preset: GTX 1060 6GB (192 GB/s), 16 GB DDR4 XMP
           (48 GB/s), SATA SSD (0.45 GB/s) - constants measured on the
           reference box this preset describes                           [measured]
calibration  not consulted: a preset machine was named. For YOUR box,
           run without --machine after `quantprobe calibrate`.

## Predicted placements (single user, decode)

  RECOMMENDED   1.8 tok/s  split: 15/65 layers -> VRAM, rest -> RAM      [derived]
                0.1 tok/s  stream from disk - exceeds RAM, capacity demo,
                           not usable inference                          [derived]
  ------------- not runnable on stock llama.cpp -------------
                0.1 tok/s  layer-by-layer streaming - UPPER BOUND: needs a
                           layer-streaming runtime (airllm-class); PCIe
                           transfer per layer is unpriced and a measured
                           1.82x streaming gap applies. Expect materially
                           slower. Shown for capacity context only.      [derived, upper bound]

## What limits this machine

CAPACITY-BOUND (RAM): this configuration is 5.1 GB over the RAM boundary, and
crossing the boundary is worth 1.2x.                                     [derived]

  cross it by   shaving the file (next quant tier down)  -> ~2.3 tok/s   [derived]
                (quality cost of that shave: see Quality, below)
           or   fitting 21.1 GB of RAM                   -> ~1.8 tok/s   [derived]
                (no speed gain: RAM bandwidth then binds instead)
  until then    system RAM bandwidth is where the time goes - 88% of
                every token                                              [derived]

Levers that will NOT help here (exact ceilings, not opinions):           [derived]
  faster GPU memory bus        capped at 1.14x (12% of the token)
  faster storage / NVMe        NO effect (0% of the token)
  more/faster CPU threads      NO effect (0% of the token)
  +16 GB RAM                   re-evaluated: does not move this placement
                               (the shave is worth more than the fit)

Scope: DECODE only. Reading long prompts binds differently (compute, not
bandwidth) and is not classified here.

## The command

  llama-server -m Qwen3.8-27B-Q4_K_M.gguf -ngl 15 -b 2048 -ub 2048 --threads 4

  --threads 4 = this machine's logical cores. llama.cpp's own auto-detect may pick
  physical-only and cost 2x on CPU-bound decode - verify on your build if unsure.

Optional lever, measured on this placement class (not on this file): a small
same-family draft model (`-md draft.gguf -ngld 0 --spec-draft-n-max 2`) bought +33%
decode on a 14B dense split on this box, at zero VRAM cost. Keep K=2; K>=3 measured
at or below baseline. Treat as an experiment to run, not a promised number. [measured, other model]

## Measured check

llama-bench on this machine, build 0278d8362, same flags (-ngl 15):      [measured]

  decode  tg128   2.04 +/- 0.02 tok/s    predicted 1.8 -> ratio 1.13, IN BAND
  prompt  pp512   44.22 +/- 39.26 tok/s  error bar is 89% of the value: quoted
                                         for completeness, not usable as a number

(bench reports the file as 15.92 GiB = 17.1 GB - same file, two units.)

## Quality: what this quant costs, and what the recommended shave would

this file    5.01 bits/weight -> est. quality cost x1.03                 [est]
             A fitted perplexity band, not a task score.

fragile band measured for THIS model: layers 51-64 break first under
             low-bit quantization (2.35x the median band), probed
             2026-08-15 on WikiText-2                                    [measured]
             Scope, from the measurement itself: the probe ran from a Q4
             source (a BF16 probe is infeasible on the probing box), so
             the band LOCATION is trustworthy; the absolute loss vs the
             BF16 original is not measured.

the shave    crossing the RAM boundary needs ~30% off the file: ~3.5
             bits/weight. At 3-bit-class, dense fitted cost is x1.12     [est]
             Below ~3 bits at this size: UNVALIDATED. The measured points
             are a 4B (2-bit fails even with the fragile band protected:
             50.2 vs 81.0 on MATH-500) and a 35B (the protected 2-bit
             holds: 81.0 vs 57.0 naive). This 27B sits between them -
             unmeasured. If you shave past Q3, build depth-aware with the
             measured band (quantprobe quantize --gguf <source>
             --recipe qwen3.8-27b) and treat the result as unproven until
             scored on your own tasks.

Capability numbers above were measured on other models (Qwen3.5-35B/4B, full
MATH-500/GSM8K/IFEval, temp 0). They scope the method; they do not score this file.
None of the above is a test of your workload. quantprobe cannot yet run your tasks.

## Scope - read before forwarding

- Every number here is ONE user's speed. Aggregate throughput under concurrent users
  is a different quantity, and on the one machine where we measured it, concurrency
  INVERTED the right choice of model: a dense 7B in VRAM served 219 tokens/s
  aggregate at 32 streams while a RAM-offloaded 30B MoE capped near 40 - yet the MoE
  wins at 1 stream. Do not size a multi-user service from this page; measured
  multi-user notes, where they exist for this placement, appear above (one
  Pascal-class box - treat as indicative, not a prediction for your hardware).
- Speeds are decode (writing). Prompt reading is a different regime; the one
  measured row above shows why its number is not quotable here.
- Quality lines are proxies or other-model measurements, labeled as such. The
  deciding test for a business decision is your own workload.
- Machine constants are the named preset's, measured on the box that preset
  describes - not on the reader's machine.

## Reproduce

  quantprobe report --gguf Qwen3.8-27B-Q4_K_M.gguf --machine 2016-xmp \
                    --bench-log qwen38_bench.log
  quantprobe bench  --gguf Qwen3.8-27B-Q4_K_M.gguf --machine 2016-xmp
                    (replaces the predictions above with your measurements)

Sources: Law 4 (tok/s = eta x bandwidth / active bytes); placement engine and
binding-constraint decomposition, quantprobe v1.29.0; fragile-band probe
weights/data/prereg101_probe_qwen38_q4.log; capability tables
docs/QUANT_QUALITY.md. Predictions +/-25%; misses published at the same size
as hits.
```

## 8. Implementation checklist

- quantprobe/plan.py: extract build_rows(args) (compute prefix through cap_find/bc,
  returns the dict of section 2); run() consumes it. No print moves.
- quantprobe/spec.py: apply() also sets a.arch (one line).
- quantprobe/report.py: new module (~300 lines [est]) - build_rows + pure advice
  functions + recipes.find + calibrate.load + optional bench-log parse -> Markdown.
- quantprobe/cli.py: subparser + dispatch (~10 lines).
- 340-cell byte-identity check: RUN pre-merge, 340/340 identical (see section 2 - the
  result lives there; no script is kept because post-merge it could only diff the tree
  against itself).
- tests/smoke.py, all DELIVERED (mutation-checked 2026-08-16, 4/4 mutations killed):
  (1) run-vs-build_rows row parity on 3 preset cells
  (t_plan_stdout_is_byte_identical_after_report_refactor); (2) report-vs-plan number
  parity anchored to the Verdict line and RECOMMENDED row on TWO preset cells with
  different winners, so a memorized artifact cannot pass
  (t_report_writes_a_forwardable_markdown); (3) template guard + bench-log refusal in one
  artifact (t_report_template_guards_and_bench_refusal) - M1 verbatim as a block, M2 line
  by line (the scope renderer bullet-prefixes its lines, so `M2 in md` can never pass),
  no register IDs/prereg/weights-paths in the body, ASCII-only, every placements row
  labeled on its final line, and the 8.03B-vs-27.3B param mismatch refusing the ratio.
  NOTE the labeling rule as SHIPPED (and mocked in section 7): every number-bearing line
  or wrapped block closes with its label; the Verdict speeds additionally carry the word
  PREDICTED or [measured]. "Every speed line carries PREDICTED or [measured] on the line
  itself" (this doc's first phrasing) overstated - placement rows and counterfactual
  speeds are [derived], and a wrapped row's label sits on its final line.
- README.md Commands block + QUICKSTART: one line each - DELIVERED.
