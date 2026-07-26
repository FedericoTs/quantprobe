# Changelog

## 1.11.1 - 2026-07-26

**`quantprobe auto` was recommending a placement 20% slower than the one `plan` finds for the
same model** — on every preset MoE, via the flagship one-command path. On the reference box it
proposed the hybrid placement at 22.4 tok/s when the expert-split placement, which it could not
see, runs **26.9**.

`auto` sets `a.model = None` to hand the law explicit parameters instead of a preset name. That
also discarded the preset's verified layer count, and without a layer count the planner
deliberately suppresses the MoE split row rather than print `-ot` flags it cannot ground. So the
row silently vanished from the entire `optimize` frontier that `auto` drives.

### The class this belongs to, and how it is now closed

Classifying all 22 defects shipped since v1.5.2 by root cause puts **7 of them (~32%) in one
class: a fact expressed in two places with nothing forcing them to agree.** Inside it, one shape
recurred **four times** — a value with a fallback gets a second reader written without the
fallback (v1.9.0 `target.py`, v1.10.5 `runtime.py`, `plan`'s layer-count note, and now `auto`).
Each was fixed where it was found; the shape never was.

- **One resolver.** `plan.effective_n_layer()` is now the only place that fallback exists.
  `plan`, `optimize`, `runtime` and `auto` all call it.
- **`audit.py` check C** fails the build if any module re-implements the fallback, and compares
  the machine table in `plan.MACHINES` field-by-field against the copy embedded in the published
  simulator's JavaScript. The existing parity test compared `evalCore` with hardware passed in
  explicitly, so a drift in that table was invisible to it — and it is already 5 presets behind.
  Both guards are mutation-tested.

Also fixed: `plan` on a preset printed exact `-ot` flags for layers 10–47 and then told the user
the layer count was missing and to re-run with `--gguf`. Same shape — the note read the raw CLI
flag while the placement rows read the effective value.

73 tests green; all five verify layers pass.


## 1.11.0 - 2026-07-26

**quantprobe was telling everyone running a sub-4-bit quant that their GPU was useless for it,
and recommending a placement 2.4x slower than the one it rejected.** On `gemma4-12b` at 3.51
effective bits it predicted 1.0 tok/s all-in-VRAM and pointed users to pure CPU at 3.9. Measured
on the GPU placement it discarded: **9.56 tok/s.** Two compounding bugs, one of them a finding
this project had already published and never wired into the code.

**1. There is no low-bit GPU decode collapse.** The planner gated efficiency on bit-width
(`geta if bits >= 4 else gl`, gl = 0.04 on this card), so 3.99 bits predicted 8.75x slower than
4.00. [Pre-registration #16](preregistrations/2026-07-26-gl-format-not-bitwidth.md) measured the
same 7B in three quantizations, all in VRAM, changing nothing else:

| format | bits | decode | prefill pp2048 |
|---|---|---|---|
| Q4_K_M | 4.5 | 20.03 +/- 0.04 | 27.49 |
| Q2_K | 2.8 | 19.17 +/- 0.03 | 17.71 |
| IQ3_XS | 3.3 | 18.11 +/- 0.05 | **4.04** |

A 10% band across 2.8-4.5 bits. The lowest efficiency ever measured on this path is 0.272; the
constant sat 6.8x below that floor. The collapse is real but lives entirely in **prefill**, where
IQ3_XS pays 6.8x - dequantization is compute, prefill is compute-bound, decode is not. LAWS.md
had already recorded that this was format-dependent rather than bit-width-dependent back on
2026-07-25; the finding simply never reached the code. **That gap - a measured result sitting in
a document while the tool keeps shipping the thing it disproves - is the real defect here.**

**2. Dense model sizes were inflated by up to 125%.** Size came from params x bits with attention
held at >=4.5 bits; for a dense model the tables set always-active = total, so the whole model was
priced at 4.5 bits and size stopped responding to bit-width at all. A real 12B at 3.51 bits read
as 7.2 GB against an actual 5.2 GB file, which pushed it past the VRAM ceiling and deleted the
all-in-VRAM row entirely. Capacity checks now use the real file size whenever the GGUF is on disk.

### What this means if you run local models

- **Sub-4-bit quants are fine on old GPUs.** Q2_K, Q3_K_M and Q3_K_L are what people run when a
  model nearly fits, and quantprobe was steering all of them to their CPU. Re-run `plan`.
- **Once a model fits in VRAM, quantizing further buys almost nothing.** Q4_K_M -> Q2_K here was
  36% smaller and **4% slower**. Quantize to make a model FIT; once it fits, take the highest bits
  that still fit. `plan` now says so, because the ranking alone would tell you otherwise.
- **Avoid IQ-format quants on Pascal-class cards if you send long prompts.** Decode is unaffected;
  prompt processing falls off a cliff (6.8x).

### Verification

- Every published anchor is **bit-identical** - largest movement across all anchor rows 0.0000%.
- 68 tests green. Two of them had **asserted the collapse** and were replaced; one demanded that a
  2-bit model be slower than a 4.5-bit one, which is backwards on bytes alone.
- New `VRAM_GAPS` ratchet covers all seven all-in-VRAM points measured on the reference box.
  Every MEASURED_ANCHOR was a MoE-hybrid or disk-stream row - **not one covered all-in-VRAM**,
  which is exactly why a 9.5x error survived there through a public release. The law is knowingly
  pessimistic in this regime (-2% to -67%, always under, never over); rather than refit on a
  pattern with no clean form, the errors are ratcheted so they may shrink but never grow.
- `verify.py` now finds llama.cpp and a model itself, and **exits 2 if the end-to-end layer did
  not run**. It had been reporting "all layers passed" while that layer silently skipped on every
  single run, because it needed a `--llama-dir` nobody remembered to pass.

### Also

- `auto --custom` declined the surgery based on the speed-winning bit-width, which only ever
  fired *because* of bug 1. It now asks whether a >=3.5-bit build is viable on hardware the user
  already owns.
- Em-dashes removed from every module that prints to a terminal: the Windows console cannot
  render them, so the tool's own advice displayed a broken character.
- Recorded: the tool's bench protocol (`-n 32`) agrees with the research protocol (`-n 128`)
  within 5%, so contributed data is comparable. And a GPU entering a run at a boosted clock
  (1847 vs 936 MHz) inflated one measurement by 28% - the H3a lesson in the dangerous direction,
  since a flattering number invites no scrutiny. Clock state is now logged alongside memory.


## 1.10.5 — 2026-07-26

**Two commands in this tool disagreed about the same input, and the disagreement was corrupting
our own accuracy reporting.** On an identical model and machine, `plan` predicted 19.6 tok/s and
`bench` predicted 22.1. Measured: 19.88. `plan` was right to within 1.4%; `bench` was 11% wrong —
so every predicted-vs-measured figure the tool reported depended on which command produced it.
Predicted-vs-measured is the entire product, which makes this the worst class of bug this
project can have.

Two independent causes, both introduced by recent work:

- **Double correction.** `bench` applied a file-size calibration on top of `autospec`, which
  already derives effective bits from that same file — correcting one discrepancy twice. The
  calibration is now skipped whenever the spec came from the file (it still applies when a
  preset's assumed size must be reconciled against a real one).
- **Divergent fallbacks.** `runtime` did not fall back to a preset's verified layer count the
  way `plan` does, so `run`/`bench` silently dropped the MoE split placement for preset models
  (18.9 vs 16.6). The n_layer threading was "fixed" earlier the same day in `plan` and
  `optimize` and written differently here — a checklist found the first instances and missed
  this one.

**The structural guard:** `plan`, `run` and `bench` must now produce identical predictions for
identical input — asserted across preset models, custom specs and multiple hardware classes in
the suite, and against a real GGUF in `verify.py`'s end-to-end layer (the calibration path only
exists when a file does, so the offline suite cannot reach it). It caught the second bug within
seconds of being written, and mutation-testing it exposed a hole in the guard itself, which is
why the file-based assertion lives in the gate.

After the fix: predicted 19.6, measured 19.22 ± 0.7 — **−2%**. The law was accurate the whole
time; we were misreporting it. 67 tests.

## 1.10.4 — 2026-07-26

**A stated scope limit instead of a silent error.** We do not model multi-token prediction or
speculative decoding. A user measured 29-30 tok/s on Qwen3.6-35B-A3B where this tool predicted
~16 for the placement alone — MTP emits several accepted tokens per weight read, so the law
reads LOW by roughly the acceptance multiplier (~1.8x in his case).

That is a **missing factor, not a wrong one**: the bytes-and-bandwidth arithmetic is unchanged
and the multiplier sits on top of it. Now stated in README's limitations and in LAWS.md at the
point where Law 4 is defined, so nobody discovers it by being surprised.

Measuring it is staked as a new arm (S-e) of the Law 6 pre-registration, written **before**
acquiring any MTP-capable model — no GGUF on this machine has MTP heads. The sign is genuinely
open: our own measured corollary found draft-model speculation **2.3x SLOWER** on MoE, and MTP
should escape that because its heads reuse the forward pass. "Should" is not "does", which is
why it is staked rather than assumed.

## 1.10.3 — 2026-07-26

**`python verify.py` — the pre-release gate, and the answer to "how do we not break this as it
evolves".** Every bug that reached users was caught by a different layer, and never by the one
before it. So the gate runs all four, and a skip is never reported as a pass:

1. **unit + invariant tests** — properties over ~380 placement rows, mutation-tested
2. **installed artifact** — must import and run from site-packages, not a repo cwd, and the
   installed version must MATCH the repo (this check immediately caught the gate verifying a
   stale 1.10.0 against a 1.10.2 tree, and passing)
3. **end-to-end against real llama.cpp** — runs the tool's own recommendation and compares
   predicted vs measured; fails on >15% measurement spread or a >25% prediction miss. This is
   the layer that caught the 82%-below-prediction config that 54 green unit tests slept through
4. **measured anchors** — every published number is now a regression test. If a change makes
   the law stop retrodicting reality, this fails. Seeded with the flagship 19.3 tok/s (which
   was *not* previously covered), the corrected 18.35 baseline, and the two disk-stream anchors

Add a row to `MEASURED_ANCHORS` whenever a number is published, and the claim can never
silently drift away from the code. 66 tests.

## 1.10.2 — 2026-07-26

**Correction: the MoE partial-offload gain is +12.4%, not the +34.7% shipped in 1.8.0.**

The original measurement compared our split against an all-experts-to-CPU baseline running
**without `--no-mmap`** — a flag this tool has recommended on that very row for many versions,
and which llama.cpp itself warns about. The control was therefore a worse-than-recommended
version of the thing being beaten. Re-measured with every cell configured the way the tool
actually tells you to run it (warm cache, r=3):

| config | tok/s | vs correct baseline |
|---|---|---|
| baseline, all experts to CPU | 18.35 ± 0.48 | — |
| split at the cutoff the tool picks | 19.47 ± 0.77 | +6.1% |
| split at the measured peak | 20.62 ± 0.26 | **+12.4%** |
| one step past the ceiling | 10.59 ± 0.07 | −42.3% (cliff reproduced) |

About 21 of the original 34.7 percentage points were `--no-mmap` alone. The mechanism, the
capacity cliff, and the prefill result are unchanged; only the size of the decode win moves.
README, QUICKSTART and pre-registration #13 all corrected, with the original left visible
beneath the correction.

**Found via a community report.** A user pointed out that llama.cpp's own `-fit` auto-placement
worked well on his 12 GB card. Checking whether `-fit` beat our recommendation (it does not
here — 4.06 tok/s on a 6 GB card) led to re-examining our own baseline, where the real problem
was. Also noted for future work: **we do not model multi-token prediction at all**, and a user
running MTP measured ~1.8x above our estimate — a missing term, not an error, but a real gap.

Process lesson worth recording: a benchmark's *control* deserves the same scrutiny as its
treatment. No unit test catches this; it is a measurement-design error. 65 tests.

## 1.10.1 — 2026-07-26

**Tensor-role registry — the part of a recipe that legitimately transfers.** A recipe has two
halves with opposite properties: the machine half is *computed* (Law 4 already prices any
hardware combination nobody has ever contributed), and the model half is *measured* (Law 3:
fragility is not predictable). Between them sits structure — which tensor classes a model has
and which are always-active — and that IS a property of the architecture, knowable for a model
nobody has probed.

- The builder now classifies every tensor by role before quantizing, and **reports any weight
  class it has no protection rule for** instead of silently compressing it. This is the exact
  bug we shipped once: hybrid SSM models name their state tensors `ssm_*`, our pattern matched
  only `attn_*`, and every one landed at the aggressive base level (v1.6.4, −24% ppl).
- Verified across three architecture families (dense, classic MoE, SSM-hybrid MoE) with no
  unrecognised classes.

**Why recipes are not synthesizable from other recipes** — worth stating, since it is tempting:
using our four measured bands to predict a fifth model would have protected Mistral-7B's layers
24-31 (all three precedents are late-fragile, and it is a dense 7B like Qwen2.5-7B). Mistral is
**early**-fragile: quantizing layers 0-7 costs +6.53 ppl versus +0.26 for 24-31. The synthesizer
gives away the expensive band and protects the cheap one — a **25x** error, in our own atlas.
Structure transfers. Fragility does not. 65 tests.

## 1.10.0 — 2026-07-26

**`quantprobe recipes` — the community fragility atlas.** The expensive part of the custom
pipeline is the probe (hours on a large model). But Law 3's result is a property of the MODEL,
not your machine: Qwen3-30B's fragile band is layers 36-47 whether you measured it on a GTX 1060
or an H100. So it needs measuring **once, globally, ever** — and everyone after that skips
straight to the build.

- `quantprobe recipes` lists what has already been measured. Seeded with **four models, every
  band read from a raw log in this repo** — Mistral-7B (0-7, **early**-fragile, 27x the median
  band), Qwen2.5-7B (21-27, late, 2.5x), Qwen3-30B-A3B (36-47, late, 3.7x), Qwen3.5-35B-A3B
  (30-39, late, 2.9x). Gemma-4-12B is claimed in LAWS.md but its band log was not located in
  this sweep, so it is **not** included — a recipe without evidence does not ship.
- `quantize --recipe <key>` builds using a measured band instead of the default guess, and
  **refuses** if the recipe's layer count does not match your file, because a band is only
  meaningful for the model it was measured on.
- When you quantize a model someone has already probed, the tool **says so** and names the
  recipe — the default `--protect-late 12` is a guess, and a measurement beats it.
- Every recipe carries its evidence: the raw log, the eval corpus, the hardware, the method.
  Contributions are held to the same bar. A recipe you cannot check is a recipe nobody should use.

Why this matters: Mistral is **early**-fragile while its architectural near-twin Qwen is late.
The default guess protects the wrong end of that model entirely — which is the whole point of
Law 3, now enforceable in one flag instead of six hours. 64 tests.

## 1.9.2 — 2026-07-26

Consistency audit. The trigger: 54 green unit tests sat there while the tool shipped a config
that measured 82% below its own prediction. Case-by-case tests cannot catch that class.

- **Invariant tests over the whole configuration space** (~380 placement rows swept across MoE
  and dense, five GPU classes, three RAM sizes, four bit-levels, with and without context).
  They assert properties, not examples: any row overriding tensors to CPU carries `--no-mmap`;
  no row's flags contain prose; `-ngl` is always a valid integer; split regexes reference only
  real layer indices; rows are sorted and positive.
  **Each was mutation-tested** — the two bugs that actually shipped (v1.6.5 prose-in-flags,
  v1.8.0 missing `--no-mmap`) were deliberately reintroduced and each invariant caught its own
  bug by name, then went green on restore. A test that cannot fail is not a test.
- **Browser calculator parity restored.** The simulator did not know about MoE partial expert
  offload, so it could not show the +34.7% the CLI recommends. Added, with the same 1 GB desktop
  reserve, and verified numerically identical to the CLI (0.00% delta on the reference case).
- Dense paths swept and confirmed unregressed across four hardware classes.

60 tests (counted, not estimated).

## 1.9.1 — 2026-07-26

Both fixes found by an end-to-end run, not by the unit suite — the tests were all green while
the real pipeline was wrong.

- **The MoE split placement was missing `--no-mmap`.** The all-experts-to-CPU row has carried it
  for versions; the new split row (1.8.0) did not, even though llama.cpp itself warns that
  tensor overrides to CPU with mmap enabled cost performance. Measured on the split placement:
  **16.45 tok/s with mmap vs 18.70 without (+13.7%)**.
- **`bench` now refuses to report a number whose own error bar is huge.** A first read from a
  cold file cache produced `4.01 ± 2.16 tok/s` (54% spread) against a warm value of 18.7, and
  the tool printed it as a result. It now flags any run with >15% spread as unreliable and tells
  you to re-run rather than letting someone quote an artifact.

After both: predicted 22.1, measured **19.36 ± 0.61** (−12%, inside the stated ±25% band). The
prediction is on the optimistic side of measurement, which is recorded rather than tuned away.

## 1.9.0 — 2026-07-26

**Honest time estimates.** This tool told users a probe takes "~30-60 min". On a 35 GB source
it actually took **5h40m**, and the importance-matrix pass another **4h30m** — so
`auto --custom` on a large model was a >10-hour commitment advertised as under an hour. That is
the exact kind of unmeasured claim this project exists to catch, and it was ours.

- **Every long operation now estimates itself up front**, derived from measured throughput
  (2.8 GB/min quantize; perplexity ~0.55 min/GB when the working file fits memory, ~2.4 when it
  spills). On the case above the estimate reads ~6.0 h against 5h40m actual.
- **It says WHY**, not just how long: the dominant term is whether the Q6_K working file exceeds
  your RAM, because then every perplexity run pages from disk. Users get told that a smaller
  source or more RAM turns hours into minutes.
- **Progress with a live ETA**: `step N/M`, elapsed, and remaining — recomputed from *your*
  machine's measured pace after the first step rather than trusting our constants.
- **Runs over 2 hours ask for confirmation** (`--yes` / `-y` to skip). No more discovering the
  cost after committing to it.
- The imatrix pass carries its own estimate and points at `--no-imatrix` and `--imatrix-chunks`
  as the levers.

Audit fixes shipped alongside: `probe`'s printed recipe had gone stale by two versions (missing
the SSM and shared-expert protections — anyone copying it built the old recipe); it is now
generated by the same builder `--apply` uses, so it cannot drift again. `target` was missing the
layer count and so would silently withhold the MoE offload placement. Dead no-op branch removed
from `bench`. 54 tests.

## 1.8.0 — 2026-07-26

**MoE partial expert offload** — the placement the planner never offered, measured in
pre-registration #13 and now shipped.

- Until now, MoE models got all-experts-to-CPU or nothing, so a mid-size GPU sat half empty.
  The planner now also evaluates keeping the **first K expert layers on GPU** and emits the exact
  `-ot` regex. Measured on the reference box: **+34.7% decode** (15.18 → 20.44 tok/s) and
  *[CORRECTED 2026-07-26 to +12.4% — that baseline lacked `--no-mmap`; see pre-registration #13]* and
  **~2-3x prefill** vs all-experts-to-CPU. Flagged by three separate users in one session.
- **The cutoff is computed conservatively on purpose.** Overshooting VRAM is a cliff, not a
  taper: one step past the ceiling measured **−29%**. A 1 GB desktop reserve is subtracted
  (measured 0.8–1.5 GB held by Explorer/compositor/browser during the sweep), because a real
  machine is not an empty GPU.
- **The row is suppressed when the model's layer count is unknown**, with a note telling you to
  pass `--gguf`. A speed figure whose printed command cannot deliver it is the v1.6.5 bug class;
  the planner will not do that again. Verified layer counts added for `qwen3-30b` and
  `deepseek-16b` (read from real GGUFs, not guessed).
- `optimize` and `run`/`bench` see the new placement too, so the recommendation and the launched
  command agree.
- **Test honesty:** the optimizer backtest previously asserted the 2.5-bit hybrid (18.9) in the
  top-2. That config is now legitimately superseded by a measured-better one, so the test was
  rewritten to assert what must remain true (top pick grounded in a measured mechanism,
  realizable by stock llama.cpp, beating the number it replaced) rather than pinned to an
  obsolete answer. 54 tests.

## 1.7.0 — 2026-07-26

Everything measured in pre-registration #12, now in the tool:

- **Importance-matrix calibration.** `probe --imatrix auto` generates one and builds with it;
  `quantize --imatrix FILE` uses an existing one; `auto --custom` now does it automatically
  (`--no-imatrix` to skip). Measured **−8.5% perplexity at ~3 bits, at zero file-size cost and
  no measurable speed cost** — the single largest quality lever in the recipe, and one this
  project had never used.
- **Always-active tensor protection.** Shared-expert tensors (`ffn_*_shexp`) fire on every token
  while routed experts fire ~8/256, and they are heavy-tailed. They are now pinned at q8_0 in
  every depth-aware build: **−3.2% ppl for ~0.65% more bytes**. The rule is emitted FIRST because
  llama.cpp resolves `--tensor-type` first-match-wins — placed last it is a silent no-op.
- Combined, these reverse a head-to-head loss: our build went from +8.9% worse than a strong
  competing recipe at matched bytes to **1.1% better**. Full method, numbers and scope limits:
  `preregistrations/2026-07-25-recipe-upgrade-shexp-imatrix.md`.
- LAWS.md: Law 3 refinement (structural and statistical allocation are orthogonal, stack with
  diminishing returns) and a Law 4 scope confirmation (weight *content* does not affect the
  speed law — staked and measured). 53 tests.

## 1.6.5 — 2026-07-25

- **`run`/`bench` no longer try to launch a placement stock llama.cpp can't execute.** The
  three-tier expert-cache row's "flags" field is a human-readable description
  (`-ngl 99 + runtime-managed expert cache`), not argv — when that row ranked first, `run`
  split it into arguments and handed llama-cli a bare `+`, which died with
  `error: invalid argument: +`. `optimize` and `auto` already filtered unrunnable rows;
  `run`/`bench` now do the same, print a note naming the faster-but-unrunnable placement, and
  launch the fastest stock-llama.cpp placement instead. Reported by a user running
  GLM-4.5-Air-UD-TQ1_0. New regression test (51 total) asserts no prose can ever reach the
  launch command.

## 1.6.4 — 2026-07-25

- **Depth-aware builds now protect SSM (state-space) tensors, not just attention.** Hybrid
  SSM+attention+MoE architectures (e.g. Qwen3.5-class) name their recurrent-state tensors
  `ssm_*`, which the `attn_.*` protection pattern never matched — so every custom build left
  them at the aggressive base level (Q2_K) with zero protection. Found running our own APEX
  comparison: it cost a measured wikitext ppl of 8.81 on a 35B-class model, worse than a plain
  community IQ2_M reference at a SMALLER size. The fix costs ~56 MB on a 13 GB file (SSM tensors
  are tiny) and is a straight quality win with no real tradeoff. Manually verified (dry-run
  command + a live rebuild) before shipping. 50 tests unchanged (same reasoning as 1.6.3 - this
  path needs a real hybrid-architecture GGUF to exercise end-to-end).

## 1.6.3 — 2026-07-25

- **`probe` no longer silently fails on large models with small VRAM.** Its internal perplexity
  measurement hardcoded `-ngl 99` (full GPU offload) for every intermediate test file — but a
  probe's Q6_K reference build can be far bigger than the model's final compressed size (e.g.
  ~23 GB for a 35B-class source), so on a 6 GB card it OOMs, llama-perplexity exits without a
  result, and probe silently reported `PPL None` for every band with zero diagnostic. Found
  running our own APEX-vs-depth-aware comparison. Now: on an OOM-shaped failure, probe retries
  once at `-ngl 0` (CPU, slower but always fits) before giving up, and prints the last lines of
  llama.cpp's own output if it still can't parse a result — never another silent `None` again.
  Manually verified against the real failure case (recovered PPL 6.5161 on a file that OOM'd at
  ngl99). 50 tests unchanged (this path needs a real GGUF + llama.cpp to exercise, consistent
  with the existing probe test coverage).

## 1.6.2 — 2026-07-25

- **7 new presets** (13 total): `qwen3-235b`, `glm-4.7` (358B), `glm-744b` (= GLM-5.2, exact total
  753.3B from HF safetensors), `kimi-k2.6` (1058.6B), `gpt-oss-120b`, `llama-70b`, `deepseek-16b` —
  every repo verified live before shipping. The verification caught a real trap: unsloth/GLM-4.7-GGUF
  is the 358B model; the 744B-class one is GLM-5.2. Estimated fields are marked [est] in-source.
- **No more currency figures** in `optimize` output: upgrade suggestions stay ("+16GB RAM", "NVMe"),
  prices don't (region- and time-dependent guesses). Internal relative-cost ranking unchanged.
- **Split multi-part GGUFs supported** (`-00001-of-000NN`): grouped into one logical file
  (bits computed from the SUMMED size — honest), all parts fetched, recursive repo listing
  (big repos keep quants in subfolders). Repos that only have Q8/BF16 so far (brand-new
  giants) get an honest "no ready-to-run quant yet" with the `--custom`/plan escape hatches
  instead of a silent 800 GB pick. 50 tests.

## 1.6.1 — 2026-07-25

- **`python -m quantprobe` now works** — the PATH-proof entry point. On Windows, `pip install`
  frequently lands `quantprobe.exe` in a user-site Scripts folder that is not on PATH ("'quantprobe'
  is not recognized"); `python -m quantprobe <anything>` is identical and always works. Found when
  it happened on our own machine. 46 tests.

## 1.6.0 — 2026-07-25

The full-customization pipeline, delivered as one decision-making command:

- **`quantprobe auto` with no arguments is now interactive**: detects the machine, asks for the
  model (preset or any HF GGUF repo), asks one question — best standard quant (skip
  quantization), full custom probe-and-build, or a speed target — and offers to launch when
  ready. Clean one-line refusal when there is no terminal to ask.
- **`--custom` is now machine-gated by the laws.** If the optimizer wants ≥3.5 bits on your
  hardware, the surgery doesn't pay (Laws 1–2: the fragile-band fix matters below ~3 bits) —
  auto says so and fetches the optimal standard quant instead. `--force-custom` overrides.
  The same command on a 6 GB/16 GB box still builds the depth-aware file, because there it wins.
- 4 new tests (45 total, recounted: the historical "41" was itself off by 4 — real ladder 37 → 41 → 45): gate, force-override, wizard with piped answers, wizard EOF.

## 1.5.2 — 2026-07-25

- **Unknown preset names now fail loudly.** `plan/target/optimize --model <unknown>` used to fall
  back silently to a 13B default and produce plausible-looking wrong numbers (found while
  answering a real user question). Now: clean error listing the presets, plus the two escape
  hatches (`--total/--active` to describe any model, `--gguf` to read the exact spec from the
  file). Same for `--machine`. 4 new tests (41 total).
- Docs: the one-command pipeline (`auto <model> --custom --run`) is now the first thing in
  QUICKSTART; bit-level selection clarified (quantize = fixed validated recipe, `auto`/`fetch`
  = standard quants).

## 1.5.1 — 2026-07-25

**`auto --custom` — the personalized recipe, now truly one command.** Fetches the best
requantizable source from the repo (prefers Q8-class over f16: half the download, identical PTQ
quality), auto-fetches the WikiText-2 eval corpus (1.3 MB, once), probes YOUR model's fragile
band (~30-60 min), and builds the depth-aware GGUF - personalized to the model, sized by the
optimizer. The fast path (closest community quant) remains the default; every fast-path run
advertises the upgrade. 41 tests.

## 1.5.0 — 2026-07-25

**One command from empty machine to running model — and the law, watchable.**

- `quantprobe auto <model> [--tps N] [--run]`: machine auto-detected -> optimizer picks the
  effective bits -> the HF repo's file list is scanned and the closest quant matched BY SIZE
  (bits = size x 8 / params; format-agnostic) -> resumable fetch -> run command (or --run
  launches). First live run picked the exact file independently measured at 18.32 tok/s.
  The custom probe path (better quality at the same bytes) is advertised on every run.
- Dashboard v2.1: single-viewport app (fixed sidebar, internal chat scroll), streaming replies
  with VISIBLE thinking, a thinking TOGGLE + per-reply anatomy (TTFT / thinking / answer), and
  the NEURON GALAXY - every expert of every layer as a dot, lit per generated token, colored by
  its memory tier. Honesty printed on the panel: uniform sampling is the statistically exact
  picture under the measured flat-routing law; stock llama.cpp exposes no router telemetry.
- Hardening from real use: completion-probe readiness (llama-server reports healthy before
  weights load), exclusive port bind (Windows silently allowed double-binds), RTX 50-series in
  the detect table, per-card multi-GPU bandwidth aggregation.
- 40 smoke tests. Eleven commands.

## 1.4.0 — 2026-07-24

**`quantprobe optimize` — the cheapest path to a target speed.** A pure search layer over the
frozen law (no physics touched; anchors untouchable by construction): bits ladder x placement x
KV levers x hardware deltas, Pareto-ranked by quality cost then euros, with realize-commands.

- Backtested: blind on the reference box it rediscovers the measured-best config (2.5-bit
  depth-aware hybrid, 18.9 predicted / 19.30-20.02 measured) on the frontier.
- Boundary-aware: on a 16 GB card with a just-over file it picks the bits-shave that crosses into
  all-in-VRAM (x4+), the pre-registration #8 lesson operationalized.
- Measured lever gates: KV-q8 blocked on weak-decode GPUs (measured -83% at 16k on Pascal,
  2026-07-24); REAP-class pruning never ranked without --allow-prune (+39% OOD ppl measured).
- Realizable-by-default: only stock-llama.cpp placements unless --any-runtime.
- 37 smoke tests. Ten commands.

## 1.3.1 — 2026-07-24

**Tier-boundary advisor.** Corollary of Law 4 made explicit: decode speed is a step function of
placement, so the marginal value of a gigabyte is ~zero mid-tier and enormous at a boundary. When a
config sits within 30% over a tier boundary, `plan` now names the gap and prices the promotion
("1.6 GB over the VRAM boundary - shave it -> ~67.6 tok/s (x4.3)"). Works for any shave lever:
quant step, tighter probed band, pruned variant, KV quantization. Validated on the pre-reg #8 REAP
pair: fires on the 14.7 GB parent, silent on the promoted 11.5 GB prune. 32 smoke tests.

## 1.3.0 — 2026-07-24

**Any hardware combination.** Prompted by a wild 744B rig (72 GB VRAM + 128 GB RAM + RAID-0 Gen5
NVMe at 3.6 tok/s) that the two-tier model under-predicted.

- Three-tier expert cache: new ADDITIVE placement row "stream from disk (VRAM+RAM expert cache)" —
  models what expert-caching runtimes (ktransformers/colibri-class) achieve; the stock-llama.cpp
  rows are untouched (validated: retrodicts the 3.6 tok/s rig at 2.9, within the law's +/-25%).
- Multi-device inputs: comma lists aggregate — `--vram 24,24,24 --vram-bw 936,936,936` (x0.85 TP
  efficiency [est]) and `--disk-bw 14,14` (x0.75 stripe [est from the RAID-0 eta 0.66 datapoint]).
- Simulator carries the same three-tier row (CLI parity).
- Validation matrix green: every measured anchor identical to the digit (30B hybrid 18.9, ctx-16k
  15.4, 110B 0.2, Laguna 0.3) with the new rows strictly additive. 31 smoke tests.

## 1.2.0 — 2026-07-24

**Zero-configuration.** The minimal command is now `quantprobe plan --gguf model.gguf` — nothing else.

- New `quantprobe hw`: detects RAM (sticks + configured MT/s -> peak GB/s), GPU(s) (nvidia-smi +
  name->bandwidth/eta table; multi-GPU aggregated at 0.85 TP efficiency), Apple unified memory.
  Every value tagged [os]/[table]/[default]; nothing leaves the machine. `--measure FILE` adds a real
  sequential-read disk measurement.
- GGUF autospec: `--gguf` alone yields total/active params (tensor sums + expert metadata), TRUE
  effective bits (file size), EXACT KV bytes/pos (MLA-aware). Explicit flags always override.
- Auto-detection engages only when no `--machine` and no hardware flags are given — presets/flags
  are unchanged and remain the way to estimate a machine you are not running on.
- `--bits` freed to continuous values (e.g. 2.88) + nearest-key quality lookup.
- Verified: auto-detected reference box reproduces the hand-measured `2016-xmp` preset exactly
  (17.6 == 17.6 tok/s on the same GGUF). 28 smoke tests green.
- Pre-registration #7 HIT: Laguna S 2.1 (118B) on the 2016 desktop — staked 0.2-0.4 tok/s before
  the download, measured 0.38 +/- 0.17 (llama-bench, mainline b10098, no draft).

## 1.1.0 — 2026-07-23

**Law 4 v2: the context term.** Prompted by u/RogerAI--fyi's observation that the decode law omitted
per-token KV reads; measured same-day on the reference box (tg32 clean 20.02 ± 0.02 → 16.12 ± 0.06
at depth 16384, −19.5% vs pre-registered −8…−15% — a published near-miss; η_kv ≈ 0.70 single-point
calibration).

- `--ctx N` on `plan` / `target` / `run` / `bench`: adds per-token KV reads (served from the tier KV
  lives on) **and** KV memory to the fit check — large contexts can flip the winning placement.
- `bench --depth N`: measure the context term on your box (llama-bench `-d`); prediction follows depth.
- Per-model KV bytes/pos in presets (MLA ≈10× smaller: DeepSeek 31 KB vs Qwen3-30B 98 KB; SWA [est]).
- `--kv-per-pos KB` override for custom models; `run --ctx` launches llama.cpp with `-c` set.
- Simulator: context-depth input, same math, CLI-parity verified.
- Chart I (`weights/data/x_chart_I_kvdepth.png`): measured KV-depth slope vs the law.
- New pre-registered prediction: 30B-A3B Q4 pure-CPU on DDR4-45 at 16k = −29% (8.0 → 5.7 tok/s).
- 6 new smoke tests (19 total): ctx=0 identity, monotonicity, placement-dependence, calibration
  anchor, fit-flip, `bench --depth --dry`.

## 1.0.0 — 2026-07-22

Initial public release: four placement laws, 8-command CLI (plan / target / fetch / quantize / probe /
run / bench / dashboard), depth-aware GGUF compression verified end-to-end, browser calculator,
opt-in community datapoint loop, validation bundle for the 19 tok/s claim.
