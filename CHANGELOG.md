# Changelog

## v1.30.0 - 2026-08-19

- **`plan` now states the expert-count CEILING instead of offering the dial** - `--override-kv
  expert_used_count` is widely traded as a free speed knob for MoE on small hardware. Prereg #107
  measured it end to end and it is bounded by the always-active floor: on Qwen3.6-35B-A3B the
  routed experts own 22% of the active bytes, so the knob cannot beat ~1.24x even at k=1. Law 4
  predicted the curve from the file to within 2% (k=4 1.146x vs 1.125 predicted, k=2 1.175x vs
  1.200), and every point costs more quality than the speed is worth: k=4 buys 15% for +1.51 PPL,
  k=1 buys 45% and the model is destroyed (PPL 2277). So the tool prints the ceiling computed from
  YOUR file and cites the evidence. New `spec.expert_ceiling()`; dense models and unreadable
  expert metadata produce no line at all. (L-30, V-22)

## v1.29.0 - 2026-08-18

- **Every measured tok/s now carries a residency verdict** - `bench` prints free RAM against the
  model's size and refuses to present a figure as stable when the model does not fit. This came
  from failing to reproduce our own headline (C-32): a 13.15 GiB build published at
  14.86 +/- 0.36 tok/s returned 11.0 days later, same box, same binary, same command. The
  existing guard only catches *noisy* runs, and that measurement was 2.4% spread - it sailed
  through. Variance and reproducibility are different questions: where the file is larger than
  free RAM, part of every decode pass streams from disk, and the number describes that minute's
  machine state rather than the model. New `detect.ram_free_gb()` and `detect.residency()`;
  unreadable free RAM reports as *unknown*, never as fine.

- **A recipe with a published build says so before `quantize` rebuilds it** - the atlas exists to
  skip work already done, and skipping the build saves hours plus a high-precision source
  download several times larger than the output. `quantprobe recipes` lists the prebuilt file and
  `quantize --recipe` names it before starting.

- **The binding-constraint line now carries a validation scope label** - prereg #95's
  variance-attribution arm (Sobol, decided 1000/1000) did NOT confirm the staked
  flag-level mapping: the placement lever (-ngl) carried the variance, not the mapped
  flags. Per the staked kill rule, every `plan`/`report` classification now prints
  `validation: derived from the law, not confirmed by variance attribution` at full
  prominence, and the README and report-card assets carry the same line. The time
  decomposition itself is unchanged arithmetic; the label prices exactly the part no
  measurement confirms yet.

- **`quantprobe report`** - the one-page forwardable answer. `plan` talks to the person at
  the terminal; `report --gguf model.gguf` writes the same answer as ONE Markdown file for
  the person who was not there - the IT manager sizing a hardware buy, a consultant's
  client, an ISV writing hardware requirements. Every number carries an honesty label
  ([measured] / [derived] / [est] / UNVALIDATED) on the line or block it qualifies; the
  verdict speeds are spelled PREDICTED or [measured] in words; register IDs stay out of the
  body (the reader has no FINDINGS.md - the failure mode issue #1 demonstrated); two
  mandatory misread-prevention blocks (predictions-were-not-run, one-user-not-throughput)
  render verbatim in every artifact; `--bench-log` quotes a llama-bench run beside the
  prediction and REFUSES the ratio when the log's param count is not this model's. It is a
  renderer over `plan.build_rows()` - the same engine, so the two commands cannot disagree
  about the same file (plan's stdout verified byte-identical across the full 340-cell
  preset grid through the refactor). Full contract: docs/DESIGN_REPORT_CMD.md. Built
  design-first by a 7-agent workflow whose adversarial verify pass mutation-tested the
  tests themselves; all 4 staked mutations now kill (the first parity test provably could
  not tell a hardcoded report from a real one - that version never shipped).

## 1.28.2 - 2026-08-17

**Hotfix: v1.28.1 was broken on machines with no GPU. Upgrade if you installed it.**

The AMD refactor wrapped the non-NVIDIA detection branches in an `if gs:` guard with no
`else`, so a box with no GPU at all fell through every arm: `detect()` returned hardware
with **no `vram`/`vram_bw` keys and no GPU line at all**, and `None` propagated downstream -
the same failure mode issue #1 reported. Every CPU-only user of 1.28.1 was affected; every
GPU user was not.

- `detect()` always settles `vram`/`vram_bw` and always emits exactly one `GPU:` line. The
  empty case now falls through the same `else` as every other dead end (`_price_gpus([])`
  returns `([], [])`), which is what the pre-refactor code did.
- A test pins it **on any box**: the three probes are stubbed empty in-process, so the
  no-GPU path is exercised on GPU machines too. Verified against the actual released
  1.28.1 file - it fails there, passes here.

**How this got out, since the misses are published here too:** the full suite ran green
locally before the release, and this box has an NVIDIA card, so the broken branch was
never reached. CI - which runs on GPU-less runners - caught it within minutes on four
tests, but the release had already been tagged and uploaded in the same sequence. The rule
that follows: **CI green is a release gate, not a formality that runs after the upload.**
No release from now on precedes its CI run.

## 1.28.1 - 2026-08-17

**The first external code contribution: AMD GPUs are detected, not guessed at.**
[fboudra](https://github.com/fboudra) built the rocm-smi path end to end and reported it
from an RX 9070 XT ([PR #4](https://github.com/FedericoTs/quantprobe/pull/4), closing
issue #2).

- **AMD detection via `rocm-smi`** (Linux, amdgpu driver): name, VRAM, sclk/mclk, temp and
  the supported-frequency ceiling, parsed by a pure function that is testable without a GPU
  - the same shape the Windows registry parser uses. `detect()` tries NVIDIA first, so no
  existing machine changes behaviour; `calibrate` gets the same clock/boost instrument on
  AMD that it has on NVIDIA (identical dict shape, so ClockSampler, boost_verdict and cal_id
  were untouched); `audit-ollama` can verify a clean GPU on AMD; and the stuck-boost advice
  names `rocm-smi --showclocks`. Five parser tests shipped with the patch.
- **Follow-up on the merge (ours, not his):** `unload()` reported "cannot read VRAM" for
  BOTH failure modes, which made the "ollama is still holding the GPU" refusal unreachable -
  a working nvidia-smi plus a squatting ollama would have sent the user to fix the wrong
  thing. The tristate is restored and pinned by a test that fails if the distinction
  collapses again (mutation-checked).

## 1.28.0 - 2026-08-16

**The planner now prices hybrid linear-attention models correctly - two days after the class
shipped.** Qwen3.8-27B landed 2026-08-14 with 48 of its 64 layers as linear attention; every
prior model this tool had met was full-attention throughout.

- **KV is priced on full-attention layers only** (U-51, from prereg #101 P-5, staked before
  measuring). The old formula multiplied by n_layer, assuming every layer caches K+V with
  position - a measured 4x over-estimate on the hybrid (260 KB/pos read, ~64 real). The count
  now comes from the FILE: a block carrying an `attn_k` projection caches K+V, a linear/SSM
  block does not. Full-attention models are byte-identical to the old formula (pinned by a
  regression test against a real file); hybrids print
  `KV 68 KB/pos (hybrid: 17 of 65 layers cache KV)`.
- **The recipes atlas gains Qwen3.8-27B** - the first measured depth-aware band published for
  it, probed launch-day+1 (band 51-64, back-fragile, 2.35x under the atlas ratio convention;
  the probe binary's own banner prints 2.04 off a different median - both documented so the
  two numbers do not read as a discrepancy). Sixth entry; `quantprobe recipes` lists it.
- **Linux end-to-end CI** (.github/workflows/linux-e2e.yml): a GPU-less ubuntu runner installs
  the package, runs `calibrate` and asserts it exits 0 with "GPU: none detected" - the exact
  failure class issue #2 reported on AMD/Linux - then runs a real llama-bench and a real
  `plan --gguf` against a downloaded model. Green on its first run (34s). The AMD/Vulkan
  detection path itself arrives with issue #2's contributed patch, as v1.28.1.
- The speed receipt behind the release: quantprobe predicted 1.80 tok/s for Qwen3.8-27B
  Q4_K_M on a 2016 GTX 1060 before the model generated a token; llama-bench measured
  2.04 +/- 0.02 (+13%, floor direction) - weights/data/qwen38_plan.log and qwen38_bench.log,
  both committed. The same run is what exposed the KV gap this release fixes.
- Also in the repo since 1.27: the quant-quality campaign page (docs/QUANT_QUALITY.md - the
  35B naive-vs-recipe verdict, the capability-not-format decomposition, the 4B ceiling chain
  and its size-dependence law, the hybrid fragility result), and prereg #102's answer to
  whether effective rank could replace the probe: it cannot (P-C - 2 of 6 models, the
  front-fragile control flat), so the probe remains the honest instrument. The negative is
  published at the same size as the hits, as always.

## 1.27.0 - 2026-08-05

**AMD and Intel GPUs are now detected.** The tool was nvidia-smi-only: issue #1's contributor
ran an RX 5700 XT and `hw` printed "GPU: none detected", forcing them to hand-pass flags for
the very card class the tool exists to serve. Now:

- Non-NVIDIA adapters are read from the **Windows driver registry** (`qwMemorySize` - the CIM
  `AdapterRAM` field is a uint32 that caps at 4 GB and under-reports every modern card), with
  virtual/remote adapters filtered.
- The bandwidth table gains **29 AMD/Intel entries** (RDNA1-4, Vega, Arc A/B) at spec-sheet
  peaks, same `[table]` convention as the NVIDIA rows. The field case prices at exactly the
  448 GB/s the contributor had to type by hand.
- A detected card whose bandwidth is NOT in the table is **named, with its VRAM**, and asked
  for `--vram-bw` - never a silent "none". Capacity without bandwidth is not priced (a GPU
  tier with unknown bandwidth would be an invented number).
- Eta on RDNA/Arc backends is flagged UNVALIDATED in the output; `calibrate`'s anchors and the
  size-classed floor do the honest work - which is precisely how E-13's +0.1% prediction
  landed with zero GPU-side calibration.

## 1.26.3 - 2026-08-04

**Split GGUFs half-specced, and the contribution payload could ship `total=None`.** Found by
the tool's first external datapoint (issue #1, an RX 5700 XT measuring +0% against prediction -
thank you): the submission arrived titled `total=None active=None @ 2.5-bit` for a Q4_0 7.6B,
because the file was a 2-part split. Two distinct bugs:

- **Multi-part GGUFs (`-00001-of-00002.gguf`) were specced from one part.** Autospec saw half
  the tensors, and every `os.path.getsize` site (file-size calibration, tier placement, anchor
  records, probe estimates) priced half the model. `from_gguf` now enumerates every part
  (metadata from part 1, tensors from all), any part maps to the full set, and a missing
  sibling refuses to spec rather than speccing a half model. New `spec.split_siblings` /
  `spec.gguf_size` used at all seven size sites.
- **The contribution payload's model side read raw CLI args** - `None` whenever resolution
  happened anywhere but the flags (the mirror of the 1.26.2 hardware fix). `bench` now stashes
  the exact spec the prediction used at its resolution moment, and the payload carries it plus
  the GGUF filename - the field a human reader actually recognises.

Two new smoke tests pin both regressions, including the mutation direction.

## 1.26.2 - 2026-08-04

**`bench --contribute` sent the machine as `None`.** Under auto-detect - the default path,
i.e. nearly every contributor - the payload and issue title read `vram=None vram_bw=None
ram=None...`: a datapoint whose entire purpose is the machine, arriving without one. The
formatter read raw CLI args instead of the resolved hardware the prediction itself used.
Caught by a pre-launch gauntlet walking the exact commands the launch post asks strangers to
run; fixed by re-resolving through the same `resolve_hw` path, guarded with the exact failing
shape. If you contributed on 1.26.0/1.26.1 (nobody has yet), re-run on 1.26.2.

## 1.26.1 - 2026-08-04

**The 1.26.0 wheel self-reports 1.25.0.** The release bumped `pyproject.toml` but not the
`__version__` literal, and the clean-venv verification *printed* the version without *asserting*
it - so a package whose `--version` lies reached PyPI. Functionally 1.26.0 is complete (the
clean-venv check proved the new advisories present); only the string is wrong. 1.26.1 fixes the
string, and the smoke suite now asserts `pyproject.toml` and `__version__` agree, so this class
cannot pass a gate again. Prefer 1.26.1; 1.26.0 remains installable but misreports itself.

## 1.26.0 - 2026-08-04

### Two published numbers corrected against primary sources - one ours, one an anomaly that dissolved

**E-12 (Kimi K3) byte model corrected.** Our note modelled expert-only movement (23.8
GB/token). The repo's own `docs/data` states the engine re-reads the **trunk in full every
token: 108.81 GB**, plus ~25.8 GB of touched experts - 134.6 GB/token, trunk-dominated. Under
the corrected two-tier arithmetic the laptop arm lands at **32.6 s/tok predicted vs 32.69
measured (0.3%)** with ordinary bandwidths; the high-RAM arms run ~1.7x slower than
full-trunk-caching predicts, *consistent with* their ladder's stated hard cgroup caps (their
harness property; we say consistent-with, not proven). The discrimination is restated honestly:
the naive add-RAM rival (15.6x) stays refuted by the measured 1.63x, but our original "Law 4
predicts ~1x because experts cannot be cached" was also wrong - corrected bytes predict ~3.8x
at the capped end. A smaller, honest win; the withdrawn implied-bandwidth column and the
correction are both in the register and MATRIX.md. The 200-320x unit-inversion finding is
untouched.

**The Gemma-4-26B "C-02 violation" is resolved - no exception exists.** The model's own GGUF
header (read remotely from an ungated mirror, parsed by our shipped autospec) shows active
params were fine (3.82B vs our ~4.0B) but **gemma4 carries 480 KB of KV per position - 5x
Qwen-class**. Our 67.4 tok/s was a zero-depth floor; priced at the reporter's plausible 1-2k
context the same floor gives 58.0-49.2, bracketing the 51.6 report inside the C-02 band. New
scoring rule shipped into MATRIX.md: third-party reports get scored at their stated context
with kvp read from the header, or they do not get scored.

### The ladder becomes a four-model comparison surface - and catches its own compromised task

Rows added on the same 52 predicates, same box: Qwen2.5-7B @ Q4_K_M scores 30/40 staked
(T1 100%, T2 50%); the 7B at 2-bit (both byte-equal quants) 27/40; Qwen3-0.6B @ Q8 22/38 with
thinking-model truncations quarantined - and the 0.6B fires the suite's own kill rule (57.9% <
60%), the instrument correctly refusing to call a 0.6B business-usable. The 30B's 40/40 now sits
atop a measured gradient instead of standing alone.

The ladder also caught its own defect: **the only T4 task ever solved (t4l1, the 5-house logic
puzzle) has now been solved by the 30B AND the 0.6B while both 7Bs failed it.** Non-monotonic in
model capability is the signature of training-data recall (the 0.6B's chain-of-thought is a
co-factor), exactly the caveat logged when the 30B first solved it. t4l1's scores stand as
recorded; a generated-novel 5-house variant (fresh constraints, brute-force-verified unique)
replaces it for future rows per the roadmap's recall-immunity item.

### Prereg #92 scored: per-shape calibration KILLED by its own gates - nothing ships, and that is the system working

The dangling 07-31 stake is closed as a FAIL, published at the same size as the wins. Phase A's
replication gate did exactly what it exists for: the productized probe (compiled at runtime on
the user path, as a shipped `calibrate --shapes` would have to) reads systematically ~7% below
the logged research curve (staked +/-5%, 8 violations), and the curve's own shape missed the
staked characterization (span 1.41x vs >=2.0x; knee at 512 rows, outside [2048..8192]; knee not
rows-keyed). Phase B never runs; no per-shape term enters the planner; U-32's separately staked
prediction half is untouched. First attempt exited 2 (nvcc missing its host compiler) - a
precondition, not a result - and only after fixing it did the stake produce its verdict.

### A2A scored, same day: depth-aware beats uniform at equal bytes - 3x over its staked bar - and one prediction missed

The benchmark the community asked for, run end to end on the shipped product path. Uniform
`Q2_K` vs a depth-aware build whose fragile band came from a **fresh, blind probe run that
landed on the identical band the stored recipe carries** (21-27, delta +1.01 vs median +0.44) -
the instrument replicating its own historical measurement from a re-downloaded source.

| equal bytes (+0.48%) | uniform | depth-aware |
|---|---|---|
| perplexity | 9.579 | **8.319 (-13.2%; staked >=4%)** |
| KLD median vs Q8_0 teacher | 0.268 | **0.162 (-39.5%)** |
| same top token | 70.3% | **75.5%** |
| tok/s (tg128) | 21.41 | **22.82 (+6.6%)** |

P1 and P2 confirmed; KR-1 does not fire. **P3 missed and is published as a miss**: speed was
staked invariant to +/-3% and the depth-aware arm is 6.6% *faster* - consistent with the
documented Pascal format effect (fewer q2_k bytes; q2_k decodes slower per byte on this card),
stated as consistent-with. P4 (exploratory) returned the most instructive null of the day:
**52/52 suite verdicts identical across the arms** - a measured instrument-sensitivity
ordering, KLD >> perplexity >> binary task predicates. The ladder separates *models* (30B
2.95-bit: 40/40 staked; 7B 2-bit: 27/40, T4 0/6 again) and cannot see these two byte-equal
quants. KR-2 earned its keep: the first build came out +13.69% and was rejected and rebuilt
to +0.48% before any quality number was read. Full verdict in the prereg; README carries the
row.

### Staked: the apples-to-apples benchmark the community asked for

`preregistrations/2026-08-04-a2a-depth-aware-vs-uniform.md` - depth-aware vs uniform Q2_K,
same 7B, same box, same context, byte-matched to ±2%: perplexity, full-distribution KLD,
tok/s, and the 40-task business suite on both arms. P1 stakes ≥4% ppl advantage (not the -9%
headline - that was another family and staking it here would pretend transfer); KR-1 kills the
claim's scope and qualifies the README at equal prominence if depth-aware fails to beat
uniform on both ppl and KLD. The exllamav2 arm is stated not runnable on Pascal and goes to
`bench --contribute`.

### U-38 scored: our batching hypothesis is refuted - and the "2x ceiling" it tried to explain is overturned with it

The stake said batched decode crosses smoothly from bandwidth-bound to compute-bound at N~4.5 on
this card. The sweep (N=1,2,4,8,16, staked kill rules, 7B all-in-VRAM) killed it on both halves
of K-1: agg(4)/agg(1) = **2.256** against a staked >=2.5, and agg(16)/agg(8) = **3.257** against
a staked <=1.25. No coefficient taken.

What the sweep found instead was not in the hypothesis: aggregate decode runs 23 -> 40 -> 52 ->
54 tok/s through N=8 - exactly C-06's "2x and flat" - then **jumps to 175.6 at N=16**
(replicated twice to 0.1%) **and 219.4 at N=32**. Per-stream speed *recovers* from 6.7 to 11.0
across the jump, which no smooth crossover can do and a batch-width kernel switch can. The
mechanism is registered as a hypothesis, not a claim.

Two consequences, both bigger than the refuted stake:

- **C-06 is overturned.** The "2x aggregate ceiling by ~4 slots" was a sweep-range artifact -
  every earlier sweep stopped inside the flat region and we generalised the plateau into a law.
  The register entry now says so, and the public replication ask under that ID is updated to
  sweep to N=32 minimum.
- **The 2016 GTX 1060 serves 32 concurrent streams of a 7B at 219 aggregate tok/s** - 6.9 tok/s
  per stream, 9.5x the single-stream figure this project quotes as the card's speed. Law 4
  remains a single-stream law and says so; a batch axis is now evidence-backed roadmap work
  (docs/ROADMAP.md, Track A).

Evidence: `weights/data/u38_np_sweep.log`, `weights/data/u38_confirm.log`.

### X-1 confirmed: draft length is a kernel decision - and every speculation guide tunes the wrong knob

Staked before the sweep (`preregistrations/2026-08-04-x1-verify-width-cliff.md`): speculation's
verify pass is a fused multi-token step, so the U-38 kernel cliff (mat-vec kernels serve widths
<=8, mat-mat >=9) should appear INSIDE single-stream speculation at the staked draft length.
It does, exactly there:

| draft m | verify width | tok/s |
|---|---|---|
| off | - | 22.8 |
| 4 / 6 / 7 | 5-8 | 48.9 / 51.2 / 48.2 - the classic ~2x plateau, stuck in the slow kernel |
| **8** | **9** | **88.5** - 1.836x in one step, at the staked boundary |
| 12 / 24 | 13 / 25 | 124.3 / **132.1** |

P1 held (jump 1.836x >= staked 1.25 at exactly width 8->9); P2 held (below-boundary ratios
1.047 and 0.941 <= 1.15). The acceptance confound is excluded by arithmetic, not assumption:
at 100% acceptance a width-9 verify at slow-kernel step time cannot exceed 60.8 tok/s, and
88.5 was measured. Outputs are byte-identical across all nine arms - speculation's invariant,
verified, so the workload is constant by construction.

**The rule no guide states: on pre-Ampere cards, drafts of 4-7 leave ~2.5x on the table.**
Total effect off -> draft 24: **5.8x single-stream** (22.8 -> 132.1 tok/s) on copy-heavy work.
Draft length is a kernel decision first, an acceptance decision second. Ships as a planner
advisory with the anti-valley rule (widths 2-8 dominated; 1 or >=9 only).

Also registered from the same line of thought: **U-40** draft-driven expert prefetch (run the
router on ngram drafts to prefetch RAM experts during the current step - attacks the exact
wall U-39 confirmed) and **U-41** expert-coherent sampling (steer best-of-N lanes to share
expert working sets), both with kill rules staked before any build.

The sweep also ate two harness lessons, both now structural: killing a stuck benchmark CHILD
revives its parent loop (three contaminated windows this session - every overlap-window arm
was deleted, never explained; runners now carry a lockfile + per-arm timeout), and this
build's `llama-cli` spins forever at its prompt on stdin EOF.

### U-39 confirmed: batching does NOT survive expert offload - and model choice inverts with user count

Staked before the sweep (`preregistrations/2026-08-04-u39-moe-batching.md`), with a refutation
condition that would have made the 1060 a multi-user 30B server. It did not fire. Measured, same
session, dense anchor replicated a third time to 0.4%:

| concurrent users | 30B MoE (experts in RAM) | dense 7B (all in VRAM) |
|---|---|---|
| 1 | **19.7** tok/s | 23.1 tok/s |
| 8 | 37.5 aggregate | 53.9 aggregate |
| 32 | **40.0** aggregate (1.25/user) | **219.4** aggregate (6.9/user) |

P1 held (agg8/agg1 = 1.905 in the staked [1.0, 2.5]); P2 held (agg32/agg16 = 1.079 < 1.5 - no
jump exists on this placement at any N). Mechanism as staked: routed-expert reads from system
RAM do not amortise across streams - each user summons different experts every step - while
dense weights read once serve everyone. Prefill is the exception and batches fine even on the
MoE (45 -> 225 tok/s): compute amortises, expert bandwidth does not.

**The practical sentence: at 1 user this box's best model is the 30B MoE; at 32 users it is the
dense 7B, by 5.5x.** Same hardware, same budget - the right model depends on how many people
share it, and no consumer tool prices that today. The planner's batch axis (ROADMAP Track A) now
has its first two measured curves.

Bonus from the edge probe: the dense jump sits exactly at the 8->9 stream boundary (53.9 ->
107.7 aggregate), matching llama.cpp's mat-vec kernel cap of batch 8; per-stream speed above the
switch is nearly flat (~11-12 tok/s each from N=9 to 16). Consistent-with, not proven - kernels
were not individually forced.

**A claim that inverted, and five harness bugs caught before any of them became a finding.**

### E-12: the Kimi K3 number is a unit error, and the repo's own data tests our law

[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) reached us as "running Kimi at
10 tok/s". Its README reports **seconds per token** - `~32 s/token` laptop, `~19-21 s/token`
server, and a run line reading *8 tokens in 261.5 s*. That is 0.03-0.05 tok/s: the claim as
relayed is the reciprocal, off by 200-320x. **We made the same slip on first read** and called
the preset ladder "backwards" because speed appeared to fall as RAM rose. Under the correct
units it rises, which is the expected direction.

The four presets are an out-of-sample test that **discriminates**, with no free parameter. Kimi
routes 16 of 896 experts; solving the 1.56 TB checkpoint for the expert/trunk split gives a
1.33 TB expert store, so 23.8 GB moves per token, from disk, in every preset.

| model | predicts | |
|---|---|---|
| Law 4 - bytes/token constant, store uncacheable | **~1x** | |
| "it does not fit, add RAM" - speed tracks resident set | **15.6x** | |
| measured, 8.2 GB -> 128 GB | **1.63x** | Law 4 |

Which 16 of 896 a token needs changes every token, so the working set over a generation is the
whole store and no preset caches a meaningful slice. Capacity is not the lever. Reaching the
claimed 10 tok/s needs 238 GB/s against 1.33 TB - roughly 16x H100 before the store is
resident, at which point the C implementation is beside the point.

The per-preset implied bandwidths (0.73-1.19 GB/s) are *derived* from the measurements, so that
column is a consistency check with one free parameter per row and is disclosed as such. The
1.63x-vs-15.6x discrimination is the load-bearing result.

### Business tasks: the instrument was rebuilt before it was allowed to produce a number

The old set graded against prose ("sendable with at most a name edit"), which cannot compare two
models - it puts a human who already knows which model wrote the output back in the loop. Now
**40 auto-scored tasks with executable predicates** (JSON shape and exact values, exact
arithmetic, single-label classification, code that must run and pass assertions, word and
sentence caps) plus `nonums`, a deterministic hallucination check that fails any number not
present in the source. 8 rubric tasks are retained and never enter the headline.

### The worst bug: the hallucination detector was inventing the hallucinations

Two tasks were reported as failures for "inventing numbers". Both models outputs were correct.

| output | what the checker claimed | why |
|---|---|---|
| `Q3 revenue rose, up YoY but below plan.` | contains the number **3** | pulled the digit out of the quarter label `Q3` |
| `EMEA,Q3,1200000` (a byte-perfect CSV) | contains **31200000** | matched across the field separator as `3,1200000`, then stripped the comma |

A detector that manufactures the thing it detects is the worst failure this harness can have: it
publishes a confident verdict against a model that did the work correctly. Two rules fix it - a
number may not begin immediately after a word character or a dot (so the `3` in `Q3` and the `24`
in `v1.24.0` are not numbers), and a comma groups digits only in threes (`47,500` is one number,
`Q3,1200000` is not). Guarded by a test built from those exact two strings.

A third "failure" was also ours. `analysis/n4` asks whether 3.1% monthly churn loses more than a
third of a cohort in 12 months. `1 - 0.969^12 = 31.47%`, against a third at `33.33%` - the answer
is **NO**, the model answered **NO**, and the staked key said YES. The key is corrected. Changing
a key after seeing outputs is normally forbidden; it is allowed here only because the arithmetic
is objectively wrong and the correction makes the task *harder* to pass, not easier.

A fourth, `code/p1`, was an ambiguous spec rather than a wrong answer: the prompt said "rate"
without stating whether `0.23` or `23` was meant, and the model chose the other convention. The
prompt now states it, so that task is excluded from this run and must be re-run - a stored answer
to a different question cannot be graded.

**Because scoring is deterministic and outputs are stored, the whole run was re-graded with the
corrected checks without touching the model** (`--rescore`). That is the payoff of executable
predicates: when a check turns out to be wrong, every past run can be re-scored consistently
instead of re-generated or quietly dropped.

### Five more defects found while standing it up - each would have published a confident wrong answer

- **No chat template.** The raw `/completion` endpoint applies none: asked to reply
  `ACKNOWLEDGED`, the model answered *"I am a student who is struggling with my homework"*.
  All 40 tasks would have failed and the verdict would have read *"2.5-bit cannot do business
  work"*. Fixed to `/v1/chat/completions`, plus a preflight canary using that exact prompt -
  the run refuses to start unless the endpoint proves it follows instructions.
- **A truncated answer scored as a wrong answer.** Qwen3 is a reasoning model; the token budget
  covers thinking. At 1024 the arithmetic tasks burned the whole budget mid-thought and returned
  empty content, scoring as five confident failures. Truncation is now detected, quarantined,
  and the shrunken denominator is stated out loud along with the worst case - because quietly
  dropping the hard tasks is how a headline flatters itself.
- **Two runners, one results file.** A kill that did not take left an older run alive writing
  the same path as a new one. The merge looked like a finished 40-task run at the wrong token
  budget, with nothing in the output saying so. The runner now refuses to start if its output
  was touched in the last 180s, and results go to timestamped filenames.
- **A verdict from a run that never happened.** After the preflight aborted, `score()` still
  read the results *file* and printed "KILL RULE FIRED (33.3%)" from stale data. An empty run
  is no longer scored.
- **Two of eight arithmetic answer keys were wrong** (a1, a2), caught by recomputing every key
  before the run. A wrong key invalidates the task for every model that ever runs it.

Four new guards cover these, each mutation-tested: re-injecting its defect fails that guard and
only that guard. Suite at 144 tests.

### The verdict on the staked set - and what the five truncations turned out to be

On the corrected instrument the recommended config - Qwen3-30B-A3B Q2_K on the GTX 1060 at the
planner's own placement - scored **34/34 = 100% of gradeable staked tasks**, against a bar of
>=80% staked before any output existed. The honest floor: with all 5 truncated tasks counted as
failures the score is **85.0%**, still above the bar, so **P1 is confirmed under either
reading**.

The truncations themselves resolved cleanly: they were the CONTEXT WINDOW, not the model. All
five had burned the full budget reasoning at `-c 4096` (the stop at 4044 tokens gave it away -
that is the window minus the prompt, not our `--npredict`). Re-run at `-c 16384`, **all five
pass, plus the re-specified code task: 6/6** (`bt_retry16k.json`). The token counts show why 4k
could never work: the annual-retention question used 7,417 tokens end to end, the churn
projection 5,275 - a reasoning model's budget is thinking plus answer, and the thinking alone
overflows a 4k window on genuinely multi-step arithmetic. So the complete picture for the
staked set is **40/40 tasks correct** once the harness stopped standing in the way - every
earlier deficit traced to the instrument (wrong keys, missing chat template, false-positive
hallucination regex, context window), not the 2.5-bit model.

Cost of the bigger window on a 6 GB card, measured in passing: decode fell from 22.69 tok/s at
`-c 4096` to ~11.7 at `-c 16384` - the KV cache displaces expert weights out of VRAM, exactly
the trade the planner's depth term prices. Practical advice that falls out: run the 30B at 4k
for interactive work and open the window only for jobs that need long chains.

### A difficulty ladder, up to a deliberate ceiling (tiers, 2026-08-03)

The staked set measures "can the cheap quant do routine business work" - it cannot rank models
that all pass it. Every auto task now carries a difficulty tier, and the set gains 12 new tasks:

- **T1 routine / T2 standard** - the original 40, unchanged. The staked 80%/60% bar is computed
  over these and only these, forever; folding new tasks into a staked bar would move the
  goalpost in whichever direction they score.
- **T3 hard (6 tasks)** - ledger reconciliation to exact deltas, NPV to the cent, amortization
  code against exact assertions, a brute-force-verified logic puzzle, compound format
  constraints, and a hallucination-capture task where inventing one ticket ID fails exactly.
- **T4 ceiling (6 tasks)** - designed so today's best models fail while staying 100%
  machine-checkable: exact 9x9-digit multiplication, 96-month amortization interest to the
  cent, a 45-word lipogram with mandatory vocabulary, a 5-house logic puzzle, ISO-8601 week
  numbers implemented with no imports, and an exact string transformation. **0/6 is the
  expected score for every model that exists today.** The tier is there so the first model to
  score above zero does it against a bar that predates it.

Validity is enforced mechanically, because two of the original eight arithmetic keys were wrong
when first written and a wrong key at T4 would be invisible forever: the self-test *recomputes
every tier key* (the multiplication, the NPV, the amortization chain, the ISO weeks against
`datetime`, the string transform) and *brute-forces both logic puzzles*, refusing to run unless
each has exactly one solution equal to the staked key. It also caught the author's own T4
sample containing a forbidden letter - the checks discriminate against the person who wrote
them, which is the point.

**First datapoint** - the 2.5-bit 30B on the 1060, 16k context (`bt_tier_30b.json`,
`bt_tier_30b_t4rerun.json`):

| tier | score | composition |
|---|---|---|
| T3 hard | **5/6** | solved NPV to the cent (8,476 tokens of work), amortization code, ledger reconciliation, hallucination-capture, the 3-house puzzle; missed only a 12-word bullet cap |
| T4 ceiling | **1/6** | solved the 5-house puzzle; wrong answers on the 9x9-digit product (emitted an exact-looking but wrong 18-digit result), the lipogram and the string transform; burned the full 15,000-token budget without answering on the interest chain and the ISO-week code |

Two honest annotations. The staked expectation was 0/6 on T4 and it came in 1/6 - **the
expectation missed, and that is published, not adjusted**. The solved task is the Einstein-style
puzzle, the one T4 task where training-data familiarity plausibly helps; the exact-arithmetic
tasks, where recall cannot help, all held. Second, the 9x9 product failure is the ceiling
working exactly as designed: the model *announced* it would need a calculator, then printed a
confident 18-digit answer that is wrong in the 5th digit - a perfect specimen of why these
tasks are machine-checked to the digit rather than eyeballed.

The tiers ship with the instrument, so any model anyone runs lands on the same ladder.

### Also

- `docs/MATRIX.md` gains the Kimi section beside the existing "70B Q4 at 35-45 tok/s on a Spark
  is physically impossible" callout - one claim that was wrong, one that was right and got
  relayed wrong.
- 22.69 tok/s re-confirmed for the recommended 30B placement on a quiet box.

## 1.25.0 - 2026-08-03

**A new command that found two bugs in itself before it found anything about your setup.**

### `quantprobe audit-ollama`

Prices the models already sitting in your ollama store. Its blobs *are* GGUFs, so the real
header is already on disk - no name parsing, no guessing. Your tag says `7b`; the header says
7.62B total at 4.92 effective bits.

Measured on a real store, both placements timed by llama-bench on the **same blob at the same
depth**:

| | |
|---|---|
| ollama's own eval rate | 18.3 tok/s |
| ollama's placement | 16%/84% CPU/GPU at ctx 4096 |
| that split, timed here (`-ngl 24`) | 11.43 tok/s |
| **all layers on the GPU (`-ngl 99`)** | **18.73 tok/s - 1.64x, and it fits** |

ollama holds layers back on a model that fits in VRAM. One Modelfile line fixes it:
`PARAMETER num_gpu 28`.

**No "+X% on the table" claim without `--measure`.** Without it the output is labelled a
prediction and says so. With it, both placements are timed, and the note states plainly that a
prediction-vs-ollama gap is a claim about *the prediction* as much as about ollama.

### The two bugs it shipped with, caught by running it

- **It read the wrong number.** ollama prints `prompt eval rate:` *before* `eval rate:`, so a
  bare regex took the prompt rate - 186.59 where the truth was 19.92. It doesn't look wrong; it's
  a plausible tok/s. The tool reported ollama at 205.6 and concluded nothing was worth
  recommending.
- **It contaminated its own comparison.** ollama keeps a model resident ~5 minutes - measured,
  5209 MiB of a 6144 MiB card. The audit measured ollama, then immediately benched the same GPU,
  so `-ngl 99` couldn't fit, silently spilled to CPU, scored 4.56, and the tool confidently
  recommended *against* a config measured by hand at 18.83 minutes earlier. It now stops ollama,
  polls until the GPU is actually free, and **refuses to compare** when it isn't - including when
  `nvidia-smi` is absent and cleanliness can't be verified.

Neither was visible by reading the code. Both were obvious in one real run.

### Also

- **`auto` no longer trusts a partial download.** An interrupted `fetch` leaves a partial GGUF
  where `auto` looks. Unguarded it crashed the command; on a different truncation it *succeeded*
  and returned a plausible-but-wrong spec - a confident tok/s for a model that isn't there. Now
  size-checked and guarded.
- **Every evidence file the register cites is now in the repo, and a gate enforces it.** 37 of
  109 citations pointed at files that weren't there.
- **Retraction: gemma4-12B is one of the *steadiest* ladder rows**, not an unstable one. The
  earlier claim quoted three readings out of seven; six span 1.087x across wildly different
  calibrations.
- **L-28**: mmap costs 1.388x against `read()`, and access *granularity* matters far more than
  access *pattern* - 18x at 4 KB where 512 MB units showed nothing.
- **U-37 not confirmed**: the `--no-mmap` residency hypothesis failed on a quiet box, and its
  pilot's signal turned out to be contention.

## 1.24.0 - 2026-07-31

**Three defects that were live in 1.23.0, and the one number this release still cannot stand behind.**

Every fix below is a correction to something we already shipped. Read the last section first.

### The disk tier is now MEASURED - and the law missed it by 30%

We corrected the disk-bandwidth probe (see Fixed), then measured a real disk-tier row against a
band staked in advance. **The prediction missed, and it missed upward: real hardware is 30%
faster than the law says.**

| | |
|---|---|
| predicted | **0.331505 tok/s** (staked band [0.265204, 0.442007]) |
| measured | **0.476225 tok/s** |
| error | **−30.4% — OUTSIDE the band. KR1 FAIL.** |

The miss is worth trusting because the three guard rules passed *first*, which is exactly what
separates this run from the previous attempt: **KR3** the harness demonstrably varies (control
7.570 tok/s against a 7.09 anchor), **KR4** steady state reached (rep spread 1.172×, against a
2.09× failure the day before), **KR2** decode not load (5.7% / 6.0%, both denominators). One
`cal_id` at both ends, gates open on the first attempt at 0.2% CPU.

Inverting the term, the I/O component is **1.485× faster than modelled**. That was one of exactly
two things — a wrong *access pattern* in our probe, or a wrong *miss fraction* in the law. We
staked the discriminator before running it, with the inconclusive band declared in advance so the
answer could not be chosen for convenience, and **it came back decisive.**

### The disk calibration is now confirmed correct — the residency model is what's wrong

Prereg #94, same drive, disjoint cold regions:

| arm | result |
|---|---|
| scattered — 8 × 512 MB at random offsets | **0.452 GB/s** |
| contiguous — one 4 GB sequential read | **0.459 GB/s** |
| ratio | **1.015× — access-pattern hypothesis REFUTED** |
| warm re-read (cannot-vary guard, must exceed 2.0) | 2.802 GB/s — **guard passed** |

This drive does not care about access pattern at this size. Two consequences, and the second
matters more than the first:

1. `measure_disk()` needs no change. The probe was measuring the right thing.
2. **Both arms land within 4% of the 0.47 GB/s that `calibrate` reports.** The C-17-corrected
   disk figure is independently confirmed by a probe carrying a working falsifier. Yesterday this
   changelog said *"proving the old number wrong is not the same as proving the new one right"* —
   the new one is now right.

By elimination the 30% miss belongs to the **miss fraction**: the law assumes 0.637 of each
token's bytes are re-read from disk; the measurement implies **~0.429**. That is exactly what
expert-usage skew predicts — if MoE routing concentrates on a hot subset, those experts stay
page-cached permanently and never pay disk at all.

**We still have not changed a coefficient**, and won't until skew is measured directly. Elimination
is not evidence: H2 stands because its only rival is dead, not because anyone has watched the
routing. That measurement is owed, and **0.429 is the number it has to hit.**

Why this went undetected until now: all 14 rows of our validation ladder are VRAM- or RAM-resident.
**Zero read the disk tier.** A component no validation row touches can stay wrong indefinitely.

Earlier context, retained because the correction is the point: the probe read a fixed 512 MB tail
region jittered by at most 7 MB, so ~98.6% of it overlapped between calls, and `buffering=0` does
**not** bypass the OS page cache. Cold 0.44 GB/s, then 2.99 and 2.99 on re-reads. The warm figure
is RAM. It shipped as a disk-tier input, **6.8× too fast**.

**The first attempt at this measurement was thrown away, and that is why the second one counts.**
It returned 0.34 tok/s — nominally 2.5% *inside* the band, the most tempting number of the whole
release. It was discarded because KR4 fired (`0.34 ± 0.17` over 2 reps implies reps of ~0.22 and
~0.46, a 2.09× spread against a >2× rule) and because the run overlapped a concurrent ladder
holding ~13.7 GB on a 16 GiB box, violating C-14 on a row whose entire premise is the page-cache
miss fraction. A number landing near prediction under contamination is worth less than no number.
Re-run serially on an idle box, the same row measured 0.476 and failed the band outright.

### A ladder median can hold still while every row under it moves

All 14 ladder rows measured **faster** than the reference pass taken under the same `cal_id` — not
13 of 14, all of them. Median **+4.6%**, range +1.4% to +27.5%, prediction playing no part.

The median |err| moved only 9.0% → 8.4%, because the errors carry mixed signs and a uniform
speed-up improves the over-predicted rows while worsening the under-predicted ones. **The median
was stable by cancellation, not because the machine was.** A median inside C-18's ±1 point noise
floor is not evidence that the measurement basis held still; the per-row measured-vs-measured diff
is the sensitive detector, and it is now printed alongside.

Established independently of any of this, because it involves no timing at all: across both
passes **every prediction field is bit-identical** — 14 tok/s, 14 placements, 14 emitted commands.
**1.24.0 changed no shipped prediction.**

Consequences we are acting on rather than noting:

- The 8.4% median passes its staked band [6.8, 10.8] and is reported as **unchanged**, not as an
  improvement. 0.6 points is inside the noise floor.
- **Published headline speeds stay at the reproducible range, not the scrubbed-box ceiling.**
  Qwen3-30B-A3B measured 22.94 tok/s on this pass, above the README's stated 20.4–22.2. The README
  is unchanged: that pass required stopping services and closing everything, and a user with a
  browser open cannot reproduce it. A number we can only obtain under lab conditions fails the
  standard we set for ourselves.
- **gemma4-12B: one anomalous reading, not an unstable row — a correction to what this changelog
  said first.** The original text called it "a 27% spread… not measuring a stable quantity," from
  three readings. There are seven. Six of them — across partial-disk, idle pre-reboot, post-reboot
  stale, RAM-only, uncalibrated and locked calibrations — span **12.17–13.23 tok/s, a 1.087×
  spread**, which makes it one of the *steadiest* rows we have. Only the scrubbed-box run's 15.62
  sits outside, 1.18× above the cluster. That run is also where all 14 rows sped up and gemma moved
  most (+27.5%); a dense 12B split across VRAM and RAM has the most to gain from page-cache
  residency, which is what U-37 predicts. Possibly the same effect from another angle, not a defect.

### The disk probe no longer asks you to spot its own bad readings

`measure_disk()` took **one** timed read, and its docstring told *you* to "treat a single number
above ~1.5 GB/s as evidence of a warm cache rather than a fast disk." That is a defect wearing a
disclosure's clothes - the check belongs in the code. Hours after that sentence was written our
own release gate caught it: reads of `[0.413, 3.171, 0.415]` GB/s on one file, because an
experiment had streamed 15 GB of it minutes earlier.

Measured, 8 draws per arm on a 13.7 GB GGUF:

- **73% of the file deliberately warmed:** 6 of 8 single draws returned >1.5 GB/s, max **2.854** -
  RAM reported as disk, a **6.3x error**.
- **After evicting 16 GB of an unrelated file:** *still* 1 of 8 draws at **2.092**, a 4.7x error.
- **Minimum over the draws:** 0.4499 and 0.4537 - both correct against the independent raw-read
  baseline, and 0.8% apart despite wildly different cache states.

That second line is the one that matters: a cold read cannot be *guaranteed* on this machine at
all, so single-sample probing is structurally unreliable, not merely unlucky after a download.
`measure_disk` now returns the **minimum over N disjoint random regions** - nothing reads faster
than the device except cache. `calibrate` and `hw --measure` print the individual draws.

**One staked prediction was refuted and is published here at the same size.** We expected a
`max/min > 2.0` spread test to flag warm files and stay silent on cold ones. It fired on *both*
arms - correctly, since neither arm was truly cold. The warm *fraction* does discriminate (1/8 vs
7/8), but that criterion was chosen after seeing the numbers; it is shown to you, labelled
post-hoc, and never used to alter the estimate.

We also did **not** nudge this probe toward the ~0.25 GB/s llama.cpp actually achieves while
streaming. That gap is a runtime inefficiency and belongs in the law, not in a probe whose job is
to measure the device. Two errors that cancel are still two errors.

### Also not validated in this release

- **Expert-usage skew is inferred, never observed.** The miss fraction survives only because its
  rival is dead — nobody has watched MoE routing to confirm that a hot subset stays resident. Until
  that is measured against the 0.429 target, the disk-tier term keeps under-predicting and the
  mechanism behind it is a hypothesis with one competitor eliminated, not a finding.
- **L-26's "+4.3% prefill" is law-mediated, not a measured A/B.** 360.76 tok/s was measured at
  pp4096; the 345.89 baseline is pp2048 - a different prompt length - and the like-for-like
  pp4096-at-ub2048 control has never been run. Because `-ub` is a cap, the flag also cannot do
  anything on prompts of 2048 tokens or fewer. The tool now says all of this itself.
- **`measure_disk` is still a single sample** spanning ~2.1x on repeat. Making it a median is a
  method change that needs its own stake, so for now it is disclosed rather than repriced.

### Fixed

- **The upgrade advisor could only ever suppress good upgrades, never invent bad ones** (C-19) - a
  one-directional error, which is why it survived: every symptom looked like conservatism. The
  counterfactual dropped `n_layer` and `true_size_gb`, so `evaluate()` re-derived them from presets
  and priced *a different model with more RAM*. Affected **31 of 340** grid cells. Verified by a
  staked 340-cell re-sweep: 0 invented pairs against a live population of 24 cells where the check
  is not vacuous, 0 identity mismatches over 700 counterfactual calls, 0 arithmetic mismatches over
  131 fired upgrades. The auditor was falsified first - re-injecting the defect makes it exit 1 and
  reproduces the historical signature to six decimals.
- **The speculation block printed a speedup the row could not reach** (C-20). Called without the
  row, it quoted its constant headline regardless of the row's own ceiling. On the **17 of 127**
  affected cells the true bound was 1.025x-1.229x against a printed 2.10x. Now prints "NOT
  REACHABLE ON THIS ROW" with the bound. Scope worth knowing: all 22 qualified cells are the dense
  2.10x headline; the 4.7x n-gram branch has synthetic-probe coverage only.
- **The C-17 disk fix had shipped half-done.** `os.urandom(4)` capped the random offset at 4 GiB, so
  on any larger file - precisely the size class the disk tier exists to model - the probe could
  never read past the 4 GiB mark, and the reachable prefix is exactly what a partial download has
  already warmed. Found by the new falsification test, not by inspection.
- **A `+73%` prefill figure leaked onto every row it printed on**, including all-in-VRAM rows where
  the same flag measured **-39%**. The scope footnote now names the CPU-expert MoE placement and
  shows the opposite-sign control.
- **The `-ub` prefill percentage ignored which ubatch was actually emitted.** A 3 GB card sized to
  `-ub 1024`, where the sweep measured +38.7%, was told +73%. Each ubatch now selects its own
  measured cell; a ubatch with no cell gets prose, not a borrowed number.
- **Partial calibration printed the same "calibration applied" line as a complete one**, though
  RAM-only calibration measures 12.5% median error against 8.8% for the fully-preset baseline -
  worse than no calibration at all for the components you skipped. Unmeasured components are now
  named at equal prominence, with the penalty and the command to finish.
- **Two regression tests were decorative.** The guard for the 6.8x disk error was defined *below*
  the runner block and had never executed on any commit; its only assertion was that repeated
  timings agree, which a fully cached file satisfies perfectly. The channel-count guard - the one
  protecting against a 2x RAM-bandwidth input error from an external replication - re-implemented
  the rule in its own body and never touched `detect.py`; restoring the shipped bug left it green.
  All 14 guards for the audited behaviour are now mutation-tested; none are unfalsifiable.
- **`verify.py` layer 1 counted a skip as a pass**, contradicting the "a skip is not a pass"
  doctrine layer 3 has enforced since 1.12. Skips now surface in the release gate.
- **Two pre-registration documents both claimed #92**, so the "every staked prereg is cited" gate
  was satisfied for both by a single citation, and prereg #92 was in fact cited by nothing. The
  later document is renumbered #93 with a visible correction note (no stake or threshold changed,
  nothing had run). `findings.validate()` now rejects duplicate numbers outright.
- **A measurement harness reported "disk counter unreadable - NOT substituted with an estimate"**
  while 83 populated rows sat in its log. The sampler emits an empty timestamp on this locale, so
  rows arrived with two fields instead of three and the parser discarded all of them - then printed
  a sentence that reads like scientific restraint. A silent fallback wearing a disclosure's clothes
  is worse than a crash.

### Added

- `detect.ram_channels()` and `detect.probe_offset()` extracted so the rules are testable against
  the shipping function rather than a copy.
- `plan.calibration_gap_warning()` + `CAL_COMPONENTS`.
- `weights/resweep340_audit.py`, re-runnable with `--inject upgrade` / `--inject spec` to prove it
  can still fail.

## 1.23.0 - 2026-07-30

**A feature we promised, tested, and did not ship - plus the finding that replaced it.**

We told an external reporter (E-08, RTX 5070) that this release would turn the all-in-VRAM
FLOOR into a point estimate, using the 61 public benchmarks we had collected. Pre-registration
#72 staked that with their measurement as the holdout. **It failed its own gate and does not
ship.** On one basis (efficiency = tok/s x file bytes / spec bandwidth) the public corpus spans
0.512-0.812 across 8 GPU architectures - and our 2016 Pascal card measures 0.509-0.577, INSIDE
that range. Their box sits at 0.810: the population maximum. Fitting a constant until their
number landed in band would have been the exact failure C-02 taught us to refuse.

- **L-18 (new law):** the all-in-VRAM spread is a property of the POPULATION, not of our old
  GPU and not of any architecture. 1.6x wide, irreducible from public data, collapsed only by
  measuring the specific machine. The floor advice now cites those 61 benchmarks as evidence
  and points at `calibrate` as the concrete fix instead of apologising.
- **U-23 (shipped, with its limit stated):** `--no-mmap` is no longer unconditional on expert
  splits. Above 75% of the usable RAM pool the emit keeps mmap so pages stay evictable, and
  says why - naming the +13.7% decode it is giving up. Honest limit, in the register: the gate
  reasons about model share, so it does NOT catch the reporter's own case (their 330 MiB free
  came from other processes and the mapped file). Reading actual available memory is U-24.
- **The mark:** quantprobe has an icon - old-style pixel art, one 12x12 bitmap that generates
  both the SVG and the terminal banner. Bare `quantprobe` now prints it (with plain-text
  fallbacks for pipes, NO_COLOR, and legacy Windows consoles).
- Register at 92 entries: E-08 (the first out-of-sample validation on hardware we do not own -
  58.1 predicted vs 54-57 measured), E-09 (rabbit: disk-tier bandwidth-indifference), L-18,
  U-23 closed, U-24 opened.


## 1.22.0 - 2026-07-30

**The IQ ladder priced in, and the speculation map's first version qualifier.**

- **IQ formats are measured, not guessed** (prereg #70): FORMAT_EBW gains IQ2_XS 51.1 / IQ3_S
  61.1 / IQ3_XXS 68.3 / IQ4_NL 117.0 - the divide is CODEBOOK vs not, and IQ4_NL's kernel is
  Q4_0-class. The format advice now offers IQ4_NL (+14% over Q4_K_M at the same size class WITH
  imatrix quality) and warns off codebook IQ for VRAM decode on pre-Ampere. The staked split-arm
  retrodiction MISSED and is published with its diagnosis (U-19 refuted as hypothesized; the
  residual is structural, U-20 - which now also carries public out-of-sample support from a
  four-point residency curve on RDNA3).
- **Native MTP is the first positive MoE-split speculation** (prereg #71): models with a
  trained-in MTP head (Qwen3.6 class) measure **+11.4% at --spec-draft-n-max 2** (93.2%
  acceptance) on the expert-split placement where every external draft loses. K MUST stay at 2:
  K=4 measured 0.61x on the same pair - the expert-union tax is alive, the head just out-accepts
  it at short drafts. speculation_advice prints the exception with its limits.
- Register at 86 entries: E-07 (colibri cross-engine confirmations), the 61-entry public
  benchmark corpus + audit script ship in-tree (weights/public_bench_audit.py), and the
  draft-note %% display fix.


## 1.21.0 - 2026-07-29

**The dense-split release: speculation's best cell shipped as advice, and three registered defects closed.**

The headline is pre-registration #69: on a dense model split across GPU/CPU, a small same-family
draft model is the BEST speculation cell this box has measured - **+33.5% decode at K=2**
(5.54 -> 7.40 tok/s on a 14B, 76% acceptance, novel code) with the draft running on CPU
(`-ngld 0`), so it costs zero VRAM. The mechanism is the mirror of the MoE union tax: a K+1-token
verify batch reads each CPU-resident layer once, so the CPU share of every token amortizes. The
staked K-shift MISSED and is published as such: every K>=3 lands at or below baseline (llama.cpp's
default 3 is a measured LOSS here), because the CPU draft spends the same RAM bandwidth the
amortization saves. plan now prints the whole recipe on dense-split placements
(`dense_draft_note`), completing the speculation map: dense split 1.335x > dense all-in-VRAM
1.11x > MoE split 0.74-0.81x (never pays).

Three queued defects closed, each with its test:

- **C-11** - the dense split budgeted `vc*0.9` with no desktop reserve and no compute buffer, so
  at 16k depth it emitted a config that overcommitted VRAM and measured **-58%** (driver memory
  fallback). The budget now subtracts both (the reserve, and the #23 measured 0.5874 MiB/ub-token
  slope at the default -ub), counted once, same discipline as the MoE path: a 14B emits 21/48
  layers at ctx 0 and 16/48 at 16384 instead of a flat 28/48. Disclosed cost: ~5 layers more
  conservative at shallow ctx than a config that measured healthy; the cliff config is gone.
- **U-17** - the IQ-on-CPU warning is now priced, not prose: every RAM weight read is discounted
  by a per-IQ-byte penalty on the file's measured IQ share. The retrodiction gate rewrote the
  plan: the hypothesized 2.7x would have overshot ~7x; the calibrated e2e value is **k = 0.242**
  (pure-CPU IQ arm now exact: 11.4 vs 11.44 measured). The split IQ arm improves (28.0 -> 26.9 vs
  23.60) but stays over - its GPU share is still priced at K-quant eta, registered as U-19 with
  V-11's 1.55x pointing the way.
- **U-18** - `fetch` no longer treats "a file with that name exists" as "the download is
  complete": it compares local size against the remote and fails loudly on mismatch, naming the
  collision; `--force` replaces the file. This is the exact failure that cost pre-registration
  #69 three crashed runs (a June-era 0.5B under the target's filename fed llama-speculative an
  incompatible tokenizer).

Also in this release: the U-16 split-ub gate (split placements emit `-ub 1024`, recovering the
+29% prefill the flat cap left on the table), run/bench/plan flag unification for `--threads` and
`-ub` (no command drops what another prints), and the residency buffer term measured rather than
estimated (0.9 GB = #23 buffer + margin, emitting 15 residents where the sweep measured the
optimum). Pre-registrations #67-#69 and their raw logs ship in-tree.


## 1.20.2 - 2026-07-28

**The end-to-end consistency audit: six defects found, fixed, and now structurally prevented.**

Asked "are we sure the tool works end to end without misconfigurations or miscalibrations?",
the audit answered no six times, then yes: optimize and auto bypassed calibration entirely;
`run` dropped the `--threads` that plan printed; bench could not forward it; the anchor
size-class band (4x) let a 0.5B anchor mis-price the 30B flagship (+27% over); the GPU anchor
ratio was priced against a different eta than the prediction rows use; and small targets mixed
the two references (halving their predictions in one intermediate). All five commands now
resolve constants through the same three shared functions (apply_calibration_overrides,
resolve_gpu_eta, resolve_cpu_bw), and the guardrails that caught the intermediates - the
plan/bench parity layer and the all-in-VRAM ratchet - remain armed.

Accuracy claims corrected accordingly: v1.20.1's briefly-published 8.6% median leaned on the
inconsistent boost; the principled column is ~12% median with every big-model miss an
under-promise and anchor-class predictions exact by construction. MACHINE_LADDER.md carries the
correction. Known remaining gap, stated: a live llama-server launch test (run --dry and the
bench real-binary loop are verified; the server binary build is environment-dependent).

## 1.20.1 - 2026-07-28

**The machine ladder found the anchors' two blind spots; both fixed, gated, and disclosed.**

The 0.5B-30B ladder (MACHINE_LADDER.md) measured the v1.20 anchored predictions at median error
14.8% with one -34% miss. Both causes were mechanical: anchors were priced without the same
active-byte convention every row uses, and a 0.5B GPU anchor was pricing 7B+ targets - small
models pay a size floor big models do not (#59), so the anchor conflated format eta with it.

Fixed: anchors carry their own spec (active bytes + format mix) and big-model GPU eta now comes
from the measured per-format ladder (spec.FORMAT_EBW, L-16 made actionable), with the anchor's
machine ratio applied only within the anchor's size class. Post-fix: median 8.6%, worst 14.3%,
DS-Lite (quasi-out-of-sample mix) at -2.5%. U-15 closed.

Two guards earned their keep in the same hour: the plan/bench parity check (layer 3) and the
all-in-VRAM ratchet test both BLOCKED intermediate versions of this change - the format eta is
calibration-class and is now gated off preset estimation entirely.

## 1.20.0 - 2026-07-28

**Anchored predictions: your own two benchmark runs cut the tool's median error from 19% to 6%.**

calibrate's anchor runs (pure-CPU + all-in-VRAM on your own GGUF) now become tier-local
correction ratios - measured-vs-law for the anchor arm, clamped 0.70-1.40 - that scale that
tier's constants for every OTHER prediction. The target arm is never consulted: this is a
calibrated law, not a lookup.

Shipped default-on because it passed the gate it pre-registered (prereg #64) BEFORE any number
existed: leave-one-out across 5 same-state arms (anchor arms excluded), anchored median |error|
5.8% vs the plain law's 19.0%, every arm improved, and every miss in the under-promise
direction. --no-anchors restores the plain law.

Known limit, stated: one GPU ratio prices all formats alike, so Q4_0 (measured 1.8x healthier
than Q2_K-class, L-16) is under-promised by ~43%. Per-format anchoring is U-15, next.

Also in this release:
- The calibration+anchor logic lives in ONE function shared by plan and bench - layer 3 of the
  verification gate caught the two commands disagreeing (the v1.10.5 bug class) within hours of
  the anchors existing, and the release was blocked until unified.
- The boost-state verdict gained a minimum-sample guard after false-positiving on its own short
  anchor run (model-load clocks at 1506 MHz read as "stuck" while real benchmarks sustained
  1873-1898). GPU anchor is now tg128 with 1 s sampling and a 3-sample minimum.
- Prereg #59's DS-Lite "structural miss" is REVISED: at healthy clocks it measures inside its
  staked band - that miss was substantially the stuck-boost state. The out-of-sample kill of the
  L-17 constant stands regardless, on the other arm.

## 1.19.0 - 2026-07-28

**The first external replication found five real defects. All five are fixed, plus the two the
reconciliation found underneath them.**

u/MoneroApe ran the tool on an RTX 3090 + Ryzen 8600G + 64GB DDR5 with a 117.6B MoE (register
E-06). The tool nailed the placement (32%-expert split from --gguf) and missed the speed 9x. The
reconciliation put the miss in OUR detector, not the law - corrected inputs predict 9.3 tok/s
against 9.26-11.36 measured on same-class tuned hardware. Law 4 survived its first external
contact; the inputs did not.

### Fixed, each with a test named after the report

- **Channel count is not stick count.** detect.py treated his 4 DIMMs as 4 memory channels on
  dual-channel AM5 and quoted 173 GB/s where the platform peaks at ~86 - a 2x input error.
  Consumer platforms now default to dual-channel regardless of stick count (HEDT/server CPUs
  recognized by name go wider), and the RAM note says what was assumed.
- **`quantprobe calibrate` (new command): measure, don't assume.** RAM stream (a real read, not
  the spec sheet - the reference box delivers 26.1 of its 48 GB/s "peak"), disk on your own
  file, GPU boost-state health (catches the stuck-clock failure mode that costs 25-30% silently;
  preregs #60/#61), and an optional pure-CPU anchor on your own GGUF. plan consumes it
  automatically, tagged [calibrated].
- **The ubatch cap no longer assumes a 6 GB card.** safe_ubatch's cap rises to 4096 where the
  buffer math allows it - on his 24 GB card the old 2048 cap was the limiter (external datapoint:
  prefill 90 -> 470 tok/s at -b/-ub 4096 on a same-class card; buffer-fit still gates, so tight
  cards never see it).
- **Pinned-memory warning.** `-ot ...=CPU` host buffers are CUDA-pinned; the advice tried to pin
  36.5 GB of his 64 GB, which fails under memory pressure. Any -ot row pinning >45% of system RAM
  now says so and names the fallback (drop -ot, let auto-placement decide).
- **`--threads` in emitted commands.** His fork auto-detected 6 of 12 threads ("decode struggled
  at 2 t/s ... this flag alone helped it jump past 9"; our C-07 measured the same class of swing).
  CPU-resident placements now carry --threads <logical cores> with the caveat spelled out.
- **Speculation reality is TOP-LINE.** The "novel generation drafts 0 tokens" fact he
  independently replicated (D-10) was buried ten paragraphs down and cost him a debugging
  session. It now prints directly under the placement list.
- **pp numbers carry their measurement conditions.** He compared a 22-token prompt against our
  pp2048 figure - partly our fault for publishing the number bare. It now says pp2048, and that
  a 22-token prompt measures startup, not prefill.

Also in this release: the resident-expert sweep fix (prereg #62: the pattern generator's double
VRAM discount cost +15.3% prompt processing and +6.3% generation against its own measured
frontier - now one reserve, counted once).

## 1.18.0 - 2026-07-28

**The format lever: on old GPUs the quantization FORMAT sets decode speed, not just the bytes.**

### New advice: prefer Q4_0 over Q4_K_M on pre-Ampere; never Q2_K when a 4-bit file fits

Measured (pre-registrations #52/#53, same card, same session, interleaved):

    Qwen2.5-7B all-in-VRAM:   Q4_K_M  22.72 tok/s     Q4_0  26.87 tok/s   (+19%, bytes explain 5.7%)
                              Q2_K    21.67 tok/s     <- SLOWER than Q4_0 while 32% smaller

`plan` now surfaces this whenever the all-in-VRAM row wins at <=5.0 bits, with its scope stated:
one Pascal-class card, speed-only (Q4_K_M is higher quality per byte), and explicitly unverified
on Ampere+ where the ranking may invert.

### The mechanism, isolated at the metal (Law 4 amended: L-15)

A standalone CUDA benchmark with zero llama.cpp (`tools/kernelprobe/`) shows a matvec with NO
unpacking runs at 95% of the streaming ceiling, while the same bytes unpacked naively run at 42%.
The decode wall on ALU-weak GPUs is unpack instruction cost, not bandwidth. This names the cause
of C-05 ("a quantized byte is not a byte"), sighted six times before without a mechanism.

### Recorded with equal prominence: what was refuted this session

Fragmentation (#51, +6.5% at a 30x contrast), our own Q2_A format (#54, killed by its own
fairness control), multi-row mmvq blocking (#55, -22%; upstream's Pascal carve-out validated),
the "MoE gather penalty" (an occupancy artifact of our benchmark - scattered expert reads are
free at real block counts), the K-quant min-term tax at real geometry (#56, 0.9-1.8% - counting
instructions is not measuring them), the layout walk (#57, bitwise-matched pairs at 0.99-1.02),
and TWO of our own laws killed by their own out-of-sample kill rules (#56, #59).

### The mechanism that survived every control: metadata application density (L-16)

Q2_K's definition forces a scale+min chain every 4 bytes at 2 bits - 4x Q4_K's density per byte.
Confirmation arm with identical loads and identical dp4a count: +23%. The full decomposition of
real Q2_K decode lands within 9-12% of measurement, closing a contradiction open for weeks.

### Two shipped-copy corrections from a fresh original-case retest (#60/#61)

The plan output now states plainly that a fixed -ngl split measures EQUAL generation speed to the
-ot placement (three measurements, degraded and full clocks) - the -ot advice is earned on prompt
processing (2.2x), KV-in-VRAM safety, and speculation, not raw tg. And a new machine-state
diagnostic: a 25-30% sag after hours of GPU churn was diagnosed (clock polling) and confirmed
(cold-boot A/B) as a STUCK BOOST STATE - SM 1506 vs 1835+ MHz at cool temps. The tool now tells
users to check clocks and reboot; consumer cards cannot reset it in place. After reboot the
original calibration reproduced to 0.5%, and a pristine zero-patch llama.cpp build agreed with
our instrumented build within 1.4% - the whole measurement corpus stands on clean footing.

Register: 75 entries, 11 pre-registrations this cycle (#51-#61), 8 kill rules fired and honoured,
5 verification layers green. New: tools/kernelprobe (the standalone CUDA measurement harness,
zero llama.cpp) and GROK_KERNEL_BRIEF.md (the full kernel ladder with the retraction log, for
external red-teaming).

## 1.17.0 - 2026-07-27

**A new warning that saves CPU-tier users 2.7x, the honest closure of novel-generation
speculation, and the complete physics-vs-code ledger of the CPU decode token.**

### New warning: I-quant files on CPU tiers (the user-facing fix)

Measured (pre-registration #31): on the CPU tier, IQ formats deliver **10.6 GB/s where K-quants
deliver ~29** at the same size - a silent 2.7x decode penalty for anyone who downloaded the IQ
file because it was smaller. `plan` now reads the bytes-weighted I-quant share from the GGUF and
warns when >30% lands on a host-resident placement. It stays silent in VRAM, where IQ formats
measured mid-pack - warning there would be crying wolf.

### Novel-generation speculation: CLOSED, by kill rule

Pre-registration #30 tested every candidate for accelerating FRESH generation: statistical ngram
variants (1.03x best, zero drafts fired), draft-MTP (unmeasurable - three attempts), an external
draft model (net negative, from #28). The kill rule fired and the line is closed. One arm briefly
showed 50 tok/s on a novel task at 100% acceptance - it was `ngram-mod`'s PERSISTENT cross-request
store replaying an identical second request. The third spectacular number this project has killed
by reading the model's actual output; "read what the model wrote" is now a protocol rule.

Scope for users: speculation pays 2.4-2.5x when output REUSES context (edits, refactors, RAG
quoting - most of what coding agents emit), and exactly nothing on fresh reasoning. The plan
output says both, with this model's own numbers.

### The CPU decode token: zero unattributed milliseconds

Four pre-registrations (#27, #31, #32, #33) took the flagship's 84.7 ms CPU token apart:

| component | ms | verdict |
|---|---|---|
| expert weight reads (23.1 GB/s marginal, k-sweep) | 33.5 | at physics |
| always-active weight reads | 19.2 | at physics |
| thread-sync growth, 1 -> 4 threads, identical work | ~13 | code (ggml per-op barriers) |
| serial small-op chain | ~19 | mixed |

Along the way, four attractive stories died by measurement: memory-scatter (shuffled 2 MB
expert-slab reads lose nothing), expert-major dispatch (null, with proof of mode), scheduling
flags (six arms, all flat - llama.cpp's defaults are right), and fat-file K-formats (Q2_K and
Q4_K_M tie at the wall on CPU). The remaining prize is +18-35% via graph-executor sync reduction,
staked before building, upstream-PR shaped - never a fork.

### Also

- `demo/`: reproduce the 50-59 tok/s speculation result yourself from cmd, payloads included,
  with the honest control and the VRAM caveats.
- The fork question (D-05) is reopened SCOPED: the memory system is measured indifferent to
  expert-slab scatter, so the CPU MoE deficit is code - but every other component measured at the
  wall, which is precisely why the vehicle is an upstream PR and not a runtime rewrite.

## 1.16.0 - 2026-07-27

**The wall is measured, and then it is exceeded: 50 tok/s effective decode on a 2016 GPU.**

Three pre-registrations in one session, chasing one goal - get the flagship as close as possible
to its physical decode ceiling, or past it.

### #27: the gap is decomposed, and the wall comes DOWN

The DDR4-3000 spec sheet promises 48 GB/s. Measured: **26.1 GB/s** pure read (one thread already
saturates the controller at 30). So the "realistic wall" we published (52.9 tok/s) was computed on
bandwidth this box cannot deliver - the true raw-decode wall is **41.1 tok/s**, and we sit at
22.25 (54%). Decomposed by ablation: the CPU path runs at 66% of real stream (kernel+MLP share),
and GPU<->CPU sync costs 17-25% of each token. Consequence: no runtime on earth gets this box past
41.1 for raw decode. Anything more requires breaking the every-byte-every-token axiom.

### #28: the axiom is broken - 50.04 tok/s, 22% above the wall

`--spec-type ngram-simple` on llama-server, on the flagship, on the shipped placement: **2.41x
decode (20.7 -> 50.0 tok/s, 89% draft acceptance)** on an edit task, output-identical, one flag,
no download. And two corrections that matter more than the headline:

- **The axis is COPY vs NOVEL, not code vs prose.** Novel generation gets 0% acceptance and
  gains nothing; output that reuses its context (edits, refactors, RAG quoting) gets 89%. Our
  earlier "if you write CODE" advice was a proxy for the real variable.
- **The staked harness was measuring garbage until we read the output.** Temp-0 raw continuation
  degenerated into a repetition loop; ngram feasts on loops (100% acceptance, 1.82-1.96x on
  gibberish). Everything was re-measured under the chat template on coherent output. Also:
  llama-cli silently ignores --spec-type; only llama-server speculates.
- A 0.6B **draft model is NET NEGATIVE** (0.72x) despite 81% acceptance - its own forward passes
  and the VRAM it displaces cost more than verification saves.

### #29: prefix caching moves the asymptote

`cache_prompt: true`: the second question against a 2k-token document pays **183 ms instead of
5381 ms** of prompt time (29x; identical repeat 110x), decode untouched. For RAG and document QA
this dwarfs every placement decision in the register.

### The honest map of decode on this box, all measured

| regime | tok/s | limited by |
|---|---|---|
| raw decode, measured | 22.25 | CPU path at 66% of real stream |
| raw decode, wall | 41.1 | measured DRAM (26.1 GB/s) |
| copy-regime speculation | **50.04** | acceptance x verify cost |
| novel generation | 21.3 | the raw wall - nothing to draft |

Plus the one measurement we cannot make ourselves: whether the ~2x batching ceiling (C-06) is
Pascal or physics. FUTURE.md now carries the one-command replication ask for anyone with a modern
GPU.

## 1.15.0 - 2026-07-27

**Two shipped numbers were wrong, the Pareto frontier turned out to be an artefact, and our
published accuracy band was false for the most common configuration. All three are corrected here.**

### Correction 1: the +/-25% band did not apply to models that fit in VRAM

We published one symmetric +/-25% accuracy band for every placement. For all-in-VRAM it was simply
wrong. Thirteen benchmarks across eight models put the real spread at **-9% to +84%** — real speed
lands between **0.91x and 1.84x** our prediction, usually faster.

`verify.py` now applies a band **per regime**: all-in-VRAM is -15/+90, everything else keeps
+/-25%, which is where the layer-4 anchors validate it. The lower bound is the load-bearing half —
it is what fails if the tool ever becomes *optimistic*, the direction that actually costs a user
something.

We have refuted **six** candidate explanations for the gap, four of them our own favourites:
fixed overhead (pre-reg #15), GPU clock state (cold 144.21 vs warm 143.37 — refuted by its own
prediction), bytes-per-token (refuted by a control with *fewer* bytes and *higher* efficiency),
monotone-in-bit-width (Q8_0 at 8.5 bits is less efficient than Q2_K at 2.8), a per-format constant
(two models both labelled Q4_K_M differ by 21%), and a bytes-weighted mixture over the actual
tensor types — the most promising of them, which forces a *negative* efficiency for Q5_K when you
solve the system. Within one architecture the pattern is clean and monotone in the dominant tensor
type; across architectures it does not transfer at all.

Also corrected: we previously wrote that all-in-VRAM models ran faster than predicted **every
time**. One does not. `gemma4-12b` (sliding-window attention) is over-predicted by 9%, so "floor"
is a floor with one measured exception, not a guarantee.

### Correction 2: both GLM KV-cache figures were wrong, in opposite directions

| model | was | now | why |
|---|---|---|---|
| `glm-744b` | 188,416 | **89,856** | GLM-5.2 is MLA+DSA: 78L x (512+64) latent |
| `glm-air` | 94,208 | **188,416** | GLM-4.5-Air is plain GQA: 46L x 8KV x 128d |

`188,416` is *exactly* Air's correct value — it had been written into the 753B row and halved to
produce the Air row. A transposition, not two independent estimates. Both verified against the
`zai-org` config.json files directly, never back-solved from an observed datapoint, and the MLA
convention checks out against our own DeepSeek-V2-Lite row (27 x 576 x 2 = 31,104, the shipped
value). We had believed this error was "30-60x"; it is 2.10x. Our estimate of our own error was
wrong by more than an order of magnitude, because it reasoned from a symptom instead of reading the
architecture.

### Correction 3: there is no Pareto frontier — one configuration wins

v1.14.x offered a workload-dependent choice between three placements. Re-measured with **all four
cells in one session** — necessary because between-session drift is 10-13% against sub-1%
within-session error bars — two of the three are dominated outright:

| configuration | pp2048 | tg128 | |
|---|---|---|---|
| split, `-ub 1024`, KV in VRAM | **386.04** | **21.58** | **the winner** |
| split, `-ub 512` | 307.13 | 22.02 | wins decode by 0.96 sigma — noise |
| all experts to CPU, `-ub 2048` | 381.82 | 19.79 | dominated |
| split, `-ub 1024`, KV evicted | 381.60 | 17.95 | dominated |

The frontier existed because the winning cell had only ever been benchmarked at `-ub 2048` — past
a compute-buffer cliff — and was then recorded as "dominated". That claim silently inherited the
scope of the sweep that produced it: it really meant *dominated at ub 2048*. **Excluding a
configuration is also a claim, and it needs a scope like any other.**

The whole "evict KV to buy prompt speed" trade was never necessary. Keeping KV in VRAM gives *more*
prefill and 20% more decode. The claim shrank three times before it died — 2.25x, 1.33x, 1.23x,
gone — and every shrink came from fixing one of our own measurement errors.

### New: the ubatch cliff, and `-ub` is now sized rather than pinned

llama.cpp's CUDA compute buffer is linear to four figures (0.5874 MiB per ubatch token). Demand
grows smoothly, VRAM ends abruptly, so prefill falls off a **cliff**: 381.21 -> 209.64 tok/s in one
`-ub` step, then flat forever. v1.14.x quoted 391.72 for a command that delivers 209 on the same
machine — the difference was roughly 250 MiB of desktop VRAM, one browser window. `-ub` is now
derived from measured headroom.

### New: your numbers are single-stream, and a server is ~2x faster

Every figure this tool prints was measured at one request at a time. With 8 parallel slots,
aggregate throughput is ~2x higher and saturates by about 4 slots — the *same* ratio on the MoE
split (2.03x), on all-experts-to-CPU (1.90x) and on a dense fully-resident control (2.25x). Two
explanations died: host-residency amortisation was refuted **by direction** (the arm with more
host-resident weight gained *less*), and MoE routing divergence was refuted by the dense control.
We are not naming a mechanism. The plan output now says so.

### New: a findings register the release gate enforces

`findings/REGISTER.json` is the canonical machine-readable record of every law, lever, dead end,
contradiction and untried lever, each carrying the scope it was measured in. `FINDINGS.md` is
generated from it. `findings.py` fails the build if a staked pre-registration is uncited, a claim
lacks a scope, or a "wired into" target no longer exists — and `verify.py` layer 5 runs it. Writing
it immediately caught four pre-registrations mis-numbered from memory and five that nothing
referenced.

### Withdrawn as measured dead ends

- **Frequency-ranked expert residency.** Not expressible in stock llama.cpp — all experts of a
  layer live in one fused tensor and `-ot` matches names, so its finest unit *is* the contiguous
  split we already ship. ktransformers had to patch llama.cpp for their own comparison, and at our
  VRAM ratio their matched-memory table shows frequency ranking buys +2.5% to +5.3%, not the +15%
  to +40% we had staked.
- **Pinned host memory.** Already in use: `-ot ...=CPU` routes to `CUDA_Host` (`cudaMallocHost`)
  whenever `--no-mmap` is set, which quantprobe always sets. Predicted +30% is 0%.

## 1.14.1 - 2026-07-27

**Correction: v1.14.0 shipped a dominated point on the frontier, and fixing it makes the feature
less impressive than advertised.**

The frontier was built by sweeping placement and KV while holding `-ub` pinned at 2048. Measured
afterwards, the point shipped as the *decode champion* is beaten by its own small-batch form:

| split, KV in VRAM | pp2048 | tg128 |
|---|---|---|
| `-ub 512` | **280.64 ± 1.32** | **20.25 ± 0.16** |
| `-ub 2048` (shipped in v1.14.0) | 163.39 ± 0.36 | 20.13 ± 0.23 |

72% more prompt processing for the same generation, free. Pinning one dimension while sweeping the
others is exactly the error the Law 4 fungibility corollary warns about — made in the code that
implements that corollary. **A frontier is only Pareto-optimal with respect to the dimensions
actually swept.**

Consequence, stated plainly: because the worst available choice is now much better than we
thought, **the benefit of choosing correctly drops from 2.25× to 1.33×** on a long-prompt
workload. That is still above the ≥15% bar pre-registration #20 set for keeping the feature, but
it is a materially smaller claim than v1.14.0 made and the smaller number is the true one.

Corrected frontier (all measured, reference box):

| configuration | pp2048 | tg128 | best for |
|---|---|---|---|
| split, KV in VRAM, `-ub 512` | 280.64 | **20.25** | chat, short prompts |
| all experts → CPU, `-ub 2048` | 345.41 | 18.68 | mixed |
| split, KV evicted, `-ub 2048` | **391.72** | 16.54 | RAG, documents, agents |


## Research note - 2026-07-27 (no code change)

**The last placement dimension is closed, and the answer is "don't use it."**

[Pre-registration #18](preregistrations/2026-07-26-topk-decode.md) measured MoE expert-count
reduction on decode — never tested before, since every log in this repo showed
`expert_used_count = 8`.

| | tok/s | WikiText-2 ppl |
|---|---|---|
| k=8 | 20.32 ± 0.14 | 9.2364 ± 0.358 |
| k=4 | **27.13 ± 0.33** | **11.1411 ± 0.436** |

All three speed stakes hit, and the byte model predicted **27.7** against a measured 27.13 —
accurate to 2%. The lever is real and correctly modelled.

**It is also strictly dominated.** On the same placement, with no metadata surgery:

| lever | speed | quality cost |
|---|---|---|
| top-k 8 → 4 | ×1.335 | **×1.206** |
| bits 2.95 → 2.0 | **×1.424** | **×1.048** |

Quantizing further is faster *and* four times cheaper in quality. Top-k=4 costs more quality than
the entire 2-bit quantization of the model, and breaches `optimize`'s ×1.12 ceiling on its own.
There is no operating point where a user should prefer it — so nothing ships, and it is recorded
in LAWS.md as a measured dead end so it is not re-derived.

The quality figure was measured on **this** file rather than cited: it independently reproduces
H6's +20.7% from a different model, which is worth knowing separately.

With this, all four dimensions the placement search was blind to are closed — **batch** (#19),
**phase** (#20) and **KV placement** (#21) shipped in v1.13.0–v1.14.0; **expert count** measured
and rejected.


## 1.14.0 - 2026-07-27

**There is no single best placement. There is a frontier, and the right point on it depends on
how much prompt you read per token you write.**

[Pre-registration #21](preregistrations/2026-07-27-kv-placement.md) added the last missing
dimension — KV placement, which llama.cpp exposes via `-nkvo` **independently of where the weights
live**, and which our law had welded to the layers it serves. Measured on the reference box,
`Qwen3-30B-A3B Q2_K`, one session, `-ub 2048`, r=3:

| placement | KV | pp2048 | tg128 | |
|---|---|---|---|---|
| split K=16 | VRAM | 161.59 ± 0.09 | **20.14 ± 0.24** | decode champion |
| split K=16 | host | **391.72 ± 2.80** | 16.54 ± 0.03 | prefill champion |
| all → CPU | VRAM | 345.41 ± 0.36 | 18.68 ± 0.27 | balanced |
| all → CPU | host | 336.31 ± 1.71 | 15.82 ± 0.08 | **dominated — never choose it** |

`plan` now prints the frontier and which end suits which workload:

| workload | best | vs worst choice |
|---|---|---|
| chat (0.5 : 1) | split, KV in VRAM | 1.23× |
| coding (10 : 1) | all → CPU, KV in VRAM | 1.35× |
| RAG (50 : 1) | split, KV evicted | 1.91× |
| document QA (200 : 1) | split, KV evicted | **2.25×** |

### The finding underneath

**The VRAM claimants are fungible, and fungibility is placement-specific.** Evicting KV recovers
**2.42×** of prompt processing on the split — where VRAM binds — and essentially nothing on
all-experts-to-CPU (345 → 336), where it doesn't. Weights, KV cache and compute buffer draw on one
budget; only the configuration that is actually starved can spend the refund.

That closes the four dimensions the search was blind to. Batch and KV are now measured and
disclosed; phase is disclosed; expert count remains open (#18, blocked on a harness).

### A miss worth recording

I staked that `-nkvo` would be **neutral at zero context** — no cache, nothing to read, nothing to
place. It costs **19.5%** there. The penalty is largely *fixed*, not depth-proportional, which
means the flag moves where attention is **computed**, not merely where the cache is stored.
Anyone modelling it as pure cache-read bandwidth, as I was, will mis-predict it.


## 1.13.1 - 2026-07-27

**Correction to v1.13.0, found by measuring the same lever on a placement we had not tested.**
v1.13.0 recommended `-b 2048 -ub 2048` on the MoE **split** placement — its own default for the
flagship model — where it is now measured to cost **42% of prompt processing**.

The gate tested *"is anything host-resident"*, which the split satisfies
(`split experts: N%->VRAM, rest->RAM`). But the split exists to fill spare VRAM with experts, and
that is exactly the VRAM the larger compute buffer needs. Same flag, opposite sign, depending on
placement:

| placement | pp2048 @ub512 | pp2048 @ub2048 |
|---|---|---|
| all experts → CPU | 199.90 | **349.59 (+75%)** |
| split, K=16 → VRAM | **279.07** | 161.87 (**−42%**) |

#19 measured `-ub` on all-experts-to-CPU and on a fully-VRAM-resident control, reached the right
conclusion about the mechanism, and *still* shipped a wrong gate — because the split is neither
of those cases and nobody looked. **A double dissociation proves a mechanism; it does not
enumerate a decision surface.**

### New: `plan` says which phase its command optimises

[Pre-registration #20](preregistrations/2026-07-27-phase-split-placement.md), 4/4 stakes hit:
prompt processing and generation want **different placements**.

| placement | pp2048 @ub2048 | tg128 |
|---|---|---|
| all experts → CPU | **349.59 ± 1.78** | 18.54 ± 0.16 |
| split, K=16 → VRAM | 161.87 ± 0.24 | **20.16 ± 0.18** |

The split wins generation by 9% and loses prompt processing by **2.16×**. Ranking by decode — what
this tool has always done — silently hands long-prompt users the worse configuration by a factor
of two. `plan` now says so, and prints the alternative.

Also notable: **adding a lever inverted which placement is fastest at prefill.** At the default
ubatch the split wins (279 vs 200); at `-ub 2048` the all-CPU placement wins (350 vs 162).
Placement and batch are not independent dimensions — they compete for one VRAM budget.


## 1.13.0 - 2026-07-27

**+73% prompt processing, one flag, no download — for anyone running a MoE with experts in RAM.**

`plan` now emits `-b 2048 -ub 2048` on placements that leave weights in **host memory**. Measured
on the reference box ([pre-registration #19](preregistrations/2026-07-27-ubatch-cpu-resident.md),
r=3, warm-up discarded):

| `-ub` | Qwen3-30B, `-ot exps=CPU` | dense 7B **fully in VRAM** |
|---|---|---|
| 512 | 199.90 ± 1.42 | 329.80 ± 0.90 |
| 1024 | 277.17 ± 1.70 | 333.07 ± 0.27 |
| 2048 | **345.89 ± 0.88  (+73%)** | **200.31 ± 0.17  (−39%)** |

**The control is the result.** Same flag, same box, same session — opposite signs. A speedup alone
would be consistent with "bigger batches are just better"; a 39% *regression* on the VRAM-resident
model is not. Weight residency is the only difference, and it is exactly what the mechanism
predicts: with `-ot exps=CPU` the experts live in a host buffer, so CUDA is offered the op and
accepts it once the ubatch clears 32 tokens (`ggml-cuda.cu`, `MUL_MAT_ID → op->ne[2]`). Those
weights then cross PCIe **once per ubatch instead of once per token**. With nothing host-resident
there is no transfer to amortise and the larger compute buffer just costs VRAM.

So the flag is **gated, never defaulted** — host-resident weights, and only with VRAM headroom.
Decode is unaffected (18.46 → 18.76): a ubatch cannot be filled one token at a time.

**This is a search dimension, not a law change.** `plan.evaluate()` is untouched and all four
published anchors are bit-identical. Batch size was simply never an input to the placement search,
which is why a 1.73× effect sat unmeasured on the most-used MoE path.

### Also fixed: `bench` silently dropped any flag it did not hand-list

`runtime.py` forwarded `-ngl` and `-ot` into `llama-bench` and dropped the rest. Shipping `-ub`
through that forwarder would have had `bench` quietly measure the **un-flagged** configuration and
report the law as wrong by exactly the size of the new lever — while both printed commands still
looked correct. Unknown flags now raise instead of vanishing.


## 1.12.1 - 2026-07-27

**The dense layer-split row emitted an `-ngl` that tells llama.cpp to put every layer on a GPU
that cannot hold them.** It computed `int(g * 99)` where `g` is a *fraction* and `99` is the
all-layers sentinel used elsewhere in the same file.

Two failures, the second serious:

- The split was misreported. `llama-70b` (80 layers) printed *"split: 50% layers→VRAM"* and
  emitted `-ngl 49` — 61%. Prediction and command disagreed.
- For any model with `<= 99·g` layers the flag **exceeds the layer count**, so llama.cpp places
  *every* layer on the GPU — on a row that exists **only** when the model does not fit in VRAM.
  A 32-layer model does this for any `g > 0.32`. That is an OOM, or a silent thrash on Windows
  with driver memory fallback.

Now emits `int(g · n_layer)`, and **suppresses the row** when no layer count is grounded rather
than printing a command it cannot honour. `--gguf` always unlocks it (autospec reads
`block_count`). `mistral-7b` (32) and `llama-70b` (80) gain `nl` from the exact counts already
written in the table's own comments. The row label now reads `split: 40/80 layers→VRAM` so the
prediction and the flag are checkable against each other by eye.

Same shape as v1.10.5 and v1.11.1: a layer count with a proper resolver, at a call site that
didn't use it. Found by a 25-agent survey auditing our own emitted flags.

### The test harness was validating two different copies of the code

Chasing this exposed something worse than the bug. `python tests/smoke.py` puts `tests/` on
`sys.path[0]` — **not** the cwd — so in-process `from quantprobe.plan import ...` resolved to
**site-packages** while the subprocess CLI tests resolved to the **repo**. Half the suite silently
tested stale installed code, so an edit that hadn't been reinstalled was invisible to it.

That cost a real mutation test: re-introducing this bug reported `ok` while the same function
failed when called directly, and three rewrites of the assertion went by before the harness
turned out to be at fault. `smoke.py` now puts the repo first; `verify.py` layer 2 still covers
the installed artifact deliberately and separately.

The surviving assertion is physical rather than formal: **the layers sent to the GPU must fit in
its VRAM.** Checking the printed label against the emitted flag does not work — both derive from
the same variable, so a wrong value agrees with itself — and a bare range check does not either,
since `-ngl 49` is comfortably under 80 layers while asking a 24 GB card to hold 26.3 GB.


## 1.12.0 - 2026-07-26

**A dense model's predicted speed did not respond to its quantization at all.** Reported from the
published calculator: Gemma 4 12B (dense, 11.9B active) shown at 28 tok/s on a DGX Spark while
GLM-4.5-Air 106B (MoE, 12B active) showed 42 — a 12B behind a 106B with the same active
parameter count.

The tables set `ne = t` for dense models. That is true for **activation** (every parameter is
read per token) and false for **quantization**. Since the law prices always-active parameters at
the recipe's protected precision, `max(bits, 4.5)`, the entire dense model was priced at ≥4.5
bits: Gemma came out at 7.70 GB/token — and therefore identical tok/s — at 2.5 bits *and* 4.5.

The recipe protects `attn_`/`ssm_` only, not embeddings and not the FFN. Measured from real GGUF
tensor shapes that share is 10.8% (Qwen2.5-7B), 20.2% (gemma4-12B), 25.1% (Qwen3.5-4B), 29.6%
(Qwen3-0.6B). `DENSE_PROTECTED_SHARE = 0.214` applies the mean, **dense models only** — MoE is
untouched, since there `ne` already names the protected set exactly.

Gemma at 2.5 bits now predicts **43.1**, marginally ahead of the 106B MoE with the same active
parameters, and responds to bit-width at all (43.1 at 2.5 vs 28.0 at 4.5).

### Validation

[Pre-registration #17](preregistrations/2026-07-26-dense-activation-model.md), staked before the
held-out measurement. Mean |error| over the six dense in-VRAM points falls **18% → 10%**.
Held-out on `Qwen2.5-7B IQ3_M`, never benchmarked before: **P-1 hit, P-2 hit decisively (error
−22% → −7%), P-3 miss, P-4 hit** (all four published anchors are MoE and are bit-identical).

The P-3 miss is the more useful result. Measuring the held-out model three times gave **17.53
(cold, 810 MHz) → 17.03 → 16.89 (settled, 72 °C)**, a 3.8% decay with non-overlapping error bars.
The band I staked came from siblings measured earlier and evidently cooler, so part of that miss
is my own protocol. Third GPU-state effect after orphaned-process contention and boost-clock
inflation; temperature now joins the logging convention.

### New: `weights/plausibility_sweep.py`

Cross-model relational invariants over every model × machine × bit-width the calculator can show
— 2,987 comparisons. The five verification layers each check the law against something *we*
chose. None of them asks the question a reader asks: **do these numbers make sense next to each
other?** That question is what found this bug, so it is now a check.

Its most important invariant is the one that reaches *outside* the law, to declared parameter
counts: bytes-per-active-parameter must rise with the protected share. The other five compare the
law's outputs against each other and are therefore blind to an error in how those outputs are
computed — verified by mutation test, which also caught an earlier version narrowed until it
could no longer see the bug it was written for.


## 1.11.2 - 2026-07-26

**Says out loud where the law is weakest, and asks for the datapoint that would fix it.**

All-in-VRAM is the most common placement for anyone with adequate VRAM, and it is the one this
law knows least well. Across all seven models measured there on the reference box, the real speed
came in **faster than predicted every single time — by 2% to 67%.** That is a one-directional
bias, not noise, and it has been sitting in a pre-registration rather than in front of the people
it affects.

`plan` and `optimize` now both say so whenever the winning placement is all-in-VRAM: read that
number as a floor rather than a ceiling. Below 4.5 bits they add the companion finding — the model
already fits, and a lower quant buys almost nothing once it does (the same 7B at Q2_K vs Q4_K_M is
36% smaller and **4% slower**).

**And they ask.** Seven points on one GPU cannot identify a better functional form, and the last
two times a constant moved on thin evidence it cost a public correction. So instead of quietly
shipping a number we know is low, the note ends in a request: `quantprobe bench --contribute` on a
GPU-resident model turns your machine into the datapoint that settles it. Results landing
*outside* the predicted band are the most valuable ones we can receive.

The regime stays pinned by the `VRAM_GAPS` ratchet, so it can improve but never silently worsen.

Also: the consistency test that guards this now pins the *invariants* — that both commands
disclose the floor and both make the ask — rather than exact prose, which is what broke it when
the wording changed.


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
