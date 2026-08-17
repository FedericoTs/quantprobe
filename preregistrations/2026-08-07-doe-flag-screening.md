# Pre-registration #95: which flags actually matter — Morris screening, and the binding constraint checked a second way

**Author:** Federico Sciuca · **Date staked:** 2026-08-07, **BEFORE any screening run.** **STAKED.**

**Method and tooling credit: [bigattichouse](https://github.com/bigattichouse)** —
[`llama-optimize`](https://github.com/bigattichouse/llama-optimize) (llama.cpp flag tuning by
Design of Experiments) and [`robust`](https://github.com/bigattichouse/robust) (the general DoE
toolkit under it: Morris screening, Sobol variance attribution, Taguchi arrays, in C, CC0-1.0
public domain). Registered as E-16. The design below is theirs; the hardware, the flag set and
the predictions are ours, and the predictions are what is being staked.

## Why this, and why now

Two independent reasons, one of which can embarrass us.

**1. Our autotune searches; it does not explain.** Prereg #71 shipped a fixed-budget flag
search. It returns a config and no understanding — we still cannot say which knobs carry the
effect on this card, or which ones interact. Morris elementary effects answer exactly that in
~R(k+1) runs, ranking factors by **μ\*** (magnitude of effect) and flagging interaction or
non-linearity via **σ**. That is a *finding*, not just a faster sweep.

**2. Our binding-constraint classifier has never been checked against measured variance.**
`plan` prints which resource binds (VRAM bandwidth / system RAM bandwidth / capacity / CPU
compute) **derived from the law**. Sobol first-order and total-order indices measure which
factor actually explains the variance in tok/s. These are two routes to the same claim and they
have never been compared. If they agree, the classifier gains an independent basis. If they
disagree, that is the most interesting result this project could get this month.

**Independent convergence worth recording:** `llama-optimize` settles GPU temperature between
runs and records start temperature per run, as routine hygiene. We reached the same conclusion
the hard way (preregs #60/#61: a stuck boost state cost **28%** and looked like nothing).
Two projects, different methods, same conclusion — a benchmark without machine-state control
measures the state, not the config.

## Protocol

- Factors (the knobs we already ship advice about): `-ngl`, `-ub`, `-t`, KV cache type
  (f16/q8_0), `-ot` expert-offload fraction, `--no-mmap`, `-np` concurrency, `-fa`.
- Two models spanning the regimes that behave differently here: **Qwen2.5-7B Q4_K_M**
  (all-in-VRAM) and **Qwen3-30B-A3B Q2_K_L** (CPU-expert split). The law says these are bound by
  different resources; if the screening does not separate them, the law is in trouble.
- Response: `tg128` tok/s, one machine state throughout (C-14), clocks logged before and after
  every run, thermal settle between runs **per the llama-optimize practice**.
- Stage 1 Morris screening → rank by μ\*, flag σ. Stage 2 Sobol on the survivors for variance
  attribution with bootstrap CIs. Taguchi confirmation only if stage 1 and 2 agree.
- Runs are serial and lock-guarded like every other measurement here. Raw CSV/JSON committed.

## Staked expectations

- **P-1 (concentration).** The top **3** factors carry **≥70%** of total μ\* on both models.
  If effects are spread evenly across 8 knobs, per-machine tuning is worth far more than we
  have been telling people, and our "free speed you already have" framing is too modest.
- **P-2 (the regimes separate).** The top-ranked factor **differs** between the all-in-VRAM
  model and the CPU-split model. Same tuning advice for both regimes would contradict the
  placement physics we publish.
- **P-3 (the classifier holds).** For each model, the factor with the highest Sobol
  total-order index maps to the resource `plan` names as binding. Stated as a mapping table in
  advance: capacity-bound → `-ngl`/`-ot`; RAM-bandwidth-bound → `-ot`/KV type; VRAM-bandwidth
  bound → KV type/`-ub`; CPU-compute-bound → `-t`.
- **P-4 (interaction warning).** `-ub` × `-ngl` shows σ above the median — the batch/placement
  interaction we already measured indirectly (prereg #19: `-ub 2048` is **+73%** on the CPU-expert
  split and **−39%** all-in-VRAM — the same flag with opposite signs by placement).

## KILL RULES

- **If P-3 fails**, the binding-constraint line — shipped in `plan`, quoted in the README, drawn
  on the pipeline chart — is **not validated by measurement**, and it gets a scope label
  ("derived from the law, not confirmed by variance attribution") the same day, at full
  prominence, until re-derived.
- **If P-1 fails** (effects spread evenly), autotune's fixed-budget design is wrong in principle,
  not just in efficiency, and the DoE funnel replaces it rather than supplementing it.
- **If Morris and Sobol disagree on the top factor**, neither is published as a finding until a
  Taguchi confirmation run adjudicates. Two methods disagreeing is a reason to measure again,
  not to pick the flattering one.

## What this changes if it works

`quantprobe autotune` becomes screen-then-optimise instead of search: Morris to drop dead knobs,
Sobol to spend the budget where the variance is, Taguchi to land the config — the funnel
bigattichouse built, seeded by our law so the search starts inside the plausible region instead
of the full grid. Their tool needs 25–125 GPU runs because it knows nothing about the machine
beforehand; ours knows where to look. That is the collaboration, and it points both ways.

**Wired into:** pending — E-16, `autotune` successor design, the binding-constraint scope note,
and `docs/ROADMAP.md`.


---

## Pre-data amendment - 2026-08-16, before any stage-1 run

Design doc: `docs/DESIGN_DOE_MORRIS.md` (committed with this amendment, together with the
harness `weights/doe_morris.py` and the pre-committed scorer `weights/prereg95_score.py` -
scorer in the repo before the first CSV row exists, per house rule). Deviations found
while making the staked protocol executable on this box, declared before data:

1. **Model substitution:** Qwen3-30B-A3B **Q2_K** (11,258,610,240 bytes on disk), not the
   staked Q2_K_L - the Q2_K_L we hold is the Coder finetune, which prereg #102's verify
   pass established must never be joined across. Same base model, same regime.
2. **`-np` is not exercisable in stage 1:** llama-bench b10098 rejects it
   (`error: invalid parameter for argument: -np`) - it is a llama-server concurrency
   flag. Stage 1 screens 7 of the 8 staked factors; `-np` screening is deferred to a
   server-harness arm (U-05 lineage), not silently dropped.
3. **`-ot` dropped for the 7B only:** dense model, no `_exps` tensors; probed inert
   (override recorded, zero tensors matched, tok/s unmoved). 7B design is k = 6.
4. **KV factor narrowed to `-ctk`:** `-ctv q8_0` requires `-fa` on (probed:
   context-creation failure with fa off), which would confound the KV and fa factors on
   a hypercube. `-ctv` pinned f16. U-01 measured K+V jointly; stage 1 screens K only.
5. **`-fa` levels are {0, 1}, never `auto`** (the build default): auto lets the build
   decide per-config and hides the factor.
6. **`-t` range is {1..4}:** i5-7600K is 4C/4T; the staked range implied oversubscription
   levels that do not exist on this part.
7. **30B expert-offload fraction restricted to [0.75, 1.0] CPU-side** ({36,40,44,48} of
   48 layers), exercised via generated `-ot` patterns in the shipped direction (early
   blocks' experts stay on GPU). Lower fractions OOM the 6 GB card at the max-ngl
   corner. Build's `-ncmoe` not used: it offloads the FIRST n layers - a different
   layer set than the shipped recipe.
8. **Chosen design constants:** Morris R = 10, p = 4, delta = 2/3, seeds
   `"prereg95:{7B|30B}:20260807"`; response tg128 r = 3 (per-rep samples read from
   JSON); `-b` pinned 2048; runs 70 + 80 = 150; timeouts 240 s / 360 s with DNF rows;
   thermal settle to <= 52 C (min 30 s, cap 180 s) between runs.

The harness cannot write this prereg; this amendment was appended by the operator before
the first designed run. P-1/P-2/P-4 remain scoreable exactly as staked on the 7-factor
stage 1; P-3 (Sobol) is stage 2 and its scorer will be committed before stage 2 runs.


---

## SCORED - stage 1 (Morris), 2026-08-16, by the pre-committed scorer

150 of 150 designed runs on disk, 0 DNF, one machine state, thermal settle 38-49 C
throughout. CSV sha256 c35c7e9d718b3d80..; verdict json committed beside it.

| stake | verdict | evidence |
|---|---|---|
| P-1 concentration | **PASS** | top-3 mu_star share: 7B **0.972** (ngl, t, fa), 30B **0.768** (t, mmp, ngl) - both over the 0.70 bar |
| P-2 regimes separate | **PASS** | 7B top factor **-ngl** (13.5 tok/s per unit range); 30B top factor **-t** (10.7) - the all-in-VRAM-class dense model and the CPU-expert split want different knobs first, as the placement physics requires |
| P-4 interaction warning | **FAIL** | sigma(-ub): 7B 0.076 vs median 1.028; 30B 0.706 vs median 4.256 - BOTTOM of both rankings, not above median |
| P-3 classifier vs Sobol | deferred | a Sobol claim; Morris cannot score it and the scorer says so verbatim. Stage 2. |

**The P-4 miss, diagnosed:** the stake imported prereg #19's `-ub 2048` asymmetry (+73%
CPU-split / -39% all-in-VRAM) as evidence of a `-ub x -ngl` interaction - but #19's
asymmetry is a PREFILL effect, and this experiment's response is tg128, decode only
(`-p 0`). On decode, `-ub` ranked DEAD LAST by mu_star on both models (7B 0.069, 30B
0.411 tok/s per unit range). The interaction warning was staked onto the wrong phase.
Published at full size: the flag most guides tell you to tune first is the flag that
moves single-user decode the least.

**Unstaked observations, recorded as observations only:** (a) `--no-mmap` is the #2
factor on the 30B split (mu* 7.0) with the highest sigma (6.0) - strongly interacting,
consistent with its role gating whether expert reads hit page cache or disk; a Sobol
stage-2 candidate. (b) The 7B's top-3 concentration (97.2%) is extreme: ngl, t and fa
are effectively the whole story on the dense model; ctk/ub/mmp are noise-level there.

**Kill rules:** P-1 passed, so autotune's fixed-budget design survives as a funnel stage;
P-3's kill rule (scope-label the binding-constraint line) stays ARMED pending stage 2 -
nothing about stage 1 discharges it. Morris-vs-Sobol adjudication also waits for stage 2.

Chain of custody: staked 2026-08-07 (before any screening run) -> amended pre-data
2026-08-16 with 8 declared deviations -> harness + scorer committed before the first CSV
row (31dd79c) -> 150 runs, one night, one machine state -> scored by the frozen scorer,
rc 0. The miss publishes at the same size as the hits.


---

## Pre-data amendment - stage 2, 2026-08-16, before any stage-2 run

Design doc: `docs/DESIGN_DOE_SOBOL.md` (committed with this amendment, together with the
`--stage2` harness mode and the pre-committed scorer `weights/prereg95_sobol_score.py` -
scorer in the repo before the first stage-2 CSV row, per house rule). Deviations and
constants, declared before data:

1. **Survivor rule:** factors covering >= 90% of stage-1 total mu_star, UNION the
   model's P-3 mapping-set factors (a design that excludes the mapping factors could
   never PASS the stake it scores). Result: 7B k=4 {ngl, t, ctk, ub} (95.7% of mu*);
   30B k=5 {t, mmp, ngl, moe_cpu_frac, ctk} (96.2%). Survivor level lists identical to
   stage 1.
2. **Non-survivors fixed at stage-1 best-observed levels:** 7B `fa=0, mmp=0` (best ok
   row 22.486 tok/s); 30B `ub=2048, fa=1` (best ok row 20.497 tok/s). Every fixed factor
   is outside both mapping sets, and max(mu*, sigma) of each is <= 10.7% of its model's
   top-survivor mu*, so fixing removes only FAIL modes it could not plausibly have won.
   The 7B fa x ngl interaction is thereby excluded from measured variance, recorded not
   estimated.
3. **Design:** Saltelli N*(k+2), seeded stdlib Monte Carlo (no QMC dependency), seeds
   `"prereg95:stage2:{7B|30B}:20260816"`, N_start 32 (7B) / 40 (30B), N_cap 64, runs
   192 + 280 = 472, extension in +16-block steps of the same stream on an UNDECIDED
   gate. Estimators: Saltelli-2010 Table-2(b) first-order, Jansen-1999 total-order;
   bootstrap 1000 block resamples, seed `"prereg95:stage2:bootstrap:20260816"`; argmax
   DECIDED iff rank-1 retention >= 950/1000.
4. **Two nights by resume, declared:** 11.9 h nominal at stage-1's measured per-run
   costs (84.4 s 7B at fixed mmp=0, 95.3 s 30B). Per-model scoring as each design
   completes; a publishable model-FAIL short-circuits overall P-3 (the 7B can fire the
   kill rule after night 1).
5. **P-3 ground truth frozen:** plan classes measured 2026-08-16 on this box
   (7B VRAM-bandwidth-bound at 100%; 30B system-RAM-bandwidth-bound at 51%, margin
   1.03x recorded as context), embedded in the scorer as constants; mapping sets
   {ctk, ub} and {moe_cpu_frac, ctk}. The near-tie does not widen the 30B set.
6. **Morris-vs-Sobol disagreement** is handled per kill rule 3: any model where the two
   argmaxes differ is HELD_FOR_TAGUCHI and nothing about its ranking publishes until a
   Taguchi arm adjudicates - except that a mapping refuted under BOTH candidate
   argmaxes is a publishable P-3 FAIL with only the factor identity held.
7. **Stricter than staked, accepted:** if the argmax is still UNDECIDABLE at N_cap=64,
   the binding-constraint scope label ships anyway - the label text ("derived from the
   law, not confirmed by variance attribution") is literally true in that branch, and
   perpetual undecidability must not become a shield. This can only add a label the
   staked rules would not force; it can never suppress one.
8. `-np` remains not exercisable (llama-bench rejects it; server-only), unchanged from
   stage-1 amendment item 2.
9. **As-built scorer strengthening, declared:** DECIDED additionally requires the
   bootstrap-modal argmax to EQUAL the full-data argmax - a night whose gate passes but
   whose modal and point-estimate argmaxes disagree reads UNDECIDED, never decided.
   Strictly conservative: it can only defer a verdict, never manufacture one.

**The stage-1 consequence this arm inherits, stated before its own data:** the P-3
mapping sets and the Morris argmaxes (7B ngl, 30B t) are already disjoint, so stage 2
cannot produce a P-3 PASS - every branch ends in a publishable FAIL, a HELD_FOR_TAGUCHI,
or an UNDECIDABLE-at-cap, and all three ship or hold exactly per items 6-7. Written down
now so the outcome cannot be mistaken for a post-hoc choice.

The harness cannot write this prereg; this amendment is appended by the operator before
the first stage-2 run.


---

## SCORED - stage 2 night 1 (7B Sobol), 2026-08-17, by the pre-committed scorer

**P-3: FAIL. The kill rule fired, and the scope label shipped the same day.**

192 of 192 designed runs (N=32 blocks), 0 DNF, one machine state, finished 4.4 h into the
9.5 h window. CSV sha256 4234b358..; verdict json committed beside it.

| item | value |
|---|---|
| Sobol total-order, 7B | ngl **1.131** [0.795, 1.469] >> t 0.033 >> ctk 0.0001, ub 0.0001 |
| decidability gate | ngl holds rank 1 in **1000/1000** bootstrap resamples - DECIDED |
| methods | Morris argmax ngl, Sobol argmax ngl - AGREE (no Taguchi hold) |
| staked mapping (VRAM-bw-bound) | {ctk, ub} - the measured argmax is in neither |
| verdict | publishable P-3 FAIL; overall P-3 FAIL (per-model short-circuit, amendment item 4) |
| 30B | REFUSED (no night-2 data yet) - the refusal is the correct behavior, night 2 queued |

**What the FAIL means, precisely.** The classification's time decomposition is untouched
arithmetic and still stands - on this 7B cell the law says VRAM bandwidth takes 100% of
the decode token, and nothing here contradicts that. What failed is the staked FLAG-LEVEL
mapping: the prereg bet that the binding resource's within-placement levers (KV type,
ubatch) would carry the measured variance. They carried none of it (ST 0.0001 both).
The variance lives in the PLACEMENT lever (-ngl spans 0..99 and moves the config across
placements), which the mapping table never listed. A better-constructed stake would have
separated within-placement variance from across-placement variance; this one did not, it
lost, and the label prices that.

**Kill rule executed (same day, full prominence):**
- `quantprobe plan` / `quantprobe report`: every binding-constraint print now carries
  `validation: derived from the law, not confirmed by variance attribution` directly
  under the headline (plan.py binding_report, report.py _limits; report body keeps the
  plain-words form, the register pointer lives in its Sources).
- README quickstart example + prose: the label line added to the example block, and the
  prose now explains what is and is not confirmed.
- Both report-card assets (weights/data/card_flagship.svg, media/card_flagship.svg via
  make_report_card.py) draw the label under their binding line.
The label stays until a re-derivation - a variance design that separates placement from
within-placement factors - survives a Taguchi arm.

Night 2 (30B, 280 runs) queued; its verdict can refine the story but cannot un-fire the
kill rule. Chain of custody: mapping staked 2026-08-07; ground truth frozen and scorer
committed 2026-08-16 before any stage-2 row; measured overnight; scored by frozen code
2026-08-17. The third staked miss of prereg #95, published at the same size as its hits.


### stage 2 night 2 (30B Sobol), 2026-08-17: UNDECIDED at N=40 - extending, not deciding

280 of 280 designed runs, 0 DNF, 7.5 h. Total-order: mmp **0.914** [0.571, 1.289] vs t
**0.869** [0.597, 1.184] - a statistical dead heat, and the bootstrap agrees: the modal
argmax holds rank 1 in only **569 of 1000** resamples against the pre-committed 950 gate.
ngl 0.325, ctk 0.312, moe_cpu_frac 0.168.

The scorer refused to issue a 30B verdict from an undecided argmax and printed the
pre-declared remedy verbatim: extend the SAME seeded stream to 56 blocks (amendment item
3, +16-block steps, cap 64). That extension is running; a design that cannot separate its
top two factors does not get to pick the flattering one.

Note what is NOT affected: the overall P-3 verdict was already FAIL on the 7B alone
(per-model short-circuit, amendment item 4), and the scope label shipped 2026-08-17.
Nothing the 30B can return will un-fire that. What the extension buys is the honest
answer to a different question - whether --no-mmap or -t carries decode variance on a
CPU-expert split - which stage 1 already flagged as the strongest interaction candidate
on the board.


---

## SCORED - stage 2 complete, 2026-08-18: the 30B is UNDECIDABLE at the cap

448 of 448 designed runs, 0 DNF, three sessions, one machine state throughout. The
extension ladder ran exactly as pre-declared (amendment item 3), and it never decided:

| N blocks | runs | modal top factor | rank-1 retention | gate (950/1000) |
|---|---|---|---|---|
| 40 | 280 | --no-mmap | 569 | not decided |
| 56 | 392 | -t | 663 | not decided |
| **64 (cap)** | **448** | **-t** | **765** | **not decided -> UNDECIDABLE** |

Final total-order indices, 30B: `-t` **0.830** [0.610, 1.080] and `--no-mmap` **0.710**
[0.507, 0.950] - overlapping across most of their range - then `-ngl` 0.335, `-ctk` 0.292,
`-ot` 0.184.

**What this is, precisely.** Not a null result and not a failure of the instrument. On a
CPU-expert MoE split, thread count and page-cache behaviour carry comparable decode
variance, and 448 measurements on this box cannot separate them. The modal winner even
CHANGED between the first extension and the second - which is exactly why the gate exists.
A pass/fail scorer would have named `--no-mmap` the top carrier at N=40, and 168 further
runs would have quietly made that published claim wrong.

**Consequences, all pre-declared:**
- **P-3 overall: FAIL**, unchanged - it was decided on the 7B (`-ngl`, 1000/1000, outside
  the staked mapping set) and short-circuited per amendment item 4.
- **The 30B branch ships the scope label anyway**, per amendment item 7 (stricter than
  staked): "derived from the law, not confirmed by variance attribution" is literally true
  when the attribution never resolved. The label went live on 2026-08-17 across `plan`,
  `report`, README and both card assets; nothing further is owed.
- **Stage 3 (Taguchi) is NOT triggered.** Kill rule 3 adjudicates a Morris-vs-Sobol
  DISAGREEMENT; here Morris and Sobol agree on `-t` and the block is decidability, not
  conflict. Separating `-t` from `--no-mmap` needs a different design - a two-factor
  targeted experiment at depth, not more Saltelli blocks - and that is a NEW prereg if it
  is ever worth the nights. It is recorded as an open question, not as work in flight.

**The stage-1 observation that survives intact:** `--no-mmap` was flagged there as the
strongest interaction candidate on the board (mu* 7.0, the highest sigma of any factor).
Stage 2 neither confirmed nor refuted it; it established that the question is harder than
the design that asked it.

Chain of custody: mapping staked 2026-08-07 -> amended pre-data 2026-08-16 with the
extension ladder and the UNDECIDABLE branch both declared in advance -> scorer committed
before the first stage-2 row -> 448 runs measured -> scored by the frozen scorer, which
refused three times and was right to.
