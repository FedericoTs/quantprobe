# DESIGN: Prereg #95 stage 2 — Sobol variance attribution on the Morris survivors

**Serves:** `preregistrations/2026-08-07-doe-flag-screening.md` (STAKED 2026-08-07; pre-data
amendment 2026-08-16; stage 1 SCORED 2026-08-16: P-1 PASS, P-2 PASS, P-4 FAIL, P-3 deferred
to this stage). This document designs the measurement; it never edits the stake. Every
deviation is listed in section 6 as a ready-to-append amendment, which the OPERATOR appends
to the prereg before the first stage-2 run. The harness never touches the prereg.

**Design date:** 2026-08-16, before any stage-2 data exists. The scorer spec (section 5) is
precommitted now; the scoring code must exist and be frozen before the first stage-2 CSV row.

**Inputs this design is computed from (all committed):**
`weights/data/prereg95_verdict.json` (stage-1 mu_star/sigma, csv sha256 c35c7e9d718b3d80..),
`weights/data/doe_morris_stage1.csv` (walls, settles, best-observed rows),
`weights/data/doe_morris_20260816_161754.log` (the real night: 150/150 runs, 16:17:54 to
20:04:17 = 3h46m), `docs/DESIGN_DOE_MORRIS.md` (machine facts, timeouts, settle policy,
llama-bench templates — reused verbatim, not restated), `weights/doe_morris.py` (the harness
stage 2 extends), and `quantprobe plan` run fresh on 2026-08-16 against both exact staked
GGUFs (quoted in full in section 1).

Box, build, models, timeouts, settle: **unchanged from stage 1** (design doc box-state
section). Same exe `tools/llamacpp-b10098/llama-bench.exe`, same two GGUFs byte-asserted at
startup, same 240 s / 360 s caps, same 52 C / 30 s / 180 s settle, same tg128 r=3 response.

---

## 1. THE POINT: P-3 is the last unscored stake, and its kill rule is ARMED

P-3 as staked: *"For each model, the factor with the highest Sobol total-order index maps to
the resource `plan` names as binding. Stated as a mapping table in advance: capacity-bound ->
`-ngl`/`-ot`; RAM-bandwidth-bound -> `-ot`/KV type; VRAM-bandwidth bound -> KV type/`-ub`;
CPU-compute-bound -> `-t`."* Kill rule: *"If P-3 fails, the binding-constraint line — shipped
in `plan`, quoted in the README, drawn on the pipeline chart — is not validated by
measurement, and it gets a scope label ('derived from the law, not confirmed by variance
attribution') the same day, at full prominence, until re-derived."*

### 1.1 Ground truth: what `plan` actually classifies, on this box, for the exact staked configs

Run 2026-08-16, `python -m quantprobe plan --gguf <exact stage-1 file>`, no hardware flags
(auto-detect: `vram 6GB@192 | ram 16GB@48 | disk 0.5 GB/s`, calibration 2026-07-31 state
2dc97d41, anchors on). Binding-constraint lines verbatim:

**7B — `D:/evo-compress-data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf` (4,683,074,240 bytes):**

    binding constraint: BANDWIDTH-BOUND (VRAM bandwidth) - 100% of every decode token is spent there.

**30B — `D:/evo-compress-data/gguf/Qwen3-30B-A3B-Q2_K.gguf` (11,258,610,240 bytes):**

    binding constraint: BANDWIDTH-BOUND (system RAM bandwidth) - 51% of every decode token is spent there.

Committed corroboration: `weights/data/ladder_20260731_scored.json` carries the same
classes for the same rows — "Qwen2.5-7B Q4_K_M": *"BANDWIDTH-BOUND (VRAM bandwidth) - 100%
of every decode token"*; "Qwen3-30B-A3B Q2_K": *"BANDWIDTH-BOUND (system RAM bandwidth) -
54% of every decode token"* (54% then, 51% now — same class, drift inside the recalibrated
constants). The class, not the percentage, is what the mapping table keys on.

Applying the prereg's own mapping table (KV type = `ctk`, per pre-data amendment item 4;
`-ot` = the `moe_cpu_frac` factor):

| model | plan class | staked mapping set: Sobol ST argmax must be in |
|---|---|---|
| 7B  | VRAM-bandwidth-bound | { `ctk`, `ub` } |
| 30B | RAM-bandwidth-bound  | { `moe_cpu_frac`, `ctk` } |

One recorded caveat, pinned NOW so it cannot become an escape hatch later: the 30B margin
line reads *"1.03x - system RAM bandwidth must get that much faster before VRAM bandwidth
(49%) takes over"* — the classification is near-tied (51/49). The scorer keys on the class
`plan` PRINTS. Widening the 30B mapping set to include the runner-up class's factors after
seeing Sobol data would be exactly the pick-the-flattering-reading move the prereg forbids.
The margin is recorded in the verdict JSON as context, nothing more.

### 1.2 The tension stage 1 already exposed

Morris argmax (from `prereg95_verdict.json`): **7B = `ngl`** (mu* 13.52), **30B = `t`**
(mu* 10.73). Neither is in its model's mapping set. Sharper still: the 30B `plan` output
prints, in its own other-levers line, *"more/faster CPU threads: NO effect (0% of the
token)"* — while Morris measured threads as the single biggest knob on that model. The 7B
prints the same NO-effect-for-threads line against a t that Morris ranks #2 (mu* 4.31).

The mechanism is no mystery and the design does not hide it: `plan` classifies the STARRED
placement (7B at `-ngl 99` all-in-VRAM; 30B at the shipped expert split), while
Morris/Sobol attribute variance over the WHOLE tuning hypercube, which includes `ngl 0`
pure-CPU corners where threads dominate trivially. P-3 as staked compares a global variance
attribution against a placement-local classification. That was the stake, it was staked
with eyes open, and it is scored as written. If the global-vs-local gap is what kills it,
the label ships and the re-derivation path (kill rule: "until re-derived") is a
placement-conditional Sobol — a future prereg, not a rescue of this one.

### 1.3 What fires, mechanically

Per model, with `s` = Sobol ST argmax among the stage-2 survivors (section 2) after the
decidability gate (section 5.4), `m` = Morris argmax (frozen constants: 7B `ngl`, 30B `t`),
and `MAP` = the mapping set from the table above:

| case | model verdict | publishable now? |
|---|---|---|
| `s == m` and `s in MAP` | PASS | yes (methods agree) |
| `s == m` and `s not in MAP` | FAIL | yes (methods agree) |
| `s != m` and `s not in MAP` | FAIL | yes — the mapping fails under EITHER method's argmax; only the top-factor IDENTITY is held for Taguchi |
| `s != m` and `s in MAP` | HELD_FOR_TAGUCHI | no — Sobol would pass P-3, Morris disagrees; prereg kill rule 3: neither is published until a Taguchi confirmation run adjudicates |
| gate never decides at N_cap | UNDECIDABLE | see section 6 item 7 (label ships; stricter than staked, declared) |

Overall P-3: **FAIL** if either model is a publishable FAIL; else **HELD_FOR_TAGUCHI** if
either model is held (or undecidable-pending); **PASS** only if both models publishably
pass.

**A theorem of the frozen constants, stated so scoring is mechanical:** both Morris
argmaxes (`ngl`, `t`) are OUTSIDE their mapping sets, so the `s == m` PASS row is
unreachable on both models. Therefore stage 2 alone can produce only FAIL or
HELD_FOR_TAGUCHI (or UNDECIDABLE). **P-3 can no longer PASS out of stage 2 by itself** — a
PASS requires Sobol to overturn Morris's argmax on BOTH models (7B to `ctk` or `ub`, 30B to
`moe_cpu_frac` or `ctk`) and then a Taguchi arm to side with Sobol. Conversely the
likeliest branch — Sobol confirming `ngl` top on the 7B or `t` top on the 30B — is an
immediate, publishable, same-day FAIL: methods agree, the mapping is refuted, kill rule
fires. One model suffices; the 7B can fire it the morning after night 1.

**The label, exact text and placements (same day as a scored FAIL):**

    derived from the law, not confirmed by variance attribution (prereg #95 P-3)

applied at full prominence to all three shipped surfaces the kill rule names:
1. the `binding constraint:` print in `quantprobe plan` / `report` (quantprobe/plan.py,
   quantprobe/report.py);
2. the README example block that quotes it (README.md, the *"binding constraint:
   BANDWIDTH-BOUND (system RAM bandwidth) - 51% ..."* line);
3. the pipeline chart asset that draws it (`weights/data/card_flagship.svg` — one
   `binding constraint:` line confirmed in the asset).

The scorer prints the verdict; the OPERATOR ships the label. The scorer edits nothing.

---

## 2. Survivors, and what gets fixed

Rule, pinned before data: survivors = the smallest mu_star-descending prefix reaching
**>= 90% of total stage-1 mu_star**, UNION the model's mapping-set factors (a P-3 whose
mapping factors are frozen out of the design could never PASS — argmax cannot land on a
factor with zero designed variance — and a design that pre-decides its own stake is not a
measurement). All numbers from `prereg95_verdict.json`:

**7B** (total mu* 19.1253): `ngl` 13.5165 (70.7%), + `t` 4.3068 (cum 93.2%) — the 90% rule
stops at TWO factors. Union with mapping set {`ctk`, `ub`} gives:

| survivors (k=4) | mu* | levels (identical to stage 1) |
|---|---|---|
| ngl | 13.5165 | 0 / 9 / 19 / 99 |
| t | 4.3068 | 1 / 2 / 3 / 4 |
| ctk | 0.4111 | f16 / q8_0 |
| ub | 0.0691 | 128 / 512 / 1024 / 2048 |

Coverage 18.3036/19.1253 = **95.7%**. Fixed: **`fa` = 0, `mmp` = 0**.

**30B** (total mu* 29.5582): `t` 10.7270 (36.3%), `mmp` 7.0417 (60.1%), `ngl` 4.9440
(76.8%), `moe_cpu_frac` 3.3809 (88.3% — still under the bar), `ctk` 2.3353 (96.2%) — the
90% rule needs FIVE factors. Mapping set {`moe_cpu_frac`, `ctk`} is already inside:

| survivors (k=5) | mu* | levels (identical to stage 1) |
|---|---|---|
| t | 10.7270 | 1 / 2 / 3 / 4 |
| mmp | 7.0417 | 1 / 0 |
| ngl | 4.9440 | 0 / 16 / 32 / 99 |
| moe_cpu_frac | 3.3809 | 0.75 / 0.833 / 0.917 / 1.0 |
| ctk | 2.3353 | f16 / q8_0 |

Coverage **96.2%**. Fixed: **`ub` = 2048, `fa` = 1**.

Survivor level lists are the stage-1 lists UNCHANGED — narrowing a range after seeing which
regime carries the variance would quietly change what "top factor" means mid-stake.

**Fixed levels come from the stage-1 best-observed row** (max ok tok_s in
`doe_morris_stage1.csv`):

- 7B best row 22.486 tok/s: `ngl=99 ub=1024 t=4 ctk=q8_0 mmp=0 fa=0` -> fixed `fa=0`,
  `mmp=0`. `fa=0` is unanimous across the top-4 rows; `mmp` splits 0/1 across them but its
  mu* is 0.053 (0.28% share) — the level barely matters, the rule picks 0.
- 30B best row 20.497 tok/s: `ngl=99 ub=2048 t=3 ctk=f16 mmp=0 fa=1 moe=0.75` -> fixed
  `ub=2048`, `fa=1`. That exact config completed ok in stage 1, and stage 1's max-vram
  pre-flight corner (`ngl 99, ub 2048, moe 0.75`) passed, so the fixed levels create no
  new feasibility corner.

**Why fixing these is safe for the stake, quantitatively:** a fixed factor cannot become
argmax, so fixing can only remove FAIL modes (every fixed factor is outside both mapping
sets — `fa` and `mmp` appear in NO row of the staked mapping table; `ub` is mapping-relevant
only for the 7B, where it stays varied). It would bias the verdict only if a fixed factor
would actually have won total-order. Bounding each fixed factor by max(mu*, sigma) against
the top survivor's mu*: 7B `fa` 1.449 vs `ngl` 13.52 (10.7%), `mmp` 0.064 (0.5%); 30B `ub`
0.706 vs `t` 10.73 (6.6%), `fa` 0.911 (8.5%). Even attributing every fixed factor's entire
sigma to interactions leaves it an order of magnitude below the leader. The one honest
cost: fixing 7B `fa` freezes the fa x ngl interaction (fa sigma 1.449) out of the measured
variance; it is recorded here as excluded, not estimated.

---

## 3. Run budget, honestly

**Method: Saltelli sampling, N*(k+2) runs** per model — base matrices A and B (N rows
each) plus one A_B^(i) matrix per factor (A with column i taken from B), giving first-order
AND total-order from the same runs. Sampling is seeded stdlib Monte Carlo, not QMC — the
harness stays zero-dependency like stage 1, and at these N the bootstrap CI (section 5.3),
not the sequence discipline, is what bounds the error [est].

**Per-run cost, measured from the stage-1 night** (150/150 ok, log
`doe_morris_20260816_161754.log`, walls from `doe_morris_stage1.csv`):

| block | wall mean | wall p90 | settle mean | per-run mean |
|---|---|---|---|---|
| 7B, all rows | 51.3 s | 88.9 s | 30.7 s | 82.0 s |
| 7B, mmp=0 rows only (stage 2 fixes mmp=0) | 53.7 s | 88.9 s | 30.7 s | **84.4 s** |
| 30B, all rows (mmp still varies in stage 2) | 64.9 s | 99.8 s | 30.4 s | **95.3 s** |

Cross-check: the whole stage-1 night was 3 h 46 m for 150 designed runs + 8 pre-flight
probes = 90.6 s per designed run — consistent with the weighted row means (89.1 s).

**The (k, N, runs, hours) table** (hours = runs x per-run mean; pre-flight adds ~10 min
per model per launch, same corners as stage 1):

| model | k | runs/block (k+2) | N_start | runs | nominal hours | pessimistic (p90 walls) |
|---|---|---|---|---|---|---|
| 7B | 4 | 6 | **32** | **192** | **4.5 h** | 6.4 h |
| 30B | 5 | 7 | **40** | **280** | **7.4 h** | 10.1 h |
| total | | | | **472** | **11.9 h** | 16.5 h |

**11.9 h nominal does not fit one 9.5 h deadline night: this is a TWO-NIGHT design,
declared now.** Block order 7B then 30B, same as stage 1; night 1 completes the 7B
(~4.5 h) and runs down the 30B until the deadline guard stops it cleanly; night 2 resumes
the 30B remainder (~2.4 h nominal). The stage-1 resume discipline (run_ids skipped, DNFs
count as done, scorer refuses per-model partials) makes the split harmless, and per-model
scoring (section 5) means the 7B verdict — and, if it is a publishable FAIL, the overall
P-3 verdict and the label — can land the morning after night 1. The operator may instead
run `--model 30B` first; the design does not care, the deadline guard does the same job.

**Why N is allocated unevenly:** the 7B argmax race is a landslide (ngl 70.7% of mu* vs t
22.5%; ngl also has the largest sigma) — N=32 decides that argmax with room to spare
[est]. The 30B race is the close one (t 36.3% vs mmp 23.8%), so it gets the larger N=40.

**What small N buys, and the refusal it forces:** we do not pretend to know CI widths
before the data — with stage-1-like concentration, bootstrap 95% CIs on ST at N=32-40 are
expected around +/-0.10-0.20 absolute [est], which decides a 3x argmax gap and may NOT
decide a 1.5x one. The design's answer is a pre-committed decidability gate instead of a
promised width: the ST argmax counts as DECIDED iff it holds rank 1 in >= 950/1000
bootstrap resamples (section 5.4). If UNDECIDED, the harness extends the SAME seeded
design (`--n-blocks +16`; blocks are drawn sequentially from one stream, so old rows stay
valid and resume adds only the new blocks) up to **N_cap = 64** per model (7B +2.3 h per
step, 30B +3.0 h per step). Still undecided at N_cap -> UNDECIDABLE, section 6 item 7. A
design that could neither decide nor extend would be refused here; this one always
terminates in a verdict or a declared, labeled undecidability — never a wasted night.

Worst-case arithmetic, for the record: every run DNF at cap would cost 192*(240+31)s =
14.5 h (7B) + 280*(360+30)s = 30.3 h (30B). Cannot happen short of a broken build (stage 1:
0 DNF in 150), and the deadline guard + resume make it harmless anyway.

---

## 4. Harness delta — `--stage2` in `weights/doe_morris.py`

An extension, not a new file: stage 2 reuses `bench_cmd`, `ot_pattern`, `classify`,
`measure_one`'s sequence, `settle`, `gguf_block_count`, the lock, the orphan kill, the
deadline guard and the startup assertions AS THEY ARE. No second physics. The delta is
design generation + CLI + CSV constants (~150 lines [est]).

1. **CLI:** `--stage2` switches to the Sobol design. `--n-blocks N` (per the active model;
   default = N_start table below), `--model {7B,30B}` (optional gate for single-model
   nights), `--dry-run` and `--deadline-hours` behave exactly as stage 1.

       N_START = {"7B": 32, "30B": 40}     # section 3
       N_CAP   = 64                        # extension ceiling, both models

   `--n-blocks` outside [N_START[tag], N_CAP] is refused at startup.

2. **Design generation** (`build_stage2_design(tag, n_blocks)`), seeded and regenerable
   from this paragraph alone: seed string **`"prereg95:stage2:{tag}:20260816"`** (the date
   this design was frozen — stage 1's 20260807 date is NOT reused, because this design did
   not exist on 08-07 and a seed should not claim it did). Survivor order = the FACTORS
   tuple order restricted to survivors (7B: ngl, ub, t, ctk; 30B: ngl, t, ctk, mmp,
   moe_cpu_frac — the stage-1 pinned order, NOT mu_star order, so the stream layout cannot
   drift if a mu_star tie is ever re-ranked). For block b = 0..N-1, strictly in this order
   from that one stream: k uniforms U[0,1) for A_b, then k uniforms for B_b. Level map:
   4-level factors take index `min(3, floor(u*4))`; 2-level factors take
   `min(1, floor(u*2))`. Fixed factors (section 2) are constants in every config. Runs per
   block, in this order: A_b, B_b, then AB_i_b for each survivor i in survivor order,
   where AB_i_b = A_b with factor i's level taken from B_b. Blocks are generated in
   sequence, so any prefix of blocks is itself a valid (smaller-N) design — that is what
   makes `--n-blocks` extension resume-safe.

3. **run_id** = `sha256(f"{tag}|s2|{block}|{matrix}|{canonical_config_json}")[:16]` with
   `matrix` in `{"A", "B", "AB_ngl", "AB_ub", ...}`. The `s2` infix keeps stage-2 ids
   disjoint from stage 1 by construction.

4. **CSV:** `weights/data/doe_sobol_stage2.csv`, append-only, per-row fsync, own pinned
   header (stage-1 columns with `traj,pos,changed_factor` replaced by `block,matrix`; 30
   columns):

       run_id,model,block,matrix,ngl,ub,t,ctk,mmp,fa,moe_cpu_frac,status,tok_s,stddev_ts,reps_tok_s,wall_s,settle_s,free_ram_gb_pre,temp_pre,sm_mhz_pre,mem_mhz_pre,vram_mib_pre,power_w_pre,temp_post,sm_mhz_post,mem_mhz_post,vram_mib_post,power_w_post,ts_utc,cmd

   Pinned hash (computed 2026-08-16 from that exact line, no trailing newline):
   `STAGE2_HEADER_SHA256 = 95fe4b76345921ea95d6a86cc83b30caad741f799515716d0d07f687c2d89ade`.
   The stage-1 `_check_header_pin` pattern applies verbatim: the harness refuses to start
   (and to resume) on drift. `doe_morris_stage1.csv` is never opened in stage-2 mode.

5. **Unchanged discipline, byte for byte:** `.doe_lock` via `runner.owns_the_box` (already
   in `runner.LOCK_NAMES`), orphan kill of both llama images, unique log
   `doe_sobol_<stamp>.log` via `runner.make_log`, startup byte/block_count assertions,
   thermal settle before each run with settle_s recorded in the row it protects, timeouts
   240 s / 360 s, DNF-as-row (never a retry, never a hole), per-row fsync, resume by
   run_id, deadline default 9.5 h. Pre-flight: the same 4 corner specs as stage 1 with the
   stage-2 FIXED levels substituted in, per model per launch, log-only.

6. The harness writes measurements and logs ONLY. Scorer coupling mirrors stage 1: the
   scorer imports `build_stage2_design` and refuses to score if its own frozen copy does
   not regenerate identical run_ids — drift is caught actively, and the frozen copy wins.

---

## 5. Scorer spec — `weights/prereg95_sobol_score.py` (PRECOMMITTED)

Frozen before any stage-2 row exists. Reads `weights/data/doe_sobol_stage2.csv`, writes
`weights/data/prereg95_sobol_verdict.json`, prints the table. It scores ONLY P-3 (plus the
Morris-vs-Sobol adjudication state); it refuses everything else, including any request to
re-score stage-1 stakes.

### 5.1 Embedded ground-truth constants (from section 1, quoted, never recomputed at score time)

    PLAN_BINDING = {
      "7B":  "binding constraint: BANDWIDTH-BOUND (VRAM bandwidth) - 100% of every decode token is spent there.",
      "30B": "binding constraint: BANDWIDTH-BOUND (system RAM bandwidth) - 51% of every decode token is spent there.",
    }                                   # quantprobe plan, 2026-08-16, auto-detected hw, calib 2026-07-31 state 2dc97d41
    PLAN_MARGIN_30B = "1.03x - system RAM bandwidth must get that much faster before VRAM bandwidth (49%) takes over"
    MAPPING_SET   = {"7B": ("ctk", "ub"), "30B": ("moe_cpu_frac", "ctk")}   # prereg mapping table applied to PLAN_BINDING
    MORRIS_ARGMAX = {"7B": "ngl", "30B": "t"}                               # prereg95_verdict.json, frozen
    N_START, N_CAP = {"7B": 32, "30B": 40}, 64

### 5.2 Refusals (house style, checked in this order)

1. CSV absent:

       REFUSED: weights/data/doe_sobol_stage2.csv not found. Stage 2 has not produced data; the scorer never invents a verdict.

2. Header drift (sha256 of the header line != `95fe4b76345921ea..`):

       REFUSED: CSV header hash <found> != design hash 95fe4b76345921ea95d6a86cc83b30caad741f799515716d0d07f687c2d89ade. This file was not written by the staked harness; scoring it would score a different experiment.

3. Design regeneration mismatch (scorer's frozen generator vs `doe_morris.build_stage2_design` import):

       REFUSED: the harness and the frozen scorer no longer regenerate the same design. The frozen copy wins; un-edit the harness.

4. Per-model completeness: N_complete(tag) = the largest N in [0, N_CAP] such that ALL
   run_ids of blocks 0..N-1 are present (a declared DNF row counts as present, a missing
   row does not). If N_complete < N_START for a model:

       REFUSED (<tag>): incomplete design: largest complete prefix is <n> blocks, N_start is <N_START>. Resume weights/doe_morris.py --stage2; a partial night is not the staked design.

   A model at or past N_START is scored at its N_complete even while the other model is
   refused — per-model verdicts are the point of the two-night split (section 3).

5. Validity floor: for each survivor, ST needs the pair (A_j, AB_i_j) ok and S needs the
   triple (A_j, B_j, AB_i_j) ok, per block. A survivor with fewer than
   `ceil(0.75 * N_complete)` valid ST pairs is UNSCOREABLE; the model's P-3 is then VOID,
   never guessed (stage-1 floor discipline, restated for blocks).

### 5.3 Estimators (exact, named)

With f = tok_s of ok rows, N = N_complete, V(Y) = sample variance (ddof=1) over all ok A
and B values pooled:

- **First-order, Saltelli et al. 2010 (Table 2, estimator (b)):**
  `S_i = (1/N_i) * sum_j [ f(B_j) * ( f(AB_i_j) - f(A_j) ) ] / V(Y)` over valid triples.
- **Total-order, Jansen 1999:**
  `ST_i = (1/(2*M_i)) * sum_j [ ( f(A_j) - f(AB_i_j) )^2 ] / V(Y)` over valid pairs.

Both reported per survivor per model, with N_i/M_i counts. P-3 keys on **ST argmax**, as
staked ("highest Sobol TOTAL-ORDER index").

### 5.4 Bootstrap CIs and the decidability gate

1000 resamples of block indices 0..N-1 with replacement, seed
`"prereg95:stage2:bootstrap:20260816"`, recompute every S_i and ST_i per resample,
report percentile 2.5/97.5 CIs. **Gate:** the ST argmax is DECIDED iff the modal top
factor holds rank 1 in >= 950/1000 resamples. Otherwise the scorer prints:

    UNDECIDED (<tag>): ST argmax rank-1 retention <r>/1000 < 950. Extend the same seeded design: python weights/doe_morris.py --stage2 --model <tag> --n-blocks <N+16> (cap 64). No verdict is issued from an undecided argmax.

At N_complete = N_CAP still undecided -> verdict `UNDECIDABLE` for that model (consequence
pinned in section 6 item 7).

### 5.5 The P-3 verdict and the Taguchi hold

Per model, the section-1.3 decision table verbatim, then the overall aggregation (FAIL if
any publishable model-FAIL; else HELD_FOR_TAGUCHI if any model held or undecidable-pending;
PASS only if both publishably pass). On any `s != m` the scorer MUST print, and the verdict
JSON must carry:

    HELD: Morris says <m>, Sobol says <s> on <tag>. Per prereg #95 kill rule 3, neither ranking is published as a finding until a Taguchi confirmation run adjudicates. This tool does not pick.

The scorer never chooses between the methods, never re-reads `plan` at score time (the
ground truth is the frozen constants above, so a later quantprobe release cannot move the
goalposts), and never edits the prereg, the README, or the chart. On a publishable FAIL it
prints the kill-rule reminder and the label text from section 1.3 for the OPERATOR to ship
the same day. Output JSON: per-model S/ST tables with CIs and valid-block counts, bootstrap
retention, per-model and overall verdicts, the embedded constants echoed, csv sha256,
scored_utc, rc 0 only when every non-refused model reached a verdict.

---

## 6. Amendment draft (operator appends to the prereg BEFORE the first stage-2 run)

> ### Pre-data amendment — stage 2, 2026-08-16, before any stage-2 run
> Design doc: `docs/DESIGN_DOE_SOBOL.md` (committed with this amendment, together with the
> `--stage2` harness mode and the pre-committed scorer `weights/prereg95_sobol_score.py` —
> scorer in the repo before the first stage-2 CSV row, per house rule). Deviations and
> constants, declared before data:
>
> 1. **Survivor rule:** factors covering >= 90% of stage-1 total mu_star, UNION the
>    model's P-3 mapping-set factors (a design that excludes the mapping factors could
>    never PASS the stake it scores). Result: 7B k=4 {ngl, t, ctk, ub} (95.7% of mu*);
>    30B k=5 {t, mmp, ngl, moe_cpu_frac, ctk} (96.2%). Survivor level lists identical to
>    stage 1.
> 2. **Non-survivors fixed at stage-1 best-observed levels:** 7B `fa=0, mmp=0` (best ok
>    row 22.486 tok/s); 30B `ub=2048, fa=1` (best ok row 20.497 tok/s). Every fixed factor
>    is outside both mapping sets, and max(mu*, sigma) of each is <= 10.7% of its model's
>    top-survivor mu*, so fixing removes only FAIL modes it could not plausibly have won.
>    The 7B fa x ngl interaction is thereby excluded from measured variance, recorded not
>    estimated.
> 3. **Design:** Saltelli N*(k+2), seeded stdlib Monte Carlo (no QMC dependency), seeds
>    `"prereg95:stage2:{7B|30B}:20260816"`, N_start 32 (7B) / 40 (30B), N_cap 64, runs
>    192 + 280 = 472, extension in +16-block steps of the same stream on an UNDECIDED
>    gate. Estimators: Saltelli-2010 Table-2(b) first-order, Jansen-1999 total-order;
>    bootstrap 1000 block resamples, seed `"prereg95:stage2:bootstrap:20260816"`; argmax
>    DECIDED iff rank-1 retention >= 950/1000.
> 4. **Two nights by resume, declared:** 11.9 h nominal at stage-1's measured per-run
>    costs (84.4 s 7B at fixed mmp=0, 95.3 s 30B). Per-model scoring as each design
>    completes; a publishable model-FAIL short-circuits overall P-3 (the 7B can fire the
>    kill rule after night 1).
> 5. **P-3 ground truth frozen:** plan classes measured 2026-08-16 on this box
>    (7B VRAM-bandwidth-bound at 100%; 30B system-RAM-bandwidth-bound at 51%, margin
>    1.03x recorded as context), embedded in the scorer as constants; mapping sets
>    {ctk, ub} and {moe_cpu_frac, ctk}. The near-tie does not widen the 30B set.
> 6. **Morris-vs-Sobol disagreement** is handled per kill rule 3: any model where the two
>    argmaxes differ is HELD_FOR_TAGUCHI and nothing about its ranking publishes until a
>    Taguchi arm adjudicates — except that a mapping refuted under BOTH candidate
>    argmaxes is a publishable P-3 FAIL with only the factor identity held.
> 7. **Stricter than staked, accepted:** if the argmax is still UNDECIDABLE at N_cap=64,
>    the binding-constraint scope label ships anyway — the label text ("derived from the
>    law, not confirmed by variance attribution") is literally true in that branch, and
>    perpetual undecidability must not become a shield. This can only add a label the
>    staked rules would not force; it can never suppress one.
> 8. `-np` remains not exercisable (llama-bench rejects it; server-only), unchanged from
>    stage-1 amendment item 2.
>
> The harness cannot write this prereg; this amendment is appended by the operator before
> the first stage-2 run.

---

*Design sources: `prereg95_verdict.json` (survivor arithmetic, Morris argmaxes, sigma
bounds); `doe_morris_stage1.csv` (walls, settles, best rows, mmp-conditional means);
`doe_morris_20260816_161754.log` (night span 16:17:54-20:04:17, 150/150 ok, 0 DNF);
`quantprobe plan --gguf` fresh runs 2026-08-16 (binding lines, margin, other-levers
quotes); `ladder_20260731_scored.json` (committed corroboration of both classes);
`docs/DESIGN_DOE_MORRIS.md` + `weights/doe_morris.py` (everything reused). Estimator
names: Saltelli et al. 2010 "Variance based sensitivity analysis of model output", Jansen
1999. Numbers without a committed source are labeled [est].*
