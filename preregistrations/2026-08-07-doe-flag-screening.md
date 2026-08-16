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
