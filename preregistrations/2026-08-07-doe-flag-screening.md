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
