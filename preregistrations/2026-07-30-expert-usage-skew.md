# Pre-registration #87: is expert routing skewed enough to justify hot-expert caching?

**Author:** Federico Sciuca · **Date staked:** 2026-07-30, BEFORE any of the four measured arms
were run. **STAKED.** · **Task:** #52 · **Gates:** U-34, and E-02 (PowerInfer's frequency-ranked
residency claim)

We currently pin roughly 32% of the expert tensors to VRAM with an `-ot` regex over fixed layer
indices — a choice made on VRAM arithmetic, not on measured usage. BigMoeOnEdge (U-34) reports
that caching the *most-used* experts hits 76-84%, and that cache size is the dominant lever.
Before we build anything of the sort, the premise has to be measured on **our** models: does
routing actually concentrate, and does it concentrate on a *stable* set?

---

## AMENDMENT 1 — adversarial review, 2026-07-30, still BEFORE any measurement

This prereg was adversarially reviewed for the one defect that matters most in a
pre-registration: *ways the experiment cannot fail*. `weights/data/exp52_imatrix/` did not exist
at the time of review — **no arm had been run and no number existed** — so the corrections below
are amendments to an unmeasured stake, not a rewrite after seeing a result. They are listed here
in full because the alternative (quietly tightening a prereg) is the thing this protocol exists
to prevent. Every change tightens the experiment; none loosens a gate or moves a threshold.

Six ways the kill rules could have been evaded, all now closed:

1. **KR-2 could not fire in the 0.53–0.57 band.** The MARGINAL check ran before the G2 verdict
   and returned unconditionally, so a retention of 0.20 next to a headline of 0.556 would have
   been reported "too close to call" instead of a decisive stability refutation.
2. **KR-2 could pass vacuously.** With a single domain the cross-domain dict is empty and
   retention defaulted to `1.0` — an unevaluable gate scored as a passed gate.
3. **KR-3's minimum-data check was skipped whenever `imatrix.chunk_count` was absent** (`if
   chunk_count is not None and …`). A build that does not write the field would have scored a
   1-chunk run silently.
4. **A broken parse would have been published as a refutation.** A zero-mass layer scored share
   `0.0`, propagating to a headline of `0.0` and a confident FAIL. This is the same failure shape
   as the profiler-on-an-oversized-model incident: an artifact one step from being published as a
   measurement. It is now VOID.
5. **The instrument could be swapped silently** — a missing staked binary fell through to
   whatever `llama-imatrix` was on `PATH`; and a cached imatrix was reused regardless of the ctx,
   chunk budget, ngl, threads, corpus, or model that produced it.
6. **The controls did not exist.** The "negative control" numbers in the Method section were
   asserted in prose with no code behind them. They are now `--selftest`, they run inside every
   scored session, and a positive control was added so that a FAIL carries information.

The 0.55 and 0.90 thresholds, the ±0.02 band, the 0.32 budget, the corpus, and all six stakes
are **unchanged**.

---

## DISCLOSURE — what the author had already seen when staking this

Protocol requires predictions before measurement. Two things were learned while building the
instrument, and both bear on the outcome. Staking without saying so would make this prereg
theatre.

**(1) A pilot read of a pre-existing imatrix file.** To validate the parser, the author read
`D:\evo-compress-data\gguf\qwen35-35b.imatrix.gguf` — an imatrix produced earlier for
quantization work, on an **unknown calibration corpus**, 40 layers x 256 experts. Measured
there, the top 32% of experts by frequency carry:

| | top-32% share |
|---|---|
| minimum layer | 0.5198 |
| median layer | 0.7374 |
| maximum layer | 0.8670 |

with concentration rising monotonically with depth. That file is a `qwen35moe` sibling of the
**q35-A-shexp** arm below. So for that arm, "is there raw concentration at all" is largely
pre-answered, and P-2 is a **confirmation, not a discovery**. It says nothing about the
Qwen3-30B-A3B arm (different architecture, 128 experts), and — because that corpus is unknown
and single-domain-unlabelled — **nothing at all** about the cross-domain question (P-3), which
is the stake that actually decides the lever.

**(2) A structural fact, discovered while parsing, that reframes the question.** Total routing
mass is **exactly identical in every MoE layer** (409600 in all 40 layers of the pilot file).
This is not an empirical accident: every token traverses every MoE layer and selects exactly
`expert_used_count` experts, so per-layer mass is `n_tokens x top_k` by construction. Therefore:

> **Choosing *which layers* to keep resident by measured usage is not a lever and never can be.**
> Every MoE layer carries the same routing mass. The shipped index-based regex is not "arbitrary"
> in any way that usage data could repair. The only exploitable skew is *within* a layer, across
> experts.

This is why the experiment below measures per-expert skew only, and why a positive result does
**not** authorise a change to the regex (see *Implementability*).

Everything else — all four measured arms, both models, both domains, every held-out and
cross-domain number — is unmeasured at staking time.

---

## Why this is not already answered by `route_locality.py`

`weights/route_locality.py` measured DeepSeek-V2-Lite and concluded *"hot-expert cache NOT
viable (paper-2 thesis REFUTED)"*. That verdict stands, but it tested a **different claim**:
temporal locality — consecutive-token stickiness (2.5x base rate) and working-set size (~51 of
64 experts within 32 tokens). Those numbers kill a *small online* cache that must be refilled
inside a short window.

A **static frequency-ranked pin** needs none of that. It needs only that aggregate usage over a
corpus is concentrated and stable. A model can have a large short-window working set and still
have a heavy aggregate tail — the pilot above shows exactly that shape. Different claim,
different measurement, and the earlier refutation does not transfer. It does, however, set the
prior: this project has already refuted one version of this idea, so the bar here is set high
deliberately.

---

## Implementability — the cost side, established before measuring

In llama.cpp all experts of a layer live in **one fused 3-D tensor**. Verified on the target
file: `blk.0.ffn_gate_exps.weight` has shape `[2048, 768, 128]` — all 128 experts in a single
tensor. `-ot` / `--override-tensor` matches *tensor names*, so it can move a whole layer's
experts and nothing finer.

**Per-expert residency is therefore not expressible in the tool we ship.** It requires a change
to expert dispatch inside ggml/llama.cpp — the same class of work D-05 priced at 1-6% for the
VRAM/host-resident regimes, and the reason BigMoeOnEdge had to write a custom engine. This is
what justifies the demanding gate below: a cheap win would deserve a cheap bar, and this is not
a cheap win.

---

## Method

**Instrument.** No patch and no custom build. `llama-imatrix` already counts, per MoE layer and
per expert, how many `(token, slot)` pairs the router assigned — `e.counts[ex]++` under
`GGML_OP_MUL_MAT_ID` in `tools/imatrix/imatrix.cpp` — and writes them to the output GGUF as
`blk.<N>.ffn_<gate|up|down>_exps.weight.counts`, one F32 per expert. We drive the stock
prebuilt `tools/llamacpp-b10098/llama-imatrix.exe` and read those tensors.

**This is not a timing measurement — but C-14 is not fully waived either.** Counts do not depend
on clocks, thermals, or placement, so the *timing* form of C-14 does not bind and no `cal_id` is
required. The earlier draft went further and said "C-14 does not bind", full stop. That was too
broad. Counts still depend on the binary, the model file, the corpus, the chunk budget, and —
through CPU float reduction order at router near-ties — on thread count and offload split. The
idempotent cache made it easy to produce shard A under one setting and shard B under another and
then compare them, which is the cross-state comparison C-14 exists to forbid. So each imatrix now
carries a **run stamp** (`<file>.stamp.json`: binary SHA-256, model path/size/mtime, corpus
SHA-256, ctx, chunks, ngl, threads). A cached file with no stamp, or a stamp that differs from
the current run, is **refused** rather than reused. Runs use `-ngl 0` (pure CPU) for determinism
and to sidestep the 6 GB VRAM limit on a 30B file.

**The binary is part of the instrument.** `llama-imatrix` is taken from the staked path only.
The script previously fell back to whatever `llama-imatrix` was on `PATH` if the staked path was
missing — a silent instrument swap that would have been invisible in the headline. It now
refuses, and records the binary's SHA-256 in the output JSON.

**Models (2 arms).**

| arm | file | arch | layers | experts | top-k |
|---|---|---|---|---|---|
| `qwen3-30b-a3b-q2k` | `Qwen3-30B-A3B-Q2_K.gguf` | qwen3moe | 48 | 128 | 8 |
| `q35-A-shexp` | `q35-A-shexp.gguf` | qwen35moe | 40 | 256 | 8 (+shared) |

Shared experts are always-on and are excluded by construction (the regex matches `_exps` only,
never `_shexp`).

**Prompt set (fixed and hash-pinned before running).** Routing is input-dependent, so a
code-only sample would not generalise. Two domains, two independent shards each:

| shard | bytes | SHA-256 | provenance |
|---|---|---|---|
| `code_A` | 26,915 | `7fe9821f1570dabdae866a4ad19eceead74475b831028813c0b2a03f28429bd3` | hand-authored .NET/C# (no C# existed on this box) |
| `code_B` | 25,550 | `dccbb1bd4af5b47b363d6cce7090bbebd73c9cf112249583f92a98618f5a1593` | idem |
| `prose_A` | 119,318 | `09e3648bef826e9ca14744249aaf3cbb5c0a5455efdde7a9138663fbfc9dea42` | WikiText-2 raw test, bytes 0-119,318 |
| `prose_B` | 119,800 | `5fb4087ab6b5c048366614e0525127f3adb70055eb1efeead2e9eb9ac785d304` | WikiText-2 raw test, next slice |

The C# corpus spans ASP.NET Core controllers and minimal APIs, EF Core mapping and migrations,
LINQ projection, async/await with `CancellationToken`, `Span`/`stackalloc`/`ArrayPool`,
`System.Text.Json` converters and source-gen contexts, Channels, Polly pipelines, SignalR hubs,
FluentValidation, and xUnit/Testcontainers tests. Source: `wiki.test.raw`, SHA-256
`173c87a5...dd08`, also checked at runtime. The script **refuses to run** on any hash mismatch,
so the prompt set cannot be swapped after seeing a result.

**Budget.** 8 chunks x 512 tokens = 4096 tokens per shard, **equal for both domains** (capped by
the code corpus, which tokenises to ~5.0-5.4k tokens per shard). Equal budgets matter: the
cross-domain comparison must not be confounded by one domain contributing more routing events.
At 4096 tokens x top-8 this is 32768 routing events per layer — a mean of 256 (128-expert arm)
or 128 (256-expert arm) selections per expert.

**Every headline number is held out.** Ranking experts on a sample and scoring the same sample
is a winner's-curse estimate: the top-k of a noisy count vector always looks hotter than it is.
So hot sets are chosen on shard **A** and scored on shard **B**, always. Four quantities per
arm, at a pinned budget of `k = round(0.32 x n_expert)` experts (41 of 128; 82 of 256 — the same
VRAM the shipped regex spends):

1. **in-sample** — rank and score on A. Reported only as the inflated bound it is.
2. **held-out, same domain** — rank on A, score on B.
3. **cross-domain** — rank on domain X's A, score on domain Y's B.
4. **static pin (the headline)** — rank *once* on the pooled A shards of both domains, score on
   each domain's held-out B shard. This is exactly what a static frequency-ranked pin would
   deliver, and the reported figure is the **worst domain**.

**Controls — reproducible, and run automatically inside every scored session.** An earlier draft
of this prereg asserted two synthetic controls with numbers in the text and *no code in the
script*. That was an unfalsifiable claim and it has been replaced. The controls are now
`weights/exp52_expert_usage_skew.py --selftest`; they drive the same `analyse_model` /
`verdict_for` path the real arms drive, and the measured run refuses to emit any verdict unless
all four reproduce their ground truth in the same session.

| control | construction | headline | retention | verdict |
|---|---|---|---|---|
| `uniform` | flat routing | 0.3187 | 1.0005 | **FAIL** (uplift 0.99x — the metric is not inflated by construction) |
| `disjoint_hot` | hot sets share no experts | 0.4408 | 0.1177 | **FAIL** |
| `partial_overlap` | 28 of 41 hot experts shared, 13 domain-specific | 0.7629 | 0.6984 | **FAIL** — G1 *passes*, G2 fires |
| `shared_hot` | strong and domain-stable | 0.7995 | 1.0000 | **PASS** (uplift 2.50x) |

`partial_overlap` is the load-bearing one: it is the only demonstration that KR-2 can overturn a
*strong* headline on its own, which is precisely the case where a stability gate earns its keep.
`shared_hot` is the positive control — without it, "the gate can fail" and "the gate always
fails" would be indistinguishable, and a FAIL on the real arms would carry no information.

**Reproduce:**

```
cd C:\Users\Federico\Documents\evo-compress\.claude\worktrees\law5-prefill
python weights\exp52_expert_usage_skew.py --selftest   # controls only; no model, no GPU
python weights\exp52_expert_usage_skew.py              # the staked run (controls run first)
```

Idempotent (completed imatrix runs are reused; `--force` redoes them). Raw output:
`weights/data/exp52_expert_usage_skew.{json,log}` and `weights/data/exp52_imatrix/*.gguf`.

---

## Where the 55% bar comes from

U-34's standing wording is "materially more than 32%". That is too weak to act on, and I am
**raising it before measuring** rather than after.

On a MoE split the streamed-expert term dominates decode: `t_cpu ∝ (1 - f) x B_expert / BW_cpu`,
where `f` is the share of routing mass served from VRAM. Because per-layer routing mass is
exactly uniform (Disclosure 2), the shipped index-chosen 32% serves **exactly** `f = 0.32`. So a
frequency-ranked pin at share `X` cuts that term by `0.68 / (1 - X)`:

| X | cut in the streamed-expert term |
|---|---|
| 0.40 | 1.13x |
| **0.55** | **1.51x** |
| 0.68 | 2.13x |
| 0.74 (pilot median) | 2.62x |

I require **1.5x on paper**, i.e. `X >= 0.55`, for two reasons. First, this buys a runtime
change, not a flag change (see *Implementability*). Second, our own track record says byte-term
arithmetic over-promises on tiers we have not calibrated: U-06 predicted 2.0 tok/s on the
streaming tier and measured 0.66, a 3x over-promise. A 1.5x paper win is already thin once that
haircut is applied; anything below it would not survive contact with a measurement.

Note the bar is genuinely discriminating rather than a gimme: the pilot file's *worst* layer
(0.5198) sits **below** it.

---

## Stakes

Point predictions with bands. Blindness is labelled per stake.

- **P-1 (Qwen3-30B-A3B, fully blind).** Held-out static-pin share on the worst domain:
  **0.62**, band 0.50-0.75. Direction is genuinely uncertain — top-8-of-128 activates 6.25% of
  experts per token against 3.1% for the 256-expert file, so there are fewer experts to
  concentrate among; I expect somewhat less concentration than the pilot's 0.74 median.
- **P-2 (q35-A-shexp, SEMI-BLIND — sibling pilot seen).** Held-out static-pin share on the worst
  domain: **0.70**, band 0.60-0.80. This is a confirmation that the pilot's concentration
  survives (a) a known, fixed, two-domain corpus and (b) a held-out ranking. It is not evidence
  of discovery and will not be reported as such.
- **P-3 (cross-domain retention — fully blind, and the stake that decides the lever).** Worst
  retention across both directions and both models: **0.93**, band 0.85-0.98. A static pin is
  chosen once; if the set fitted on C# loses its mass on prose, the concentration is real but
  unusable statically.
- **P-4 (code concentrates more than prose — fully blind).** Held-out same-domain share is
  higher for `code` than for `prose` on **both** models, because code is lower-entropy and more
  repetitive. Directional only; no band.
- **P-5 (depth gradient — semi-blind, direction seen on the pilot).** `late_minus_early > 0` on
  both models and both domains, replicating the pilot out of sample on two new
  (model, corpus) pairs.
- **P-6 (instrument).** Routing mass identical across all MoE layers, and `gate`/`up`/`down`
  counts elementwise identical per layer. Near-certain; its job is to fail loudly if the parse
  is not measuring routing.

---

## KILL RULES

Mechanically checked by `weights/exp52_expert_usage_skew.py`; the verdict is printed and written
to `weights/data/exp52_expert_usage_skew.json` as `overall`.

- **KR-1 (headline).** If the held-out pooled static-pin share on the worst domain is
  **< 0.55**, hot-expert caching is **REFUTED for that model at a 32% budget**. We keep the
  static `-ot` regex, U-34's caching lever is marked not-transferable to our rows, and **task
  #55 (cache-aware dropping) loses the gate it was waiting on** and must be re-justified
  independently or dropped.
- **KR-2 (stability).** If worst-case cross-domain retention is **< 0.90**, **static** frequency
  pinning is refuted *even where KR-1 passes*: the hot set would be an artifact of whichever
  domain we calibrated on. Only a *dynamic* cache could exploit it, that is out of scope here,
  and it must be staked separately — with BigMoeOnEdge's own refuted prefetch A/B as the warning
  that online expert machinery can lose to its own overhead.
  **KR-2 outranks KR-5.** A retention failure is decisive on its own and is never softened into
  MARGINAL by a headline that happens to land near 0.55. (The first draft returned MARGINAL
  whenever the headline sat in 0.53–0.57 *regardless of retention*, which meant KR-2 could not
  fire anywhere in that band. Fixed; the `partial_overlap` control exists to keep it fixed.)
- **KR-3 (instrument VOID).** The run is **VOID** — no verdict, no register entry, fix the
  instrument and rerun — if any of the following hold. A void is not a failure and must not be
  reported as one.
  - `gate`/`up`/`down` counts disagree for any layer;
  - per-layer routing mass is not constant;
  - any shard yields fewer than 8 chunks, **or `imatrix.chunk_count` is absent** (an absent field
    used to skip this check silently, so a 1-chunk run would have scored), **or
    `imatrix.chunk_size` is absent or differs from the requested `-c`**;
  - any layer has **zero total routing mass** (this used to score as share 0.0 and would have
    been published as a *refutation* caused by a broken parse);
  - the ranking and scoring shards cover **different layer sets** (previously the intersection was
    scored silently and reported as the whole model), or the four runs of an arm disagree on
    `n_expert` / `n_moe_layers`;
  - fewer than two domains are present, so KR-2 cannot be evaluated (retention used to default to
    1.0 — a vacuous pass of the stability gate);
  - the four synthetic controls do not reproduce their ground truth in the same session.
- **KR-4 (anti-fudge).** The four corpus shards and `wiki.test.raw` are pinned by SHA-256 above
  and re-checked at runtime; the script refuses to run on mismatch, and a shard with **no** pinned
  hash is now also refused (it previously logged a warning and continued, so deleting one hash
  disabled the gate). The prompt set cannot be reselected after seeing a result.
  The staked budget is `frac=0.32, ctx=512, chunks=8, min_chunks=8, ngl=0`. **Any** departure —
  not just `--frac` — demotes the run to **SWEEP**: it is labelled in the log, tagged
  `is_staked_headline: false` in the JSON, and written to `…SWEEP.json` so it cannot overwrite or
  be mistaken for the staked result.
- **KR-5 (marginality).** If the headline lands within ±0.02 of 0.55 **and KR-2 has passed**, the
  script reports **MARGINAL**, not PASS/FAIL. The documented remedy is to lengthen the C# corpus
  and re-stake; a marginal result must not be rounded into a verdict in either direction.
- **KR-6 (no split-spinning).** KR-1 is per-model, so the two arms can disagree. A `SPLIT` is
  reported as what it is: hot-expert caching **refuted for the failing model** and surviving only
  on the other. It does not license a general claim, the failing arm is published at equal
  prominence with the passing one, and task #55 stays gated to the surviving arm alone. The
  script prints this explicitly rather than leaving a split to be summarised by its winner.

---

## What would REFUTE the idea

Stating this plainly, because the interesting outcome here is the negative one and this project
publishes misses at equal prominence.

**Flat skew refutes hot-expert caching for our models.** If the top 32% of experts by frequency
carry near 32% of routing mass, then frequency-ranked residency is worth exactly nothing over
the index-based regex we already ship, the BigMoeOnEdge 76-84% hit rate is a property of their
models and not of ours, and the correct engineering decision is **to keep the static `-ot`
regex and build nothing**. That would also retire task #55 and close E-02's question in the
negative for MoE experts.

There is a second, subtler refutation that this design is built to catch: **concentration that
is real but domain-specific**. High raw skew with retention below 0.90 means the top-32% set is
a fingerprint of the calibration corpus, not a property of the model. A static pin fitted on
code would then *underperform* the index regex on prose, and shipping it would be an
over-fitting error dressed up as a measurement. KR-2 exists solely to make that outcome visible
rather than to let a strong KR-1 number carry the decision on its own.

**And a passing result does not authorise the obvious change.** Because llama.cpp fuses each
layer's experts into one tensor, no `-ot` pattern can express per-expert residency. A PASS
authorises a *runtime* change and a follow-up prereg that measures actual tok/s; it does not
authorise editing the regex, and it does not by itself claim any end-to-end speedup. The
quantity measured here is routing mass, not time.

**Symmetric reporting, committed in advance.** Whatever `overall` comes back — PASS, FAIL,
SPLIT, MARGINAL, or VOID — the register entry is written from the JSON verbatim and the outcome
is published in the same place with the same prominence. Concretely: a FAIL is written up as a
refutation of E-02's frequency-ranked-residency claim for our rows, not filed as "inconclusive";
a SPLIT names the failing model in its headline sentence (KR-6); a MARGINAL is published as
*undecided* with the remedy, not rounded; a VOID is published as an instrument failure and
explicitly *not* as evidence either way. The only outcome that produces no register entry is
VOID, and VOID is itself reported.

**Wired into:** nothing yet. Decides U-34's caching lever for our rows, gates task #55, and
answers E-02's open question for MoE experts. `findings/REGISTER.json` is updated only after
this is scored.
