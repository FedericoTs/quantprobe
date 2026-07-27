# Pre-registration #21: KV placement is separable from weight placement — and shares the same budget

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## The claim

The law welds KV to the layers it serves: `kv_gb` is added to whichever tier holds the weights,
and `ETA_KV = 0.70` prices it there. But llama.cpp exposes `-nkvo/--no-kv-offload`, which keeps
the KV cache in host memory **independently of where the weights live**. That is a placement
dimension the search does not have.

It matters more now than it did yesterday. Pre-registration #20 established that **placement and
batch compete for one VRAM budget** — the expert split fills VRAM, which starves the large-ubatch
compute buffer and costs 42% of prompt processing. KV is a third claimant on that same budget, and
at depth it is not small: 96 KB/token on this model, so 16k of context is ~1.5 GB — comparable to
the entire compute-buffer headroom the ubatch lever needs.

So the interesting question is not "is CPU-side KV slower" (it must be). It is whether **giving up
KV residency buys back enough VRAM to unlock a lever worth more than it costs.**

## Configurations, `Qwen3-30B-A3B-Q2_K`, reference box

Split placement (`-ot "blk\.(16..47)\.ffn_.*_exps\.=CPU"`), which #20 showed is VRAM-starved.

## Stakes

- **P-1 (the cost is real).** At `-d 16384`, `-nkvo 1` **hurts decode by ≥10%** versus `-nkvo 0`.
  KV moves to a tier ~4× slower and is re-read every token; if this does not show, either the KV
  term or my understanding of the flag is wrong.
- **P-2 (the cost is depth-dependent).** At `-d 0` the same flag is **neutral, within ±3%**. With
  no context there is no cache to read, so placement of an empty thing cannot matter.
- **P-3 (the joint-budget test — the reason this is worth doing).** On the split placement at
  `-ub 2048`, `-nkvo 1` **recovers ≥20% of prompt processing** versus `-nkvo 0`, because the VRAM
  released by evicting KV is exactly what the compute buffer was short of. This is the claim that
  the three claimants are **fungible**: if it holds, the search must optimise them jointly rather
  than pick a placement and then bolt levers on.
- **P-4 (no anchor moves).** A flag, not a law change. All four published anchors bit-identical.

## Refuted if

**P-3 fails.** That would mean VRAM freed from KV does not convert into usable compute-buffer
headroom — the budget is shared but not fungible — and the dimensions can go on being optimised
one at a time. That is a genuinely useful negative result and would simplify the search.

P-1 failing would instead mean I have misread what `-nkvo` does, and P-3 would be uninterpretable.

## What ships if it holds

Nothing automatic. A KV-placement recommendation is **depth-dependent and workload-dependent** —
it trades generation speed at long context for prompt speed — so at most it becomes a disclosed
option alongside the phase advice, in the same shape as v1.13.1: state the trade, name the
alternative command, let the user choose. `plan.evaluate()` stays frozen either way.

---

## Scored (2026-07-27, log: `weights/data/prereg21_kv_placement.log`)

**Verdict: P-1 HIT, P-2 MISS, P-3 HIT far beyond the stake, P-4 HIT.**

### P-1 and P-2 — decode cost by depth

| `-nkvo` | tg32 @ d0 | tg32 @ d16384 |
|---|---|---|
| 0 (KV in VRAM) | **20.35 ± 0.30** | **6.30 ± 0.18** |
| 1 (KV in host RAM) | 16.39 ± 0.11 | 3.36 ± 0.06 |

- **P-1 (≥10% cost at 16k): HIT.** −47%.
- **P-2 (neutral at d0, ±3%): MISS.** −19.5%. I reasoned that with no context there is no cache
  to read, so its placement could not matter. Wrong: the cost is **largely fixed, not
  depth-proportional**. `-nkvo` evidently changes where the attention *computation* happens, not
  merely where the cache is stored — otherwise an empty cache would be free. The flag is not a
  "free below some depth" lever, and anyone reasoning about it as pure cache-read bandwidth (as I
  was) will mis-predict it.

### P-3 — the joint-budget test, and the result of the whole exercise

Split placement, `-ub 2048`, one invocation:

| `-nkvo` | pp2048 |
|---|---|
| 0 | 85.47 ± 0.27 |
| 1 | **327.81 ± 4.58** |

**HIT — 3.84× against a staked ≥20%.** The VRAM claimants are **fungible**: evicting KV frees
exactly the headroom the large-ubatch compute buffer was short of.

### The joint matrix, and the frontier

All four cells, one session, `-ub 2048`:

| placement | KV | pp2048 | tg128 | |
|---|---|---|---|---|
| split K=16 | VRAM | 161.59 ± 0.09 | **20.14 ± 0.24** | Pareto-optimal — decode champion |
| split K=16 | host | **391.72 ± 2.80** | 16.54 ± 0.03 | Pareto-optimal — prefill champion |
| all → CPU | VRAM | 345.41 ± 0.36 | 18.68 ± 0.27 | Pareto-optimal — balanced |
| all → CPU | host | 336.31 ± 1.71 | 15.82 ± 0.08 | **dominated** — never choose it |

Two things follow, and neither was visible from any single dimension:

1. **There is no single best placement — there is a frontier.** Three configurations are
   Pareto-optimal and the right one depends on the prompt-to-generation ratio:

   | workload | best configuration | vs the worst choice |
   |---|---|---|
   | chat (0.5 : 1) | split, KV in VRAM | 1.23× |
   | coding (10 : 1) | all → CPU, KV in VRAM | 1.35× |
   | RAG (50 : 1) | split, KV evicted | 1.91× |
   | document QA (200 : 1) | split, KV evicted | 2.25× |

2. **Fungibility is placement-specific.** Evicting KV recovers 2.42× on the split, where VRAM
   binds, and *nothing* on all-experts-to-CPU (345 → 336, −3%), where it does not. The budget is
   shared, but only the configuration that is actually starved can spend the refund.

### Note on reproducibility

The split/`-nkvo 0`/`ub 2048` prefill cell measured **85.47** when swept against `-nkvo` alone and
**161.59** in the full matrix. Both are internally consistent within their invocation and the
`-nkvo` contrast holds in both (3.84× and 2.42×), but the absolute value is not stable across
invocation shapes. Flagging rather than averaging: the *ratios* are the result here, and any
future use of the absolute figure needs its own controlled run.

**Wired into:** `quantprobe/plan.py:workload_frontier` · `tests/smoke.py:t_workload_frontier_is_pareto`
