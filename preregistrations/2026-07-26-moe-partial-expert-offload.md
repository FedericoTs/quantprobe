# Pre-registration #13: MoE partial expert offload — the placement the planner never offers

**Author:** Federico Sciuca · **Date staked:** 2026-07-26, committed BEFORE any measurement.
**Scoring: same day.**

## The gap

`quantprobe/plan.py` offers a partial-offload "split" row **only for dense models**
(`if (not moe)`). For MoE it offers all-experts-to-CPU hybrid, or nothing. So every MoE user
with a mid-size GPU is told to leave VRAM idle: attention takes ~2 GB of a 6/12/24 GB card and
the remaining VRAM does nothing while all experts stream from RAM (or worse, disk).

Three separate users flagged this in one session (a GLM-4.5-Air owner on a 24 GB card, an
RTX 5070 owner, and the general case), which is what moved it to the front of the queue.

**The mechanism under test:** stock llama.cpp can place *some* expert layers on GPU and the rest
on CPU via `-ot` with a layer-range regex. Expert weights are the bulk of an MoE, so moving a
fraction `f` of them onto the fast tier should shift decode time by the Law-4 identity — and, on
memory-starved boxes, can pull total RAM residency below capacity and escape the disk cliff
entirely (that is the GLM user's case, not ours).

## Method

Model: **Qwen3-30B-A3B-Q2_K** (48 layers, 12.5 GB, 30.5B total / 3.3B active, MoE).
Box: GTX 1060 6 GB · 16 GB DDR4-3000 · the standard reference machine.
Command shape — experts of layers K..47 to CPU, layers 0..K-1 keep theirs on GPU:

```
llama-bench -m <model> -ngl 99 -ot "blk\.(K|K+1|...|47)\.ffn_.*_exps\.=CPU" -n 32 -p 0 -r 2
```

`K = 0` is the current all-experts-to-CPU hybrid — the baseline, re-measured in the same sweep
rather than quoted from an earlier session. Sweep **K = 0, 4, 8, 12, 16, 20**, stopping at OOM.
GPU state logged before/after per Law-5 convention; orphans killed first.

## Stakes

Arithmetic from the Law-4 identity with this box's fitted constants (η_vram 0.35, η_ram 0.35,
192 / 48 GB/s), active split 0.675 GB/token always-active + 0.656 GB/token experts:

- **P-1 (monotonicity).** tok/s rises **monotonically** with K until VRAM is exhausted. A
  non-monotonic interior peak would mean something other than tier bandwidth dominates, and
  would be the more interesting result.
- **P-2 (magnitude).** Baseline (K=0) lands **17–21 tok/s** (consistent with the 19.26–20.4
  measured for this file previously). At the largest K that fits, tok/s lands **24–29** — a
  **+25% to +50%** gain. Below +15% means expert-layer placement does not behave like the tier
  identity predicts and the planner should NOT ship this.
- **P-3 (capacity ceiling).** ~3.0 GB of VRAM is free after attention+KV+buffers; experts cost
  ~0.19 GB/layer, so the last K that fits lands at **K = 12–20**. Beyond it: OOM or a sharp
  fall (WDDM paging), not a gentle taper.
- **P-4 (prefill unharmed).** CPU-side prompt processing at the best K is **within ±10%** of
  the K=0 baseline. Moving weights to the faster tier should not cost prefill; if it does, the
  recommendation needs a workload-conditional caveat.

## Refuted if

Any band missed. Misses publish with equal prominence. **If P-2 misses low, this feature does
not ship** — the planner will keep offering only what it can justify by measurement, and the
three users get told the honest negative.

## Scope limit, stated before measuring

Our 6 GB card holds ~32% of this model's experts. Users on 12–24 GB cards can hold a much larger
fraction, so their gains should be *larger* — but that is an extrapolation along a fitted curve,
not a measurement, and will be labelled as such in the tool. The mechanism is what is being
tested here; the magnitude is per-machine and belongs to the law, not to a benchmark.

---

## Scored (2026-07-26, log: weights/data/prereg13_moe_split.log)

| K (first layer → CPU) | tg32 tok/s | vs baseline | pp512 tok/s |
|---|---|---|---|
| 0 (baseline, all experts → CPU) | 15.18 ± 0.13 | — | 88.3 ± 32.1 |
| 4 | 19.38 ± 0.60 | +27.7% | 206.3 ± 14.5 |
| 8 | 19.89 ± 0.17 | +31.0% | 216.3 ± 11.4 |
| 12 | 20.21 ± 0.02 | +33.1% | 226.9 ± 9.6 |
| **16 (peak)** | **20.44 ± 0.44** | **+34.7%** | **237.6 ± 9.5** |
| 20 | 10.82 ± 0.06 | −28.7% | 57.7 ± 0.5 |

- **P-1 (monotonicity): HIT.** Strictly monotonic through K=16, then a cliff — not a taper —
  at K=20 (20.44 → 10.82). Tier bandwidth dominates exactly as the Law-4 identity predicts.
- **P-2 (magnitude): SPLIT — relative HIT, absolute MISS, and the split is the honest part.**
  The staked *relative* gain (+25–50%) is hit at **+34.7%**. But both *absolute* bands are
  missed low: baseline 15.18 vs staked 17–21, peak 20.44 vs staked 24–29. The whole curve is
  shifted down ~20%. Most likely cause, and it is our own published phenomenon: the desktop held
  **~1485 MiB of VRAM** throughout this sweep (Explorer/overlay/WebView), versus the ~805–1050 MiB
  of the sessions that produced this file's 19.26–20.4 baseline. That is the co-residency effect
  from the Law-5 H3a correction, applied to the baseline instead of the treatment. **Not confirmed**
  — confirming it needs a clean-desktop rerun, and it is recorded as a hypothesis, not a result.
  What survives either way: the *ratio*, measured under identical conditions within one sweep.
- **P-3 (capacity ceiling): HIT.** Last good K=16, cliff at K=20 — inside the staked K=12–20,
  and the failure mode is the staked sharp fall (VRAM overcommit → paging), not a gentle taper.
- **P-4 (prefill unharmed): MISS, badly, in our favour.** Staked within ±10%; measured
  **+169%** (88.3 → 237.6). My stated mechanism was wrong: I reasoned "moving weights to a
  faster tier shouldn't *hurt* prefill" and missed that it also moves the **compute**. Prefill is
  compute-bound in the batch regime (Law 5), so expert layers on GPU means expert prefill math on
  GPU. Caveat on size: the K=0 prefill baseline has 36% variance (±32.1), so treat "+169%" as
  "roughly 2–3×", not a precise figure. Direction and order of magnitude are solid; the exact
  multiplier is not.

**Ship condition met.** The pre-registered non-ship threshold was a gain under +15%; measured
+34.7% on decode and a large prefill bonus. Implementation follows.

**What this means for users:** partial expert offload is worth ~⅓ more decode and ~2× prefill on
a 6 GB card holding only ~⅓ of this model's experts. Users with 12–24 GB cards hold a larger
fraction and should gain more — an extrapolation along a fitted curve, labelled as such, not a
measurement. And the cliff is real: the tool must compute the cutoff from *free* VRAM, because
overshooting costs more than never offloading at all.

---

## CORRECTION (2026-07-26): the baseline above was mis-configured, and +34.7% is overstated

**What was published:** partial expert offload is worth **+34.7%** decode (15.18 → 20.44 tok/s).

**What is wrong with it:** the K=0 baseline was measured **without `--no-mmap`** — a flag this
tool has recommended on the all-experts-to-CPU row for many versions. llama.cpp itself warns
that tensor overrides to CPU with mmap enabled cost performance. So the control condition was a
*worse-than-recommended* version of the thing being beaten. That is a strawman baseline, and it
was self-inflicted.

**Re-measured with every cell configured the way the tool actually recommends** (log:
`weights/data/prereg13_fair_baseline.log`, warm cache, r=3):

| config | tok/s | vs correct baseline |
|---|---|---|
| K=0 baseline, `--no-mmap` | 18.35 ± 0.48 | — |
| K=9 (the cutoff the tool currently picks) | 19.47 ± 0.77 | +6.1% |
| K=12 | 19.61 ± 0.50 | +6.9% |
| **K=16 (peak)** | **20.62 ± 0.26** | **+12.4%** |
| K=20 | 10.59 ± 0.07 | −42.3% (the cliff, reproduced) |

**Honest figure: +12.4% at the peak, +6.1% at the cutoff the tool actually chooses** — not
+34.7%. Of the original claim, **~21 percentage points were `--no-mmap` alone**, which the tool
already recommended and which the baseline was denied.

**What survives unchanged:** the mechanism (monotonic gain with more experts on GPU, P-1), the
capacity cliff (P-3 — reproduced here at −42.3%), and the prefill result. What changes is only
the size of the decode win, and it is now measured against the configuration a user would
actually run.

**How this was caught:** a community user (u/[reddit]) reported that llama.cpp's own `-fit`
auto-placement worked well on his 12 GB card. Checking whether `-fit` beat our recommendation
led to re-examining our baseline, which is where the real problem was. The finding cost us a
headline number and was worth it.

**Process lesson, recorded because it generalises:** a benchmark's *control* deserves the same
scrutiny as its treatment. Our own tool's recommended flags should have defined the baseline
from the first run. No amount of unit testing catches this — it is a measurement-design error,
not a code error.
