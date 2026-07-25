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
