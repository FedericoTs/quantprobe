# Pre-registration #72: turning the all-in-VRAM floor into a prediction — the modern-class eta

**Author:** Federico Sciuca · **Date staked:** 2026-07-30, BEFORE implementation. **STAKED.**

## Why, and the promise this honours

The all-in-VRAM row publishes a one-sided FLOOR ("at least X, typically 1.1–1.8× more") because a
single Pascal-fitted eta produced a −9%/+84% spread across models (C-02). That was the honest
answer with one card. We now have 61 verified public benchmarks (`weights/data/public_bench_corpus.json`)
plus E-08's first out-of-sample validation, and we publicly promised the reporter that the next
version turns the floor into a point estimate with a band.

## The honest structure (decided BEFORE seeing whether it validates)

The corpus's per-architecture cells are mostly **n = 1**. Fourteen per-arch constants fitted on
single points would be the exact failure C-02 named. So the shipped object is ONE modern-class
band, not a per-arch table:

- dense Q4-class all-in-VRAM, spec-bandwidth basis, 8 architectures with n≥1 each:
  ada 0.706 · rdna4 0.812 · rdna3 0.665 · ampere 0.645 · blackwell 0.618 · intel-xe 0.588 ·
  apple 0.581 · hopper 0.512 → **median 0.62, band 0.51–0.81**
- excluded, with reasons stated in the code: cdna3 (MI300X, 0.168 — a ROCm kernel gap, not an
  efficiency; its own sibling RDNA3 sits at 0.665) and gcn (MI50, ancient). Excluding low outliers
  makes the prediction MORE aggressive, i.e. it can only hurt us if wrong.
- MoE-resident, same basis: apple 0.319 · ampere 0.409 · blackwell 0.347 · ada 0.501 →
  **median ~0.37** (thinner; ships only if P-3 holds).
- Applies to NON-Pascal presets and auto-detected non-Pascal GPUs on the UNCALIBRATED path only.
  Calibrated/anchored predictions are untouched (they already measure the machine). Pascal keeps
  its own measured constants — this box's ladder must not move at all.

## Stakes

- **P-1 (THE PROMISE, out-of-sample).** E-08's rig (RTX 5070 12GB, 32GB): the 9B Q6_K all-in-VRAM
  row becomes a point estimate whose ±20% band CONTAINS their measured 71–76 tok/s. Their 35B
  split row must STAY in band (54–57 measured) — the new eta must not break what already worked.
- **P-2 (no home regression).** Every arm of the nine-model home ladder moves by **< 2%**: the
  modern-class eta must not touch Pascal. The all-in-VRAM ratchet test is the automated guard.
- **P-3 (the MoE half earns its place, or is withheld).** The MoE modern-class median retrodicts
  the corpus's own MoE rows with median |error| ≤ 25%. If it misses, the MoE half does NOT ship
  and the floor stays for MoE-resident rows — half a feature beats a fitted one.
- **P-4 (honesty surface).** Wherever the new number prints, it names its basis
  ("modern-class eta, derived from N public benchmarks, spec-bandwidth basis") and `calibrate`
  is still recommended as the way to replace it with a measurement.

## KILL RULE

**If P-1 fails — their 71–76 falls outside the new band — the modern-class eta does NOT ship**;
the floor language stays, the miss is published with the corpus row that misled us, and we tell
the reporter we owed them a better number and did not have one yet. **If P-2 fails, the change
is reverted outright**: no external improvement is worth silently moving the measured home
ladder. Publishing the miss carries the same prominence as shipping the hit.

## Also in this release (gated separately, not part of #72's score)

U-23: `--no-mmap` becomes conditional on RAM headroom (E-08's critique — their 20 GB working set
left 330 MiB free, where non-evictable pages are an OOM risk), printing the trade-off both ways:
the measured +73% prefill against evictability.

**Wired into:** pending; `plan.resolve_gpu_eta` (modern-class branch), `spec`/preset path, and a
smoke test pinning both the band and the Pascal no-op.

---

## SCORED — 2026-07-30, before a line of implementation shipped

The derivation and the diagnosis were run against the corpus and E-08's arithmetic BEFORE
touching `resolve_gpu_eta`. The result kills the feature as specified, and the reason is more
useful than the feature would have been.

**Everything on one basis (eta against FILE bytes / spec bandwidth):**

| box | measured eta_file |
|---|---|
| our GTX 1060, Q4_0 / Q4_K_M / IQ4_NL | 0.577 / 0.509 / 0.551 |
| corpus modern cards (8 architectures) | 0.512 (hopper) … 0.812 (rdna4), **median 0.62** |
| E-08's RTX 5070 (73 tok/s x 7.46 GB / 672) | **0.810** |
| what `plan` currently applies (table 0.60 ÷ the 1.147 act convention) | 0.523 |

- **P-1 MISS — and the kill rule fires.** The defensible central value (corpus median 0.62,
  i.e. 0.71 in plan's convention) predicts **56.1 tok/s** for their 9B Q6_K. Their measured
  71–76 sits at the very TOP of the entire public range (0.810, matching the corpus maximum),
  outside the staked ±20% band (44.9–67.3). Raising the constant until their number lands in
  band would be fitting a single point — the exact C-02 failure this prereg named in advance.
  **The modern-class eta does not ship.**
- **The finding that replaces it.** Pascal's measured 0.509–0.577 sits INSIDE the modern-card
  range (0.512–0.812). The spread across modern GPUs is **1.6×** — the same width as the
  "typically 1.1–1.8× above the floor" disclosure we already publish. So the floor is not a
  Pascal artefact and cannot be replaced by any architecture constant: **the variance is real,
  irreducible from public data, and only a per-machine measurement collapses it.** 61 public
  benchmarks now independently justify both the one-sided floor AND `calibrate`'s existence.
- **P-2/P-3/P-4 not evaluated** (no implementation shipped). U-23's `--no-mmap` gating is
  independent and proceeds.
- **What ships instead:** the floor's band language gains its evidence (n=61 public benchmarks,
  measured spread 0.51–0.81 across 8 architectures), and the calibrate recommendation gains the
  specific reason. Registered as **L-18**.

**We owe E-08's reporter a correction**: we promised "~64 ±20%" and the honest answer is that no
constant can produce their 71–76 without fitting to them. What we can offer is `calibrate`,
which measures their box instead of guessing it.
