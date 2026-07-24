# Pre-registration #9: Law 5 (prefill) — first staked predictions from the pilot fit

**Author:** Federico Sciuca · **Date staked:** 2026-07-25, committed before the measurements ran
**Protocol:** [weights/LAW5_PROTOCOL.md](../weights/LAW5_PROTOCOL.md) (frozen before the pilot)
**Pilot results:** [weights/data/law5_pilot.log](../weights/data/law5_pilot.log), law5_p5p6.log
**Status: SCORED same day (P-a/P-b/P-c); P-d remains community-gated.**

- **P-a: HIT.** Measured **31.11 ± 0.21** vs staked 28.4–34.7 — dead center. The per-architecture
  fit reproduces across the Coder variant.
- **P-b: MISS (−18%).** Measured **223.58 ± 1.31** vs staked 265–280. Deep batch is not flat —
  and the residual is identifiable: attention's context-quadratic FLOPs, excluded by the 2×params
  convention, grow to a visible share by 8k. **Prefill has its own context term**, the compute-side
  sibling of Law 4 v2's KV term. Fitting it is the next protocol step (the 512/2048/8192 sweep
  already brackets it: 276.7 → 272.7 → 223.6).
- **P-c: HIT on the staked fork branch.** Measured **277.15 ± 2.67** — statistically identical to
  Q4's 272.7. **The dequant tax is CPU-only**; CUDA kernels amortize format cost. The
  η_pp(format, device) table has its decisive cell.

Running tally for Law 5's birth: two sharp claims killed (bits-invariance, hybrid-hurts), two
staked hits, one productive miss naming the next term — all before the law is allowed anywhere
near LAWS.md. Raw logs: [law5_stakes.log](../weights/data/law5_stakes.log).

## What the pilot established (and killed)

- **Killed: bits-invariance.** Q2 prefills 35% *slower* than Q4 on CPU at identical FLOPs
  (17.60 vs 27.13) — dequant-format cost rules the compute regime. Same mechanism family as the
  Pascal decode collapse.
- **Killed: "hybrid hurts prefill."** Hybrid (`-ngl 99 -ot exps=CPU`) prefills at **193 tok/s**
  vs 31.6 CPU-pure — batch amortization of the tier transfer beats the expert-union tax at
  agent-scale batches. The tax survives only at small batch (pp32 ≈ 24) and as a ~0.78×
  per-FLOP MoE efficiency on CPU-pure.
- **Confirmed: the transfer→compute batch crossover**, predicted at batch ≈ 104 from
  PCIe-bandwidth-vs-compute arithmetic, measured bracketing [32 → 42, 512 → 277].

## Staked predictions (fit: η_pp is format- and device-dependent; per-FLOP MoE tax ×0.78 on CPU)

- **P-a — same-class reproduction.** Qwen3-**Coder**-30B-A3B **Q2_K_L**, CPU-pure, pp2048:
  same architecture and bit-class as the pilot's P5 → **28.4–34.7 tok/s** (P5 ± 10%).
- **P-b — deep-batch flatness.** Dense 7B Q4, GPU-assist (`-ngl 0`), **pp8192**: past the
  crossover the compute ceiling holds → **265–280 tok/s** (flat vs pp2048's 272.7, ±3%).
- **P-c — is the dequant tax universal, or CPU-only?** Dense 7B **Q2**, GPU-assist, pp2048.
  Sharp fork, staked without hedging: if CUDA dequant kernels amortize the format cost, this lands
  **250–280** (GPU bits-invariant); if the tax is universal, it lands near **175** (0.65 × Q4).
  **I stake the first branch: 250–280.** A miss here rewrites the η_pp(format, device) table, and
  I'll publish it as such.
- **P-d — community band (the Continue-user case).** MoE 30B-A3B class at **Q4**, CPU-pure,
  pp2048, on DDR4 desktop CPUs (4–8 cores): from per-FLOP fit (dense-Q4 413 GFLOPS-eff × 0.78 MoE
  tax ÷ 6.6 GFLOP/token) → **43–55 tok/s** — notably *faster* than its Q2 sibling (31.6), the
  bits-invariance kill applied in reverse. Anyone: `llama-bench -dev none -ngl 0 -p 2048 -n 0`.

## Refuted if

P-a outside ±10% (the per-arch reproduction fails); P-b falls >5% below pp2048 (an unmodeled
deep-batch cost); P-c lands in neither band (mechanism misunderstood); P-d outside 38–60 on
comparable CPUs. Misses publish with the same prominence as hits — they are the point.
