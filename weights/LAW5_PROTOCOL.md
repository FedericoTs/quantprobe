# Law 5 protocol — prefill (prompt processing) on commodity hardware

**Status: PROTOCOL. Nothing here is a law yet.** This document is written *before* the pilot
matrix runs, per house rules: define conventions and falsifiable structure first, measure second,
stake third, confirm fourth. Law 5 stays out of LAWS.md until it has survived its own staked
predictions. It never modifies Laws 1–4.

## Why a fifth law

Everything measured so far — here and across the field's calculators — is *decode*. But 2026's
dominant local workload is agentic (Cline/Continue/Aider dumping 4–20k-token payloads), and agents
live on **prefill**. We have already measured the pain in the wild: a Continue user's "2–5 minutes
per request" decomposed exactly into a 4–8k payload at CPU prefill speed. Decode has a law;
prefill has folklore.

## Hypothesis (to be tested, then staked)

Prefill differs from decode in kind: tokens are processed in large batches, so each weight read is
amortized over the whole batch. That predicts a **two-regime law**:

> **H1 (compute regime).** When weights are co-located with the computing device (or the batch is
> large enough to amortize transfer), prefill is compute-bound:
> `pp tok/s ≈ η_pp × OPS_eff ÷ (2 × active-params)` — and, sharply: **independent of quantization
> bit-width** (the FLOPs don't change when the bytes shrink).
>
> **H2 (transfer regime).** When weights live on a slower tier than the compute (hybrid expert
> offload, disk streaming), prefill becomes transfer-bound on the weight traffic that the batch
> forces across the boundary — MoE batches union experts (the Law-4 corollary), so hybrid
> placements can make prefill *worse* while making decode better. The crossover batch size is
> predictable from bandwidths.

## Conventions (fixed before any measurement)

- Work unit: `2 × active-params` FLOP per token (standard forward-pass counting; integer paths may
  yield η_pp > 1 against an f32 peak — η_pp is a fitted utilization constant, not an efficiency
  percentage, exactly like Law 4's η).
- Standard batch: `-p 2048` (agent-payload scale); batch sweeps use `-p 32,512,2048`.
- CPU-pure means `-dev none` (**`-ngl 0` alone still GPU-accelerates prefill** — measured artifact,
  2026-07-24, and the reason several public "CPU" prefill numbers are wrong).
- llama-bench, `-r 2` minimum, first-run warm-up discarded where it matters (run-order artifact,
  measured 2026-07-23).
- Quality is out of scope; engine-specific sparse-prefill (learned indexers) is out of scope —
  this law prices *dense exact* prefill on stock llama.cpp.

## Points already in hand (raw logs in-tree)

| config | pp tok/s | source log |
|---|---|---|
| dense 7B Q4, CPU-pure (`-dev none`) | 27.23 ± 0.02 | cpu_prefill_true.log |
| MoE 30B-A3B Q2, CPU-pure | 31.70 ± 0.06 | cpu_prefill_true.log |
| dense 7B Q4, GPU-assisted (`-ngl 0`, CUDA prefill) | 277.8 ± 2.7 | cpu_repro.log |
| MoE 30B-A3B Q2, GPU-assisted | 221.4 ± 0.9 | cpu_repro.log |
| MoE 30B-A3B Q2, hybrid, live chat prompt | ~28.4 | dashboard session |
| Laguna 118B, disk-streamed | 0.47 | laguna_text3.log |

First-glance structure the pilot must confirm or kill: GPU-assisted ≈ 10× CPU-pure on the same
file; MoE CPU prefill lands *near dense* despite 2.3× fewer active params (batch expert-union tax,
consistent with H2's mechanism); disk prefill is catastrophic (0.47), consistent with H2.

## Pilot matrix (this box: i5-7600K · GTX 1060 6GB · 16GB DDR4-3000 · files on disk)

Batch sweep `-p 32,512,2048`, `-r 2`, per configuration:

| # | file | device path | tests |
|---|---|---|---|
| P1 | dense 7B **Q4_K_M** | CPU-pure | bits-invariance arm A |
| P2 | dense 7B **Q2_K** | CPU-pure | bits-invariance arm B (H1's sharp claim) |
| P3 | dense 7B Q4 | GPU-assist (`-ngl 0`) | transfer-amortization vs batch |
| P4 | dense 7B Q4 | all-in-VRAM (`-ngl 99`) | pure GPU compute ceiling |
| P5 | MoE 30B Q2 | CPU-pure | MoE union tax vs P1/P2 |
| P6 | MoE 30B Q2 | hybrid (`-ngl 99 -ot exps=CPU`) | H2: does hybrid *hurt* prefill? |
| P7 | Bonsai 27B Q1 (linear-hybrid) | CPU-pure + all-in-VRAM | third architecture class |

## After the pilot (in order, separate commits)

1. Fit η_pp per (device, regime) from the pilot; write the two-regime model down.
2. **Pre-register** (repo, timestamped) predictions for configs NOT in the pilot — candidates:
   Coder-30B Q2 CPU-pure (must equal P5's class within ±10% — same arch, same bits),
   7B Q4 GPU-assist at batch 8192 (H2 crossover extrapolation), and one community-hardware band.
3. Measure the staked configs. Score in public, hits and misses.
4. Only then: Law 5 enters LAWS.md, `plan --prefill` enters the tool, and the Continue-class
   answer ("your first request will take N minutes") becomes a computed number.

---

## Phase 2 — the allocation frontier (stakes written 2026-07-25, before the runs)

**H3 (VRAM-residency tax).** The pilot anomaly (ngl0 273 vs ngl99 166 at pp2048) has a proposed
mechanism: resident weights compete with prefill compute buffers on a 6 GB card. Staked: (a) the
ngl-sweep peak is NOT at full offload; (b) raising `-ub` to 1024 at ngl99 recovers at least half
the gap (≥ 200 tok/s). If -ub does nothing, the mechanism is wrong and gets published as such.

**H4 (coverage kills selective prefill placement — by arithmetic).** Expected expert coverage in
a batch of n tokens is 1−(1−k/E)^n: for k=8, E=128 → 90% at n=32, ~100% at n=512. Prefill
touches everything; there is nothing selective to place. Corollary staked without measurement
(the arithmetic plus measured flat routing suffice): dynamic expert placement can only pay below
batch ≈ 8 — i.e. decode, where the static version is already measured-dead. Empirical coverage
curve deferred to an eval-callback source build (prebuilts ship no router telemetry).

**H5 (pruning does not speed prefill).** Active-k is unchanged by expert pruning, so prefill
FLOPs are unchanged: REAP-50-class pp2048 within ±10% of its parent, any device. Community-checkable.

**H6 (k-reduction endpoints; the asymmetric idea).** Using stock `--override-kv
qwen3moe.expert_used_count=int:4` on the k=8 file: staked — CPU-pure pp2048 rises to **43–50**
(FLOPs ×0.68 of the 31.6 baseline), and WikiText-2 ppl degrades by **+8–25%** (k-halving is not
free; consistent with the dynamic-top-k dead end and Lucebox's admitted trade). The genuinely
novel design this bounds: **asymmetric top-k** (prefill at k=4, decode at k=8) — if context
ingestion tolerates reduced k better than generation does, agents get +40% prefill nearly free.
Not expressible in stock llama.cpp (no per-phase k); requires a small patch; designed, not run.

### Phase 2 — scored (same day, logs in weights/data/law5_phase2.log)

- **H3a HIT, and a discovery:** the ngl sweep is wildly non-monotonic — 273 (ngl0) → **322.5
  (ngl16, the peak)** → 180 (ngl32) → 166 (ngl99). **Prefill-optimal offload is PARTIAL** on this
  card: +18% over stream-everything, +94% over full offload. Nobody sweeps ngl for prefill;
  decode-optimal and prefill-optimal placements are different configurations — the phase-dependent
  placement corollary now has its first measured demonstration.
- **H3b MISS:** ub-1024 at ngl99 recovered only 166→181.7, far short of the staked ≥200. The
  buffer-squeeze mechanism is minor at best. Revised hypothesis for Phase 3 (unstaked, to be
  staked before testing): the ngl16 peak is CPU+GPU **pipeline overlap** — partially-resident
  layers compute on both devices concurrently; full offload serializes onto the weaker GPU.
- **H6 quality HIT:** k=4 override costs **+20.7%** WikiText ppl (8.31 → 10.03), inside the staked
  +8–25%. k-halving is not free — the asymmetric-top-k design's value now hinges entirely on
  whether *context ingestion* tolerates what generation doesn't (the patch-gated experiment).
- **H6 speed: blocked by tooling** — `--override-kv` is supported by llama-perplexity (the quality
  runs prove it) but not llama-bench; speed endpoint recovered via perplexity prompt-timing
  (law5_h6speed.log).
- **H6 speed HIT (recovered via perplexity pass-timing):** k=4 CPU-pure prefill = **47.2 tok/s**
  (2048/43.41s) vs k=8's 33.0 (2048/62.08s — cross-validating llama-bench's 31.6 from a second
  tool) — **+43%**, inside the staked 43–50. Both asymmetric-top-k bounds are now measured: the
  prize (+43% prefill) and the price-if-global (+20.7% ppl). The patch-gated experiment — k=4
  during prefill only — decides whether the prize can be taken without the price.
