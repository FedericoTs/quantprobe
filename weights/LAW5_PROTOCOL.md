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

---

## Phase 3 — phase-split placement and the prefill knob map (stakes written 2026-07-24, before any run)

Phase 2 proved decode-optimal and prefill-optimal placements are different configurations.
Phase 3 tests whether the two can be *composed* on stock llama.cpp, and turns the knobs
(offload fraction, KV persistence, format, fa, KV-quant) into a per-machine map. All runs:
dense 7B Q4 = qwen7b-Q4_K_M (28 layers, kvp 57,344 B/token f16), MoE = Qwen3-30B-A3B Q2_K
(kvp 98,304 B/token). b10098 build. New artifacts on C: (D: full).

**H7 (phase-split serving via slot save/restore).** llama-server `--slot-save-path` serializes
sequence state device-agnostically (claim under test).
- **H7a staked (binary):** a slot saved on a prefill-tuned instance (ngl16) restores into a
  decode-tuned instance (ngl99) of the same GGUF (same -c, same cache types); the follow-up
  request with the identical prompt reports prompt_n ≈ 0 (no recompute) and generates coherently.
  If the state format embeds device layout, this fails and is published as the kill.
- **H7b staked (bridge price is transfer-priced):** 2048-token dense-7B state file lands
  **100–140 MB** (ctx x kvp ≈ 117 MB + overhead); save and restore each sustain **≥ 200 MB/s**
  effective warm (≤ 0.7 s); restore ≥ **2,000 tok/s** equivalent — ≥ x60 over CPU-pure compute
  (33), ≥ x6 over the ngl16 peak. Below 2,000 → serialization overhead dominates and gets named.
- **H7c staked (end-to-end):** pp8192+tg256 on dense 7B: phase-split total (prefill@ngl16 →
  save → restore → decode@ngl99) ≤ **0.85 x** the best single-config total. Arithmetic behind
  the stake: split ≈ 31+2+5 s vs single-config ≈ 52 s (ngl16 decode ~12 tok/s: 12 CPU layers
  ≈ 1.87 GB/token over RAM) or ≈ 59 s (ngl99 prefill ~150 @8k). Robust across decode-rate
  uncertainty ±50%.

**H8 (KV persistence beats recompute for agent turns — the prefix answer).** Same-config
save/restore across server restarts, MoE 30B on its decode config (ngl99 -ot exps=CPU, 193
pp measured).
- Staked: 8k-token state file **0.75–0.95 GB**; restore-then-decode beats recompute-then-decode
  on time-to-first-token by **≥ x10** (compute ≈ 42 s; restore ≤ 4.2 s; cold-disk floor
  1.6 s from 0.5 GB/s SATA). Effective restored-prefix rate staked **2,000–20,000 tok/s** —
  the "turn 2+ is free" claim, priced by tier.

**H9 (the ngl-peak formula).** Pre-stated: the naive load-balance overlap model (peak at
ngl 25–26, ceiling ≈ 300 from R_cpu 27.2 + R_gpu-stream 273) is ALREADY inconsistent with
the measured ngl16 = 322.5 > 300 — throughput at the peak is super-additive. The fine sweep
maps the correction term.
- Staked: sweep ngl ∈ {4,8,12,14,16,18,20,24} at pp2048 shows a single interior peak within
  **ngl 12–20**, peak rate **310–335**, and both ngl8 and ngl24 sit ≥ 8% below the peak.
- Interleave fork (activation-handoff arithmetic: 27 CPU↔GPU crossings ≈ 1.6 GB/pass ≈ 2% of
  pass time — cheap): **staked branch:** interleaved offload at the same fraction
  (ngl99 + -ot routing even-index blk to CPU, 14/28) lands **within ±10%** of the contiguous
  curve at ngl14–16. If it instead collapses ≥ 30%, per-crossing sync cost (not bandwidth) is
  the named term.

**H10 (context-quadratic term — owed to P-b).** Per-token linear model fitted on measured
512/2048/8192 (t/tok = 3.614 ms + 1.117e-4 ms x ctx):
- Staked: pp4096 @ ngl0 = **242–256 tok/s**; pp12288 @ ngl0 = **195–211 tok/s** (fa at build
  default, matching Phase 1/2 conditions).
- fa pilot cells (no prior data → measured unstaked, Phase-1 style): pp2048 and pp8192 at
  -fa on / off / default on Pascal → coefficient c per fa-mode. If c moves < 10%, the fa knob
  is dead on this hardware and recorded as such.

**H11 (KV-quant prefill tax — one cell, forked).** ctk/ctv = q8_0, ngl0, pp2048, dense 7B
(with -fa on if quantized V requires it; fallback cell ctk-only without fa).
- **Staked branch:** within **[−20%, +5%]** of f16 — batched dequant amortizes, same family as
  P-c. The collapse branch (−50%+, mirroring Pascal's −83% decode) would make the KV-q8 gate
  phase-independent.

**H12 (format x device η_pp table — prefill-friendly formats).** CPU-pure pp2048, dense 7B.
Known: Q4_K_M 27.2, Q2_K 17.6. New cells: Q4_0, Q5_0, Q8_0 (requantized from qwen7b-Q4_K_M —
speed-only cells, no quality claims, disclosed), IQ3_M and IQ3_XS (existing Instruct files,
identical architecture and shapes → identical FLOPs, disclosed).
- Staked: **Q8_0 fastest overall** (+10–35% over Q4_K_M — trivial dequant, and bytes don't
  matter in the compute regime); **Q4_0 ≥ Q4_K_M** (+5–20%); **IQ3 family slowest, ≥ 25%
  below Q4_K_M** (LUT dequant tax, ≤ 20.4 tok/s).
- Output if it survives: per-format η_pp column and the recipe "AVX2-era CPUs: prefer _0
  formats for CPU-resident tensors during prefill-heavy workloads."

---

## CORRECTION (2026-07-24): Phase-2 H3a "partial-offload peak" was a VRAM-contention artifact

**What was published:** prefill peaks at ngl16 (322.5) vs ngl99 (166) — "+94% over full offload,"
attributed to a residency mechanism.

**What is true:** on a clean GPU, full offload wins: **ngl99 = 377.5 ± 0.9**, ngl16 = 316.6–322.5.
The Phase-2 sweep (and the pilot's P4 = 166) ran while an orphaned llama-server from the previous
day's dashboard session (30B hybrid, ~1.6 GB VRAM) was resident. Controlled A/B, same build, same
file, same command, run 2026-07-24:

| GPU state (nvidia-smi before run) | ngl99 pp2048 | ngl16 pp2048 |
|---|---|---|
| clean (927 MiB baseline) | **377.51 ± 0.90** | 316.56 ± 1.36 |
| deliberate squatter resident (2560 MiB: 30B hybrid, the dashboard config) | **166.50 ± 0.71** | 317.76 ± 1.02 |

The squatted number reproduces the pilot's 166 to ±0.5 tok/s; ngl16 is invariant across states.
Attribution of the original contamination is forensic reconstruction (the orphan was found and
killed at today's session start); the mechanism is proven by the controlled reproduction above.

**What replaces the claim:** placement is **co-residency-conditional**. A ~1.6 GB VRAM squatter
(a dashboard, a browser, a game) halves full-offload prefill and flips the optimal placement from
ngl99 to partial offload — while partial offload is contention-immune. This is a real, useful,
measurable phenomenon; it was just not the phenomenon we claimed. H3b's miss is explained (ub
buffers were never the mechanism), and H9's staked sweep will be scored against the CLEAN curve
(expected outcome: MISS of the staked interior peak — publishing it as such) plus a squatted twin
sweep to map the co-residency geography.

**Contamination audit triggered:** every GPU-path prefill number from the pilot and Phase 2
(P3 ngl0 273/277.8, P-b 223.58@8k, P-c 277.15, P6 hybrid 193.24, Phase-2 sweep) is being
re-measured on a verified-clean GPU. H10's staked bands were derived from potentially
contaminated points; they will be scored against clean re-measurements and re-staked if the
underlying curve moves. CPU-pure numbers (-dev none) are unaffected by construction.

**New convention (binding from now on):** `nvidia-smi memory.used` is logged immediately before
and after every GPU-path measurement batch; a prefill run is valid only if the pre-run baseline
is the clean-desktop value (~0.9 GB on this box). Kill-orphans-first is now a protocol step, not
just hygiene.

### Phase 3 — scored (2026-07-24, same day; raw logs law5_clean_audit.log, law5_h12_formats.log, server logs in session scratch)

- **H7a HIT.** State saved on the ngl16 instance restored into the ngl99 instance: 2036/2036
  tokens, follow-up request recomputed exactly **1 token**, coherent continuation. The
  phase-split bridge exists in stock llama.cpp.
- **H7b HIT, above band.** State file = tokens x kvp to within a header (116,785,660 B measured
  vs 116.75 MB arithmetic at 2k; 778,007,340 B vs 777.9 MB at 8k-MoE — the format is pure KV).
  Save 1.98–2.14 GB/s, restore 2.18–2.50 GB/s warm (staked floor 0.2 GB/s). Restore-effective
  rate 16,800–39,800 tok/s.
- **H7c MISS (ratio 1.10 vs staked <= 0.85).** Split 46.8 s vs best-single 42.6 s at pp8192+tg256
  on dense 7B. Cause: the premise (ngl16 prefill >> ngl99) was the Phase-2 contention artifact;
  on a clean GPU ngl99 wins both phases on this model/box, so there is no gap to arbitrage.
  Unstaked corollary noted for future work: under measured co-residency the gap reopens
  (squatted ngl99 = 154 vs ngl16 = 315), and the split premise revives conditionally.
- **H8 HIT, x159 vs staked x10.** 30B hybrid, 8k context: compute TTFT 74.56 s (106 tok/s
  server-path at depth) vs restore TTFT **0.47 s** (311.7 ms restore of 778 MB + 0.16 s eval).
  Including full server restart: 6.4 s → x11.7, still above stake. Effective restored-prefix
  rate 16,800 tok/s (staked 2,000–20,000). Decode after restore coherent at 15.36 tok/s
  (8k-depth KV drag, consistent with Law 4 v2). **The agent recipe is measured: persist KV
  between turns and turn-2+ context cost drops from ~75 s to ~0.5 s.**
- **H9 MISS.** Clean-GPU fine sweep is MONOTONIC: 281.4 (ngl4) → 292.9 → 305.0 → 309.5 → 314.3
  → 321.9 → 328.5 → 345.3 → 363.0 (ngl28) → 368.0 (ngl32/99). No interior peak; the staked
  12–20 peak was the artifact's geography. More offload = faster prefill, saturating at full.
- **H9 squatted twin — the replacement law, quantified.** With a 1.6 GB co-resident (nvidia-smi
  2311 MiB): ngl0/8/16 unchanged (261/290/315) but ngl24 **collapses to 145.0** and ngl99 to
  153.6 — the knee is exactly VRAM overcommit (layers + squatter + buffers > 6 GB → WDDM
  paging). **Prefill-optimal ngl under co-residency = the largest ngl that still fits free
  VRAM** — computable from nvidia-smi, the Law-4 fit-check applied per phase and per machine
  STATE. Contention-immune region: any ngl whose footprint fits beside the squatter.
- **Interleave fork: VOID.** It was designed to discriminate mechanisms of an interior peak
  that does not exist on a clean GPU; the parent phenomenon was retracted, so the fork is moot
  (not run, per protocol economy).
- **H10 double HIT — the context term survives the audit.** Clean ladder (ngl0): 512 → 285.6,
  2048 → 271.6, 4096 → **254.1** (staked 242–256), 8192 → 224.4, 12288 → **201.1** (staked
  195–211). Per-token linear fit: t/tok = 3.44 ms + 1.24e-4 ms x ctx — attention doubles
  prefill cost at ~28k tokens on this path. The ngl0 cells were only mildly contaminated
  (small VRAM footprint), which is why P-b/P-c reproduce.
- **H11 HIT (staked branch).** KV-q8 prefill: q8/q8+fa 272.8 vs f16+fa 262.7 (**+3.8%**),
  ctk-q8 261.5 vs f16 271.6 (−3.7%) — inside [−20%, +5%]. Pascal's −83% decode collapse has
  no prefill twin: **the KV-q8 gate is phase-dependent** (batched amortization, P-c family).
  fa pilot cells: fa costs −1..−3% on Pascal prefill in every pairing → knob verdict: minor,
  prefer off on this hardware.
- **Contamination audit results.** P-c reproduces clean (276.5 vs 277.2 published ✓); pilot
  30B ngl0 reproduces (213.6 vs 221.4, −3.5% ✓); P6 hybrid corrected upward to **209.3 ± 12.1**
  (193.2 published was ~8% low, inside today's noise band); P6 pp32 re-cell (6.8) flagged as
  cold-cache warm-up artifact, excluded pending a warmed re-run. CPU-pure numbers unaffected
  by construction. H12 running.
