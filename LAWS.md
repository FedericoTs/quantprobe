# The Four Placement Laws

*Every law below states its claim, the measurement that established it, and a falsifiable prediction
anyone can test. All numbers from one 2016 desktop (GTX 1060 6 GB · 16 GB DDR4 · SATA). Full logs in-tree.*

---

## Law 1 — Rotation is rank-conditional
**Incoherence rotation — the foundation of modern quantization codecs (QuIP#, QTIP, QuaRot) — helps
full-rank tensors and destroys low-rank bottlenecks.**

- *The measurement:* the same orthogonal rotation costs **+0.006 ppl** on a full-rank MLP
  (eff. rank 1168) and **+1623 ppl** on the low-rank KV-latent (eff. rank 394) — a **~270,000×** swing
  on effective rank alone. Every gauge tried on the bottleneck (Hadamard, SVD, diagonal) made it worse;
  only native-basis precision repairs it.
- *The prediction:* any architecture that manufactures low-rank structure (MLA latents, LoRA merges,
  GQA projections, SSM states) will be damaged, not helped, by incoherence processing at low bits.

## Law 2 — Trained networks are dense everywhere
**Load-balanced training fills every axis a data-free method can reach: there is no free lunch left in
the weights, the routing, or the activations.**

- *The measurement:* routed experts sit exactly at the Gaussian rate-distortion floor
  (rel-MSE 0.069 = D(R=2), identical across all 64); 1-bit collapses (+253 ppl) under every codec;
  routing is flat (a token needs ~5.3 of its top-6 experts); activations are diffuse (72–84% of neurons
  carry 90% of energy); and expert usage is **domain-flat** — prose and code use *identical* expert sets
  (Jaccard 1.00). Thirty candidate levers, twenty-nine measured dead.
- *The prediction:* 2-bit is the data-free floor for any load-balance-trained model; task-trimming
  experts and semantic "brain-region" paging will fail on any of them.

## Law 3 — Fragility is measurable, not predictable
**Where a model breaks at low bits is model-specific: no configuration flag, architecture family, or
weight statistic predicts it — but a 30-minute functional probe measures it exactly.**

- *The measurement:* the depth-fragility atlas — Gemma-4-12B **late**-fragile (~4×), Qwen2.5-7B late
  (~2–3×), Qwen3-30B-MoE late (~2.3×), **Mistral-7B early-fragile (~25×)** despite being Qwen's
  architectural near-twin. Weight kurtosis points the *wrong way* on Gemma. Placement by the probe:
  byte-identical GGUF files **2.25 ppl apart** (10.02 vs 12.27); the depth-aware recipe halves Gemma's
  2-bit gap (1.91× → 1.45×) and, data-free, edges an imatrix-calibrated community quant at 30B scale.
- *The prediction:* for any new model, the band probe (`quant_probe.py`) beats every static allocation
  rule; guessing the fragile end without probing risks forfeiting up to a 25× fragility differential.

## Law 4 — The tiered decode law
**Decode speed is a placement identity: `tok/s = η(tier) × bandwidth ÷ active-bytes-per-token`, with
the utilization constant η collapsing per memory tier.**

- *The measurement:* η = 0.56 (VRAM) · 0.29–0.68 (RAM: dense ≈0.65, MoE ≈0.35 — the scatter penalty)
  · 0.88–1.0 (disk), across 7B→744B **including colibri's independently published tiers** (his 0.48 and
  0.88 sit inside our bands). Pre-registered hits: a 110B model streamed from SATA at **0.19 tok/s**
  (predicted 0.2–0.3); a RAM overclock (2133→3000) delivered **×1.52** on dense (predicted ×1.41+);
  and when bandwidth rose, the 30B's bottleneck *migrated* to RAM capacity — exactly as a law-governed
  system should behave.
- *Corollaries, each measured:* on poor-decode GPUs, experts belong on the CPU (+54%, one flag);
  batch-union returns scaling on the CPU tier (4.5× at batch 8); **speculative decoding is antagonistic
  to MoE sparsity** on bandwidth-bound tiers (verify-batches union ~40 experts vs 8 — measured 2.3×
  *slower* with a draft); and the MoE scatter penalty is a memory-system property (slab-hopping defeats
  prefetch), not scheduling or sync — both eliminated experimentally.
- *The prediction:* measure any machine's tier bandwidths and any model's active bytes, and this
  equation prices its decode speed before you download a single weight.

### Law 4, general form (v1.3–v1.4, 2026-07-24) — a restatement, not a revision

Every formulation below NESTS: the v1.0 statement is the single-dominant-tier special case, v2 adds
the KV term, and the general form covers any tier set. **No measured anchor moved at any step** —
the anchor suite re-proves all of them on every commit.

**General form.** For a machine described as memory tiers *i* (aggregated devices count as one tier:
multi-GPU bandwidth sums x ~0.85 tensor-parallel efficiency [est], striped disks x ~0.75 [est,
from the RAID-0 eta 0.66 datapoint]), and a placement assigning bytes to tiers:

**tok/s = [ Σ_i  bytes-read-per-token from tier i ÷ (η_i · BW_i) ]⁻¹**

where bytes-per-token = always-active weights + routed-expert reads (hit-rate = resident fraction,
by routing flatness, Law 2) + ctx · kv-bytes/pos on KV's tier. Fit is checked per tier with KV
counted. The v1.0 form is recovered when one tier dominates the sum.

**Corollary (tier boundaries).** Speed is a step function of placement, so the marginal value of a
gigabyte is ~zero mid-tier and enormous at a boundary (measured: a one-quant-step shave across our
RAM boundary is worth x4-6). All size levers should be priced by boundary distance.

**Corollary (lever gates).** Lever validity is hardware-conditional — Law 1's shape recurring at the
systems level. Measured example: quantized K-cache costs -83% at 16k depth on Pascal-class GPUs
(no flash attention -> per-token dequant tax) while being a plausible win on FA-capable hardware
[est]. Optimizers over the law must carry measured gates, not assume levers are universal.

### Law 4 v2 — the context term (v1.1, 2026-07-23)

The formulation above is the **short-context law**. u/RogerAI--fyi (Reddit) correctly observed it
omits per-token KV reads: every generated token re-reads the entire KV cache, so at depth the byte
budget gains a second term —

**`tok/s = η(tier) × BW ÷ (active-bytes + ctx × kv-bytes/pos ÷ η_kv-adjusted)`**, where the KV term is
served by **whichever tier KV lives on** (VRAM in hybrid placements, RAM on CPU-only boxes — placement
matters for context too, i.e. the law recurses).

- *The measurement (same 2016 box, warm-up-controlled):* tg32 clean baseline **20.02 ± 0.02** →
  at depth 16384 **16.12 ± 0.06** (**−19.5%**). Pre-registered −8…−15% — scored honestly as a
  **near-miss**: the pure bandwidth term (16384 × 98 KB/pos off the 192 GB/s tier) explains ~70% of
  the slope; the residual is depth-dependent attention *compute* on Pascal. Shipped calibration:
  η_kv ≈ 0.70 (single point — falsify or refine it: `quantprobe bench --depth N --contribute`).
- *Architecture matters:* kv-bytes/pos is per-model (Qwen3-30B 98 KB exact; MLA models ~10× smaller —
  DeepSeek-V2-Lite 31 KB; SWA models slope on global layers only [est]). KV also **consumes capacity**
  on its tier: at 16k the 30B no longer fits a 16 GB box as pure-CPU — the fit-checker knows.
- *The prediction:* CPU-only boxes (KV in RAM at ~45 GB/s) degrade **steeper** with context than
  hybrid placements (KV in VRAM at 192): the law says **−29% at 16k** for 30B-A3B Q4 pure-CPU on
  DDR4-45 (8.0 → 5.7 tok/s) vs −19.5% for the hybrid — pre-registered, unmeasured as of v1.1;
  band ±10 points. A `bench --depth 16384` on any CPU-only box settles it.

---

**The umbrella claim:** at low bits on commodity hardware, *placement beats budget* — which layers get
the bits (Law 3), which tier serves the bytes (Law 4), where rotation is applied (Law 1) — because the
budget itself has no slack left to give (Law 2).
