# Gemma 4 12B port — first-principles go/no-go (autonomous-loop findings)

Scoping the carve-out/rank-robustness method for **Gemma 4 12B** on a 6 GB GPU, testing our axioms cheaply
before committing to long compression runs. Status: **axioms partially pre-tested on local proxies; the
decisive test is blocked on the gated Gemma weights.**

## Architecture (recon, Phase 0)
- **Dense**, 11.95 B params, **48 layers**, **262K vocab**, 256K context.
- **Alternating attention**: 1024-token local sliding-window layers interleaved with global full-context.
- **Per-Layer Embeddings (PLE)**: per-layer dim 256, packed table ≈ 262K × 48 × 256 ≈ **3.2 B params
  (~27% of the model)** — but PLE is a *memory-optimization by design*: only the current token's per-layer
  embedding is needed, so it is meant to **stream from CPU/disk, not sit in VRAM**.
- Encoder-free multimodal (image/audio projected into embedding space) — out of scope for a text eval.
- Exact hidden_size / head counts: read from `config.json` on download (not disclosed in docs).

## Axiom test results (what the loop established)

| Axiom | Test | Result |
|---|---|---|
| **A1** the dense bulk is 2-bit-compressible (weight) | trellis rel-MSE on local dense tensors | **CONFIRMED but VACUOUS** — the codec whitens every tensor to rel-MSE 0.069 (= D(R=2)); weight-MSE does **not** discriminate fragility |
| **A1′ (redirect)** the dense bulk is 2-bit-tolerant *functionally* | needs a dense model + eval | **OPEN — the make-or-break; needs Gemma weights.** Yellow flag: dense MLPs are more concentrated (flatness 0.38–0.59) and heavier-tailed (kurtosis up to +4.2) than MoE experts (0.57–0.63 / ~0), so dense 2-bit may cost more than our MoE numbers |
| **A2** a small subset carries the collapse (carve-out lever) | flatness/kurtosis ranking + small causal decomposition | data-free part ready (`gemma_probe.py`); functional part needs weights |
| **A3** the harness reproduces fp16 | port + small-subset fp16 ppl vs published | needs port + weights |
| **A4** memory closes at ≤6 GB | budget from config | **likely YES** — PLE (~3.2 B) streams, so resident core is well under 12 B; our "17 B-per-6 GB" ceiling (all-resident MoE) does not apply |

**Key methodological lesson (folded back into the plan):** because the trellis equalizes weight distortion,
the Phase-1 data-free probe must rank by **effective-rank × kurtosis, not weight rel-MSE**. `gemma_probe.py`
implements the corrected version.

## Revised gate sequence (cheapest/most-decisive first; no long run until each passes)
- **Phase 0 (done, ~mins):** config recon + PLE-aware memory budget. → A4 likely green.
- **Phase 1 (ready to run on weights, ~mins, CPU):** `gemma_probe.py` — flatness/kurtosis per tensor type;
  identify Gemma's low-rank/heavy-tailed tensors to protect. Confirms whether a carve-out *candidate set*
  exists. (Weight-MSE dropped — proven uninformative.)
- **Phase 2 (port + fp16 sanity):** `evoq_gemma.py` — see port spec below — fp16 ppl on ~20 windows vs a
  published Gemma-4-12B number. **Gate A3.**
- **Phase 3 (small-eval functional gate, THE decisive one):** on ~20 windows, quantize the dense MLP bulk to
  2-bit (rest fp16) and measure Δppl → answers A1′. If a small carve-out (protect the flagged tensors + keep
  PLE/embeddings at higher bits) recovers the gap → **Gate A2/A1′ pass → commit to Phase 4.**
- **Phase 4 (only then):** full-set compression + rate-distortion sweep.

## Harness port spec (evoq_moe.py → evoq_gemma.py)
1. **Per-layer attention mask** selector: sliding-window (1024) for local layers, full-causal for global,
   keyed off the config's layer pattern. (evoq_moe uses one causal mask.)
2. **PLE**: locate per-layer embedding tensors; keep them **CPU-streamed** (per-token gather), out of the
   VRAM budget; decide their precision separately (candidate for low-bit, but they feed every layer).
3. **Gemma RMSNorm** `(1 + weight)` scaling and any logit soft-capping — subtle, would silently break Gate A3.
4. **Tied embed/lm_head** (one 262K×hidden table) — big; per the embed-harvest result, 4-bit costs ~+0.45,
   so it's a real budget lever, not a footnote.
5. **TARGETS** = `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj` (no experts branch); protect the
   Phase-1-flagged low-rank/heavy-tailed tensors.

## Decision — GREEN (validated on a dense proxy, no download needed)
The make-or-break gate (A1′) was run on the **local dense Qwen2.5-7B** (`dense_2bit_gate.py`):

| dense Qwen-7B (trellis codec, fp16=5.13) | ppl | Δ |
|---|---:|---:|
| attention @ 2-bit | 5.68 | +0.55 |
| MLP @ 2-bit | 6.52 | +1.39 |
| MLP @ 2-bit, **fast codec (no incoherence)** | 28,119 | collapse |

**Dense 2-bit is viable with our incoherence-trellis codec** — the fast-codec collapse was the *missing
rotation* (incoherence rescued the full-rank MLPs 4,300×), validating the rank-conditional law on a third,
dense architecture. **The carve-out inverts vs MoE:** here the **MLP bulk is the main cost (+1.39, but it's
the memory bulk → must be 2-bit)** and **GQA attention is cheaper (+0.55)** — and Gemma is GQA (like Qwen,
not DeepSeek-MLA), so its attention should behave the same.

**Validated dense-Gemma recipe:** 2-bit MLP bulk + attention at 2-bit (+0.55) or 4-bit (cheap, small) + PLE
streamed from CPU + embed/lm_head 4-bit. **Expected whole-model gap ~1.25–1.35×** (worse than MoE's 1.10×
but usable), fits 6 GB, and dense = coalesced batch-1 so speed beats the MoE case.

**Only remaining blocker:** the gated weights. Accept the Gemma 4 license + create an HF token
(`huggingface-cli login`), then I pull to `D:\evo-compress-data\gemma-4-12b` and run Phase 0–3 (the
data-free `gemma_probe.py`, the harness port, the functional gate). *Caveat:* the proxy numbers are on
enwik8 3×1024 (directional); a WikiText-2 baseline on Qwen-7B would make them paper-grade (optional follow-up).
