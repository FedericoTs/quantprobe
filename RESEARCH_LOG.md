# evo-compress — Research Log & Project Map

A consolidated map of the whole project: the research threads, the strategies tried, the
headline results, and where each is documented. This is the navigable index over the detailed
records (which are preserved). Newest thread last.

---

## Lineage (one project, four threads)

```
lossless compression spike  ─►  context-mixing codec  ─►  lossless LLM-weight delta codec
       (evocompress/)                 (cmcore/)                  (wcodec / EVOLUTION_LOG)
                                                                          │
                                                                          ▼
                                          lossy LLM quantization (signed-Hadamard + AWQ + trellis/ECVQ)
                                              ├─ ~3-bit PTQ, evolved + audited   (PAPER_DRAFT.md)
                                              └─ 2-bit MoE carve-out, KV-latent  (PAPER_MOE.*) ◄ CURRENT
```

The evolvable-search philosophy (un-gameable verifier, held-out metric, honest accounting,
report negatives) carried through every thread.

---

## Thread 1 — Lossless compression pipeline spike (`evocompress/`)
**Goal:** can AlphaEvolve-style evolutionary search find lossless preprocessing *pipelines*
(transforms → entropy backend) that beat strong baselines? North star: Hutter Prize.
- **Result (GO):** on time-series, evolved `float_split(f4) → lzma-4` beats zstd-19 by +20.7%,
  lzma-9 by +6.4%, byte-exact. The preprocessing — not the codec — is the win.
- **Docs:** `README.md` (full writeup), `RESULTS.md`, `experiments/EXPERIMENT_LOG.md`, `results/`.

## Thread 2 — Context-mixing codec for the prize bridge (`cmcore/`)
**Goal:** the "if GO, then…" bridge to the Hutter Prize — evolve the **Rust source** of a bitwise
context-mixing arithmetic coder (no stored weights; size = |compressed|+|decompressor|).
- **Result:** 30 LLM evolution steps: 2.20 → **1.6749 bpc** on a 16 MB enwik8 slice (4.78×),
  beating lzma-9/zstd-19/brotli-11. Frontier (cmix ~1.0) needs a trained LSTM (out of scope).
- **Docs:** `README.md` (cmcore section), `experiments/HUTTER_NOTES.md`.

## Thread 3 — Lossless LLM-weight delta codec (`wcodec`, `EVOLUTION_LOG.md`)
**Goal:** best-in-world *lossless* delta/checkpoint codec. Single-model is an entropy wall
(~30% bf16); the open frontier is the **delta** between related models.
- **Strategies tried (KPIs in `EVOLUTION_LOG.md`):** raw/split/smart per-plane entropy coders
  (~33% bf16, the wall) → **delta vs reference** (XOR) → **low-rank delta** (adaptive numerical rank).
- **Headline results (byte-exact):** single-model 32.7% (beats ZipNN); checkpoint delta 61.3% bf16
  (≈ published 62%); **abliteration low-rank delta 97.6–99.1% (42–109×)**; scaled to 3B.
- **Docs:** `EVOLUTION_LOG.md`, `EVOLUTION_R13-21.md`, `RESULTS.md`, `ABLITERATION_DETECT_RESULTS.md`,
  `wcodec.py`, `codec_zoo.py`.

## Thread 4a — ~3-bit LLM quantization, evolved + audited (`PAPER_DRAFT.md`)
**Paper:** *"An LLM as the Mutation Operator: Evolving an Honestly-Accounted Rate-Distortion
Frontier for ~3-Bit LLM Weight Quantization."*
- **Method:** LLM as the in-loop mutation operator; un-gameable verifier (held-out enwik8 ppl +
  honest bits/weight from an actually-decoded container + decode-cost class). ~30 attack rounds +
  hostile multi-agent audits.
- **Champion:** per-group signed-Hadamard incoherence rotation + AWQ scaling + **ECVQ** (entropy-
  constrained scalar quant) + entropy-coded indices + 0.5% fp16 outliers. Qwen2.5-0.5B: ~3.00
  honest b/w at ppl ~4.58 (fp16 3.944); Pareto-dominates the best E8 lattice / trellis points.
- **Docs:** `PAPER_DRAFT.md`, `PAPER.md`, `paper_sec_01..06.md`, `paper_facts.md`,
  `QUANT_EVOLVE_RESULTS.md`, `ROADMAP_R25{,_v2}.md`, `ROADMAP_SPEED.md`, `ROADMAP_DECODE.md`,
  `TRELLIS_RUNTIME_RESULTS.md`, `VALUE.md`. Codec: `qtip_trellis.py`, `wcodec.py`, `trellis_*.py`,
  `noise_shaping.py`, `quant_dataaware.py`. Artifact: `data/qwen05b.evoq` (+ `.json`).

## Thread 4b — 2-bit MoE carve-out (CURRENT, publication-ready) — `PAPER_MOE.{md,tex}`
**Paper:** *"The KV-Latent is the Bottleneck: Data-Free 2-Bit Mixture-of-Experts Quantization on a
6 GB GPU."* Reuses the validated trellis codec from 4a, applied to MoE.
- **Headline (full WikiText-2 test set):** DeepSeek-V2-Lite carve-out **6.96 ppl, gap-ratio 1.104×**
  at 2.49 quantized b/w, **data-free**, beats MxMoE (1.184×, calibrated); fp16 6.31 = SINQ's 6.31
  (external validation). Qwen1.5-MoE replicates (1.074×).
- **Mechanism (causal decomposition):** the damage is the attention/shared **internal projections**,
  not the experts/routing; for MLA the **two KV-latent tensors carry 87% of the uniform collapse**,
  and dropping just them to 2-bit inside the carve-out costs **+5.27 ppl** (operating-regime confirm).
  Routing divergence is a 21% symptom (forced-routing control).
- **Speed:** measured negative — single-stream 5–6 tok/s bounded by the MoE batch-1 access pattern,
  not silicon (4 llama-bench controls). Paper-2 "fitting AND fast via routing locality" **refuted**
  (working set ~51/64 experts within 32 tokens).
- **Verification:** audit vs logs + cache re-exec + **from-scratch re-quant (6.2483 exact)** +
  cross-paper protocol workflow (MxMoE evals at seqlen 4096; MC-MoE/EAQuant don't eval this model).
- **Status:** publication-ready. Remaining: C1 (generative decode from our own packed weights —
  needs a deployable trellis kernel; future work), Overleaf compile, author info.
- **Docs:** `PAPER_MOE.md` (+ `.tex`), `DISCLOSURE.md`, `REPRODUCE.md`, `MOE_2BIT_RESULTS.md`,
  `MOE_GEN_DEMO.md`. Harness: `evoq_moe.py`, `evoq_moe_qwen.py`, `forced_routing.py`,
  `forced_output{,_qwen}.py`, `route_locality.py`, `m1_kvlatent.py`, `make_*.py`. Master results:
  `data/moe_results.txt`; figures `data/fig_*.png`; run logs `data/{full,fulldecomp,verify}_*.log`.

---

## Standing methodology (applied throughout)
- **Un-gameable verifier / hard gate:** byte-exact round-trip (codec) or honest streaming ppl (quant).
- **Held-out metric, honest accounting:** no hidden side-info; report bits actually decoded.
- **Leverage every finding, including negatives:** each cheap test redirects the next step. Examples
  that overturned a plausible narrative: routing-as-mechanism (16→21% only) → writers (45→49%) →
  internal/KV-latent (87%); paper-2 locality refuted; "free VRAM speeds it up" refuted.
- **Adversarial verification before committing:** hostile multi-agent audits (Thread 4a), red-team
  proofread + reviewer pass (Thread 4b).

## Test record (where the raw numbers live)
- Codec KPIs: `EVOLUTION_LOG.md`, `EVOLUTION_R13-21.md` (every variant, timestamped).
- Quant frontier: `QUANT_EVOLVE_RESULTS.md`, `data/sens_db.json`, `data/alloc_*.json`.
- MoE: `data/moe_results.txt` (master), `data/{full_*,fulldecomp_*,verify_*}.log` (per-run).
- Persistent agent memory: `~/.claude/.../memory/evo-compress-project.md` (full chronological log).
