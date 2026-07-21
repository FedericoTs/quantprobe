# Submission package — PAPER_MOE

Everything needed to get *The KV-Latent is the Bottleneck* out the door. Steps marked **[you]**
are yours (I can't compile/submit for you); the rest is ready to paste.

---

## 1. Recommended venue & timeline

| Step | Venue | Why | Deadline (verify on site) |
|---|---|---|---|
| **Now** | **arXiv** (cs.LG primary, cross-list cs.CL) | Establishes priority + visibility, citable, no gatekeeper. Post first. | none |
| **Primary** | **NeurIPS 2026 ENLSP workshop** (Efficient NLP & Speech) | Best fit: data-free *efficient* MoE quant + mechanism. Lower bar, fast feedback, a real venue line. | ~Sept–Oct 2026 |
| **Stretch** | **MLSys 2027** | Systems+ML; quant+hardware fit; more prestigious, harder. Good if you want a flagship line. | ~Oct–Nov 2026 |

**Recommendation:** arXiv immediately, then ENLSP as the peer-reviewed target (MLSys if you want to aim higher and can wait a cycle). MoE quantization is hot — the KV-latent finding is scoopable, so don't sit on it.

**Double-blind note:** ENLSP/MLSys are double-blind → submit the *anonymized* author block + an anonymized code repo; use your real name only on arXiv (and decide arXiv-before-vs-after per the venue's policy — both allow prior arXiv, neither lets you advertise it during review).

---

## 2. arXiv-ready abstract (plain text — paste as-is)

> Mixture-of-Experts (MoE) language models are large on disk but activate only a small slice of their parameters per token, which makes them attractive for memory-constrained inference -- provided they can be quantized aggressively without collapsing. We study 2-bit post-training quantization of DeepSeek-V2-Lite (15.7B parameters; 64 routed + 2 shared experts per layer; multi-head latent attention), the regime needed to make a 16B MoE fully resident on a 6GB consumer GPU. Uniform 2-bit quantization inflates WikiText-2 perplexity from 6.31 to 18.31 on the full test set, and we show the damage is NOT in the experts. Using a per-tensor, data-free trellis codec (no calibration set, no fine-tuning, no Hessian), protecting only the attention and shared-expert tensors at 4 bits -- a small fraction of parameters -- while leaving the 64 routed experts at 2 bits collapses the gap more than fifteenfold, to 6.96 ppl at 2.49 bits per quantized weight, beating the prior 2-bit result on this model (MxMoE) on gap-ratio (1.10x vs 1.18x) without using any data. A causal decomposition pinpoints the damage in the attention/MLP internal projections, not the residual writers or routing; for multi-head latent attention, two tiny low-rank KV-latent tensors carry 87% of the collapse, and dropping just them to 2-bit inside the recipe costs +5.27 ppl. Expert-routing divergence, the intuitive culprit, is only a 21% symptom. The finding is architecture-general (it replicates on Qwen1.5-MoE). We show the 2-bit weights resident at 5.64GB on a 2016-era GTX 1060, and -- by a sequence of controls -- that single-stream speed is bounded not by the silicon but by the MoE batch-1 memory access pattern.

*(~1,750 chars — under arXiv's ~1,920 limit. Adjust if you trim the paper.)*

---

## 3. arXiv metadata

- **Title:** The KV-Latent is the Bottleneck: Data-Free 2-Bit Mixture-of-Experts Quantization on a 6 GB GPU
- **Authors:** [you] — real name(s) + affiliation
- **Primary class:** cs.LG  ·  **Cross-list:** cs.CL (optionally cs.AR for the hardware angle)
- **Comments:** `N pages, 5 figures. Code & reproduction: <repo URL>`
- **License:** CC BY 4.0 (recommended for max reach) or arXiv default
- **MSC/ACM:** optional

---

## 4. Pre-submission checklist

- [ ] **[you]** Compile the PDF on Overleaf (upload `PAPER_MOE.tex` + a `data/` folder with the 5 figure PNGs); read it end-to-end as a PDF.
- [ ] **[you]** Author info: real name on arXiv; keep the anonymized block for the double-blind venue.
- [ ] Anonymize for review: no name/affiliation/self-identifying URLs in the PDF; anonymized code repo.
- [ ] Page limit: ENLSP ~4–8 pp, MLSys ~10 pp + refs. Trim §3.4/§7 first if over.
- [ ] Add the **AI-assistance disclosure** (`DISCLOSURE.md`) where the venue requires (most now ask).
- [ ] Code release: `REPRODUCE.md` + the harness; anonymized for review, public on arXiv.
- [ ] Supplementary: figures + the run logs (`data/moe_results.txt`, `data/full_*.log`).
- [ ] Run the mock-review rebuttal-prep (incoming) and fold any cheap fixes in before submitting.

---

## 5. The 30-second pitch (for the OpenReview "TL;DR" / your framing)

> Aggressive low-bit MoE quantization fails not in the experts but in the attention/shared *internal* projections — and for multi-head latent attention, in two tiny KV-latent tensors that carry 87% of the collapse. Protecting them (data-free, ~2.5 bits) fits a 16B MoE on a 6GB GPU and beats the calibrated prior result on gap-ratio. The mechanism is *measured*, not assumed: routing divergence, the obvious culprit, is only a 21% symptom.

**Why accept:** (1) a genuinely surprising, *measured* mechanistic finding (KV-latent), not just a recipe; (2) data-free yet beats the calibrated frontier on gap-ratio at a comparable bit budget; (3) honest to a fault — reports a measured-negative speed result and every refuted hypothesis. **Known limitation to own in the rebuttal:** the generative demo uses a third-party codec; we prove residency + quality of our own weights, with the deployable decode kernel as future work.
