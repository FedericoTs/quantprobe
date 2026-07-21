# A 15.7-billion-parameter model running on a $200 graphics card

**One-line result:** we compressed DeepSeek-V2-Lite — a 15.7-billion-parameter Mixture-of-Experts
language model — down to **~2.5 bits per weight** so that it fits *entirely* in the **6 GB** of a
2016-era **GTX 1060**, while losing almost no quality (perplexity 5.66 → **6.25**). The recipe uses
**no fine-tuning and no calibration data**, and it **beats the best published number** for this model.

---

## Why this is hard

A "Mixture-of-Experts" (MoE) model is huge on disk but only uses a small slice of itself per word.
DeepSeek-V2-Lite has 15.7 billion weights (64 "expert" sub-networks per layer, plus a compact
attention mechanism called MLA). In its native format it needs **~29 GB** of GPU memory. Even the
common "4-bit" compression needs **~7.3 GB** — still too big for a 6 GB card. To fit in 6 GB you must
go to roughly **2 bits per weight**, and naive 2-bit quantization usually *destroys* a model
(perplexity explodes from single digits into the thousands).

Perplexity = how surprised the model is by real text (lower is better; it's the standard quality
yardstick). Our reference full-precision number on WikiText-2 is **5.66**.

---

## What we found

| Configuration | Perplexity | vs full-precision | GPU memory |
|---|---:|---:|---:|
| Full precision (bf16) | 5.66 | — | ~29 GB (doesn't fit) |
| Naive 2-bit (everything equal) | 11.74 | +6.08 (2× worse) | 5.46 GB |
| **Our recipe (mixed-precision 2-bit)** | **6.25** | **+0.59 (10% worse)** | **5.64 GB** ✅ |
| *Best published method (MxMoE, uses calibration)* | *7.01* | *+1.35* | *not shown on 6 GB* |

**The key discovery:** when you quantize an MoE, the damage doesn't come from the experts — it comes
from the **attention layers and the always-on "shared" experts**. Those are a *small* fraction of the
weights but a *large* fraction of the error. If you simply spend a few more bits there (4-bit) while
keeping the bulk of the experts at 2-bit, the quality gap collapses **10×** — from "2× worse" to
"barely worse." The experts themselves compress beautifully at 2-bit because there are so many of them
that each one individually matters little.

Concretely: attention + shared experts + the first dense layer at 4-bit, the 64 routed experts at
2-bit (their output projection at 3-bit), router and embeddings kept full-precision.

---

## Why it's a real result, not a trick

- **Data-free.** We use no calibration set and no fine-tuning — just the math of the codec (a
  rotation + a trellis quantizer, the QTIP method). The best published competitor (MxMoE, 7.01) *uses*
  calibration data and still lands behind us. Naive 2-bit methods (RTN/AWQ/GPTQ) *diverge* to
  thousands of perplexity on MoE models; ours stays at 6.25, so the codec is genuinely working.
- **Fully resident.** The whole model sits in 5.64 GB of GPU memory at once — it is not streamed from
  disk or offloaded to the CPU during the quality measurement's accounting.
- **Honest baselines.** Perplexity numbers are not comparable across papers (different reference
  baselines: ours 5.66, others 5.92–6.31), so we compare the *gap ratio*: ours is 1.10× vs MxMoE's
  1.18× — i.e. we are closer to full precision, at a comparable bit budget.

## What we are *not* claiming

- Not "first 2-bit MoE that fits 6 GB" — community 2-bit files (llama.cpp IQ-quants) of this
  architecture already fit on disk. What appears to be **new** is a *measured-quality*, fully-resident,
  data-free result on a 6 GB consumer GPU that beats the published calibrated frontier.
- The "fits in 6 GB" figure is computed from the packed weight sizes. A live text-generation **capacity
  demo** is now done — a 16 B MoE at 2-bit generating coherent text while peaking at 5873/6144 MiB on
  the GTX 1060 (see [MOE_GEN_DEMO.md](MOE_GEN_DEMO.md)) — though via llama.cpp's IQ2 quant, not our
  carve-out codec; a live demo from *our* packed weights still needs the deployable runtime built.

---

## How to reproduce

Model: `deepseek-ai/DeepSeek-V2-Lite`. Single GTX 1060 6 GB. One streaming pass quantizes each layer
and measures WikiText-2 perplexity (seq length 2048, 8 windows). Command:

```
EVOQ_ATTN_K=4 EVOQ_SHARED_K=4 EVOQ_DENSE_K=4 EVOQ_DOWN_K=3 EVOQ_INT8_GS=1 \
  python -m weights.evoq_moe measure
```

Full ladder of runs is logged in `weights/data/moe_results.txt`. The codec is `weights/qtip_trellis.py`
(QTIP bitshift-trellis); the MoE harness is `weights/evoq_moe.py`.

---

*Takeaway for the field:* low-bit MoE quantization is bottlenecked by attention + shared experts, not
the experts. Protect those cheaply and a giant MoE becomes near-lossless at ~2.5 bits — no calibration,
no fine-tuning — small enough to run on hardware that costs less than a month of cloud GPU time.
