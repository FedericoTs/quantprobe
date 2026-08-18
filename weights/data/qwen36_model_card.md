---
base_model: Qwen/Qwen3.6-35B-A3B
tags:
- gguf
- quantized
- moe
- llama.cpp
- quantprobe
library_name: gguf
pipeline_tag: text-generation
---

# Qwen3.6-35B-A3B — depth-aware 2-bit GGUF (14.1 GB)

**A 35B mixture-of-experts model that runs at 14.86 tok/s on a 2016 GTX 1060.**

Most low-bit quants decide which layers to protect by convention. This one was built from a
**measurement**: every depth band of this specific model was pushed to 2-bit one at a time and
scored on held-out text, and the bits went where the damage actually was.

The measurement, the comparison that justifies it, and the raw logs are all linked below. If
you only read one line: at **byte-identical file size**, putting the protection where the probe
said removed **29% of the quality loss** versus spreading the same protection evenly.

---

## What was measured

**Where this model breaks.** Each band of 10 layers quantized to Q2_K in turn, rest at Q6_K,
scored against a Q6_K reference (PPL 5.4669) on held-out WikiText-2:

| layers | Δ perplexity |
|---|---|
| 0-9 | 0.0303 |
| 10-19 | 0.1429 |
| 20-29 | 0.1869 |
| **30-39** | **0.4179** ← fragile, 2.53× the median |

Monotone, back-heavy. The last quarter of the network is where 2-bit hurts, so that is what
this build protects at Q4_K while the rest of the expert FFNs go to Q2_K.

Pre-registered before the probe ran: [prereg #103](https://github.com/FedericoTs/quantprobe/blob/master/preregistrations/2026-08-18-qwen36-hybrid-moe-fragility.md) ·
raw log: [`prereg103_probe_qwen36.log`](https://github.com/FedericoTs/quantprobe/blob/master/weights/data/prereg103_probe_qwen36.log)

**Does that actually pay?** Tested against a deliberately strong control — the same ten layers
protected at the same tier, but spread evenly across depth (0, 4, 8 … 36) instead of on the
measured band. Both files are **byte-identical: 14,115,658,720 bytes**.

| build | PPL (32 chunks, held-out WikiText-2) | Δ over reference |
|---|---|---|
| **this model** (band 30-39) | **5.7796** | +0.3127 |
| control (evenly spread) | 5.9088 | +0.4419 |

**29.2% less quality loss, at zero byte cost, with decode speed unchanged (1.22% apart).**

Pre-registered before either arm was scored, including the size gate that rejected the first
control for being 9% smaller: [prereg #104](https://github.com/FedericoTs/quantprobe/blob/master/preregistrations/2026-08-18-qwen36-recipe-vs-naive.md) ·
raw log: [`prereg104_ppl.log`](https://github.com/FedericoTs/quantprobe/blob/master/weights/data/prereg104_ppl.log)

---

## Speed, measured not estimated

On a **GTX 1060 6GB / 16GB DDR4-3000 / i5-7600K**, llama.cpp b10098, tg128, N=5 reps:

| placement | tok/s |
|---|---|
| `-ngl 12` (recommended) | **14.86 ± 0.36** |
| `-ngl 0` (CPU only) | 7.40 ± 2.09 |
| `-ngl 24` | 4.84 ± 0.60 |

**Use `-ngl 12` on a 6 GB card.** Pushing more layers onto the GPU is *3× slower* here — a
VRAM-overcommit cliff, not a gentle curve. Your optimum will differ with your VRAM; find it
before assuming higher is better.

```bash
llama-server -m Qwen3.6-35B-A3B-depthaware-Q2K.gguf -ngl 12
```

**One number we are NOT publishing.** A different placement (`-ngl 99 -ot exps=CPU`) predicted
22.7 tok/s and measured 9.94 **± 5.40** — an error bar over half the value, because a 14 GB
file pinning host memory on a 16 GB box thrashes. That measurement is unusable, so the
prediction it was meant to test is recorded as *unscored*, not as a hit or a miss. It will be
re-run on hardware where the placement is stable.

---

## Honest limits

- **Perplexity is a proxy, not a task score.** 32 chunks of WikiText-2 ranks two builds; it does
  not tell you this model is good at your work. No task benchmarks (MATH, GSM8K, IFEval) have
  been run on this build yet.
- **One box, one eval, one machine state.** Everything above was measured on a single 2016
  desktop. Different hardware will give different speeds and may give a different `-ngl` optimum.
- **2-bit is 2-bit.** The reference this is measured against is Q6_K, and the gap is real
  (+0.31 PPL). This build exists for people who otherwise could not run a 35B at all. If a
  Q4_K_M fits your machine, run that instead.
- **The fragile band is measured for *this* model.** A sibling check found Qwen3.5-35B-A3B
  fragile in the same band (30-39), which is evidence recipes survive a version bump — but that
  is two models, not a law.

## Provenance

Built with [quantprobe](https://github.com/FedericoTs/quantprobe) from `Qwen3.6-35B-A3B` Q8_0:

```bash
quantprobe quantize --gguf Qwen3.6-35B-A3B-Q8_0.gguf --recipe qwen3.6-35b
```

The recipe is a committed JSON with its own evidence:
[`quantprobe/recipes/qwen3.6-35b.json`](https://github.com/FedericoTs/quantprobe/blob/master/quantprobe/recipes/qwen3.6-35b.json)

Tensor layout: attention and SSM at Q4_K · shared experts at Q8_0 · expert FFN layers 0-29 at
Q2_K · **expert FFN layers 30-39 at Q4_K** · token embedding Q4_K.

Every claim on this page regenerates from committed data by committed code. Predictions here
were staked before measurement, and the misses are published at the same size as the hits —
including the unscored one above.

**License:** inherits the base model's license from `Qwen/Qwen3.6-35B-A3B`. Check the upstream
repository before commercial use.
