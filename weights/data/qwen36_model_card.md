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

**A 35B mixture-of-experts model that runs at 11–14.4 tok/s on a 2016 GTX 1060** — a range, not a
number, and [the speed section](#speed-a-range-and-why-it-cannot-be-a-single-number) explains why
that is the honest way to state it.

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

## Speed: a range, and why it cannot be a single number

> **Corrected 2026-08-18.** This card previously said **14.86 ± 0.36 tok/s**. That was a real
> `llama-bench` run with N=5 and a 2.4% error bar — and we could not reproduce it. Six fresh
> runs of the identical command, same box, same binary, same file, never reached it. The
> correction is published here at the same size as the original claim.

On a **GTX 1060 6GB / 16GB DDR4-3000 / i5-7600K**, llama.cpp b10098, `-ngl 12`, tg128:

| | tok/s |
|---|---|
| cold — first runs after the file has been idle | **11.0 – 13.1** |
| warm — after ~5 consecutive runs | **14.2 – 14.4** |
| previously published (unreproduced in 6 attempts) | ~~14.86 ± 0.36~~ |

**Why it moves.** This file is **13.15 GiB** and the machine has **~12.2 GB of free RAM.** The
weights do not fit, so they cannot all stay resident in page cache, and part of every decode pass
streams off disk. Throughput therefore depends on how much of the file the OS is currently
holding — which *climbs across consecutive runs* and collapses when something else needs memory.
The original 14.86 was measured minutes after the quantizer wrote the file, when it was at its
hottest.

Measured, six consecutive runs of one unchanged command: `13.04 → 13.14 → 13.89 → 14.33 → 14.43
→ 14.23`. A 4.68 GB model that *does* fit free RAM, same box same command, holds a **2.1%**
spread with no ramp at all. The instability is the size relationship, not the machine.

**What this means for you.** With **more than ~14 GB free RAM**, expect the warm figures and a
stable number. With less, expect it to move, and expect your first run to be the slowest one.

```bash
llama-server -m Qwen3.6-35B-A3B-depthaware-Q2K.gguf -ngl 12
```

**Use `-ngl 12` on a 6 GB card.** Pushing more layers onto the GPU is *3× slower* here (4.84 at
`-ngl 24`) — a VRAM-overcommit cliff, not a gentle curve. CPU-only measures 7.40 ± 2.09.

**Do NOT try to "warm the cache" by reading the file first.** We tested it: `cat`-ing the whole
13.15 GiB before benchmarking gave **11.89 tok/s**, *worse* than the 14.43 the box had just
reached, and 1.95 below the six-run mean. A file bigger than RAM cannot be held whole, so a
sequential read leaves the cache holding the file's last ~12 GB, while real runs leave it holding
the pages the model actually re-reads — the hot experts. Priming swaps a frequency-adapted cache
for a position-adapted one. Just run it twice instead.

**One number we are still NOT publishing.** A different placement (`-ngl 99 -ot exps=CPU`)
predicted 22.7 tok/s and measured 9.94 **± 5.40** — an error bar over half the value. That
measurement is unusable, so the prediction it was meant to test is recorded as *unscored*, not as
a hit or a miss.

Full record, including the two predictions that went against us:
[prereg #106](https://github.com/FedericoTs/quantprobe/blob/master/preregistrations/2026-08-18-is-the-headline-reproducible.md)
(scored 2/4) and [prereg #105](https://github.com/FedericoTs/quantprobe/blob/master/preregistrations/2026-08-18-published-speed-vs-experienced-speed.md)
(VOID — its premise died at its own reference arm).

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
