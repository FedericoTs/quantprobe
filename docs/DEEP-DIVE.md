# Deep dive — positioning, parity tables, the 744B projection, and the repository map

## What's actually new here — and what isn't

**Not mine (I build on it, gratefully):** [llama.cpp](https://github.com/ggml-org/llama.cpp) and its k-quants; the incoherence-codec line (QuIP#/QTIP/QuaRot); [colibri](https://github.com/JustVugg/colibri)'s tier-streaming engine, which inspired the streaming experiments.

**Mine (measured here, to my knowledge first):**
1. The four laws — rank-conditional rotation, density-everywhere, probe-not-predict fragility, and the tiered decode equation with fitted η bands.
2. **Probe-then-quantize** as a method + the `quantprobe` tool implementing it end-to-end.
3. The **byte-identical placement experiment** (same size, 2.25 ppl apart) — the cleanest control I've seen for placement effects.
4. **Pre-registration as methodology** for systems benchmarks (predict → then measure, in public).
5. The depth-aware GGUF recipes, the placement solver (forward + inverse), and the live self-scoring dashboard.


## Where I stand at parity — same hardware, same model, same bytes

Head-to-head under identical conditions (my box, WikiText-2, same eval windows):

| comparison at parity | baseline | this work | delta |
|---|---|---|---|
| **Placement only** (Gemma 4 12B, byte-identical 5.22 GB files) | first-12 protected: 12.27 ppl | last-12 protected: **10.02** | **−2.25 ppl, same bytes** |
| **vs llama.cpp naive best** (Qwen3-30B, same GGUF, same box) | pure CPU: 12.6 tok/s | planned hybrid: **19.3** | **+53%, zero cost** |
| **Data-free vs calibrated** (Qwen3-30B, Q2-class) | imatrix-calibrated community: 11.27 ppl | data-free depth-aware: **11.08** | parity **without calibration data** (+15% size) |
| **vs calibrated SOTA** (DeepSeek-V2-Lite, 2-bit) | MxMoE (calibrated): 1.18× gap | data-free carve-out: **1.10×** | better, with zero data |
| Uniform vs depth-aware (Gemma, 2-bit class) | uniform Q2_K: 14.41 ppl | depth-aware: **10.02** | **−4.4 ppl for +0.5 GB** |

**And colibri?** No parity comparison is possible or fair — different hardware ($16k tiers vs my $0-upgrade desktop), different model (744B vs my largest, 110B). What I can say honestly: normalized by the law, colibri's published tiers land **inside my measured η bands** (his 0.48 and 0.88) — same physics, complementary work — and my concrete, falsifiable offer stands: a probed 2-bit expert tier should give **~2× on its disk-bound tiers** and ~1.5–1.7× on RAM tiers, quality held by keeping the fragile band at int4.


## Projection: running the 744B locally

The question colibri made everyone ask: *what would GLM-5.2 (744B-A32B) cost to run at home?* The law answers it per hardware class and placement strategy — same equation, same η bands, error bars ±25–40% at this extrapolation distance:

| setup | strategy | predicted tok/s |
|---|---|---|
| My 2016 desktop (16 GB, SATA) | probed 2-bit, naive streaming | **~0.07** — it *runs*; that's the whole claim |
| My desktop + NVMe (~€180 today) | probed 2-bit, naive streaming | **~0.5** — demo class |
| 128 GB DDR5 desktop | colibri engine, int4 (its published measurement) | 1.8 |
| 128 GB DDR5 desktop | colibri + **probed 2-bit experts** (my open, falsifiable offer) | **~3.5** |
| 256 GB used workstation, ~200 GB/s (Epyc/Threadripper, ~€2–3k) | probed 2-bit, RAM-resident hybrid | **~9** — the cheapest *usable* 744B |
| 512 GB Mac Studio (~800 GB/s unified) | probed 2-bit, resident | **~40–50** |
| 4× DGX Spark, TP4 (measured by [tonyd2wild](https://github.com/tonyd2wild/GLM-5.2-NVFP4-KV-4x-DGX-Spark-300kctx-42tok-s)) | W4 + NVFP4 KV | 42.5 |
| 4× DGX Spark + **this work's recipe** (probed 2-bit experts, 4-bit attention) | active bytes 20.7 → 12.9 GB/token (×1.6) | **~55–67 predicted** — or the same 42.5-class speed on **2 Sparks (~half the cost)**, or several-fold more KV/context |

Three honest caveats: (1) 2-bit quality on a 744B is *itself* a probe-first question — the fragility atlas says find the fragile band before trusting any recipe, and MoEs of this class have absorbed 2-bit at ~1.10× so far; (2) the streaming rows assume naive LRU — colibri-style lookahead prefetch (91–99% predictable, measured) is exactly what closes the gap between my naive-streaming numbers and its engine's; (3) the biggest model I have *measured* is 110B — everything above it is the law extrapolating, which is precisely what the pre-registration culture here is for: these numbers are on the record before anyone runs them.

<p align="center"><img src="weights/data/x_chart_G_744b.png" width="720" alt="Running a 744B at home: cost versus speed, measured points versus pre-registered predictions, with the placement dividend shown at fixed cost"></p>

<p align="center"><img src="weights/data/x_chart_H_laguna.png" width="720" alt="The tiered decode law predicted Laguna S 2.1 decode within 1% from its config alone; the spec-decode x MoE antagonism explains the decay under load"></p>

> The day after Laguna S 2.1 (117.6B MoE) launched, I predicted its single-Spark decode from the config alone — **~47 tok/s base, matched within 1%** by the published ×2 per-stream number — and the load-decay curve is the [spec-decode × MoE antagonism](../LAWS.md) made visible. Three independent GB10 measurements, three models, one η.


## Repository map

| path | what |
|---|---|
| [LAWS.md](../LAWS.md) | the four laws, each with measurement + falsifiable prediction |
| `weights/PAPER_MOE.md` / `.tex` | the paper — mechanism, laws, atlas, scaling law |
| `weights/quant_probe.py` | probe-then-quantize CLI (GGUF → fragility curve → recipe) |
| `weights/GGUF_DEPTH_RECIPE.md` | copy-paste llama.cpp recipes + the fragility atlas |
| `weights/scaling_law.py` · `make_*_chart.py` | the η fit and every chart |
| `docs/simulator.html` | the interactive calculator (also served via GitHub Pages) |
| `weights/*.py` · `weights/data/*.log` | every harness, and the raw log behind every number |
| `weights/REPRODUCE.md` | claim → script → log manifest + the bench protocol |
| `README_lossless_spike.md` | the project's first thread (evolutionary lossless compression) |


## Reproduce

Every headline number has its generating script and raw log in-tree — see [weights/REPRODUCE.md](../weights/REPRODUCE.md). The streaming harnesses quantize and evaluate models larger than VRAM layer-by-layer; nothing here needs more than a 6 GB GPU, 16 GB RAM, and patience.

