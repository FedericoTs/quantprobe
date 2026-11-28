# Pre-registration #115 — should a small GPU's VRAM hold whole layers of experts, or a cache of the hot ones?

**Staked 2026-11-27, before any arm ran.**

## Why this one

When a mixture-of-experts model's experts do not fit in VRAM, quantprobe today recommends a
**static split**. The experts of the first K layers live in VRAM and the rest in pinned RAM, and
the CPU computes every expert of every other layer, every token. In November llama.cpp merged an
alternative, the **MoE cache** (`--moe-cache-mib`, PRs #29887 and #30966). It keeps host-resident
experts in RAM and holds the most used ones in a VRAM cache. A miss is uploaded into the cache or,
with `--moe-cache-cpu-share F`, computed on the CPU.

quantprobe's planner still prints that an expert cache "needs an expert-caching runtime … stock
llama.cpp gets the RAM-cache row". Since b12815 that is false. Whether the cache *pays* is a
different question: per byte of VRAM it serves only the experts the model actually uses, but every
cached layer costs a host round trip per token to learn which experts were selected. Its own
tuning script runs only after download, starting the server six times. The opening for a planner
is the decision **before** download.

## Machine and model

- **Machine:** GTX 1060 6 GB (Pascal, PCIe 3.0 x16), i5-7600K (4 cores), 16 GB DDR4-3000,
  Windows 10.
- **Runtime:** llama.cpp b12815, CUDA 12.4 build.
- **Model:** `Qwen3.6-35B-A3B-OURS-depthaware.gguf`, 14,115,658,720 bytes: 40 MoE layers, 256
  experts each, 8 used per token.
  - Layers 0–29 hold Q2_K experts, 252 MiB per layer.
  - Layers 30–39 are the protected band, Q4_K experts at 432 MiB per layer.
  - That gives the cache two groups.
- **Baseline already on record (C-32):** this model decodes at 11.0–14.4 tok/s on this machine
  under the older `-ngl 12` placement. quantprobe's current plan predicts 23.4.

## Design

Four arms. Each is one `llama-completion` process that exits by itself, so llama.cpp's cache
prints its own counters when the context closes. A killed server never does on Windows.

**Common to all four:**
- `-ngl 99 -lm none -b 1024 -ub 1024 --threads 4 -c 2048`.
- 512 generated tokens at `--temp 0.7 --seed 42 --ignore-eos`, raw completion (`-no-cnv`).
- The prompt is the first 1,669 characters of WikiText-2 test (`weights/data/prereg115_prompt.txt`).

| arm | experts in VRAM | cache | CPU share |
|---|---|---|---|
| **A** | layers 0–8 (2,268 MiB) | none | — |
| **C1008** | layers 0–4 (1,260 MiB) | 1,008 MiB | 0 |
| **C504** | layers 0–4 | 504 MiB | 0 |
| **C1008s** | layers 0–4 | 1,008 MiB | 0.95 |

- **Arm A** is exactly the run line `quantprobe plan` prints for this model on this machine.
- **C1008 and C1008s** spend the same 2,268 MiB of VRAM on experts as A, with fewer layers plus
  a cache. So A against them is an **equal-VRAM** comparison.
- **C504** exists to test prediction across cache sizes.
- **Out of scope here:** the "all experts in RAM" configuration llama.cpp's docs lead with (`-cmoe`)
  would pin 11.6 GiB on a 16 GB machine. Its unpinned form would page-fault from a 0.47 GB/s disk.
- **Ordering:** three repetitions, arms interleaved, so drift lands on every arm alike.
- **Recorded per run:** decode tok/s from llama.cpp's own timing line, the cache's logged
  counters, and the routing trace (`LLAMA_MOE_CACHE_TRACE`).
- **Tools, committed with this file:**
  - `weights/prereg115_run.py` runs the arms.
  - `weights/prereg115_score.py` scores. Its self-check gives 7 fixtures and 7 distinct verdicts.
  - `weights/moe_cache_replay.py` is a port of llama.cpp's cache policy (`src/llama-moe-cache.cpp`,
    b12815). It replays a trace at any size and share.

## Predictions

- **P-1: the lever.** The faster of the two equal-VRAM cache arms decodes at least **1.20×** as
  fast as A (median of three runs each).
  - **If it holds,** the planner integration is staked: the planner recommends the cache where
    the build has it, and corrects the stale row.
  - **If it misses with a ratio of at least 1.0,** the cache helps but not enough to change the plan.
  - **Below 1.0,** the static split wins on this machine and the plan stands.
- **P-2: the port is exact.** For every cache run, replaying its own trace at its own size and
  share reproduces llama.cpp's logged counters within **0.1%** of the lookups: hits, misses, misses
  on the CPU, and MiB uploaded. If this holds, a hit-rate curve computed from one trace is what
  llama.cpp would have done.
- **P-3: one trace predicts the other configurations.** C1008 rep 1's trace, replayed at 504 MiB,
  predicts C504's logged hit rate within **±2 points**. Replayed at share 0.95, it predicts
  C1008s's within ±2 points. Sampled decoding can diverge between arms, because the CPU and GPU
  paths differ numerically, so this is a real test, not P-2 again.

Published without a staked threshold: every arm's median tok/s, the cache arms' hit rates, and the
hit-rate curve of one trace from 126 MiB to 4 GiB.

## What would make this VOID rather than scored

- Any arm with fewer than two completed runs, or a cache run without its trace or counters.
- Another llama.cpp process sharing the GPU. The runner refuses to start if one exists.

## What this cannot show

- **One machine and one model.** A 16 GB RAM box limits the configurations. On a 32 GB machine
  the all-in-RAM arm becomes possible and the comparison may move.
- **Speed here is decode only.** Prefill batches over 32 tokens bypass the cache by design.
- **A bandwidth model of the cache arms is not staked.** C-32 already shows Law 4 over-predicting
  this placement on this machine by about 60%, and the cache adds a per-layer host round trip that
  the law does not price. Modelling that comes after this measures what dominates.

## Amendment, before any scored arm ran

Pilots of arm C1008, with 16 and then 4 generated tokens, checked the harness. Their numbers are
not scored.

- `llama-completion` exits by itself, and its timing line parses.
- The cache prints its slot and counter lines only at log verbosity 4, the level llama.cpp's own
  tuner uses. **`-lv 4` is added to every arm.** It adds start-up and exit lines and nothing per
  token: a 4-token log had 240 lines, all of them one-off.
- The 4-token pilot's trace, replayed by the port at 1,008 MiB and share 0, reproduced llama.cpp's
  own line exactly: 606 and 241 slots, 325 hits, 1,064 misses, 1,275.19 MiB uploaded. P-2 is still
  scored on the full runs.
- llama.cpp's `--fit` reported the cache arm 500 MiB short of its default 1 GiB free-memory
  margin. It changed nothing, because `-ngl` is set explicitly; VRAM use was 4,673 of 6,143 MiB.
