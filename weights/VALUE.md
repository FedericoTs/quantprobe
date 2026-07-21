# Where the money is: economics of lossless lifecycle compression

A sober, numbers-first case for *who pays* and *how much*. Ratios are from our measured,
byte-exact results; scale assumptions are conservative and labeled.

## 1. Training-checkpoint storage (the frontier-lab pain)

A checkpoint = **weights + optimizer state**. For Adam the optimizer is `m` + `v` ≈ **2×
the weights** (often fp32 even when weights are bf16), so a checkpoint is **~3× the model
size** (or ~6× if weights are stored fp32 too).

| Run | params | 1 checkpoint (w+opt, bf16) | checkpoints/run (conservative) | run total | with our delta-chain (~50%)* |
|---|---:|---:|---:|---:|---:|
| 7B | 7e9 | ~42 GB | 100 | 4.2 TB | **~2.1 TB** |
| 70B | 7e10 | ~420 GB | 100 | 42 TB | **~21 TB** |
| 405B | 4e11 | ~2.4 TB | 100 | 240 TB | **~120 TB** |

\* Our **measured** full-checkpoint (weights+optimizer) delta-chain on a real training run is
**46–55%** (bf16); late-training / fine-tune checkpoints (smaller gradients) compress more.
A large lab runs *many* such runs concurrently → **petabytes**. At cloud object-store rates
(~$0.02/GB-month), 1 PB ≈ **$240k/yr**; halving it is **~$120k/yr per PB**, before egress
and replication (which deltas also shrink). This is real but **not, by itself, acquisition-
sized** — it's an internal cost line. Its value is *strategic* only if bundled with #2.

## 2. The throughput angle (what actually makes it infra-grade)

Checkpointing stalls training: GPUs idle while TB are written. If compression is **fast
enough to overlap with compute**, the same storage budget buys **more frequent checkpoints
→ better fault tolerance → less recompute after a failure**. On a 1000-GPU run, one avoided
multi-hour restart is worth more than a year of the storage savings. *Throughput*, not ratio,
is the lever for frontier labs — so our next milestone is **MB/s on the critical path**, not
another point of ratio. (Today: enc ~9–32 MB/s at zstd-19, dec ~80–180 MB/s; a fast level or
GPU encode is the path to overlap.)

## 3. Model-hub / registry storage (the broader, more realistic acquirer set)

HF hosts **>1M models**, the vast majority derivatives of a few dozen bases. Our measured
variant deltas: **bf16 99.1%, int8 97.4%, fp8 92.3%** — a variant costs **~1–8% of a model**
to store. If even 30% of the hub is delta-representable against a resident base, hub storage
(and the bandwidth of every download) drops **~10×**. Buyers who feel this: **HF, the cloud
providers who host model weights, and serving companies** (Together/Fireworks/Baseten/
Replicate). This is the path with a real product wedge and a plausible acquisition.

## 4. Multi-tenant serving (the new frontier we uniquely enable)

Keep **one base resident**; serve any **exact** variant by applying its ~1% delta on demand.
**Measured (materialize_poc.py, real Qwen abliteration):** the delta is 9 MB (0.9% of the
988 MB model), reconstructs the **bit-exact** variant in 10.4 s, and the multi-tenant memory
math is: a base + **1000 exact variants in 10 GB** vs **989 GB** of full copies — a **99×**
reduction (53× at N=100, 10× at N=10). The lossy version (BitDelta, S-LoRA) exists; the
**lossless, bit-exact** version does not — and exactness is the moat for **regulated /
reproducible / eval-critical** deployments. Caveat: 10.4 s is full-model reconstruction
(load granularity); true per-token serving needs *per-layer* on-demand materialization — the
engineering bet — but the 99× memory result holds on the math.

## 5. Honest valuation framing

- **Not** "a frontier lab acquires our codec" (they build infra internally).
- **Yes** to: a strong paper + an adopted open standard (`safetensors`/`torch.distributed.
  checkpoint` integration) → **reputation/hiring**; and a **storage/serving product** for the
  hub/cloud/serving market → a real company with a broad acquirer set.
- The defensible IP is the **lossless low-rank residual** + the **lifecycle codec** + the
  **characterization**; the defensible *moat* is **adoption** and **exactness-at-serving**.

## 6. The one-line pitch

> *Trained weights are nearly incompressible, but the **changes** between related models are
> not — and they can be captured **exactly**. We losslessly compress the model lifecycle
> (variants, checkpoints, optimizer state, across bf16/fp8/int8) by 2–100×, turning a
> petabyte of near-duplicate models into a base plus a sea of tiny exact deltas.*
