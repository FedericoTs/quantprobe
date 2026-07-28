# Pre-registration #47: C-02 and C-09 are one mechanism — a fixed per-token cost from kernel launches, not a bandwidth deficit

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the confirming measurement.
**Status: STAKED.** (The motivating fit uses existing data; every stake below is on data not yet taken.)

## The reframe

Law 4 models decode as `t_token = bytes / (eta * BW)` — **purely proportional, no intercept.**
Fitting a single η to models of different sizes therefore forces any fixed per-token cost to appear
as *size-dependent efficiency*. That is exactly the shape C-02 has reported for the whole project:
η rising 0.354 → 0.461 → 0.560 with model size, "unexplained", six candidate causes refuted.

Fit a **two-parameter** model instead — `t_token = fixed + bytes / BW_marginal` — on two models of
the same format and family, all-in-VRAM (so dequant cost per byte is constant):

| model | bytes/token | measured |
|---|---|---|
| Qwen3.5-4B-Q4_K_M | 2.640 GB | 30.89 tok/s = 32.37 ms |
| Qwen2.5-7B-Q4_K_M | 4.361 GB | 23.02 tok/s = 43.44 ms |

→ **marginal bandwidth 155.5 GB/s, fixed cost 15.40 ms/token.**

**155.5 GB/s is within 3.6% of the 161.3 GB/s that CuPy measured independently on this card (#44).**
llama.cpp streams weights at essentially hardware speed. There is no bandwidth deficit. There is a
large constant.

## The mechanism, confirmed in source

`ggml/src/ggml-cuda/ggml-cuda.cu:4539`:

```c
if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_AMPERE) {
    graph->disable_due_to_gpu_arch = true;
}
```

`GGML_CUDA_CC_AMPERE` = 800; the GTX 1060 is cc **610**. **CUDA graphs are disabled on this card by
llama.cpp's own architecture check** — not by CUDA, which has supported graphs since 10.0 on sm_35+.
So every node is an individual kernel launch, and on Windows **WDDM** each costs ~15–30 µs. 15.40 ms
÷ 20 µs ≈ **770 launches per token**, which is the right order for a 28–36 layer transformer graph.

This one mechanism would explain, at once: C-02's size-dependent η; C-09's ~20 ms unattributed in
the split token (the same constant, present in both placements); why prefill is healthy (405–445
t/s — a batched pass amortises the constant over 512+ tokens); why #15's fixed-overhead model was
"refuted" (it assumed the wrong magnitude, not the wrong shape); and why η rose with bit-width in
the format ladder (more bytes per token amortising the same constant, on top of dequant cost).

## Stakes

- **P-1 (the constant is real and placement-independent).** Fitting the same two-parameter model to
  the **split** placement across ≥3 models yields a fixed term within **±40%** of 15.4 ms. The
  constant is a property of the graph, so it must appear in both placements at similar magnitude.
- **P-2 (it scales with NODE COUNT, not bytes).** Across architectures at matched bytes/token, the
  fitted fixed term is proportional to layer count within **±35%**. If it instead tracks bytes, it
  is not launch overhead and this whole reframe is wrong.
- **P-3 (THE PRIZE — removing the arch check recovers most of it).** With CUDA graphs force-enabled
  on this Pascal card, all-in-VRAM decode improves by **≥25%**. If graphs are disabled for a *good*
  reason (a correctness bug or a real regression on Pascal), this fails and the constant is
  structural rather than a configuration accident.
- **P-4 (quality).** Output byte-identical with graphs on vs off, at matched request index.

## KILL RULE

**If P-3 fails and P-1 holds, the constant is real but not recoverable through CUDA graphs** —
the lever moves to reducing the graph's node count (op fusion on the CUDA side, which the CPU-side
E4 experiment showed is a null once barriers are cheap) or to leaving WDDM (Linux, where launch
overhead is ~2–5 µs rather than 15–30). Both are then priced by the same constant.

## What this would mean for the project's headline

If P-1 and P-3 hold, **"novel decode has no software lever left" is wrong by ~25% on the GPU side**,
the 41.1 tok/s wall was computed with a term that does not belong in it, and the tool's η constants
are describing an artifact of a disabled optimisation rather than a property of the hardware.

## Blocked on

`nvcc` targeting `sm_61` — CUDA 12.9 install (13.3 dropped Pascal; the 12.9 installer bundles an
older driver and exits 46 rather than downgrade). P-1 and P-2 need no build and can run first.

**Wired into:** pending — nothing ships until P-1 is measured.

---

## Scored (2026-07-28, log: `weights/data/prereg47_cuda_graphs.log`)

**Verdict: P-3 MISS — proven, not assumed. CUDA graphs engage on Pascal and buy exactly nothing.
The 15.4 ms constant is real but is NOT kernel-launch overhead.**

Built llama.cpp with CUDA 12.9 for `sm_61` (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61`),
patched the arch check behind `GGML_CUDA_FORCE_GRAPHS`, and A/B'd on the 7B all-in-VRAM:

| arm | tg32 |
|---|---|
| graphs default (disabled by arch check) | 22.81 ± 0.11 |
| **graphs FORCED ON** | **22.81 ± 0.02** |
| default re-run LAST (position control) | 22.66 ± 0.00 |
| forced ON, instrumented build | 22.80 ± 0.06 |

**Two false negatives were caught before this could be scored, and both are worth recording:**

1. **The first A/B measured nothing.** `ggml/CMakeLists.txt:117` sets `GGML_CUDA_GRAPHS_DEFAULT`
   to **OFF**, so `USE_CUDA_GRAPH` was undefined and the entire graph path — *including the arch
   check I had patched* — was compiled out. Rebuilt with `-DGGML_CUDA_GRAPHS=ON`.
2. **"Graphs on" is not the same as "graphs captured."** After enabling them I still could not
   see capture, so I instrumented `cudaStreamBeginCapture` directly. First attempt broke the
   build (exit 1) and silently ran the *previous* binary — the same class of error as #36's
   replay artifact. Fixed, rebuilt, and capture printed: **`[e5] cudaStreamBeginCapture fired #1`**.

Only with capture proven does the null mean anything. It does now: **graphs run, and decode does
not move.**

### What survives, and what dies

**Dies:** the launch-overhead explanation, and with it the ≥25% prize. Whatever the 15.4 ms is, it
is not per-kernel launch cost that graph replay removes.

**Survives, and is the important half:** the *two-parameter fit itself*. Marginal bandwidth
**155.5 GB/s** against an independently measured **161.3 GB/s** ceiling (#44) — 3.6% apart — is not
explained away by this null. Law 4 having no intercept is still forcing a real constant to
masquerade as size-dependent η, which is still the best account of C-02 anyone has produced in
this project. The constant's *identity* is now open again; its *existence* is not.

### The mechanism candidate this measurement leaves standing

`ggml-cuda.cu:3280` `[TAG_MUL_MAT_ID_CUDA_GRAPHS]`: `MUL_MAT_ID` **forces a stream
synchronisation** and disables graphs whenever `!ggml_is_quantized(src0)` or the batch exceeds
`mmvq_mmid_max`. Our test model is dense (no `MUL_MAT_ID`), so this did not affect the A/B — but
**the flagship is MoE and hits `MUL_MAT_ID` in every layer**. A forced stream sync per layer is a
far better candidate for a large fixed cost than launch overhead, and it is measurable with the
build that now exists.

**Wired into:** `findings/REGISTER.json:C-02` (constant confirmed, cause reopened) ·
`findings/REGISTER.json:D-15` (launch overhead refuted) · next: profile the MoE flagship on this
CUDA build, where `MUL_MAT_ID` sync is the live suspect.
