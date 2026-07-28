# Pre-registration #50 (scored inline): GPU-side CUDA event timing — the token is now 87% accounted

**Author:** Federico Sciuca · **Date:** 2026-07-28. **Diagnostic.** #49 located ~29 ms/token
outside the CPU executor but could not see inside it — E3 instruments `ggml-cpu.c` by construction.
E6 adds CUDA events around every `ggml_backend_cuda_graph_compute`, separating **device-busy** time
(events, device clock) from **wall time inside the backend** (steady clock). Their difference is
the CPU-side cost of driving the GPU.

## Measured — MoE flagship, split placement, tg128, both profilers on

| component | ms/token | note |
|---|---|---|
| CPU executor (E3) | **26.30** | 0.516 GB of experts, at physics |
| CUDA backend wall (E6) | **26.84** | over **33** `graph_compute` calls/token |
|  → device busy | **23.88** | GPU actually executing |
|  → submission / wait | **2.96** | 11% of backend time |
| residual (scheduler, sampling, glue) | 8.06 | |
| **token** | **61.20** | (16.34 tok/s — the profilers cost ~5%, 58.1 → 61.2) |

**CPU + CUDA-backend wall = 53.14 ms = 87% of the token.** The bookkeeping is essentially closed.

## The result, and it rewrites the question

**The GPU is neither idle nor stalled. It is busy for 23.88 ms doing work the byte model prices at
4.34 ms** (0.700 GB at the card's independently measured 161.3 GB/s, #44).

| | value |
|---|---|
| GPU device-busy | 23.88 ms |
| byte model at measured card speed | 4.34 ms |
| **ratio** | **5.5×** |
| effective GPU bandwidth | **29.3 GB/s** |
| η vs the card's demonstrated 161.3 | **0.15** |

Three hypotheses die at once, each having been live this session:

- **Not GPU idle.** The red-team's ~22 ms/token idle (D-14 note) and my own "nothing overlaps"
  reading from #49 are both wrong — the device is executing for 23.88 of 61.20 ms.
- **Not submission overhead.** 2.96 ms, 11% of backend time. Consistent with #47/#48: CUDA graphs
  bought +3.2% because there was only ~3 ms there to buy.
- **Not the CPU path.** 26.30 ms moving 0.516 GB = 28 GB/s, above the measured DRAM stream.

**What remains is a well-posed question this project has never been able to ask:** why do
llama.cpp's quantized CUDA kernels run at **29.3 GB/s** on a card that delivers **161.3 GB/s** to
cuBLAS on the same shape class?

## Why this is sharper than C-02 ever was

C-02 was "all-in-VRAM η is 0.32–0.56 and six explanations are refuted." It is now:

| configuration | GPU η | source |
|---|---|---|
| cuBLAS fp32 GEMV | 0.84 | #44, independent |
| llama.cpp all-in-VRAM Q4_K_M | 0.51 | #44 addendum |
| **llama.cpp split placement, GPU share** | **0.15** | this |

The split's GPU work is **3.4× less efficient than the same runtime's all-in-VRAM work**, and the
structural difference is visible in the same measurement: **33 `graph_compute` calls per token.**
The graph is fragmented into 33 subgraphs at CPU/GPU boundaries, so each GPU segment carries ~21 MB
and ~0.72 ms of work — far too little to saturate a GPU that needs large contiguous streams to
reach 161 GB/s.

**Fragmentation is a hypothesis, not a conclusion**, and it is the eighth mechanism candidate this
project has entertained. It is stated here only because it is *measurable*: it predicts that GPU η
should rise as the number of subgraphs per token falls. #43 already measured the pure layer split
(`-ngl 20`, one boundary) at the same end-to-end speed — which does **not** obviously fit, and that
tension is exactly what the next measurement must resolve rather than paper over.

## Instrument note

E6's `cudaEventSynchronize` forces a sync per call and cost ~5% end-to-end (58.14 → 61.20 ms).
Device-busy figures are unaffected (events measure device execution between records); only the
totals shift. Recorded so the numbers are not compared naively against unprofiled runs.

**Wired into:** `findings/REGISTER.json:C-09` (closed — the token is 87% accounted, and the
residual is not where anyone looked) · `C-02` (restated as a GPU-kernel efficiency question with a
measured 5.5× gap on the split path).
