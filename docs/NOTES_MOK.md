# Reading notes — Cursor's Mixture-of-Kittens (2026-08, blog + repo)

Source: cursor.com/blog/mixture-of-kittens · github.com/cursor/mixture-of-kittens (Apache 2.0,
Blackwell SM100/SM103 only). A TRAINING megakernel for GB300 NVL72 racks: all MoE
communication + computation fused into one deterministic kernel; 1.41x end-to-end at 512 GPUs.
Nothing here runs on our hardware, and none of it is inference — the value is that their
bottleneck physics rhymes with ours, one tier down the memory hierarchy.

## What transfers (their NVLink is our PCIe/RAM bus)

1. **Traffic DIRECTION is a lever, not a detail.** Their pull-based dispatch beats push by up
   to +29% NVLink utilization purely by balancing bidirectional traffic. Our analog: during
   split decode the GPU pulls expert weights from pinned host RAM (H2D) while activations
   cross the other way (D2H). Nobody — including us — has measured whether the 1060's PCIe
   link runs balanced or one-lane-saturated during a split token. NVML exposes per-direction
   PCIe throughput counters on Pascal. **Registered as U-42:** poll RX/TX during a split
   bench; if one direction saturates while the other idles, that asymmetry (a) belongs in the
   two-resource model (#53) as a real per-direction budget, (b) is upstream-relevant the same
   way our ggml sync PR (#28) was.
2. **Overlap validated from the training side.** Their inter-SM comm/compute overlap
   (saturating NVLink with <1/3 of SMs) is the same shape as our measured +18-35% ggml
   graph-executor sync win and the registered U-40 draft-driven expert prefetch. A second
   independent domain paying big for transfer/compute overlap raises U-40's priority.
3. **Sync-point elimination is where their ring buffers earn their keep** — and it is why our
   -ub/-b buffer terms move numbers (prereg #66's buffer-term fix family). Same law, their
   scale.
4. **The always-active path gets the bits — industry convergence #4.** MoK keeps the SHARED
   expert in BF16 while routed experts run MXFP8, enforced at the API level
   (mxfp8_quantize() for routed only). That is K3's protected experts, Mach-1's
   higher-precision attention floor, and our TENSOR_ROLES "shared-expert: ALWAYS ACTIVE -
   protect" rule, now appearing in a fourth independent system and on the TRAINING side.
   Recorded as external corroboration of the role-protection invariant; it also means
   MX-format GGUF conversions will proliferate (K3 = MXFP4, MoK = MXFP8), which raises
   task #49 (measure MXFP4, FORMAT_EBW entry) from backlog to soon.
5. **Determinism sold as a feature** ("for ablations and RL") — a frontier-lab kernel team
   independently arriving at our C-14 discipline: a measurement you cannot reproduce exactly
   is not a measurement. Quotable ally.

## What does NOT transfer, stated so nobody trips on it

The kernel itself (Blackwell-only, training-only), EP-64 expert parallelism (we have one
GPU), and their macrobatch regime (we care about batch 1-32 decode). The repo ships no
profiling tooling worth lifting — U-42's counters come from NVML on our side.
