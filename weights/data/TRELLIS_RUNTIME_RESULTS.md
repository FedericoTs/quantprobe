# 2-bit QTIP Trellis: Frontier Codec + Deployable Runtime (GTX 1060, sm_61)

Campaign-4 result, Llama-2-7B / WikiText-2, pure PTQ on a single 6 GB GTX 1060.

## TL;DR
A 2-bit QTIP-style trellis codec, pushed on a 6 GB GTX 1060 (Pascal sm_61, no tensor cores), with
three honest results:
1. **Compression frontier improved.** Sensitivity-based mixed precision (protect the *residual-writers*
   `down_proj`+`o_proj` at 3-bit) + int8 side-info → **6.454 ppl @ ~2.52 b/w** vs uniform 2-bit's
   6.876 @ 2.37 — QTIP-noFT-class, **pure PTQ**. The loop discovered the residual-writer rule itself and
   falsified its cheap sensitivity metric twice.
2. **Deployable where nothing else is.** Lookup-free 2-bit decode runs on Pascal (1.82 GB resident for
   7B); no published 2-bit method (QTIP/QuIP#/AQLM) runs there — they need Ampere tensor cores. Honest:
   it's a **memory** win; speed is at **parity** with 4-bit when batched (a true speed win needs a tiled GEMM).
3. **Generalizes** to a different arch family (Qwen2.5-7B, +1.9 ppl).

The cheap/low-heat probe space is exhausted; remaining gains need bigger builds (tail-biting → true 2.0 b/w;
tiled GEMM → real speed win; 13B-in-6GB capacity demo; GLVQ/fine-tuning → toward the 5.4–5.9 SOTA).

## 1. Compression frontier (quality) — DONE

| method | ppl | b/w | notes |
|---|---|---|---|
| fp16 | 5.47 | 16.0 | reference |
| **our scalar champion** | 9.63 | 2.17 | prior best (structural 2-bit cliff) |
| **QTIP trellis + AWQ + 0.5% outliers** | **6.876** | **2.365** | = **QTIP-noFT parity**, our codec |
| QTIP w/ fine-tuning (published) | 5.86 | 2.0 | needs RFT — infeasible at 7B on a 1060 |

**ppl-vs-bits curve (Llama-2-7B, seqlen 2048, fp16=5.47):** 7.127 @ 2.13 b/w (AWQ, no outliers) ·
6.876 @ 2.37 b/w (AWQ + 0.5% outliers) · 6.834 @ 2.37 b/w (+ output-gain, within noise).

**Mixed-precision (sensitivity-based bit allocation) — a real frontier improvement.** Promoting the
most quantization-sensitive tensor type (`down_proj`, which reads the heavy-tailed SwiGLU output) from
2→3-bit beats uniform allocation at the *same* bit-rate. Direct iso-b/w head-to-head @ ~2.59 b/w:
**mixed `down_proj@3-bit` = 6.528** vs **uniform 2-bit + 0.97% outliers = 6.736** → **−0.21 ppl (3.1σ)**.
So the extra bits are better spent raising the *rate* of sensitive tensors than as uniform outliers.

And it beats the **headline** at ~iso-b/w. The winning rule is **protect the residual-writers** —
the tensors that write directly to the residual stream (`down_proj` = FFN output, `o_proj` = attention
output), since their quantization error propagates ~unattenuated to the logits (attention q/k/v errors,
by contrast, are softmax-attenuated). Mixed-precision frontier (Llama-2-7B, seqlen 2048):

| config | ppl | b/w | vs uniform-at-bw |
|---|---|---|---|
| uniform + 0.5% outliers | 6.876 | 2.37 | — |
| down_proj→3-bit | 6.686 | 2.40 | −0.19 |
| **down + o_proj → 3-bit (residual-writers)** ⭐ | **6.472** | **2.51** | **−0.26 (sweet spot)** |
| down + o_proj → 4-bit | 6.331 | 2.81 | −0.08 (saturating) |

The K=4 point shows the lever **saturating**: pushing the residual-writers from 3→4-bit buys only
−0.14 ppl for +0.30 b/w (and the margin over uniform shrinks 0.26→0.08), because the bottleneck has
shifted to the 2-bit floor on the *other* tensors. **`down + o_proj → 3-bit @ 2.51 b/w (6.472) is the
mixed-precision sweet spot.** Going lower needs the major levers below (fine-tuning / learned companding).

**Side-info compression (memory axis).** The 2-bit *payload* is incompressible (symbol entropy H = 2.0000),
so memory work targets the per-group side-info. **int8 `gs` (per-tensor affine, ~0.4% step) is
near-lossless:** down+o@3-bit goes **6.472 @ 2.507b (fp16 gs) → 6.454 @ 2.444b (int8 gs)** — −0.063 b/w
at identical ppl (Δ within noise). Banked. **Honest-accounting note:** the b/w labels count the payload
as exactly K (init-free); the simple stored-init runtime adds ~0.078 b/w (a 12-bit trellis-init per group),
so the honest best is **6.454 @ ~2.52 b/w**. Removing the init *naively* (fixed-start, register=0) costs
**+0.32 ppl** (the inverse-FWHT spreads the start-ramp error across the group) — a bad trade, rejected.
Lossless init removal to a true 2.0-class rate requires proper **tail-biting** (cyclic trellis, as QTIP
uses) — a flagged complex build, not a cheap win.

### Speed/memory probe summary (thermally-constrained exploration)
Cheap microbench probes (low heat) mapped both axes and located the boundary of cheap wins:
- **Occupancy** (autopsy + `__launch_bounds__`): dead — trellis sits at F0's blocks/SM but is latency-bound.
- **Batching** (`trellis_gemv3i_batch` vs `f0_gemv3_batch`): amortizes the decode 3.45×, reaching **parity**
  with 4-bit at B≥4 (0.98× at B=16) — *not* a win, because both simple kernels are activation-bandwidth-bound;
  a real speed win needs a **tiled (Marlin-style) quantized GEMM** so the 2-bit weight traffic dominates.
- **Symbol entropy** = 2.0 (payload incompressible) → memory wins are side-info only (int8, above).

Notably `o_proj` helps even though the cheap local-output-MSE metric ranks it *dead last* — output
magnitude is not ppl importance. The metric-guided water-fill (gate/up→3-bit) actually came in *worse
than uniform* (6.926), confirming the physics beats the metric. The correct cross-type metric is
gradient/Fisher-weighted (∂L/∂y · Δy), but full backprop is infeasible on 6 GB (7B weights don't fit) —
flagged as the next major refinement, alongside GLVQ-class learned companding.

(Note: the cheap *local* output-MSE sensitivity metric mis-ranks across tensor *types* — it puts
k/q_proj on top and down_proj at median — because it measures output magnitude, not ppl impact;
softmax attenuates attention errors while down_proj writes to the residual. The correct cross-type
metric is gradient/Fisher-weighted (∂L/∂y · Δy); the win above used the physically-justified
residual-writer prior, not that metric. A proper Fisher water-fill is the next refinement.)

**Cross-architecture generality (Qwen2.5-7B — GQA, QKV bias, RoPE θ=1e6; same codec, AWQ+outliers):**
fp16 6.001 → trellis **7.910 @ 2.37 b/w (Δ = +1.91)**, vs Llama-2's Δ = +1.41. The codec
**generalizes** (works, non-catastrophic — naive 2-bit is ~30+) but degrades ~0.5 ppl more, i.e. the
AWQ/outlier hyperparameters are Llama-tuned; a per-model tune would likely narrow the gap.

**Ceiling insight:** the L=12 trellis already *saturates* the Gaussian rate–distortion bound
D(R=2)=0.069 (preflight D_trellis=0.069). So L=14/16 and 3INST give no further gain at R=2
(verified: 3INST QMAX=16 = 5.98 vs random 5.89). **6.88 @ 2.37 b/w is the pure-PTQ ceiling for
2-bit** — the only lever below it is fine-tuning. We MATCH QTIP-noFT and BEAT our scalar champion
by 2.75 ppl, as pure PTQ.

## 2. Deployable runtime (the differentiator) — kernel built + validated

No published 2-bit method ships a consumer-GPU runtime. We built one: `weights/trellis_run.py`
(serializer/decoder) + `trellis_gemv` in `weights/evoq_kernel.cu` (CUDA decode-GEMV, sm_61).

**Why the trellis is the right runtime for a bandwidth-bound card:** each group of 128 weights is a
de-Bruijn SHIFT trellis; since L=12=6·K, the window at position p is the sliding 6-tuple of 2-bit
symbols — **no sequential dependency**, so it parallelizes exactly like the 4-bit kernel but reads a
**2-bit** stream. Codebook is 4096 fp32 = 8 KB (block-shared / lookup-tiny).

| check | result | status |
|---|---|---|
| bitstream round-trip (state path) | 0 / 65536 mismatches | **integer-exact** |
| container decode vs `trellis_quant` | rel 1–3e-4 (fp32 FWHT non-assoc.) | OK (<< ppl noise 0.067) |
| rotated-space GEMV vs `wh@x` (CPU) | rel 2.3e-7 | **exact** |
| CUDA `trellis_gemv` vs `wh@x` (GPU) | rel 3.3e-7 | **PASS** |
| kernel compiles sm_61 | yes | **OK** |
| weight DRAM traffic vs 4-bit F0 | 0.56× | 44% less (but NOT the bottleneck) |
| 7B resident @ ~2.16 b/w | **1.82 GB** | (fp16 13.5 GB; F0 4-bit 3.44 GB) — **1.9× denser** |
| decode tok/s (matmul, GTX 1060) | **~12–13** (best of 4 kernel variants) | **0.52–0.60× of 4-bit F0's ~22** |

## 3. Headline — honest

QTIP-class 2-bit quality (**6.88 ppl @ 2.37 b/w**, = old QTIP-noFT class) running on a **6 GB Pascal
GTX 1060 (sm_61, no tensor cores)** — a niche **no published 2-bit method occupies** (QTIP/QuIP#/
AQLM kernels all require Ampere+ tensor cores; QTIP→llama.cpp is an open feature request since
Nov 2024). It's a **memory/capacity win, not a speed win**:
- **Memory: 1.82 GB resident** (1.9× denser than 4-bit's 3.44 GB) → lets a 6 GB card hold models
  4-bit can't fit (e.g. a 13B at 2-bit ≈ 3.4 GB fits; 4-bit 13B ≈ 6.9 GB does not).
- **Speed: ~12–13 tok/s matmul, ~0.55× of 4-bit.** The trellis decode is ~5× the instructions of a
  4-bit LUT lookup and the Pascal card is compute/occupancy-bound at 2-bit, so reading half the
  bytes does NOT translate to speed. Tried 4 kernel variants (v1 LUT / v2 coalesced-staging /
  v3i lookup-free-3INST / RPW=2,4); best ≈ 0.6× F0. Honest: this is not a speed result.

## 3b. Kernel-variant findings (sm_61, Llama-2-7B shapes)
| variant | tok/s | note |
|---|---|---|
| F0 (4-bit LUT, baseline) | ~22 | bandwidth-bound, ~44% util |
| trellis v1 (4096-LUT shared) | ~11–13 | LUT bank conflicts |
| trellis v2 (coalesced staging) | ~7 | staging syncwarp overhead made it WORSE — bottleneck wasn't coalescing |
| **trellis v3i (lookup-free 3INST)** | **~12** | removing the LUT helped (no bank conflicts); the deployable variant |

Diagnosis: ~10% util / ~24 GB/s — **compute/latency-bound, not bandwidth-bound.** Further micro-opt
(incremental window, register tuning) might add 10–20% but won't reach 4-bit speed on Pascal.

## 4. Repro

```
tools\run_trellis.cmd kcompile     # build the kernel (CUDA env)
tools\run_trellis.cmd kgate        # GPU: decode-GEMV == wh@x  (KG_OUT/KG_IN/EVOQ_VCHUNK tune memory)
tools\run_trellis.cmd kbench       # GPU: real-shape tok/s + resident GB vs F0   <-- run on a FREE gpu
EVOQ_AWQ=1 EVOQ_POUT=0.005 EVOQ_QMAX=99 EVOQ_NWIN=16 .venv\Scripts\python -m weights.qtip_trellis measure7b
```

## 5. Next levers (below the PTQ ceiling)
- **Frozen-assignment scale-learning FT**: learn per-group `gs` + AWQ scales to minimize block-output
  error, trellis path FROZEN (no re-Viterbi → 1060-feasible). The only PTQ-ceiling-breaking lever.
- Kernel speed opt if `kbench` util < F0's ~0.4 (vectorize symbol byte-loads).
- Second model family (Llama-3.2 / Mistral-7B) to show generality.
