# LUT decode-as-gather expert-GEMV (paper §9, "fitting *and* fast")

The remaining engineering prize is a batch-1 expert-GEMV that decodes a 2-bit code by **gather** (not
sequential trellis) so it runs at high bandwidth-utilization on Pascal, where `tok/s = bw × util / bytes`.
Two files:

- **`lut_decode_gemv.py`** — verified reference (CPU, no toolchain). Defines the packed format and the
  decode math; self-test passes: decode-as-gather is **bit-exact** vs dequant→matmul (3e-7), the int8/DP4A
  path is within 9e-3 (x-quant only), weight rel-MSE **0.116** (the Lloyd-Max 2-bit Gaussian scalar limit).
- **`lut_expert_gemv.cu`** — the same math with `__dp4a` (cc≥6.1), one warp per output row. **Ready to
  compile; not yet built** — the dev box this was written in has the CUDA *runtime* but no *compiler*
  (`nvcc`/`cl`/`gcc`/`ninja`/Triton all absent), and the kernel's payoff (a measured tok/s) requires it.

## Build & verify (on a box with CUDA toolkit + host compiler)

```
nvcc -O3 -arch=sm_61 -o lut_test lut_expert_gemv.cu      # sm_61 = GTX 1060 (Pascal)
```

Verify numerics against the reference: in `lut_decode_gemv.py`, dump `packed, scales, xq4 (4×int8/int32),
sx*DP4A_S, CBI` for a random expert, run the kernel, and check `max|y_kernel − y_dp4a| / max|y|` < 1e-3.
The Python `dp4a_gemv` is the exact spec; the kernel must match it bit-for-bit on integer accumulation.

## Performance model (what to expect, why it should win)

- **Format:** 2.25 b/w (2-bit codes + fp16 per-128 scale). Active bytes/token for top-6 experts of
  DeepSeek-V2-Lite ≈ 6 × (gate+up+down ≈ 3×1408×2048 × 2.25/8) ≈ **15 MB**.
- **Ceiling:** at the GTX 1060's ~192 GB/s, 15 MB/token ⇒ ~**12,800 tok/s** bandwidth ceiling for the
  expert reads alone; the realistic target is the ~100–250 tok/s whole-model active-byte bound (attention +
  router + norms dominate), i.e. **~10–40× over the current 5–6 tok/s** if the GEMV reaches even ~40% util.
- **Why gather beats trellis:** the trellis decode is sequential (state-dependent), serializing the inner
  loop and stalling the SMs (~10% util, ~12 tok/s measured). The LUT path is `byte → 4 codebook lookups →
  one __dp4a`, fully parallel and coalesced (~44% util on the 4-bit analogue). The experts' maximal entropy
  makes the four codes near-equiprobable ⇒ balanced gather, no divergence.

## Integration

Launch per active expert (gate, up, down) per layer: `launch_lut_expert_gemv(packed, scales, xq4, sx*CB,
y, out, in, stream)`. Pre-quantize the layer input `x` to int8 once (per token) and pack 4/int32; share it
across the expert's three matmuls. The router, attention (MLA), shared experts, and norms stay on the
existing carve-out path — this kernel is only for the 2-bit routed-expert GEMVs, which are the bandwidth
term.

## Quality note (the trade this code makes)

This scalar LUT code sits at rel-MSE **0.116** vs the trellis **0.069** (the vector RD floor) — it trades
~0.05 b/w-equivalent of quality for gather-decodability. Two ways to recover it without losing the gather
primitive: (1) a **vector-quantized codebook** (AQLM-style, 2-bit index → short codebook vector) approaches
0.069 while staying a pure gather; (2) keep the *protected* tensors (attention/shared/KV-latent) on the
existing trellis path and use this LUT only for the routed experts, which §6 shows are at the floor and
tolerate the scalar code. Measure end-to-end ppl with the LUT experts before committing the codec swap.
