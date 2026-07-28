# llama.cpp's Q2_K CUDA decode does 2x the dp4a work it needs to

**Status:** diagnosed at source, magnitude measured, **not yet fixed or upstreamed.**
**Hardware:** GTX 1060 6GB (cc 6.1). One card — the magnitude is arch-dependent, the defect is not.

## The defect

`ggml/src/ggml-cuda/vecdotq.cuh`, `vec_dot_q2_K_q8_1_impl_mmvq`:

```c
for (int i = 0; i < QR2_K; ++i) {
    const int sc = scales[2*i];
    const int vi = (v >> (2*i)) & 0x03030303;
    sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF));   // the real dot product

    int m = sc >> 4;
    m |= m <<  8;
    m |= m << 16;                                                    // broadcast min to 4 lanes
    sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0);                    // == m * sum(u[i])
}
```

The second `dp4a` computes `m x sum(u[i])`. **`sum(u[i])` is the sum of four ACTIVATION values —
it does not depend on which output row is being computed** — yet it is recomputed for every row of
the weight matrix. Per 16 weights that is **8 dp4a where 4 suffice**, plus 4 pure-overhead
integer ops per group to splat `m` across the lanes.

## This is not a design necessity — the same file already does it right

`vec_dot_q4_1_q8_1_impl`, llama.cpp's *other* asymmetric format, handles its min with a single
multiply against a precomputed sum carried in the activation block:

```c
const float m4s8 = dm4f.y * ds8f.y;      // block_q8_1::ds.y is the stored activation sum
```

`block_q8_1` already carries that sum. Q2_K cannot use `ds.y` **directly** — its per-16-weight
sub-block granularity is finer than the q8_1 block — but the same hoisting applies: the four
partial sums are row-invariant and belong in shared memory (or in a finer-grained q8_1 sum),
computed once per thread block instead of once per row.

## Measured magnitude

`tools/kernelprobe/bench.cu` implements Q2_K's exact cost model (2 bits/weight, 4-bit scale AND
4-bit min packed in one byte per 16-weight sub-block, superblock fp16 d/dmin = 2.625 bits/weight)
with the min term hoisted, and nothing else changed:

| | GW/s | fraction of kernel ceiling |
|---|---|---|
| llama.cpp Q4_0, real 7B end-to-end | 204.7 | 204.7 / 227.1 = **88.5%** |
| llama.cpp Q2_K, real 7B end-to-end | 165.1 | 165.1 / 352.7 = **46.8%** |
| kernelprobe, Q2_K cost model, min hoisted | **352.7** | — |

Q4_0 calibrates how much a real decode loses to attention, KV, norms and sampling against a pure
matvec: **about 11%**. Q2_K loses **53%**. The residual tracks the instruction count: Q4_0 is
symmetric and issues 4 dp4a per 16 weights, exactly like the reference kernel; Q2_K issues 8.

**Indicated headroom: ~1.9x on Q2_K decode**, at zero cost in bits, file size, or quality.

## Why it matters disproportionately

Q2_K is the format consumer hardware uses for large MoE models — it is what makes a 30B MoE fit at
all on a 6 GB card. The defect is invisible on symmetric formats (Q4_0, Q8_0) and on any hardware
with enough ALU headroom to hide it, which is plausibly why it has survived: it costs little on the
GPUs maintainers benchmark on and a great deal on the GPUs users are stuck with.

## What is NOT yet established

- **Not implemented in llama.cpp.** The 352.7 GW/s is our own kernel under a matched cost model,
  not a patched llama.cpp. Until the fix is written and A/B'd in-tree with output verification, the
  1.9x is *indicated*, not delivered.
- **One card.** On Ampere+ the extra dp4a has more room to hide and the gain should shrink.
- The same pattern should be checked on **Q3_K, Q4_K, Q5_K, Q6_K**, which are also asymmetric.
  Q4_K's measured eta 0.553 vs Q4_0's 0.622 (#52) is consistent with a smaller version of the same
  tax, but that has not been traced to source.

## Next step

Patch `vec_dot_q2_K_q8_1_impl_mmvq` and its `mul_mat_vec_q` caller to hoist the row-invariant
activation sums, A/B on the 7B and the MoE flagship, verify output, and file upstream alongside
[#26200](https://github.com/ggml-org/llama.cpp/issues/26200).
