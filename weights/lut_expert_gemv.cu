// lut_expert_gemv.cu -- batch-1 2-bit decode-as-gather expert GEMV for Pascal (cc>=6.1, __dp4a).
//
// Implements exactly weights/lut_decode_gemv.py::dp4a_gemv. The point (paper Sec.9): a 2-bit code that is
// GATHER-decodable runs the batch-1 expert-GEMV at high utilization, where the rate-optimal sequential
// trellis stalls (~12 tok/s @ ~10% util vs a LUT path ~22 @ ~44%). Weights: 4 codes/byte indexing a
// 4-entry int8 codebook (CBI, Lloyd-Max 2-bit Gaussian levels); x pre-quantized to int8 and packed 4/int32;
// inner product accumulated with __dp4a; per-group fp32 scale applied at the end.
//
// Build (needs CUDA toolkit + host compiler -- absent in the dev box this was written in; see KERNEL_README.md):
//   nvcc -O3 -arch=sm_61 --ptxas-options=-v -o lut_test lut_expert_gemv.cu
// Verify against the Python reference: dump (packed, scales, xq4, sx*DP4A_S) from lut_decode_gemv.py and
// compare y. One warp per output row; grid = out_features, block = 32.

#include <cstdint>
#include <cuda_runtime.h>

__constant__ int8_t CBI[4];                 // int8 codebook; host sets via cudaMemcpyToSymbol

extern "C" __global__ void lut_expert_gemv(
    const uint8_t* __restrict__ packed,     // [out, in/4]   4 two-bit codes per byte
    const float*   __restrict__ scales,     // [out, in/128] per-group fp32 scale (row std)
    const int32_t* __restrict__ xq4,        // [in/4]        4 int8 activations packed per int32
    const float    sx_cb,                   // sx * DP4A_S   (x int8-scale * codebook int8-scale)
    float*         __restrict__ y,          // [out]
    const int      inn) {
  const int o    = blockIdx.x;              // one warp (block of 32) per output row
  const int lane = threadIdx.x;            // 0..31
  const int ng   = inn >> 7;                // groups of 128
  const int bpr  = inn >> 2;                // bytes per row (in/4)
  const uint8_t* prow = packed + (size_t)o * bpr;

  float acc = 0.f;
  for (int g = lane; g < ng; g += 32) {     // each lane takes whole groups (G=128 -> 32 dp4a)
    int idot = 0;
    const int base = g << 5;                // 32 bytes per group
    #pragma unroll
    for (int b = 0; b < 32; ++b) {
      const uint8_t by = prow[base + b];     // 4 codes
      const int wl = ((int)(uint8_t)CBI[ by       & 3])
                   | ((int)(uint8_t)CBI[(by >> 2) & 3] <<  8)
                   | ((int)(uint8_t)CBI[(by >> 4) & 3] << 16)
                   | ((int)(uint8_t)CBI[(by >> 6) & 3] << 24);
      idot = __dp4a(wl, xq4[base + b], idot);            // 4 int8 MACs/op
    }
    acc += scales[(size_t)o * ng + g] * (float)idot;
  }
  for (int s = 16; s > 0; s >>= 1)          // warp reduce
    acc += __shfl_down_sync(0xffffffffu, acc, s);
  if (lane == 0) y[o] = acc * sx_cb;
}

// Host helpers (compiled only where a toolchain exists).
extern "C" void set_codebook(const int8_t cb[4]) {
  cudaMemcpyToSymbol(CBI, cb, 4 * sizeof(int8_t));
}
extern "C" void launch_lut_expert_gemv(const uint8_t* packed, const float* scales,
                                       const int32_t* xq4, float sx_cb, float* y,
                                       int out, int inn, cudaStream_t s) {
  lut_expert_gemv<<<out, 32, 0, s>>>(packed, scales, xq4, sx_cb, y, inn);
}
