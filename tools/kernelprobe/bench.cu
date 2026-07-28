// kernelprobe — what can this GPU actually do on a quantized decode matvec?
//
// ZERO llama.cpp, zero ggml. Our own kernels, our own memory, our own timing.
// Every previous efficiency number in this project was measured THROUGH llama.cpp, so it could
// never separate "the card cannot do this" from "this runtime does not do this well". This can.
//
// Ladder, all on the same buffer so the only thing that changes is the access pattern:
//   L0  stream      — read every byte, reduce. The true ceiling for our own code.
//   L1  contiguous  — dequantize 4.5-bit weights and matvec, reading matrices back to back.
//                     This is the all-in-VRAM dense decode pattern.
//   L2  gathered    — identical kernel, but only 1-in-G matrices are touched, chosen by an index
//                     array. This is the MoE expert pattern (8 of 128) that the flagship runs.
//
// L1 vs L2 isolates the expert-gather penalty at constant kernel code, which is the live suspect:
// llama.cpp's all-in-VRAM eta 0.51 was measured on a DENSE model, its split-placement eta 0.15 on
// a gathered MoE one. If L2 collapses relative to L1 on our own kernel too, the penalty is the
// access pattern and not the runtime.
//
// Format: 4.5 bits/weight, byte-identical in size to llama.cpp's Q4_K.
//   per 256-weight superblock: 128 B packed nibbles + 8 x fp16 sub-block scales (16 B) = 144 B
//   row layout: all quants for the row contiguous, then all scales, so loads are 4-byte coalesced.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(1); } } while (0)

// flagship-shaped: hidden 2048, expert intermediate 768
static const int K          = 2048;              // input dim  (8 superblocks per row)
static const int ROWS       = 768;               // output dim per matrix (one expert tensor)
static const int SB_PER_ROW = K / 256;           // 8
static const int QB_PER_ROW = SB_PER_ROW * 128;  // 1024 bytes of nibbles
static const int SC_PER_ROW = SB_PER_ROW * 8;    // 64 fp16 scales = 128 bytes
static const int ROW_BYTES  = QB_PER_ROW + SC_PER_ROW * 2;   // 1152 B  == 4.5 bits/weight
static const int MAT_BYTES  = ROWS * ROW_BYTES;              // 884736 B per matrix

// ---------------------------------------------------------------- L0: pure streaming read
__global__ void k_stream(const uint4 * __restrict__ p, size_t n4, unsigned long long * out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned long long acc = 0;
    for (; i < n4; i += stride) {
        uint4 v = p[i];
        acc += v.x; acc += v.y; acc += v.z; acc += v.w;
    }
    atomicAdd(out, acc);
}

// ------------------------------------------------ L1/L2: dequant matvec, one block per matrix
// 128 threads. Block loads x into shared once, then walks its matrix's rows.
// `sel` maps block -> matrix index; that is the ONLY difference between contiguous and gathered.
__global__ __launch_bounds__(128) void k_matvec(
        const uint8_t * __restrict__ base,
        const int     * __restrict__ sel,
        const float   * __restrict__ x,
        float         * __restrict__ y)
{
    __shared__ float xs[K];
    for (int i = threadIdx.x; i < K; i += 128) xs[i] = x[i];
    __syncthreads();

    const int mat = sel[blockIdx.x];
    const uint8_t * m = base + (size_t)mat * MAT_BYTES;

    for (int r = 0; r < ROWS; r++) {
        const uint8_t * row = m + (size_t)r * ROW_BYTES;
        const uint32_t * q  = (const uint32_t *)row;
        const __half   * sc = (const __half   *)(row + QB_PER_ROW);

        float acc = 0.0f;
        // 256 uint32 of nibbles per row; 128 threads -> 2 each, both coalesced
        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            const int i  = threadIdx.x + pass * 128;   // uint32 index within the row
            const int sb = i >> 5;                     // superblock (32 uint32 = 128 B each)
            const int b  = (i & 31) << 2;              // first byte within the superblock
            const uint32_t v = q[i];

            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint8_t byte = (uint8_t)(v >> (j * 8));
                const int wlo = sb * 256 + b + j;          // low nibble  -> weight b+j
                const int whi = wlo + 128;                 // high nibble -> weight b+j+128
                const float slo = __half2float(sc[sb * 8 + ((b + j)       >> 5)]);
                const float shi = __half2float(sc[sb * 8 + (((b + j) + 128) >> 5)]);
                acc += ((float)(byte & 0xF) - 8.0f) * slo * xs[wlo];
                acc += ((float)(byte >>  4) - 8.0f) * shi * xs[whi];
            }
        }
        // warp + block reduction over 128 threads
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        __shared__ float warp[4];
        if ((threadIdx.x & 31) == 0) warp[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x == 0) {
            y[blockIdx.x * ROWS + r] = warp[0] + warp[1] + warp[2] + warp[3];
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------- L1b: fp16, NO dequantization
// Same access pattern, same reduction, only the unpack is gone. The gap L1b -> L1 is the pure
// instruction cost of turning packed nibbles into floats, with memory traffic held constant.
static const int F16_ROW_BYTES = K * 2;                 // 4096 B
static const int F16_MAT_BYTES = ROWS * F16_ROW_BYTES;

__global__ __launch_bounds__(128) void k_matvec_f16(
        const uint8_t * __restrict__ base, const int * __restrict__ sel,
        const float * __restrict__ x, float * __restrict__ y)
{
    __shared__ float xs[K];
    for (int i = threadIdx.x; i < K; i += 128) xs[i] = x[i];
    __syncthreads();
    const uint8_t * m = base + (size_t)sel[blockIdx.x] * F16_MAT_BYTES;
    for (int r = 0; r < ROWS; r++) {
        const __half2 * w = (const __half2 *)(m + (size_t)r * F16_ROW_BYTES);
        float acc = 0.0f;
        #pragma unroll
        for (int p = 0; p < 8; p++) {                   // 1024 half2 per row / 128 threads
            const int i = threadIdx.x + p * 128;
            const float2 v = __half22float2(w[i]);
            acc += v.x * xs[2 * i] + v.y * xs[2 * i + 1];
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        __shared__ float warp[4];
        if ((threadIdx.x & 31) == 0) warp[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x == 0) y[blockIdx.x * ROWS + r] = warp[0] + warp[1] + warp[2] + warp[3];
        __syncthreads();
    }
}

// ---------------------------------------------------------------- L1c: int8 via __dp4a (sm_61)
// Pascal 6.1 has the 4-way INT8 dot-product instruction. 8.5 bits/weight, but the unpack is a
// single hardware op instead of shift/mask/convert. This is the actionable alternative format.
static const int Q8_ROW_BYTES = K + (K / 32) * 2;       // 2048 quants + 64 fp16 scales = 2176 B
static const int Q8_MAT_BYTES = ROWS * Q8_ROW_BYTES;

__global__ __launch_bounds__(128) void k_matvec_q8(
        const uint8_t * __restrict__ base, const int * __restrict__ sel,
        const int * __restrict__ xq, float sx, float * __restrict__ y)
{
    __shared__ int xs[K / 4];
    for (int i = threadIdx.x; i < K / 4; i += 128) xs[i] = xq[i];
    __syncthreads();
    const uint8_t * m = base + (size_t)sel[blockIdx.x] * Q8_MAT_BYTES;
    for (int r = 0; r < ROWS; r++) {
        const uint8_t * row = m + (size_t)r * Q8_ROW_BYTES;
        const int    * w  = (const int    *)row;
        const __half * sc = (const __half *)(row + K);
        float acc = 0.0f;
        #pragma unroll
        for (int p = 0; p < 4; p++) {                   // 512 int32 per row / 128 threads
            const int i = threadIdx.x + p * 128;        // covers weights 4i..4i+3, all in group i/8
            acc += (float)__dp4a(w[i], xs[i], 0) * __half2float(sc[i >> 3]);
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        __shared__ float warp[4];
        if ((threadIdx.x & 31) == 0) warp[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x == 0) y[blockIdx.x * ROWS + r] = (warp[0] + warp[1] + warp[2] + warp[3]) * sx;
        __syncthreads();
    }
}

// ------------------------------------------- L1d: THE SAME 4.5-bit format, unpacked via __dp4a
// Identical bytes on the wire as L1 — same buffer, same format, same traffic. The only change is
// that the nibbles are fed to Pascal's INT8 dot-product instruction instead of being widened to
// float one at a time. If L1d >> L1, the 4-bit deficit is an instruction-selection problem, not a
// property of 4-bit weights.
//   sum((q-8)*x) = dp4a(q,x) - 8*sum(x), and sum(x) over the 4 lanes is itself one dp4a.
__global__ __launch_bounds__(128) void k_matvec_q4_dp4a(
        const uint8_t * __restrict__ base, const int * __restrict__ sel,
        const int * __restrict__ xq, float sx, float * __restrict__ y)
{
    __shared__ int xs[K / 4];
    for (int i = threadIdx.x; i < K / 4; i += 128) xs[i] = xq[i];
    __syncthreads();
    const uint8_t * m = base + (size_t)sel[blockIdx.x] * MAT_BYTES;
    for (int r = 0; r < ROWS; r++) {
        const uint8_t  * row = m + (size_t)r * ROW_BYTES;
        const uint32_t * q   = (const uint32_t *)row;
        const __half   * sc  = (const __half   *)(row + QB_PER_ROW);
        float acc = 0.0f;
        #pragma unroll
        for (int p = 0; p < 2; p++) {
            const int i  = threadIdx.x + p * 128;
            const int sb = i >> 5, l = i & 31;
            const uint32_t v = q[i];
            const int xlo = xs[sb * 64 + l];            // weights 4i..4i+3
            const int xhi = xs[sb * 64 + l + 32];       // the same, +128
            const int qlo = (int)( v       & 0x0F0F0F0Fu);
            const int qhi = (int)((v >> 4) & 0x0F0F0F0Fu);
            const int slo = __dp4a(0x01010101, xlo, 0); // sum of the 4 x lanes
            const int shi = __dp4a(0x01010101, xhi, 0);
            acc += (float)(__dp4a(qlo, xlo, 0) - 8 * slo) * __half2float(sc[sb * 8 + (l >> 3)]);
            acc += (float)(__dp4a(qhi, xhi, 0) - 8 * shi) * __half2float(sc[sb * 8 + (l >> 3) + 4]);
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        __shared__ float warp[4];
        if ((threadIdx.x & 31) == 0) warp[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x == 0) y[blockIdx.x * ROWS + r] = (warp[0] + warp[1] + warp[2] + warp[3]) * sx;
        __syncthreads();
    }
}

// ------------------------------------------------- L1e: 2-bit, dp4a-native, PRE-PERMUTED layout
// 2.5 bits/weight: 32 weights (8 B) + one fp16 scale = 10 B per block.
//
// The trick that makes this cheap: a uint32 holds 16 two-bit weights, and (v >> 2j) & 0x03030303
// extracts four of them into the four int8 lanes dp4a wants. Naively those four are STRIDED in
// weight order (0,4,8,12), which would force a strided activation gather. So the weights are
// stored PRE-PERMUTED at quantization time -- byte j carries weights j, j+4, j+8, j+12 -- and the
// extraction then yields CONTIGUOUS weights, so each dp4a pairs with one aligned int32 of x.
//
// Symmetric values are 0..3 with an implicit -2 offset, and sum((q-2)*x) = dp4a(q,x) - 2*sum(x).
// sum(x) does not depend on the row, so it is hoisted: computed once per block into shared memory
// instead of once per row. That removes a dp4a per group per row.
static const int B2_ROW_BYTES = (K / 16) * 4 + (K / 32) * 2;   // 512 + 128 = 640 B
static const int B2_MAT_BYTES = ROWS * B2_ROW_BYTES;

__global__ __launch_bounds__(128) void k_matvec_q2_dp4a(
        const uint8_t * __restrict__ base, const int * __restrict__ sel,
        const int * __restrict__ xq, float sx, float * __restrict__ y)
{
    __shared__ int xs[K / 4];
    __shared__ int sumx[K / 4];
    for (int i = threadIdx.x; i < K / 4; i += 128) {
        const int v = xq[i];
        xs[i]   = v;
        sumx[i] = __dp4a(0x01010101, v, 0);      // hoisted out of the row loop
    }
    __syncthreads();

    const uint8_t * m = base + (size_t)sel[blockIdx.x] * B2_MAT_BYTES;
    for (int r = 0; r < ROWS; r++) {
        const uint8_t  * row = m + (size_t)r * B2_ROW_BYTES;
        const uint32_t * q   = (const uint32_t *)row;
        const __half   * sc  = (const __half   *)(row + (K / 16) * 4);

        const int t = threadIdx.x;                // one uint32 per thread == 16 weights
        const uint32_t v = q[t];
        int acc = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int qj = (int)((v >> (2 * j)) & 0x03030303u);
            acc += __dp4a(qj, xs[4 * t + j], 0) - 2 * sumx[4 * t + j];
        }
        float f = (float)acc * __half2float(sc[t >> 1]);
        for (int off = 16; off > 0; off >>= 1) f += __shfl_down_sync(0xffffffff, f, off);
        __shared__ float warp[4];
        if ((t & 31) == 0) warp[t >> 5] = f;
        __syncthreads();
        if (t == 0) y[blockIdx.x * ROWS + r] = (warp[0] + warp[1] + warp[2] + warp[3]) * sx;
        __syncthreads();
    }
}

// ============================================================ L1f: "Q2_A" — the candidate format
// Asymmetric 2-bit, group 16, byte-aligned scale AND min, superblock 256. 3.125 bits/weight.
// quality.py measures this at 0.995x Q2_K's reconstruction RMSE on real weights — parity.
//
// Per 256-weight superblock: 64 B quants + 16 B scales + 16 B mins + fp16 d + fp16 dmin = 100 B.
// Row layout keeps each field in its own plane so every load is coalesced:
//   [128 x uint32 quants][128 B scales][128 B mins][8 x fp16 d][8 x fp16 dmin] = 800 B for K=2048
//
// One uint32 IS one group of 16 weights, pre-permuted (byte j holds weights j, j+4, j+8, j+12) so
// that (v >> 2j) & 0x03030303 yields four CONTIGUOUS weights straight into dp4a's int8 lanes.
//
// The asymmetric reconstruction x = q*s - m costs nothing extra in the inner loop:
//     sum((q*s - m) * x) = s * dp4a(q, x) - m * sum(x)
// and sum(x) is row-independent, so it is hoisted into shared memory once per block. That is the
// whole trick: Q2_K's asymmetry is paid per row, this pays it once.
static const int A2_ROW_BYTES = (K / 16) * 4 + (K / 16) + (K / 16) + (K / 256) * 4;  // 800 B
static const int A2_MAT_BYTES = ROWS * A2_ROW_BYTES;

__global__ __launch_bounds__(128) void k_matvec_q2a_dp4a(
        const uint8_t * __restrict__ base, const int * __restrict__ sel,
        const int * __restrict__ xq, float sx, float * __restrict__ y)
{
    __shared__ int xs[K / 4];
    __shared__ int sumx[K / 4];
    for (int i = threadIdx.x; i < K / 4; i += 128) {
        const int v = xq[i];
        xs[i]   = v;
        sumx[i] = __dp4a(0x01010101, v, 0);
    }
    __syncthreads();

    const int t  = threadIdx.x;           // one group of 16 weights per thread
    const int sb = t >> 4;                // superblock index
    const uint8_t * m = base + (size_t)sel[blockIdx.x] * A2_MAT_BYTES;

    // per-group activation sum, also row-independent
    const int sg = sumx[4*t] + sumx[4*t+1] + sumx[4*t+2] + sumx[4*t+3];

    for (int r = 0; r < ROWS; r++) {
        const uint8_t  * row  = m + (size_t)r * A2_ROW_BYTES;
        const uint32_t * q    = (const uint32_t *)row;
        const uint8_t  * ls   = row + (K/16)*4;
        const uint8_t  * lm   = ls  + (K/16);
        const __half   * d    = (const __half *)(lm + (K/16));
        const __half   * dmin = d + (K/256);

        const uint32_t v = q[t];
        int acc = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            acc += __dp4a((int)((v >> (2*j)) & 0x03030303u), xs[4*t + j], 0);
        }
        const float s = (float)ls[t] * __half2float(d[sb]);
        const float mn = (float)lm[t] * __half2float(dmin[sb]);
        float f = s * (float)acc - mn * (float)sg;

        for (int off = 16; off > 0; off >>= 1) f += __shfl_down_sync(0xffffffff, f, off);
        __shared__ float warp[4];
        if ((t & 31) == 0) warp[t >> 5] = f;
        __syncthreads();
        if (t == 0) y[blockIdx.x * ROWS + r] = (warp[0] + warp[1] + warp[2] + warp[3]) * sx;
        __syncthreads();
    }
}

// ================================================== L1g: Q2_K-EQUIVALENT, the fairness control
// Q2_A's 1.70x was measured against llama.cpp's Q2_K running END-TO-END on a real model, which is
// not a like-for-like kernel comparison. This is: Q2_K's exact cost model - 2 bits/weight, and a
// 4-bit scale AND 4-bit min PACKED INTO ONE BYTE per 16-weight sub-block, plus superblock fp16
// d/dmin - in the same harness, same access pattern, same reduction. 2.625 bits/weight.
//
// The ONLY difference from Q2_A is that the scale and min arrive nibble-packed in one byte instead
// of from two separate byte planes: two extra ALU ops per 16 weights, and 0.5 fewer bits/weight.
// If this lands ABOVE Q2_A, then the format is not what makes llama.cpp's Q2_K slow.
static const int KQ_ROW_BYTES = (K / 16) * 4 + (K / 16) + (K / 256) * 4;   // 512 + 128 + 32 = 672
static const int KQ_MAT_BYTES = ROWS * KQ_ROW_BYTES;

__global__ __launch_bounds__(128) void k_matvec_q2k_equiv(
        const uint8_t * __restrict__ base, const int * __restrict__ sel,
        const int * __restrict__ xq, float sx, float * __restrict__ y)
{
    __shared__ int xs[K / 4];
    __shared__ int sumx[K / 4];
    for (int i = threadIdx.x; i < K / 4; i += 128) {
        const int v = xq[i];
        xs[i] = v; sumx[i] = __dp4a(0x01010101, v, 0);
    }
    __syncthreads();

    const int t = threadIdx.x, sb = t >> 4;
    const uint8_t * m = base + (size_t)sel[blockIdx.x] * KQ_MAT_BYTES;
    const int sg = sumx[4*t] + sumx[4*t+1] + sumx[4*t+2] + sumx[4*t+3];

    for (int r = 0; r < ROWS; r++) {
        const uint8_t  * row  = m + (size_t)r * KQ_ROW_BYTES;
        const uint32_t * q    = (const uint32_t *)row;
        const uint8_t  * sm   = row + (K/16)*4;                 // packed: low nibble scale, high min
        const __half   * d    = (const __half *)(sm + (K/16));
        const __half   * dmin = d + (K/256);

        const uint32_t v = q[t];
        int acc = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc += __dp4a((int)((v >> (2*j)) & 0x03030303u), xs[4*t + j], 0);

        const uint8_t p = sm[t];
        const float s  = (float)(p & 0xF)  * __half2float(d[sb]);      // <- the extra unpack
        const float mn = (float)(p >>  4)  * __half2float(dmin[sb]);
        float f = s * (float)acc - mn * (float)sg;

        for (int off = 16; off > 0; off >>= 1) f += __shfl_down_sync(0xffffffff, f, off);
        __shared__ float warp[4];
        if ((t & 31) == 0) warp[t >> 5] = f;
        __syncthreads();
        if (t == 0) y[blockIdx.x * ROWS + r] = (warp[0] + warp[1] + warp[2] + warp[3]) * sx;
        __syncthreads();
    }
}

// ===================================== L1h: warp-per-row. Removing the per-row block barrier.
// k_matvec_q4_dp4a pays a 128-thread reduction + TWO __syncthreads for EVERY output row - 768
// times per matrix. Here each warp owns a row outright: the reduction is 5 __shfl_down with no
// cross-warp traffic and no barrier at all, and the block synchronises exactly once, at load time.
// The row-invariant sum(x) terms are hoisted into shared memory as before.
// Same format, same bytes, same access pattern as L1d - only the thread mapping changes.
__global__ __launch_bounds__(128) void k_matvec_q4_dp4a_warp(
        const uint8_t * __restrict__ base, const int * __restrict__ sel,
        const int * __restrict__ xq, float sx, float * __restrict__ y)
{
    __shared__ int xs[K / 4];
    __shared__ int sumx[K / 4];
    for (int i = threadIdx.x; i < K / 4; i += 128) {
        const int v = xq[i];
        xs[i] = v; sumx[i] = __dp4a(0x01010101, v, 0);
    }
    __syncthreads();                                   // once per BLOCK, not once per row

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint8_t * m = base + (size_t)sel[blockIdx.x] * MAT_BYTES;

    for (int r = warp; r < ROWS; r += 4) {             // 4 warps -> 4 rows in flight
        const uint8_t  * row = m + (size_t)r * ROW_BYTES;
        const uint32_t * q   = (const uint32_t *)row;
        const __half   * sc  = (const __half   *)(row + QB_PER_ROW);
        float acc = 0.0f;
        #pragma unroll
        for (int p = 0; p < 8; p++) {                  // 256 u32 per row / 32 lanes
            const int i  = lane + p * 32;
            const int sb = i >> 5, l = i & 31;
            const uint32_t v = q[i];
            const int xlo = xs[sb*64 + l], xhi = xs[sb*64 + l + 32];
            acc += (float)(__dp4a((int)( v       & 0x0F0F0F0Fu), xlo, 0) - 8*sumx[sb*64 + l])
                   * __half2float(sc[sb*8 + (l >> 3)]);
            acc += (float)(__dp4a((int)((v >> 4) & 0x0F0F0F0Fu), xhi, 0) - 8*sumx[sb*64 + l + 32])
                   * __half2float(sc[sb*8 + (l >> 3) + 4]);
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) y[blockIdx.x * ROWS + r] = acc * sx;
    }
}

// ==================== L1m: prereg #56 — the K-quant min tax at llama.cpp's REAL decode geometry
// One output row per CUDA block (mmvq's rows_per_cuda_block=1 at decode, #55), 128 threads,
// activations read from GLOBAL memory (L1/L2-cached) — no smem staging, because at 1 row/block
// there is nothing to amortize it over. Three arms on the same 4.5-bit buffer:
//   (i)   sums recomputed via dp4a per group per row   <- llama.cpp's asymmetric K-quant pattern
//   (ii)  sums loaded from a per-token side buffer     <- the proposed fix (2 KB, L2-hot)
//   (iii) no offset work at all                        <- symmetric control (Q4_0-like)
// Arms i and ii are mathematically identical (checked); iii is a throughput control only.
template <int ARM>
__global__ __launch_bounds__(128) void k_mmvq_geom(
        const uint8_t * __restrict__ base, const int * __restrict__ xq,
        const int * __restrict__ sums4, float sx, float * __restrict__ y, int nmat)
{
    const int mat = blockIdx.x / ROWS;
    const int r   = blockIdx.x % ROWS;
    const uint8_t  * row = base + (size_t)mat * MAT_BYTES + (size_t)r * ROW_BYTES;
    const uint32_t * q   = (const uint32_t *)row;
    const __half   * sc  = (const __half   *)(row + QB_PER_ROW);

    float acc = 0.0f;
    #pragma unroll
    for (int p = 0; p < 2; p++) {
        const int i  = threadIdx.x + p * 128;
        const int sb = i >> 5, l = i & 31;
        const uint32_t v = q[i];
        const int xlo = xq[sb*64 + l];
        const int xhi = xq[sb*64 + l + 32];
        const int dlo = __dp4a((int)( v       & 0x0F0F0F0Fu), xlo, 0);
        const int dhi = __dp4a((int)((v >> 4) & 0x0F0F0F0Fu), xhi, 0);
        int slo, shi;
        if (ARM == 0) {                       // (i) recompute sums on the ALU port, per row
            slo = __dp4a(0x01010101, xlo, 0);
            shi = __dp4a(0x01010101, xhi, 0);
        } else if (ARM == 1) {                // (ii) cached loads from the per-token side buffer
            slo = sums4[sb*64 + l];
            shi = sums4[sb*64 + l + 32];
        } else {                              // (iii) symmetric control: no offset work
            slo = 0; shi = 0;
        }
        if (ARM == 3) {                       // (iv) K-quant-style PACKED metadata decode:
            // same math as (i) but the scale arrives nibble-packed with a min and must be
            // shift/masked out and rescaled by superblock halves - the Q2_K/Q4_K metadata cost.
            const uint8_t p1 = (uint8_t)sc[sb*8 + (l >> 3)];         // reuse buffer as packed bytes
            const uint8_t p2 = (uint8_t)sc[sb*8 + (l >> 3) + 4];
            const float d1 = (float)(p1 & 0xF) * 0.05f, m1 = (float)(p1 >> 4) * 0.01f;
            const float d2 = (float)(p2 & 0xF) * 0.05f, m2 = (float)(p2 >> 4) * 0.01f;
            acc += (float)dlo * d1 - m1 * (float)slo;
            acc += (float)dhi * d2 - m2 * (float)shi;
        } else {
            acc += (float)(dlo - 8*slo) * __half2float(sc[sb*8 + (l >> 3)]);
            acc += (float)(dhi - 8*shi) * __half2float(sc[sb*8 + (l >> 3) + 4]);
        }
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    __shared__ float warp[4];
    if ((threadIdx.x & 31) == 0) warp[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) y[blockIdx.x] = (warp[0] + warp[1] + warp[2] + warp[3]) * sx;
}

// ==================== L1n: prereg #57 — the LAYOUT WALK oracle. Matched pairs, only bytes move.
// Q2_K cost model: 256-weight superblocks, per-16 packed scale/min nibbles, fp16 d/dmin,
// QR2_K-style unpack (one u32 = 16 weights via 4 shift+dp4a), min via dp4a (free per #56).
// Thread map = llama.cpp mmvq: 16 lanes per superblock, lane reads word (t%16). 128 thr = 8 sb = K.
//
// LAY=0 planar:  [all qs words][all scale bytes][all d/dmin half pairs]  (coalesced)
// LAY=1 struct:  84-byte blocks: scales[16] | qs[64] | d,dmin            (llama.cpp block_q2_K)
template <int LAY>
__global__ __launch_bounds__(128) void k_q2k_layout(
        const uint8_t * __restrict__ base, const int * __restrict__ xq,
        float sx, float * __restrict__ y, int rows_total)
{
    const int t   = threadIdx.x;
    const int sb  = t >> 4;                    // superblock 0..7
    const int w   = t & 15;                    // word within the superblock's 64B qs
    const size_t row_bytes = (K / 256) * 84;   // 672 B either way — SAME bytes, moved
    const uint8_t * row = base + (size_t)blockIdx.x * row_bytes;

    const uint8_t  * qs_p; const uint8_t * sc_p; const __half * dm_p;
    if (LAY == 0) {                            // planar planes for the whole row
        qs_p = row;                                        // K/4 = 512 B of qs
        sc_p = row + K/4;                                  // K/16 = 128 B of packed scale/min
        dm_p = (const __half *)(row + K/4 + K/16);         // 8 x (d,dmin) half pairs
    } else {                                   // interleaved 84 B struct per superblock
        const uint8_t * blk = row + (size_t)sb * 84;
        sc_p = blk;                                        // scales[16] at the block head
        qs_p = blk + 16;                                   // qs[64]
        dm_p = (const __half *)(blk + 80);                 // d, dmin
    }

    const uint32_t v = (LAY == 1) ? ((const uint32_t *)qs_p)[w]
                                  : ((const uint32_t *)qs_p)[sb*16 + w];
    float acc = 0.0f;
    const float d    = __half2float(LAY == 1 ? dm_p[0] : dm_p[2*sb]);
    const float dmin = __half2float(LAY == 1 ? dm_p[1] : dm_p[2*sb+1]);
    if (LAY == 2) {
        // post-#57 exploratory arm (NOT staked): identical loads and dp4a count, but the
        // scale/min are applied ONCE PER u32 (16 weights) instead of once per quad — the
        // metadata-application-DENSITY test. Uses the first scale byte for all four quads,
        // so it is a THROUGHPUT arm only (different math, no bitwise check).
        int accd = 0, accm = 0;
        const uint8_t p = sc_p[sb*16 + (w >> 2)];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int qv = (int)((v >> (2*i)) & 0x03030303u);
            const int xv = xq[sb*64 + i*16 + w];
            accd += __dp4a(qv, xv, 0);
            accm += __dp4a(0x01010101, xv, 0);
        }
        acc = (float)accd * d * (float)(p & 0xF) - (float)accm * dmin * (float)(p >> 4);
    } else {
    #pragma unroll
    for (int i = 0; i < 4; i++) {              // 4 weight-quads per u32, QR2_K pattern
        const int qv = (int)((v >> (2*i)) & 0x03030303u);
        // logical weights: superblock sb, plane i, word w -> x ints at sb*64 + i*16 + w
        const int xv = xq[sb*64 + i*16 + w];
        const uint8_t p = (LAY == 1) ? sc_p[i*4 + (w >> 2)]
                                     : sc_p[sb*16 + i*4 + (w >> 2)];
        acc += (float)__dp4a(qv, xv, 0)          * d    * (float)(p & 0xF);
        acc -= (float)__dp4a(0x01010101, xv, 0)  * dmin * (float)(p >> 4);
    }
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    __shared__ float warp[4];
    if ((t & 31) == 0) warp[t >> 5] = acc;
    __syncthreads();
    if (t == 0) y[blockIdx.x] = (warp[0] + warp[1] + warp[2] + warp[3]) * sx;
}

// Q4_K pair: 144-byte blocks (dm | scales[12] | qs[128]), 16 lanes per block, lane reads
// words 2*(t%16) and 2*(t%16)+1 of the block's 128 B qs. Planar twin has the same logical map.
template <int LAY>
__global__ __launch_bounds__(128) void k_q4k_layout(
        const uint8_t * __restrict__ base, const int * __restrict__ xq,
        float sx, float * __restrict__ y, int rows_total)
{
    const int t  = threadIdx.x;
    const int sb = t >> 4;                     // superblock 0..7 (K=2048 -> 8 x 256 weights)
    const int w  = (t & 15) * 2;               // first of two words in the 32-word qs
    const size_t row_bytes = (K / 256) * 144;  // 1152 B either way
    const uint8_t * row = base + (size_t)blockIdx.x * row_bytes;

    const uint8_t * qs_p; const uint8_t * sc_p; const __half * dm_p;
    if (LAY == 0) {
        qs_p = row;                                        // K/2 = 1024 B of qs
        sc_p = row + K/2;                                  // 8 x 12 B of scales
        dm_p = (const __half *)(row + K/2 + (K/256)*12);   // 8 x (d,dmin)
    } else {
        const uint8_t * blk = row + (size_t)sb * 144;
        dm_p = (const __half *)blk;
        sc_p = blk + 4;
        qs_p = blk + 16;
    }

    const uint32_t v0 = (LAY == 0) ? ((const uint32_t *)qs_p)[sb*32 + w]     : ((const uint32_t *)qs_p)[w];
    const uint32_t v1 = (LAY == 0) ? ((const uint32_t *)qs_p)[sb*32 + w + 1] : ((const uint32_t *)qs_p)[w + 1];
    const float d    = __half2float(LAY == 0 ? dm_p[2*sb]   : dm_p[0]);
    const float dmin = __half2float(LAY == 0 ? dm_p[2*sb+1] : dm_p[1]);
    const uint8_t s0 = (LAY == 0) ? sc_p[sb*12 + (w >> 2)]     : sc_p[w >> 2];
    const uint8_t s1 = (LAY == 0) ? sc_p[sb*12 + (w >> 2) + 4] : sc_p[(w >> 2) + 4];

    float acc = 0.0f;
    const int x0 = xq[sb*64 + w],     x1 = xq[sb*64 + w + 1];      // lo nibbles: weights 4w..
    const int x2 = xq[sb*64 + w + 32], x3 = xq[sb*64 + w + 33];    // hi nibbles: +128 weights
    acc += (float)(__dp4a((int)(v1 & 0x0F0F0F0Fu), x1, __dp4a((int)(v0 & 0x0F0F0F0Fu), x0, 0)))
           * d * (float)(s0 & 0x3F);
    acc -= (float)(__dp4a(0x01010101, x1, __dp4a(0x01010101, x0, 0)))
           * dmin * (float)(s0 >> 6);
    acc += (float)(__dp4a((int)((v1 >> 4) & 0x0F0F0F0Fu), x3, __dp4a((int)((v0 >> 4) & 0x0F0F0F0Fu), x2, 0)))
           * d * (float)(s1 & 0x3F);
    acc -= (float)(__dp4a(0x01010101, x3, __dp4a(0x01010101, x2, 0)))
           * dmin * (float)(s1 >> 6);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    __shared__ float warp[4];
    if ((t & 31) == 0) warp[t >> 5] = acc;
    __syncthreads();
    if (t == 0) y[blockIdx.x] = (warp[0] + warp[1] + warp[2] + warp[3]) * sx;
}

static double bench(void (*launch)(void*), void* arg, int iters, double bytes) {
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    launch(arg); CHECK(cudaDeviceSynchronize());              // warm
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) launch(arg);
    CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return bytes * iters / (ms / 1000.0) / 1e9;               // GB/s
}

struct SArg { const uint4* p; size_t n4; unsigned long long* out; int grid; };
static void launch_stream(void* v) {
    SArg* a = (SArg*)v;
    k_stream<<<a->grid, 256>>>(a->p, a->n4, a->out);
}
struct MArg { const uint8_t* base; const int* sel; const float* x; float* y; int blocks; };
static void launch_matvec(void* v) {
    MArg* a = (MArg*)v;
    k_matvec<<<a->blocks, 128>>>(a->base, a->sel, a->x, a->y);
}

int main(int argc, char** argv) {
    int target_mb = (argc > 1) ? atoi(argv[1]) : 512;
    int gather_g  = (argc > 2) ? atoi(argv[2]) : 16;    // touch 1 in G matrices (MoE: 8 of 128)

    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    const int NMAT = (int)(((size_t)target_mb << 20) / MAT_BYTES);
    const size_t total = (size_t)NMAT * MAT_BYTES;
    printf("device      : %s (cc %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("spec peak   : %.1f GB/s  (%.0f MHz x %d-bit)\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6,
           prop.memoryClockRate / 1000.0, prop.memoryBusWidth);
    printf("buffer      : %d matrices of %dx%d @ 4.5 bit = %.1f MB\n",
           NMAT, ROWS, K, total / 1048576.0);

    // ---- host init: random nibbles, scales ~1/8 so dequantized values stay O(1)
    std::vector<uint8_t> h(total);
    for (size_t i = 0; i < total; i++) h[i] = (uint8_t)(rand() & 0xFF);
    for (int m = 0; m < NMAT; m++)
        for (int r = 0; r < ROWS; r++) {
            __half* sc = (__half*)(h.data() + (size_t)m * MAT_BYTES + (size_t)r * ROW_BYTES + QB_PER_ROW);
            for (int s = 0; s < SC_PER_ROW; s++) sc[s] = __float2half(0.125f);
        }
    std::vector<float> hx(K);
    for (int i = 0; i < K; i++) hx[i] = (float)((i % 17) - 8) / 16.0f;

    uint8_t* d_base; CHECK(cudaMalloc(&d_base, total));
    CHECK(cudaMemcpy(d_base, h.data(), total, cudaMemcpyHostToDevice));
    float* d_x; CHECK(cudaMalloc(&d_x, K * sizeof(float)));
    CHECK(cudaMemcpy(d_x, hx.data(), K * sizeof(float), cudaMemcpyHostToDevice));
    // y must hold one entry per (block, row) for the DENSEST format in the ladder: a format with
    // fewer bits/weight fits more matrices in the same byte budget, so it launches more blocks.
    const size_t max_blocks = (((size_t)target_mb << 20) / B2_MAT_BYTES) + 1;
    float* d_y; CHECK(cudaMalloc(&d_y, max_blocks * ROWS * sizeof(float)));
    unsigned long long* d_acc; CHECK(cudaMalloc(&d_acc, 8)); CHECK(cudaMemset(d_acc, 0, 8));

    // selection arrays
    std::vector<int> sel_all(NMAT), sel_gat;
    for (int i = 0; i < NMAT; i++) sel_all[i] = i;
    // scatter the gathered picks across the whole buffer, like expert ids across a weight tensor
    for (int i = 0; i < NMAT; i += gather_g) sel_gat.push_back((i * 7919) % NMAT);
    int* d_sel_all; CHECK(cudaMalloc(&d_sel_all, NMAT * sizeof(int)));
    CHECK(cudaMemcpy(d_sel_all, sel_all.data(), NMAT * sizeof(int), cudaMemcpyHostToDevice));
    int* d_sel_gat; CHECK(cudaMalloc(&d_sel_gat, sel_gat.size() * sizeof(int)));
    CHECK(cudaMemcpy(d_sel_gat, sel_gat.data(), sel_gat.size() * sizeof(int), cudaMemcpyHostToDevice));

    // ---- correctness: one row on the CPU vs the kernel, before any number is believed
    {
        MArg a{d_base, d_sel_all, d_x, d_y, NMAT};
        launch_matvec(&a); CHECK(cudaDeviceSynchronize());
        std::vector<float> hy(ROWS); CHECK(cudaMemcpy(hy.data(), d_y, ROWS * sizeof(float), cudaMemcpyDeviceToHost));
        const uint8_t* row = h.data();  // matrix 0, row 0
        const __half* sc = (const __half*)(row + QB_PER_ROW);
        double ref = 0.0;
        for (int sb = 0; sb < SB_PER_ROW; sb++)
            for (int b = 0; b < 128; b++) {
                uint8_t byte = row[sb * 128 + b];
                int wlo = sb * 256 + b, whi = wlo + 128;
                ref += ((double)(byte & 0xF) - 8.0) * (double)__half2float(sc[sb * 8 + (b >> 5)])       * hx[wlo];
                ref += ((double)(byte >>  4) - 8.0) * (double)__half2float(sc[sb * 8 + ((b + 128) >> 5)]) * hx[whi];
            }
        double err = fabs(ref - hy[0]) / (fabs(ref) + 1e-6);
        printf("correctness : row0 gpu %.6f vs cpu %.6f  rel.err %.2e  %s\n",
               hy[0], ref, err, err < 1e-4 ? "OK" : "*** MISMATCH — numbers below are meaningless ***");
        if (!(err < 1e-4)) return 1;
    }

    printf("\n%-26s %12s %12s %10s\n", "level", "GB/s", "vs spec", "vs L0");
    const double spec = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;

    SArg sa{(const uint4*)d_base, total / 16, d_acc, 2048};
    double l0 = bench(launch_stream, &sa, 20, (double)total);
    printf("%-26s %12.1f %11.2f %10s\n", "L0 stream read", l0, l0 / spec, "1.00");

    MArg ma{d_base, d_sel_all, d_x, d_y, NMAT};
    double l1 = bench(launch_matvec, &ma, 20, (double)total);
    printf("%-26s %12.1f %11.2f %10.2f\n", "L1 dequant matvec contig", l1, l1 / spec, l1 / l0);

    MArg mg{d_base, d_sel_gat, d_x, d_y, (int)sel_gat.size()};
    double gbytes = (double)sel_gat.size() * MAT_BYTES;
    double l2 = bench(launch_matvec, &mg, 200, gbytes);
    printf("%-26s %12.1f %11.2f %10.2f   (1 in %d, %.1f MB/pass)\n", "L2 dequant matvec gather",
           l2, l2 / spec, l2 / l0, gather_g, gbytes / 1048576.0);

    // ---- L1b fp16 and L1c int8: same weight COUNT, different formats.
    // Reported in GWeights/s as well, because that is the decision-relevant number: decode time
    // for a given model is weights/second, and a format that reads more bytes can still win if it
    // unpacks cheaply enough.
    const double weights_l1 = (double)NMAT * ROWS * K;
    {
        const int nf = (int)(((size_t)target_mb << 20) / F16_MAT_BYTES);
        uint8_t* d_f; if (cudaMalloc(&d_f, (size_t)nf * F16_MAT_BYTES) == cudaSuccess) {
            std::vector<__half> hf((size_t)nf * F16_MAT_BYTES / 2, __float2half(0.05f));
            CHECK(cudaMemcpy(d_f, hf.data(), (size_t)nf * F16_MAT_BYTES, cudaMemcpyHostToDevice));
            std::vector<int> s(nf); for (int i = 0; i < nf; i++) s[i] = i;
            int* d_s; CHECK(cudaMalloc(&d_s, nf * sizeof(int)));
            CHECK(cudaMemcpy(d_s, s.data(), nf * sizeof(int), cudaMemcpyHostToDevice));
            struct A { const uint8_t* b; const int* s; const float* x; float* y; int n; } a{d_f, d_s, d_x, d_y, nf};
            auto L = [](void* v) { A* a = (A*)v; k_matvec_f16<<<a->n, 128>>>(a->b, a->s, a->x, a->y); };
            double by = (double)nf * F16_MAT_BYTES;
            double g = bench(L, &a, 20, by);
            printf("%-26s %12.1f %11.2f %10.2f   %6.2f GW/s  (16.0 bit)\n", "L1b fp16 matvec (no unpack)",
                   g, g / spec, g / l0, g * 1e9 / 2.0 / 1e9);
            CHECK(cudaFree(d_f)); CHECK(cudaFree(d_s));
        }
    }
    {
        const int nq = (int)(((size_t)target_mb << 20) / Q8_MAT_BYTES);
        uint8_t* d_q; if (cudaMalloc(&d_q, (size_t)nq * Q8_MAT_BYTES) == cudaSuccess) {
            std::vector<uint8_t> hq((size_t)nq * Q8_MAT_BYTES);
            for (size_t i = 0; i < hq.size(); i++) hq[i] = (uint8_t)(rand() & 0xFF);
            for (int m = 0; m < nq; m++) for (int r = 0; r < ROWS; r++) {
                __half* sc = (__half*)(hq.data() + (size_t)m * Q8_MAT_BYTES + (size_t)r * Q8_ROW_BYTES + K);
                for (int s = 0; s < K / 32; s++) sc[s] = __float2half(0.01f);
            }
            CHECK(cudaMemcpy(d_q, hq.data(), hq.size(), cudaMemcpyHostToDevice));
            float mx = 0; for (int i = 0; i < K; i++) mx = fmaxf(mx, fabsf(hx[i]));
            const float sx = mx / 127.0f;
            std::vector<int8_t> xi(K); for (int i = 0; i < K; i++) xi[i] = (int8_t)lrintf(hx[i] / sx);
            int* d_xq; CHECK(cudaMalloc(&d_xq, K)); CHECK(cudaMemcpy(d_xq, xi.data(), K, cudaMemcpyHostToDevice));
            std::vector<int> s(nq); for (int i = 0; i < nq; i++) s[i] = i;
            int* d_s; CHECK(cudaMalloc(&d_s, nq * sizeof(int)));
            CHECK(cudaMemcpy(d_s, s.data(), nq * sizeof(int), cudaMemcpyHostToDevice));
            struct A { const uint8_t* b; const int* s; const int* xq; float sx; float* y; int n; } a{d_q, d_s, d_xq, sx, d_y, nq};
            auto L = [](void* v) { A* a = (A*)v; k_matvec_q8<<<a->n, 128>>>(a->b, a->s, a->xq, a->sx, a->y); };
            double by = (double)nq * Q8_MAT_BYTES;
            double g = bench(L, &a, 20, by);
            printf("%-26s %12.1f %11.2f %10.2f   %6.2f GW/s  ( 8.5 bit, __dp4a)\n", "L1c int8 matvec (dp4a)",
                   g, g / spec, g / l0, g / 1.0625);
            CHECK(cudaFree(d_q)); CHECK(cudaFree(d_s)); CHECK(cudaFree(d_xq));
        }
    }
    printf("%-26s %12s %11s %10s   %6.2f GW/s  ( 4.5 bit)\n", "  [L1 for comparison]", "", "", "", l1 / 0.5625);

    // ---- L1d: same buffer, same bytes, dp4a unpack. Correctness checked against a double
    // reference computed on the SAME quantized x, so only the unpack path is under test.
    {
        float mx = 0; for (int i = 0; i < K; i++) mx = fmaxf(mx, fabsf(hx[i]));
        const float sx = mx / 127.0f;
        std::vector<int8_t> xi(K); for (int i = 0; i < K; i++) xi[i] = (int8_t)lrintf(hx[i] / sx);
        int* d_xq; CHECK(cudaMalloc(&d_xq, K)); CHECK(cudaMemcpy(d_xq, xi.data(), K, cudaMemcpyHostToDevice));
        struct A { const uint8_t* b; const int* s; const int* xq; float sx; float* y; int n; } a{d_base, d_sel_all, d_xq, sx, d_y, NMAT};
        auto L = [](void* v) { A* a = (A*)v; k_matvec_q4_dp4a<<<a->n, 128>>>(a->b, a->s, a->xq, a->sx, a->y); };
        L(&a); CHECK(cudaDeviceSynchronize());
        float gpu0; CHECK(cudaMemcpy(&gpu0, d_y, sizeof(float), cudaMemcpyDeviceToHost));
        const uint8_t* row = h.data(); const __half* sc = (const __half*)(row + QB_PER_ROW);
        double ref = 0.0;
        for (int sb = 0; sb < SB_PER_ROW; sb++) for (int b = 0; b < 128; b++) {
            uint8_t byte = row[sb * 128 + b];
            int wlo = sb * 256 + b, whi = wlo + 128;
            ref += ((double)(byte & 0xF) - 8.0) * (double)__half2float(sc[sb * 8 + (b >> 5)])         * (xi[wlo] * (double)sx);
            ref += ((double)(byte >>  4) - 8.0) * (double)__half2float(sc[sb * 8 + ((b + 128) >> 5)]) * (xi[whi] * (double)sx);
        }
        double err = fabs(ref - gpu0) / (fabs(ref) + 1e-6);
        double g = bench(L, &a, 20, (double)total);
        printf("%-26s %12.1f %11.2f %10.2f   %6.2f GW/s  ( 4.5 bit, __dp4a)  rel.err %.1e %s\n",
               "L1d q4 matvec (dp4a)", g, g / spec, g / l0, g / 0.5625, err,
               err < 1e-3 ? "OK" : "*** MISMATCH ***");
        CHECK(cudaFree(d_xq));
    }

    // ---- L1e: 2-bit dp4a-native. Built with a real host-side packer so the permutation is
    // exercised, and checked against a double reference that reads the ORIGINAL weight order.
    {
        const int n2 = (int)(((size_t)target_mb << 20) / B2_MAT_BYTES);
        std::vector<uint8_t> hb((size_t)n2 * B2_MAT_BYTES);
        std::vector<uint8_t> wref(ROWS * K);              // logical weights of matrix 0, in order
        for (int m = 0; m < n2; m++) for (int r = 0; r < ROWS; r++) {
            uint8_t * row = hb.data() + (size_t)m * B2_MAT_BYTES + (size_t)r * B2_ROW_BYTES;
            uint32_t * q = (uint32_t *)row;
            __half   * sc = (__half *)(row + (K / 16) * 4);
            for (int g = 0; g < K / 16; g++) {            // one uint32 per 16 weights
                uint8_t w[16];
                for (int i = 0; i < 16; i++) {
                    w[i] = (uint8_t)(rand() & 3);
                    if (m == 0) wref[(size_t)r * K + g * 16 + i] = w[i];
                }
                uint32_t v = 0;                            // byte j <- weights j, j+4, j+8, j+12
                for (int j = 0; j < 4; j++) {
                    uint32_t byte = w[j] | (w[j + 4] << 2) | (w[j + 8] << 4) | (w[j + 12] << 6);
                    v |= byte << (8 * j);
                }
                q[g] = v;
            }
            for (int s = 0; s < K / 32; s++) sc[s] = __float2half(0.25f);
        }
        uint8_t* d_b; CHECK(cudaMalloc(&d_b, hb.size()));
        CHECK(cudaMemcpy(d_b, hb.data(), hb.size(), cudaMemcpyHostToDevice));
        std::vector<int> s(n2); for (int i = 0; i < n2; i++) s[i] = i;
        int* d_s; CHECK(cudaMalloc(&d_s, n2 * sizeof(int)));
        CHECK(cudaMemcpy(d_s, s.data(), n2 * sizeof(int), cudaMemcpyHostToDevice));

        float mx = 0; for (int i = 0; i < K; i++) mx = fmaxf(mx, fabsf(hx[i]));
        const float sx = mx / 127.0f;
        std::vector<int8_t> xi(K); for (int i = 0; i < K; i++) xi[i] = (int8_t)lrintf(hx[i] / sx);
        int* d_xq; CHECK(cudaMalloc(&d_xq, K)); CHECK(cudaMemcpy(d_xq, xi.data(), K, cudaMemcpyHostToDevice));

        struct A { const uint8_t* b; const int* s; const int* xq; float sx; float* y; int n; } a{d_b, d_s, d_xq, sx, d_y, n2};
        auto L = [](void* v) { A* a = (A*)v; k_matvec_q2_dp4a<<<a->n, 128>>>(a->b, a->s, a->xq, a->sx, a->y); };
        L(&a); CHECK(cudaDeviceSynchronize());
        float gpu0; CHECK(cudaMemcpy(&gpu0, d_y, sizeof(float), cudaMemcpyDeviceToHost));
        double ref = 0.0;                                   // reference in LOGICAL weight order
        for (int i = 0; i < K; i++)
            ref += ((double)wref[i] - 2.0) * 0.25 * (xi[i] * (double)sx);
        double err = fabs(ref - gpu0) / (fabs(ref) + 1e-6);
        double by = (double)n2 * B2_MAT_BYTES;
        double g = bench(L, &a, 20, by);
        printf("%-26s %12.1f %11.2f %10.2f   %6.2f GW/s  ( 2.5 bit, __dp4a)  rel.err %.1e %s\n",
               "L1e q2 matvec (dp4a)", g, g / spec, g / l0, g / 0.3125, err,
               err < 1e-3 ? "OK" : "*** MISMATCH ***");
        CHECK(cudaFree(d_b)); CHECK(cudaFree(d_s)); CHECK(cudaFree(d_xq));
    }

    // ---- L1f: Q2_A, the parity-quality candidate. Packed by a real host packer that exercises
    // the permutation, and checked against a double reference read in LOGICAL weight order.
    {
        const int na = (int)(((size_t)target_mb << 20) / A2_MAT_BYTES);
        std::vector<uint8_t> ha((size_t)na * A2_MAT_BYTES);
        std::vector<uint8_t> qref(K); std::vector<float> sref(K/16), mref(K/16);
        for (int mm = 0; mm < na; mm++) for (int r = 0; r < ROWS; r++) {
            uint8_t * row = ha.data() + (size_t)mm * A2_MAT_BYTES + (size_t)r * A2_ROW_BYTES;
            uint32_t * q = (uint32_t *)row;
            uint8_t * ls = row + (K/16)*4, * lm = ls + (K/16);
            __half * d = (__half *)(lm + (K/16)), * dmn = d + (K/256);
            for (int s = 0; s < K/256; s++) { d[s] = __float2half(0.002f); dmn[s] = __float2half(0.003f); }
            for (int g = 0; g < K/16; g++) {
                uint8_t w[16];
                for (int i = 0; i < 16; i++) {
                    w[i] = (uint8_t)(rand() & 3);
                    if (mm == 0 && r == 0) qref[g*16 + i] = w[i];
                }
                uint32_t v = 0;                       // byte j <- weights j, j+4, j+8, j+12
                for (int j = 0; j < 4; j++)
                    v |= (uint32_t)(w[j] | (w[j+4] << 2) | (w[j+8] << 4) | (w[j+12] << 6)) << (8*j);
                q[g] = v;
                ls[g] = (uint8_t)(rand() & 0xFF);
                lm[g] = (uint8_t)(rand() & 0xFF);
                if (mm == 0 && r == 0) { sref[g] = ls[g] * 0.002f; mref[g] = lm[g] * 0.003f; }
            }
        }
        uint8_t* d_a; CHECK(cudaMalloc(&d_a, ha.size()));
        CHECK(cudaMemcpy(d_a, ha.data(), ha.size(), cudaMemcpyHostToDevice));
        std::vector<int> s(na); for (int i = 0; i < na; i++) s[i] = i;
        int* d_s; CHECK(cudaMalloc(&d_s, na * sizeof(int)));
        CHECK(cudaMemcpy(d_s, s.data(), na * sizeof(int), cudaMemcpyHostToDevice));
        float mx = 0; for (int i = 0; i < K; i++) mx = fmaxf(mx, fabsf(hx[i]));
        const float sx = mx / 127.0f;
        std::vector<int8_t> xi(K); for (int i = 0; i < K; i++) xi[i] = (int8_t)lrintf(hx[i] / sx);
        int* d_xq; CHECK(cudaMalloc(&d_xq, K)); CHECK(cudaMemcpy(d_xq, xi.data(), K, cudaMemcpyHostToDevice));

        struct A { const uint8_t* b; const int* s; const int* xq; float sx; float* y; int n; } a{d_a, d_s, d_xq, sx, d_y, na};
        auto L = [](void* v) { A* a = (A*)v; k_matvec_q2a_dp4a<<<a->n, 128>>>(a->b, a->s, a->xq, a->sx, a->y); };
        L(&a); CHECK(cudaDeviceSynchronize());
        float gpu0; CHECK(cudaMemcpy(&gpu0, d_y, sizeof(float), cudaMemcpyDeviceToHost));
        double ref = 0.0;
        for (int i = 0; i < K; i++)
            ref += ((double)qref[i] * sref[i/16] - mref[i/16]) * (xi[i] * (double)sx);
        double err = fabs(ref - gpu0) / (fabs(ref) + 1e-6);
        double by = (double)na * A2_MAT_BYTES;
        double g = bench(L, &a, 20, by);
        printf("%-26s %12.1f %11.2f %10.2f   %6.2f GW/s  ( 3.125 bit, __dp4a)  rel.err %.1e %s\n",
               "L1f Q2_A asym (dp4a)", g, g / spec, g / l0, g / 0.390625, err,
               err < 1e-3 ? "OK" : "*** MISMATCH ***");
        CHECK(cudaFree(d_a)); CHECK(cudaFree(d_s)); CHECK(cudaFree(d_xq));
    }

    // ---- L1g: the fairness control. Same harness, Q2_K's cost model.
    {
        const int nk = (int)(((size_t)target_mb << 20) / KQ_MAT_BYTES);
        std::vector<uint8_t> hk((size_t)nk * KQ_MAT_BYTES);
        for (size_t i = 0; i < hk.size(); i++) hk[i] = (uint8_t)(rand() & 0xFF);
        for (int mm = 0; mm < nk; mm++) for (int r = 0; r < ROWS; r++) {
            uint8_t * row = hk.data() + (size_t)mm * KQ_MAT_BYTES + (size_t)r * KQ_ROW_BYTES;
            __half * d = (__half *)(row + (K/16)*4 + (K/16));
            for (int s = 0; s < (K/256)*2; s++) d[s] = __float2half(0.002f);
        }
        uint8_t* d_k; CHECK(cudaMalloc(&d_k, hk.size()));
        CHECK(cudaMemcpy(d_k, hk.data(), hk.size(), cudaMemcpyHostToDevice));
        std::vector<int> s(nk); for (int i = 0; i < nk; i++) s[i] = i;
        int* d_s; CHECK(cudaMalloc(&d_s, nk * sizeof(int)));
        CHECK(cudaMemcpy(d_s, s.data(), nk * sizeof(int), cudaMemcpyHostToDevice));
        float mx = 0; for (int i = 0; i < K; i++) mx = fmaxf(mx, fabsf(hx[i]));
        const float sxx = mx / 127.0f;
        std::vector<int8_t> xi(K); for (int i = 0; i < K; i++) xi[i] = (int8_t)lrintf(hx[i] / sxx);
        int* d_xq; CHECK(cudaMalloc(&d_xq, K)); CHECK(cudaMemcpy(d_xq, xi.data(), K, cudaMemcpyHostToDevice));
        struct A { const uint8_t* b; const int* s; const int* xq; float sx; float* y; int n; } a{d_k, d_s, d_xq, sxx, d_y, nk};
        auto L = [](void* v) { A* a = (A*)v; k_matvec_q2k_equiv<<<a->n, 128>>>(a->b, a->s, a->xq, a->sx, a->y); };
        double g = bench(L, &a, 20, (double)nk * KQ_MAT_BYTES);
        printf("%-26s %12.1f %11.2f %10.2f   %6.2f GW/s  ( 2.625 bit, __dp4a)  <- FAIRNESS CONTROL\n",
               "L1g Q2_K-equivalent", g, g / spec, g / l0, g / 0.328125);
        CHECK(cudaFree(d_k)); CHECK(cudaFree(d_s)); CHECK(cudaFree(d_xq));
    }

    // ---- L1h warp-per-row, and L2d gather+dp4a. Both reuse the L1 buffer, so bytes are identical
    // to L1d and L2 respectively and only the thread mapping / access pattern differs.
    {
        float mx = 0; for (int i = 0; i < K; i++) mx = fmaxf(mx, fabsf(hx[i]));
        const float sxx = mx / 127.0f;
        std::vector<int8_t> xi(K); for (int i = 0; i < K; i++) xi[i] = (int8_t)lrintf(hx[i] / sxx);
        int* d_xq; CHECK(cudaMalloc(&d_xq, K)); CHECK(cudaMemcpy(d_xq, xi.data(), K, cudaMemcpyHostToDevice));
        struct A { const uint8_t* b; const int* s; const int* xq; float sx; float* y; int n; };

        // reference value from the already-verified L1d kernel, to prove the remap is correct
        A ad{d_base, d_sel_all, d_xq, sxx, d_y, NMAT};
        k_matvec_q4_dp4a<<<NMAT, 128>>>(ad.b, ad.s, ad.xq, ad.sx, ad.y);
        CHECK(cudaDeviceSynchronize());
        std::vector<float> ref(8); CHECK(cudaMemcpy(ref.data(), d_y, 8*sizeof(float), cudaMemcpyDeviceToHost));

        A ah{d_base, d_sel_all, d_xq, sxx, d_y, NMAT};
        auto LH = [](void* v) { A* a = (A*)v; k_matvec_q4_dp4a_warp<<<a->n, 128>>>(a->b, a->s, a->xq, a->sx, a->y); };
        LH(&ah); CHECK(cudaDeviceSynchronize());
        std::vector<float> got(8); CHECK(cudaMemcpy(got.data(), d_y, 8*sizeof(float), cudaMemcpyDeviceToHost));
        double werr = 0; for (int i = 0; i < 8; i++)
            werr = fmax(werr, fabs(ref[i]-got[i]) / (fabs(ref[i]) + 1e-6));
        double gh = bench(LH, &ah, 20, (double)total);
        printf("%-26s %12.1f %11.2f %10.2f   %6.2f GW/s  ( 4.5 bit, warp/row)  rel.err %.1e %s\n",
               "L1h q4 dp4a WARP-PER-ROW", gh, gh / spec, gh / l0, gh / 0.5625, werr,
               werr < 1e-5 ? "OK" : "*** MISMATCH ***");

        A ag{d_base, d_sel_gat, d_xq, sxx, d_y, (int)sel_gat.size()};
        auto LG = [](void* v) { A* a = (A*)v; k_matvec_q4_dp4a<<<a->n, 128>>>(a->b, a->s, a->xq, a->sx, a->y); };
        double gbytes2 = (double)sel_gat.size() * MAT_BYTES;
        double gg = bench(LG, &ag, 200, gbytes2);
        printf("%-26s %12.1f %11.2f %10.2f   %6.2f GW/s  ( 4.5 bit, GATHERED 1-in-%d)\n",
               "L2d q4 dp4a GATHER (MoE)", gg, gg / spec, gg / l0, gg / 0.5625, gather_g);
        CHECK(cudaFree(d_xq));
    }

    // ---- L1m: prereg #56, the min-tax arms at 1-row-per-block geometry.
    {
        float mx = 0; for (int i = 0; i < K; i++) mx = fmaxf(mx, fabsf(hx[i]));
        const float sxx = mx / 127.0f;
        std::vector<int8_t> xi(K); for (int i = 0; i < K; i++) xi[i] = (int8_t)lrintf(hx[i] / sxx);
        int* d_xq; CHECK(cudaMalloc(&d_xq, K)); CHECK(cudaMemcpy(d_xq, xi.data(), K, cudaMemcpyHostToDevice));
        // per-token side buffer: sum of each aligned 4-activation group (the "once per token" work)
        std::vector<int> s4(K / 4);
        for (int i = 0; i < K / 4; i++)
            s4[i] = xi[4*i] + xi[4*i+1] + xi[4*i+2] + xi[4*i+3];
        int* d_s4; CHECK(cudaMalloc(&d_s4, (K/4) * sizeof(int)));
        CHECK(cudaMemcpy(d_s4, s4.data(), (K/4) * sizeof(int), cudaMemcpyHostToDevice));

        struct A { const uint8_t* b; const int* xq; const int* s4; float sx; float* y; int n; };
        A a{d_base, d_xq, d_s4, sxx, d_y, NMAT};
        auto L0a = [](void* v) { A* a = (A*)v; k_mmvq_geom<0><<<a->n * ROWS, 128>>>(a->b, a->xq, a->s4, a->sx, a->y, a->n); };
        auto L1a = [](void* v) { A* a = (A*)v; k_mmvq_geom<1><<<a->n * ROWS, 128>>>(a->b, a->xq, a->s4, a->sx, a->y, a->n); };
        auto L2a = [](void* v) { A* a = (A*)v; k_mmvq_geom<2><<<a->n * ROWS, 128>>>(a->b, a->xq, a->s4, a->sx, a->y, a->n); };

        // P-3: arms i and ii must agree bit-for-bit on the first 8 outputs
        L0a(&a); CHECK(cudaDeviceSynchronize());
        std::vector<float> yi(8); CHECK(cudaMemcpy(yi.data(), d_y, 8*sizeof(float), cudaMemcpyDeviceToHost));
        L1a(&a); CHECK(cudaDeviceSynchronize());
        std::vector<float> yii(8); CHECK(cudaMemcpy(yii.data(), d_y, 8*sizeof(float), cudaMemcpyDeviceToHost));
        int same = 1; for (int i = 0; i < 8; i++) same &= (yi[i] == yii[i]);

        auto L3a = [](void* v) { A* a = (A*)v; k_mmvq_geom<3><<<a->n * ROWS, 128>>>(a->b, a->xq, a->s4, a->sx, a->y, a->n); };
        double g0 = bench(L0a, &a, 20, (double)total);
        double g1 = bench(L1a, &a, 20, (double)total);
        double g2 = bench(L2a, &a, 20, (double)total);
        double g3 = bench(L3a, &a, 20, (double)total);
        printf("\n--- prereg #56: K-quant min tax at mmvq geometry (1 row/block, global x) ---\n");
        printf("%-38s %8.1f GB/s  %7.2f GW/s\n", "(i)   sum-via-dp4a  (llama.cpp K-quant)", g0, g0 / 0.5625);
        printf("%-38s %8.1f GB/s  %7.2f GW/s   arms i==ii bitwise: %s\n",
               "(ii)  sum-via-side-buffer (the fix)", g1, g1 / 0.5625, same ? "YES" : "*** NO — VOID ***");
        printf("%-38s %8.1f GB/s  %7.2f GW/s\n", "(iii) symmetric control (no min term)", g2, g2 / 0.5625);
        printf("%-38s %8.1f GB/s  %7.2f GW/s\n", "(iv)  packed nibble scale+min decode", g3, g3 / 0.5625);
        printf("      min tax (iii vs i): %.1f%%   recovered by (ii): %.0f%%   metadata tax (i vs iv): %.1f%%\n",
               100.0 * (g2 - g0) / g0, (g2 > g0) ? 100.0 * (g1 - g0) / (g2 - g0) : 0.0,
               100.0 * (g0 - g3) / g0);
        CHECK(cudaFree(d_xq)); CHECK(cudaFree(d_s4));
    }

    // ---- L1n: prereg #57 — matched-pair layout oracle. Same logical content, two byte layouts.
    {
        float mx = 0; for (int i = 0; i < K; i++) mx = fmaxf(mx, fabsf(hx[i]));
        const float sxx = mx / 127.0f;
        std::vector<int8_t> xi(K); for (int i = 0; i < K; i++) xi[i] = (int8_t)lrintf(hx[i] / sxx);
        int* d_xq; CHECK(cudaMalloc(&d_xq, K)); CHECK(cudaMemcpy(d_xq, xi.data(), K, cudaMemcpyHostToDevice));
        const size_t budget = ((size_t)target_mb << 20) / 2;   // per layout buffer
        struct A { const uint8_t* b; const int* xq; float sx; float* y; int n; };

        printf("\n--- prereg #57: layout walk, matched pairs (mmvq geometry) ---\n");
        double rq2[2] = {0, 0}, rq4[2] = {0, 0};

        {   // ---------- Q2_K-shaped pair: 672 B/row logical, planar vs 84 B struct
            const int SBR = K / 256;                       // 8 superblocks per row
            const size_t row_b = (size_t)SBR * 84;
            const int nrows = (int)(budget / row_b);
            std::vector<uint8_t> h0(nrows * row_b), h1(nrows * row_b);
            srand(1234);
            for (int r = 0; r < nrows; r++) {
                uint8_t * p0 = h0.data() + (size_t)r * row_b;      // planar
                uint8_t * p1 = h1.data() + (size_t)r * row_b;      // struct
                for (int sb = 0; sb < SBR; sb++) {
                    uint8_t S[16]; uint32_t W[16];
                    for (int j = 0; j < 16; j++) { S[j] = (uint8_t)(rand() & 0xFF); W[j] = ((uint32_t)rand() << 16) ^ rand(); }
                    memcpy(p0 + (size_t)(sb*16)*4, W, 64);                          // qs plane
                    memcpy(p0 + (size_t)K/4 + sb*16, S, 16);                        // scale plane
                    __half * dm0 = (__half *)(p0 + K/4 + K/16);
                    dm0[2*sb] = __float2half(0.002f); dm0[2*sb+1] = __float2half(0.001f);
                    uint8_t * blk = p1 + (size_t)sb * 84;                           // struct
                    memcpy(blk, S, 16); memcpy(blk + 16, W, 64);
                    ((__half *)(blk + 80))[0] = __float2half(0.002f);
                    ((__half *)(blk + 80))[1] = __float2half(0.001f);
                }
            }
            uint8_t *d0, *d1;
            CHECK(cudaMalloc(&d0, h0.size())); CHECK(cudaMemcpy(d0, h0.data(), h0.size(), cudaMemcpyHostToDevice));
            CHECK(cudaMalloc(&d1, h1.size())); CHECK(cudaMemcpy(d1, h1.data(), h1.size(), cudaMemcpyHostToDevice));
            A a0{d0, d_xq, sxx, d_y, nrows}, a1{d1, d_xq, sxx, d_y, nrows};
            auto LP = [](void* v) { A* a = (A*)v; k_q2k_layout<0><<<a->n, 128>>>(a->b, a->xq, a->sx, a->y, a->n); };
            auto LS = [](void* v) { A* a = (A*)v; k_q2k_layout<1><<<a->n, 128>>>(a->b, a->xq, a->sx, a->y, a->n); };
            LP(&a0); CHECK(cudaDeviceSynchronize());
            std::vector<float> yp(8); CHECK(cudaMemcpy(yp.data(), d_y, 32, cudaMemcpyDeviceToHost));
            LS(&a1); CHECK(cudaDeviceSynchronize());
            std::vector<float> ys(8); CHECK(cudaMemcpy(ys.data(), d_y, 32, cudaMemcpyDeviceToHost));
            int same = (yp[0] != 0.0f); for (int i = 0; i < 8; i++) same &= (yp[i] == ys[i]);
            rq2[0] = bench(LP, &a0, 20, (double)nrows * row_b);
            rq2[1] = bench(LS, &a1, 20, (double)nrows * row_b);
            printf("Q2_K-shaped  planar %7.1f GB/s   struct-84B %7.1f GB/s   ratio %.3f   bitwise %s\n",
                   rq2[0], rq2[1], rq2[1] / rq2[0], same ? "OK" : "*** MISMATCH — VOID ***");
            auto LD = [](void* v) { A* a = (A*)v; k_q2k_layout<2><<<a->n, 128>>>(a->b, a->xq, a->sx, a->y, a->n); };
            double gd = bench(LD, &a0, 20, (double)nrows * row_b);
            printf("  exploratory: scale/min per-u32 instead of per-quad  %7.1f GB/s  (density arm, throughput only)\n", gd);
            CHECK(cudaFree(d0)); CHECK(cudaFree(d1));
        }
        {   // ---------- Q4_K-shaped pair: 1152 B/row, planar vs 144 B struct
            const int SBR = K / 256;
            const size_t row_b = (size_t)SBR * 144;
            const int nrows = (int)(budget / row_b);
            std::vector<uint8_t> h0(nrows * row_b), h1(nrows * row_b);
            srand(4321);
            for (int r = 0; r < nrows; r++) {
                uint8_t * p0 = h0.data() + (size_t)r * row_b;
                uint8_t * p1 = h1.data() + (size_t)r * row_b;
                for (int sb = 0; sb < SBR; sb++) {
                    uint8_t S[12]; uint32_t W[32];
                    for (int j = 0; j < 12; j++) S[j] = (uint8_t)(rand() & 0xFF);
                    for (int j = 0; j < 32; j++) W[j] = ((uint32_t)rand() << 16) ^ rand();
                    memcpy(p0 + (size_t)(sb*32)*4, W, 128);
                    memcpy(p0 + (size_t)K/2 + sb*12, S, 12);
                    __half * dm0 = (__half *)(p0 + K/2 + SBR*12);
                    dm0[2*sb] = __float2half(0.002f); dm0[2*sb+1] = __float2half(0.001f);
                    uint8_t * blk = p1 + (size_t)sb * 144;
                    ((__half *)blk)[0] = __float2half(0.002f);
                    ((__half *)blk)[1] = __float2half(0.001f);
                    memcpy(blk + 4, S, 12); memcpy(blk + 16, W, 128);
                }
            }
            uint8_t *d0, *d1;
            CHECK(cudaMalloc(&d0, h0.size())); CHECK(cudaMemcpy(d0, h0.data(), h0.size(), cudaMemcpyHostToDevice));
            CHECK(cudaMalloc(&d1, h1.size())); CHECK(cudaMemcpy(d1, h1.data(), h1.size(), cudaMemcpyHostToDevice));
            A a0{d0, d_xq, sxx, d_y, nrows}, a1{d1, d_xq, sxx, d_y, nrows};
            auto LP = [](void* v) { A* a = (A*)v; k_q4k_layout<0><<<a->n, 128>>>(a->b, a->xq, a->sx, a->y, a->n); };
            auto LS = [](void* v) { A* a = (A*)v; k_q4k_layout<1><<<a->n, 128>>>(a->b, a->xq, a->sx, a->y, a->n); };
            LP(&a0); CHECK(cudaDeviceSynchronize());
            std::vector<float> yp(8); CHECK(cudaMemcpy(yp.data(), d_y, 32, cudaMemcpyDeviceToHost));
            LS(&a1); CHECK(cudaDeviceSynchronize());
            std::vector<float> ys(8); CHECK(cudaMemcpy(ys.data(), d_y, 32, cudaMemcpyDeviceToHost));
            int same = (yp[0] != 0.0f); for (int i = 0; i < 8; i++) same &= (yp[i] == ys[i]);
            rq4[0] = bench(LP, &a0, 20, (double)nrows * row_b);
            rq4[1] = bench(LS, &a1, 20, (double)nrows * row_b);
            printf("Q4_K-shaped  planar %7.1f GB/s   struct-144B %6.1f GB/s   ratio %.3f   bitwise %s\n",
                   rq4[0], rq4[1], rq4[1] / rq4[0], same ? "OK" : "*** MISMATCH — VOID ***");
            CHECK(cudaFree(d0)); CHECK(cudaFree(d1));
        }
        printf("P-2 ordering (q4k ratio > q2k ratio): %s\n",
               (rq4[1]/rq4[0] > rq2[1]/rq2[0]) ? "HOLDS" : "FAILS");
        CHECK(cudaFree(d_xq));
    }

    printf("\nllama.cpp, same card, measured through the runtime:\n");
    printf("  Q2_K  7B all-in-VRAM   21.67 tok/s ->  65.4 GB/s, 165.1 GW/s  (2.625 bit)\n");
    printf("  Q4_0  7B all-in-VRAM   26.87 tok/s -> 119.1 GB/s, 204.7 GW/s  (4.50  bit)\n");
    printf("  cuBLAS fp32 GEMV            161.3 GB/s\n");
    printf("  all-in-VRAM Q4_K (dense)     ~98   GB/s   (eta 0.51 of 192 spec)\n");
    printf("  split placement, GPU share    29.3 GB/s   (eta 0.15)\n");
    return 0;
}
