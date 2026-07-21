// F0 fixed-4-bit decode-LUT GEMV for sm_61 (GTX 1060). Bit-exact substrate for Campaign-2 speed.
//
// Math: after activation-side FWHT (done in torch, per token), the champion GEMV reduces to
//   y_j = sum_groups amax[j,g] * ( sum_{i in group g} lv[idx[j,i]] * xrr_i )
// where idx[j,i] in {0..15} is a 4-bit index into a per-tensor 16-entry codebook lv (<=12 used),
// amax[j,g] is a per-(row, 128-group) fp32 scale, and xrr is the FWHT-prepped activation [in].
// This is structurally identical to a K-quant MMVQ: per-block fp scale + a tiny value LUT.
//
// Layout: packed4 [out, in/2] uint8 (input 2k in low nibble, 2k+1 in high). One WARP per output
// row; the 32 lanes cooperate over each 128-group (4 inputs/lane), warp-reduce, scale by amax.
// Coalesced: lanes read consecutive packed bytes within a group.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define G 128
#define WARP 32

__global__ void f0_gemv_kernel(
    const uint8_t* __restrict__ packed4,   // [out, in/2]
    const float*   __restrict__ xrr,       // [in]
    const float*   __restrict__ lv,        // [16] per-tensor codebook
    const float*   __restrict__ amax,      // [out, in/G]
    float*         __restrict__ y,         // [out]
    int out, int in)
{
    int row = blockIdx.x * (blockDim.x / WARP) + (threadIdx.x / WARP);
    int lane = threadIdx.x % WARP;
    if (row >= out) return;

    __shared__ float s_lv[16 * 8];         // up to 8 warps/block, each its own 16-entry LUT copy
    int wid = threadIdx.x / WARP;
    if (lane < 16) s_lv[wid * 16 + lane] = lv[lane];
    __syncwarp();
    const float* LV = &s_lv[wid * 16];

    int ngroups = in / G;
    // packed row as uint16: each lane reads ONE uint16 (= its 4 consecutive nibbles), coalesced
    const unsigned short* prow16 = reinterpret_cast<const unsigned short*>(
        packed4 + (size_t)row * (in / 2));
    const float* amrow = amax + (size_t)row * ngroups;

    // Each lane pre-scales its per-group partial by amax[g] and accumulates; one reduce at end.
    // y_j = sum_g amax_g * sum_lane partial(lane,g) = sum_lane sum_g amax_g * partial(lane,g).
    float acc = 0.f;
    for (int g = 0; g < ngroups; ++g) {
        int i0 = g * G + lane * 4;               // first of this lane's 4 inputs in group g
        unsigned short v = prow16[g * (G / 4) + lane];   // coalesced: 32 lanes -> 64B
        float ginner = LV[v & 0xF]        * xrr[i0]
                     + LV[(v >> 4) & 0xF] * xrr[i0 + 1]
                     + LV[(v >> 8) & 0xF] * xrr[i0 + 2]
                     + LV[(v >> 12) & 0xF] * xrr[i0 + 3];
        acc += ginner * amrow[g];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) y[row] = acc;
}

torch::Tensor f0_gemv(torch::Tensor packed4, torch::Tensor xrr, torch::Tensor lv, torch::Tensor amax)
{
    int out = packed4.size(0);
    int in  = xrr.size(0);
    auto y = torch::empty({out}, xrr.options());
    int warps_per_block = 4;
    int threads = warps_per_block * WARP;
    int blocks = (out + warps_per_block - 1) / warps_per_block;
    f0_gemv_kernel<<<blocks, threads>>>(
        packed4.data_ptr<uint8_t>(), xrr.data_ptr<float>(),
        lv.data_ptr<float>(), amax.data_ptr<float>(),
        y.data_ptr<float>(), out, in);
    return y;
}

// ---------------------------------------------------------------------------
// v2: same math + layout, but engineered for memory-level parallelism (MLP).
//   - group-loop UNROLLED by UF: issue UF independent uint16 loads before any
//     compute, so UF memory requests are in flight per lane (hides latency).
//   - activation read float4-vectorized (xrr[i0..i0+3] is 16B-aligned: i0 = g*128 + lane*4).
//   - WPB warps/block (8) for occupancy. One warp per row, single end-of-row reduce.
// ---------------------------------------------------------------------------
#define UF 8
#define WPB 8

__global__ void f0_gemv2_kernel(
    const uint8_t* __restrict__ packed4,
    const float*   __restrict__ xrr,
    const float*   __restrict__ lv,
    const float*   __restrict__ amax,
    float*         __restrict__ y,
    int out, int in)
{
    int row = blockIdx.x * (blockDim.x / WARP) + (threadIdx.x / WARP);
    int lane = threadIdx.x % WARP;
    if (row >= out) return;

    __shared__ float s_lv[16 * WPB];
    int wid = threadIdx.x / WARP;
    if (lane < 16) s_lv[wid * 16 + lane] = lv[lane];
    __syncwarp();
    const float* LV = &s_lv[wid * 16];

    int ngroups = in / G;
    const unsigned short* prow16 = reinterpret_cast<const unsigned short*>(
        packed4 + (size_t)row * (in / 2));
    const float* amrow = amax + (size_t)row * ngroups;

    float acc = 0.f;
    int g = 0;
    for (; g + UF <= ngroups; g += UF) {
        unsigned short v[UF];
        #pragma unroll
        for (int k = 0; k < UF; ++k)
            v[k] = prow16[(g + k) * (G / 4) + lane];      // UF loads in flight, coalesced
        #pragma unroll
        for (int k = 0; k < UF; ++k) {
            int i0 = (g + k) * G + lane * 4;
            float4 xv = *reinterpret_cast<const float4*>(xrr + i0);
            unsigned short vv = v[k];
            float gi = LV[vv & 0xF]        * xv.x
                     + LV[(vv >> 4) & 0xF] * xv.y
                     + LV[(vv >> 8) & 0xF] * xv.z
                     + LV[(vv >> 12) & 0xF]* xv.w;
            acc += gi * amrow[g + k];
        }
    }
    for (; g < ngroups; ++g) {
        int i0 = g * G + lane * 4;
        float4 xv = *reinterpret_cast<const float4*>(xrr + i0);
        unsigned short vv = prow16[g * (G / 4) + lane];
        float gi = LV[vv & 0xF]        * xv.x
                 + LV[(vv >> 4) & 0xF] * xv.y
                 + LV[(vv >> 8) & 0xF] * xv.z
                 + LV[(vv >> 12) & 0xF]* xv.w;
        acc += gi * amrow[g];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) y[row] = acc;
}

torch::Tensor f0_gemv2(torch::Tensor packed4, torch::Tensor xrr, torch::Tensor lv, torch::Tensor amax)
{
    int out = packed4.size(0);
    int in  = xrr.size(0);
    auto y = torch::empty({out}, xrr.options());
    int threads = WPB * WARP;
    int blocks = (out + WPB - 1) / WPB;
    f0_gemv2_kernel<<<blocks, threads>>>(
        packed4.data_ptr<uint8_t>(), xrr.data_ptr<float>(),
        lv.data_ptr<float>(), amax.data_ptr<float>(),
        y.data_ptr<float>(), out, in);
    return y;
}

// ---------------------------------------------------------------------------
// v3: v2 + RPW rows per warp. One float4 activation load feeds RPW independent
//   FMA chains (RPW independent accumulators -> more ILP, activation/amax traffic
//   amortized across RPW outputs). The dominant packed-weight traffic is unchanged.
// ---------------------------------------------------------------------------
#define RPW 2

__global__ void f0_gemv3_kernel(
    const uint8_t* __restrict__ packed4,
    const float*   __restrict__ xrr,
    const float*   __restrict__ lv,
    const float*   __restrict__ amax,
    float*         __restrict__ y,
    int out, int in)
{
    int warp = blockIdx.x * (blockDim.x / WARP) + (threadIdx.x / WARP);
    int row0 = warp * RPW;
    int lane = threadIdx.x % WARP;
    if (row0 >= out) return;

    __shared__ float s_lv[16 * WPB];
    int wid = threadIdx.x / WARP;
    if (lane < 16) s_lv[wid * 16 + lane] = lv[lane];
    __syncwarp();
    const float* LV = &s_lv[wid * 16];

    int ngroups = in / G;
    const unsigned short* prow16[RPW];
    const float* amrow[RPW];
    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        int row = row0 + r;
        int rr = (row < out) ? row : out - 1;           // clamp tail; extra row discarded
        prow16[r] = reinterpret_cast<const unsigned short*>(packed4 + (size_t)rr * (in / 2));
        amrow[r] = amax + (size_t)rr * ngroups;
        acc[r] = 0.f;
    }

    #pragma unroll 4
    for (int g = 0; g < ngroups; ++g) {
        int i0 = g * G + lane * 4;
        float4 xv = *reinterpret_cast<const float4*>(xrr + i0);  // shared across RPW rows
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            unsigned short vv = prow16[r][g * (G / 4) + lane];
            float gi = LV[vv & 0xF]        * xv.x
                     + LV[(vv >> 4) & 0xF] * xv.y
                     + LV[(vv >> 8) & 0xF] * xv.z
                     + LV[(vv >> 12) & 0xF]* xv.w;
            acc[r] += gi * amrow[r][g];
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float a = acc[r];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            a += __shfl_down_sync(0xffffffff, a, o);
        int row = row0 + r;
        if (lane == 0 && row < out) y[row] = a;
    }
}

torch::Tensor f0_gemv3(torch::Tensor packed4, torch::Tensor xrr, torch::Tensor lv, torch::Tensor amax)
{
    int out = packed4.size(0);
    int in  = xrr.size(0);
    auto y = torch::empty({out}, xrr.options());
    int warps_per_block = WPB;
    int threads = warps_per_block * WARP;
    int warps_needed = (out + RPW - 1) / RPW;
    int blocks = (warps_needed + warps_per_block - 1) / warps_per_block;
    f0_gemv3_kernel<<<blocks, threads>>>(
        packed4.data_ptr<uint8_t>(), xrr.data_ptr<float>(),
        lv.data_ptr<float>(), amax.data_ptr<float>(),
        y.data_ptr<float>(), out, in);
    return y;
}

// ---------------------------------------------------------------------------
// v4: v3 + BANK-CONFLICT-FREE LUT. The 16-entry LUT shared by 32 lanes with
//   data-dependent indices causes ~2x shared-bank conflicts. Fix: replicate the
//   LUT across lanes as s_lv[entry*32 + lane] -> lane L always reads bank L
//   (= (e*32+L)%32 = L) regardless of the entry, so ZERO conflicts. 2KB/block.
// ---------------------------------------------------------------------------
__global__ void f0_gemv4_kernel(
    const uint8_t* __restrict__ packed4,
    const float*   __restrict__ xrr,
    const float*   __restrict__ lv,
    const float*   __restrict__ amax,
    float*         __restrict__ y,
    int out, int in)
{
    int warp = blockIdx.x * (blockDim.x / WARP) + (threadIdx.x / WARP);
    int row0 = warp * RPW;
    int lane = threadIdx.x % WARP;
    if (row0 >= out) return;

    __shared__ float s_lv[16 * WARP];                 // [entry][lane], conflict-free, block-wide
    for (int i = threadIdx.x; i < 16 * WARP; i += blockDim.x)
        s_lv[i] = lv[i / WARP];
    __syncthreads();
    // lane's conflict-free view: LV(e) == s_lv[e*WARP + lane]
    #define LV4(e) s_lv[(e) * WARP + lane]

    int ngroups = in / G;
    const unsigned short* prow16[RPW];
    const float* amrow[RPW];
    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        int row = row0 + r;
        int rr = (row < out) ? row : out - 1;
        prow16[r] = reinterpret_cast<const unsigned short*>(packed4 + (size_t)rr * (in / 2));
        amrow[r] = amax + (size_t)rr * ngroups;
        acc[r] = 0.f;
    }
    for (int g = 0; g < ngroups; ++g) {
        int i0 = g * G + lane * 4;
        float4 xv = *reinterpret_cast<const float4*>(xrr + i0);
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            unsigned short vv = prow16[r][g * (G / 4) + lane];
            float gi = LV4(vv & 0xF)        * xv.x
                     + LV4((vv >> 4) & 0xF) * xv.y
                     + LV4((vv >> 8) & 0xF) * xv.z
                     + LV4((vv >> 12) & 0xF)* xv.w;
            acc[r] += gi * amrow[r][g];
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float a = acc[r];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            a += __shfl_down_sync(0xffffffff, a, o);
        int row = row0 + r;
        if (lane == 0 && row < out) y[row] = a;
    }
    #undef LV4
}

torch::Tensor f0_gemv4(torch::Tensor packed4, torch::Tensor xrr, torch::Tensor lv, torch::Tensor amax)
{
    int out = packed4.size(0);
    int in  = xrr.size(0);
    auto y = torch::empty({out}, xrr.options());
    int threads = WPB * WARP;
    int warps_needed = (out + RPW - 1) / RPW;
    int blocks = (warps_needed + WPB - 1) / WPB;
    f0_gemv4_kernel<<<blocks, threads>>>(
        packed4.data_ptr<uint8_t>(), xrr.data_ptr<float>(),
        lv.data_ptr<float>(), amax.data_ptr<float>(),
        y.data_ptr<float>(), out, in);
    return y;
}

// ---------------------------------------------------------------------------
// FWHT activation-prep, fused: one block per (token,group) of 128.
//   xrr[base+t] = FWHT( x[base+t]/awq_s[col] * signs[t] )[t] / sqrt(G)
// Replaces ~28 tiny torch ops/group with ONE kernel launch -> removes the
// per-linear activation-prep overhead that dominates small-model decode.
// ---------------------------------------------------------------------------
__global__ void fwht_prep_kernel(
    const float* __restrict__ x,        // [B, cols] flattened
    const float* __restrict__ awq_s,    // [cols]
    const float* __restrict__ signs,    // [G] +-1
    float*       __restrict__ xrr,      // [B, cols] flattened
    int cols)
{
    int g = blockIdx.x;                 // global group index over B*ng
    int t = threadIdx.x;                // 0..127
    int base = g * G;
    int col = (base + t) % cols;
    __shared__ float s[G];
    s[t] = x[base + t] / awq_s[col] * signs[t];
    __syncthreads();
    for (int h = 1; h < G; h <<= 1) {
        float a = 0.f, b = 0.f;
        bool lead = (t & h) == 0;
        if (lead) { a = s[t]; b = s[t + h]; }
        __syncthreads();
        if (lead) { s[t] = a + b; s[t + h] = a - b; }
        __syncthreads();
    }
    xrr[base + t] = s[t] * rsqrtf((float)G);
}

// ---------------------------------------------------------------------------
// CSR outlier sidecar: y[row] += sum_{k in row} val[k] * x[col[k]]. One warp per
// output row reduces its OWN contiguous run of outliers -> NO atomics, no bucket
// contention (vs torch index_add_'s ~95 serialized atomics/bucket on down_proj).
// ---------------------------------------------------------------------------
__global__ void csr_outlier_kernel(
    const float* __restrict__ x, const int* __restrict__ row_ptr,
    const int* __restrict__ col, const float* __restrict__ val,
    float* __restrict__ y, int out)
{
    int row = blockIdx.x * (blockDim.x / WARP) + (threadIdx.x / WARP);
    int lane = threadIdx.x % WARP;
    if (row >= out) return;
    int s = row_ptr[row], e = row_ptr[row + 1];
    float acc = 0.f;
    for (int k = s + lane; k < e; k += WARP) acc += val[k] * x[col[k]];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) y[row] += acc;
}

void csr_outlier(torch::Tensor x, torch::Tensor row_ptr, torch::Tensor col,
                 torch::Tensor val, torch::Tensor y)
{
    int out = row_ptr.size(0) - 1;
    int wpb = 8, threads = wpb * WARP;
    int blocks = (out + wpb - 1) / wpb;
    csr_outlier_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), row_ptr.data_ptr<int>(), col.data_ptr<int>(),
        val.data_ptr<float>(), y.data_ptr<float>(), out);
}

torch::Tensor fwht_prep(torch::Tensor x, torch::Tensor awq_s, torch::Tensor signs)
{
    int cols = awq_s.size(0);
    auto xf = x.contiguous().view({-1, cols});
    int B = xf.size(0);
    int ng = cols / G;
    auto xrr = torch::empty_like(xf);
    fwht_prep_kernel<<<B * ng, G>>>(
        xf.data_ptr<float>(), awq_s.data_ptr<float>(), signs.data_ptr<float>(),
        xrr.data_ptr<float>(), cols);
    return xrr.view(x.sizes());
}

// ---------------------------------------------------------------------------
// D0: dp4a int8 GEMV. Activation Q8'd per group (xq int8 [in] + act_scale [ng]).
//   Codebook int8-snapped (lvq int8 [16], cb_scale). Per group, per lane:
//   load 1 uint16 (4 nibbles) -> map via int8 LUT -> pack int32 -> __dp4a with the
//   lane's 4 packed int8 activations -> int32 accumulate. 4 MACs / instruction.
//   y_j = sum_g (cb_scale * act_scale_g * amax_jg) * dp4a_int32_g.
//   dp4a quality verified +0.0061 ppl (weights/dp4a_quality.py). RPW rows/warp, WPB warps/blk.
// ---------------------------------------------------------------------------
__global__ void d0_gemv_kernel(
    const uint8_t* __restrict__ packed4,   // [out, in/2]
    const int8_t*  __restrict__ xq,        // [in] Q8 activation
    const float*   __restrict__ act_scale, // [in/G] per-group activation scale
    const int8_t*  __restrict__ lvq,       // [16] int8 codebook
    const float*   __restrict__ amax,      // [out, in/G]
    float          cb_scale,
    float*         __restrict__ y,         // [out]
    int out, int in)
{
    int warp = blockIdx.x * (blockDim.x / WARP) + (threadIdx.x / WARP);
    int row0 = warp * RPW;
    int lane = threadIdx.x % WARP;
    if (row0 >= out) return;

    __shared__ int8_t s_lvq[16 * WPB];
    int wid = threadIdx.x / WARP;
    if (lane < 16) s_lvq[wid * 16 + lane] = lvq[lane];
    __syncwarp();
    const int8_t* LVQ = &s_lvq[wid * 16];

    int ngroups = in / G;
    const unsigned short* prow16[RPW];
    const float* amrow[RPW];
    int acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        int row = row0 + r;
        int rr = (row < out) ? row : out - 1;
        prow16[r] = reinterpret_cast<const unsigned short*>(packed4 + (size_t)rr * (in / 2));
        amrow[r] = amax + (size_t)rr * ngroups;
    }
    float yv[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) yv[r] = 0.f;

    const int* xq32 = reinterpret_cast<const int*>(xq);   // 4 int8 activations per int32

    for (int g = 0; g < ngroups; ++g) {
        int apk = xq32[g * (G / 4) + lane];               // lane's 4 Q8 activations, coalesced
        float gscale = cb_scale * act_scale[g];
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            unsigned short vv = prow16[r][g * (G / 4) + lane];
            // map 4 nibbles -> 4 int8 levels, pack little-endian into one int32
            int wpk = (LVQ[vv & 0xF] & 0xFF)
                    | ((LVQ[(vv >> 4) & 0xF] & 0xFF) << 8)
                    | ((LVQ[(vv >> 8) & 0xF] & 0xFF) << 16)
                    | ((LVQ[(vv >> 12) & 0xF] & 0xFF) << 24);
            int a = __dp4a(wpk, apk, 0);                  // 4 int8 MACs in one instruction
            yv[r] += (float)a * gscale * amrow[r][g];
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float a = yv[r];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            a += __shfl_down_sync(0xffffffff, a, o);
        int row = row0 + r;
        if (lane == 0 && row < out) y[row] = a;
    }
}

torch::Tensor d0_gemv(torch::Tensor packed4, torch::Tensor xq, torch::Tensor act_scale,
                      torch::Tensor lvq, torch::Tensor amax, double cb_scale)
{
    int out = packed4.size(0);
    int in  = xq.size(0);
    auto y = torch::empty({out}, torch::dtype(torch::kFloat32).device(packed4.device()));
    int threads = WPB * WARP;
    int warps_needed = (out + RPW - 1) / RPW;
    int blocks = (warps_needed + WPB - 1) / WPB;
    d0_gemv_kernel<<<blocks, threads>>>(
        packed4.data_ptr<uint8_t>(), xq.data_ptr<int8_t>(), act_scale.data_ptr<float>(),
        lvq.data_ptr<int8_t>(), amax.data_ptr<float>(), (float)cb_scale,
        y.data_ptr<float>(), out, in);
    return y;
}

// ---------------------------------------------------------------------------
// Plain int8 GEMV (for embed/lm_head: per-row int8 weight + per-tensor Q8 activation).
//   y[row] = wscale[row] * xscale * dp4a_sum_k( Wq[row,k] * xq[k] ). Halves embed/head
//   from bf16 (2B) to int8 (1B). One warp per row, RPW rows/warp, dp4a.
// ---------------------------------------------------------------------------
__global__ void int8_gemv_kernel(
    const int8_t* __restrict__ Wq, const int8_t* __restrict__ xq,
    const float* __restrict__ wscale, float xscale, float* __restrict__ y, int out, int in)
{
    int warp = blockIdx.x * (blockDim.x / WARP) + (threadIdx.x / WARP);
    int row0 = warp * RPW;
    int lane = threadIdx.x % WARP;
    if (row0 >= out) return;
    const int* xq32 = reinterpret_cast<const int*>(xq);
    int nq = in / 4;
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        int row = row0 + r;
        if (row >= out) break;
        const int* wr = reinterpret_cast<const int*>(Wq + (size_t)row * in);
        int acc = 0;
        for (int k = lane; k < nq; k += WARP)
            acc = __dp4a(wr[k], xq32[k], acc);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if (lane == 0) y[row] = (float)acc * wscale[row] * xscale;
    }
}

torch::Tensor int8_gemv(torch::Tensor Wq, torch::Tensor xq, torch::Tensor wscale, double xscale)
{
    int out = Wq.size(0), in = Wq.size(1);
    auto y = torch::empty({out}, torch::dtype(torch::kFloat32).device(Wq.device()));
    int wpb = 8, threads = wpb * WARP;
    int warps_needed = (out + RPW - 1) / RPW;
    int blocks = (warps_needed + wpb - 1) / wpb;
    int8_gemv_kernel<<<blocks, threads>>>(
        Wq.data_ptr<int8_t>(), xq.data_ptr<int8_t>(), wscale.data_ptr<float>(),
        (float)xscale, y.data_ptr<float>(), out, in);
    return y;
}

// ---------------------------------------------------------------------------
// T0: QTIP fixed-rate bitshift-TRELLIS decode-GEMV (sm_61). The deployable 2-bit path.
//   Each group of G=128 weights is a length-128 de-Bruijn SHIFT trellis. Because L=12=6*K
//   (K=2), the L-bit window at position p is the SLIDING 6-tuple of 2-bit symbols ending at p:
//     w_p = (M[p]<<10)|(M[p+1]<<8)|(M[p+2]<<6)|(M[p+3]<<4)|(M[p+4]<<2)|M[p+5]
//   so there is NO sequential dependency -> lane L decodes positions [4L..4L+3] from 9 symbols
//   (3 packed bytes at byte-offset L), indexes the 4096-entry codebook, dots with the FWHT-prepped
//   activation, scales by gs[row,group] (=gain*std). Math validated bit-exact in trellis_run.py.
//   Weight traffic = ~34 B/group vs F0's 64 B/group -> ~1.9x LESS DRAM traffic at QTIP quality.
//   Layout: packed_M [out, ngroups*MBYTES] uint8 (per-group 34-byte master stream, contiguous);
//   code [ncode<=4096] fp32 (block-shared); gs [out, ngroups] fp32; xrr [in] FWHT-prepped activation.
//   L=12,K=2,WIN=6 hardcoded (the validated config). One warp/row, RPW rows/warp.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void trellis_decode4(
    const uint8_t* __restrict__ Mg, int lane, int& w0, int& w1, int& w2, int& w3)
{
    // lane needs symbols [4L .. 4L+8] = 9 symbols -> bytes [L, L+1, L+2] (4 symbols/byte, LSB-first)
    unsigned b0 = Mg[lane], b1 = Mg[lane + 1], b2 = Mg[lane + 2];
    int s0 = b0 & 3, s1 = (b0 >> 2) & 3, s2 = (b0 >> 4) & 3, s3 = (b0 >> 6) & 3;
    int s4 = b1 & 3, s5 = (b1 >> 2) & 3, s6 = (b1 >> 4) & 3, s7 = (b1 >> 6) & 3;
    int s8 = b2 & 3;
    w0 = (s0 << 10) | (s1 << 8) | (s2 << 6) | (s3 << 4) | (s4 << 2) | s5;
    w1 = (s1 << 10) | (s2 << 8) | (s3 << 6) | (s4 << 4) | (s5 << 2) | s6;
    w2 = (s2 << 10) | (s3 << 8) | (s4 << 6) | (s5 << 4) | (s6 << 2) | s7;
    w3 = (s3 << 10) | (s4 << 8) | (s5 << 6) | (s6 << 4) | (s7 << 2) | s8;
}

__global__ void trellis_gemv_kernel(
    const uint8_t* __restrict__ packed_M,  // [out, ngroups*mbytes]
    const float*   __restrict__ xrr,       // [in] FWHT-prepped activation
    const float*   __restrict__ code,      // [ncode] codebook (<=4096)
    const float*   __restrict__ gs,        // [out, ngroups] gain*std
    float*         __restrict__ y,         // [out]
    int out, int in, int mbytes, int ncode)
{
    extern __shared__ float s_code[];                          // ncode floats, block-shared
    for (int i = threadIdx.x; i < ncode; i += blockDim.x) s_code[i] = code[i];
    __syncthreads();

    int warp = blockIdx.x * (blockDim.x / WARP) + (threadIdx.x / WARP);
    int row0 = warp * RPW;
    int lane = threadIdx.x % WARP;
    if (row0 >= out) return;

    int ngroups = in / G;
    const uint8_t* Mrow[RPW];
    const float* gsrow[RPW];
    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        int row = row0 + r;
        int rr = (row < out) ? row : out - 1;
        Mrow[r] = packed_M + (size_t)rr * ((size_t)ngroups * mbytes);
        gsrow[r] = gs + (size_t)rr * ngroups;
        acc[r] = 0.f;
    }
    for (int g = 0; g < ngroups; ++g) {
        float4 xv = *reinterpret_cast<const float4*>(xrr + g * G + lane * 4);   // shared across RPW
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            int w0, w1, w2, w3;
            trellis_decode4(Mrow[r] + (size_t)g * mbytes, lane, w0, w1, w2, w3);
            float gi = s_code[w0] * xv.x + s_code[w1] * xv.y
                     + s_code[w2] * xv.z + s_code[w3] * xv.w;
            acc[r] += gi * gsrow[r][g];
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float a = acc[r];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            a += __shfl_down_sync(0xffffffff, a, o);
        int row = row0 + r;
        if (lane == 0 && row < out) y[row] = a;
    }
}

torch::Tensor trellis_gemv(torch::Tensor packed_M, torch::Tensor xrr, torch::Tensor code,
                           torch::Tensor gs, int64_t mbytes)
{
    int out = gs.size(0);
    int in  = xrr.size(0);
    int ncode = code.size(0);
    auto y = torch::empty({out}, xrr.options());
    int threads = WPB * WARP;
    int warps_needed = (out + RPW - 1) / RPW;
    int blocks = (warps_needed + WPB - 1) / WPB;
    size_t shmem = (size_t)ncode * sizeof(float);
    trellis_gemv_kernel<<<blocks, threads, shmem>>>(
        packed_M.data_ptr<uint8_t>(), xrr.data_ptr<float>(), code.data_ptr<float>(),
        gs.data_ptr<float>(), y.data_ptr<float>(), out, in, (int)mbytes, ncode);
    return y;
}

// ---------------------------------------------------------------------------
// T1: trellis decode-GEMV with COALESCED stream staging. v1 was 11% util / 19 GB/s because
//   each lane read overlapping bytes [lane,lane+1,lane+2] -> scattered single-byte DRAM txns.
//   Fix: per group, the 32 lanes cooperatively load each row's MBYTES-stream into shared in ONE
//   coalesced pass, then decode from shared. Converts ~96 scattered byte-reads/group into ~34
//   contiguous ones. Codebook still block-shared (4096 fp32). MBYTES<=64 assumed (=34 @ L12).
// ---------------------------------------------------------------------------
#define MBPAD 64
__global__ void trellis_gemv2_kernel(
    const uint8_t* __restrict__ packed_M, const float* __restrict__ xrr,
    const float* __restrict__ code, const float* __restrict__ gs,
    float* __restrict__ y, int out, int in, int mbytes, int ncode)
{
    extern __shared__ float s_code[];                          // [ncode]
    __shared__ uint8_t s_M[WPB * RPW * MBPAD];                 // per-warp staging for RPW row streams
    for (int i = threadIdx.x; i < ncode; i += blockDim.x) s_code[i] = code[i];
    __syncthreads();

    int wib = threadIdx.x / WARP;                              // warp index in block
    int warp = blockIdx.x * (blockDim.x / WARP) + wib;
    int row0 = warp * RPW;
    int lane = threadIdx.x % WARP;
    if (row0 >= out) return;

    int ngroups = in / G;
    const uint8_t* Mrow[RPW];
    const float* gsrow[RPW];
    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        int row = row0 + r;
        int rr = (row < out) ? row : out - 1;
        Mrow[r] = packed_M + (size_t)rr * ((size_t)ngroups * mbytes);
        gsrow[r] = gs + (size_t)rr * ngroups;
        acc[r] = 0.f;
    }
    uint8_t* myM = &s_M[wib * RPW * MBPAD];

    for (int g = 0; g < ngroups; ++g) {
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const uint8_t* src = Mrow[r] + (size_t)g * mbytes;
            for (int i = lane; i < mbytes; i += WARP) myM[r * MBPAD + i] = src[i];   // coalesced
        }
        __syncwarp();
        float4 xv = *reinterpret_cast<const float4*>(xrr + g * G + lane * 4);
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            int w0, w1, w2, w3;
            trellis_decode4(&myM[r * MBPAD], lane, w0, w1, w2, w3);
            float gi = s_code[w0] * xv.x + s_code[w1] * xv.y
                     + s_code[w2] * xv.z + s_code[w3] * xv.w;
            acc[r] += gi * gsrow[r][g];
        }
        __syncwarp();                                          // before next g overwrites staging
    }
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float a = acc[r];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
        int row = row0 + r;
        if (lane == 0 && row < out) y[row] = a;
    }
}

torch::Tensor trellis_gemv2(torch::Tensor packed_M, torch::Tensor xrr, torch::Tensor code,
                            torch::Tensor gs, int64_t mbytes)
{
    int out = gs.size(0), in = xrr.size(0), ncode = code.size(0);
    auto y = torch::empty({out}, xrr.options());
    int threads = WPB * WARP;
    int warps_needed = (out + RPW - 1) / RPW;
    int blocks = (warps_needed + WPB - 1) / WPB;
    size_t shmem = (size_t)ncode * sizeof(float);
    trellis_gemv2_kernel<<<blocks, threads, shmem>>>(
        packed_M.data_ptr<uint8_t>(), xrr.data_ptr<float>(), code.data_ptr<float>(),
        gs.data_ptr<float>(), y.data_ptr<float>(), out, in, (int)mbytes, ncode);
    return y;
}

// ---------------------------------------------------------------------------
// T2: LOOKUP-FREE trellis decode-GEMV via the 3INST COMPUTED code (QTIP's cache-limited-HW design).
//   No 4096-LUT -> no shared-bank conflicts, no 16KB shared pressure (occupancy up). The codebook
//   value for window w is computed: LCG -> mask/xor -> two fp16-reinterpreted halves summed, then
//   (raw - mu)*inv_sigma (zero-mean/unit-std, matching weights/qtip_trellis._recons_3inst).
//   Tests whether removing the LUT beats v1/v2 (which are LUT-bank-conflict / compute bound).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float recons3inst(int w, float mu, float inv_sigma)
{
    unsigned x = (unsigned)w * 89226354u + 64248484u;         // 32-bit LCG (wrap = &0xFFFFFFFF)
    unsigned bits = (x & 0x8FFF8FFFu) ^ 996162400u;
    float top = __half2float(__ushort_as_half((unsigned short)(bits >> 16)));
    float bot = __half2float(__ushort_as_half((unsigned short)(bits & 0xFFFF)));
    float r = top + bot;
    r = isfinite(r) ? r : 0.0f;
    return (r - mu) * inv_sigma;
}

__global__ void __launch_bounds__(WPB * WARP, 8) trellis_gemv3i_kernel(
    const uint8_t* __restrict__ packed_M, const float* __restrict__ xrr,
    const float* __restrict__ gs, float* __restrict__ y,
    int out, int in, int mbytes, float mu, float inv_sigma)
{
    int warp = blockIdx.x * (blockDim.x / WARP) + (threadIdx.x / WARP);
    int row0 = warp * RPW;
    int lane = threadIdx.x % WARP;
    if (row0 >= out) return;

    int ngroups = in / G;
    const uint8_t* Mrow[RPW];
    const float* gsrow[RPW];
    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        int row = row0 + r;
        int rr = (row < out) ? row : out - 1;
        Mrow[r] = packed_M + (size_t)rr * ((size_t)ngroups * mbytes);
        gsrow[r] = gs + (size_t)rr * ngroups;
        acc[r] = 0.f;
    }
    for (int g = 0; g < ngroups; ++g) {
        float4 xv = *reinterpret_cast<const float4*>(xrr + g * G + lane * 4);
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            int w0, w1, w2, w3;
            trellis_decode4(Mrow[r] + (size_t)g * mbytes, lane, w0, w1, w2, w3);
            float gi = recons3inst(w0, mu, inv_sigma) * xv.x + recons3inst(w1, mu, inv_sigma) * xv.y
                     + recons3inst(w2, mu, inv_sigma) * xv.z + recons3inst(w3, mu, inv_sigma) * xv.w;
            acc[r] += gi * gsrow[r][g];
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float a = acc[r];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
        int row = row0 + r;
        if (lane == 0 && row < out) y[row] = a;
    }
}

torch::Tensor trellis_gemv3i(torch::Tensor packed_M, torch::Tensor xrr, torch::Tensor gs,
                             int64_t mbytes, double mu, double inv_sigma)
{
    int out = gs.size(0), in = xrr.size(0);
    auto y = torch::empty({out}, xrr.options());
    int threads = WPB * WARP;
    int warps_needed = (out + RPW - 1) / RPW;
    int blocks = (warps_needed + WPB - 1) / WPB;
    trellis_gemv3i_kernel<<<blocks, threads>>>(
        packed_M.data_ptr<uint8_t>(), xrr.data_ptr<float>(), gs.data_ptr<float>(),
        y.data_ptr<float>(), out, in, (int)mbytes, (float)mu, (float)inv_sigma);
    return y;
}

// ---------------------------------------------------------------------------
// T3: BATCHED trellis decode-GEMM (weight-stationary). The decode is FIXED per weight, so decode
//   ONCE per group and MAC across BT token columns -> amortizes the decode (the batch=1 bottleneck)
//   while still reading the 2-bit stream (half of 4-bit). Lookup-free 3INST (no smem). y [BT, out].
//   per-token time should FALL faster than 1/BT until it hits the bandwidth floor -> trellis should
//   cross 4-bit at some small BT (the throughput/prefill regime). RPW=1; BT provides the ILP.
// ---------------------------------------------------------------------------
template<int BT>
__global__ void __launch_bounds__(WPB * WARP, 4) trellis_gemv3i_batch_kernel(
    const uint8_t* __restrict__ packed_M, const float* __restrict__ xrr,   // xrr [BT, in]
    const float* __restrict__ gs, float* __restrict__ y,                   // y   [BT, out]
    int out, int in, int mbytes, float mu, float inv_sigma)
{
    int row = blockIdx.x * (blockDim.x / WARP) + (threadIdx.x / WARP);
    int lane = threadIdx.x % WARP;
    if (row >= out) return;
    int ngroups = in / G;
    const uint8_t* Mrow = packed_M + (size_t)row * ((size_t)ngroups * mbytes);
    const float* gsrow = gs + (size_t)row * ngroups;
    float acc[BT];
    #pragma unroll
    for (int b = 0; b < BT; ++b) acc[b] = 0.f;
    for (int g = 0; g < ngroups; ++g) {
        int w0, w1, w2, w3;
        trellis_decode4(Mrow + (size_t)g * mbytes, lane, w0, w1, w2, w3);   // decode ONCE
        float c0 = recons3inst(w0, mu, inv_sigma), c1 = recons3inst(w1, mu, inv_sigma);
        float c2 = recons3inst(w2, mu, inv_sigma), c3 = recons3inst(w3, mu, inv_sigma);
        float gsg = gsrow[g];
        int base = g * G + lane * 4;
        #pragma unroll
        for (int b = 0; b < BT; ++b) {                                      // reuse across BT tokens
            float4 xv = *reinterpret_cast<const float4*>(xrr + (size_t)b * in + base);
            acc[b] += (c0 * xv.x + c1 * xv.y + c2 * xv.z + c3 * xv.w) * gsg;
        }
    }
    #pragma unroll
    for (int b = 0; b < BT; ++b) {
        float a = acc[b];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
        if (lane == 0) y[(size_t)b * out + row] = a;
    }
}

torch::Tensor trellis_gemv3i_batch(torch::Tensor packed_M, torch::Tensor xrr, torch::Tensor gs,
                                   int64_t mbytes, double mu, double inv_sigma)
{
    int out = gs.size(0), in = xrr.size(1), B = xrr.size(0);
    auto y = torch::empty({B, out}, xrr.options());
    int threads = WPB * WARP;
    int blocks = (out + WPB - 1) / WPB;
    auto* pM = packed_M.data_ptr<uint8_t>(); auto* px = xrr.data_ptr<float>();
    auto* pg = gs.data_ptr<float>(); auto* py = y.data_ptr<float>();
    float fmu = (float)mu, fis = (float)inv_sigma;
    #define LAUNCH_BT(BT) trellis_gemv3i_batch_kernel<BT><<<blocks, threads>>>(pM, px, pg, py, out, in, (int)mbytes, fmu, fis)
    switch (B) {
        case 1:  LAUNCH_BT(1);  break;
        case 2:  LAUNCH_BT(2);  break;
        case 4:  LAUNCH_BT(4);  break;
        case 8:  LAUNCH_BT(8);  break;
        case 16: LAUNCH_BT(16); break;
        case 32: LAUNCH_BT(32); break;
        default: TORCH_CHECK(false, "trellis_gemv3i_batch: B must be 1/2/4/8/16/32");
    }
    #undef LAUNCH_BT
    return y;
}

// ---------------------------------------------------------------------------
// F0 BATCHED (4-bit LUT) decode-GEMM, weight-stationary — the honest head-to-head for trellis batching.
//   Same structure as trellis_gemv3i_batch but 4-bit nibble + 16-LUT decode (read once, MAC over BT).
//   Trellis reads HALF the weight bytes; this lets us see at which B the 2-bit path overtakes 4-bit.
// ---------------------------------------------------------------------------
template<int BT>
__global__ void __launch_bounds__(WPB * WARP, 4) f0_gemv3_batch_kernel(
    const uint8_t* __restrict__ packed4, const float* __restrict__ xrr,   // xrr [BT, in]
    const float* __restrict__ lv, const float* __restrict__ amax,
    float* __restrict__ y, int out, int in)                                // y [BT, out]
{
    __shared__ float s_lv[16 * WPB];
    int wid = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    if (lane < 16) s_lv[wid * 16 + lane] = lv[lane];
    __syncwarp();
    const float* LV = &s_lv[wid * 16];
    int row = blockIdx.x * (blockDim.x / WARP) + wid;
    if (row >= out) return;
    int ngroups = in / G;
    const unsigned short* prow16 = reinterpret_cast<const unsigned short*>(packed4 + (size_t)row * (in / 2));
    const float* amrow = amax + (size_t)row * ngroups;
    float acc[BT];
    #pragma unroll
    for (int b = 0; b < BT; ++b) acc[b] = 0.f;
    for (int g = 0; g < ngroups; ++g) {
        unsigned short vv = prow16[g * (G / 4) + lane];
        float c0 = LV[vv & 0xF], c1 = LV[(vv >> 4) & 0xF], c2 = LV[(vv >> 8) & 0xF], c3 = LV[(vv >> 12) & 0xF];
        float am = amrow[g];
        int base = g * G + lane * 4;
        #pragma unroll
        for (int b = 0; b < BT; ++b) {
            float4 xv = *reinterpret_cast<const float4*>(xrr + (size_t)b * in + base);
            acc[b] += (c0 * xv.x + c1 * xv.y + c2 * xv.z + c3 * xv.w) * am;
        }
    }
    #pragma unroll
    for (int b = 0; b < BT; ++b) {
        float a = acc[b];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
        if (lane == 0) y[(size_t)b * out + row] = a;
    }
}

torch::Tensor f0_gemv3_batch(torch::Tensor packed4, torch::Tensor xrr, torch::Tensor lv, torch::Tensor amax)
{
    int out = amax.size(0), in = xrr.size(1), B = xrr.size(0);
    auto y = torch::empty({B, out}, xrr.options());
    int threads = WPB * WARP, blocks = (out + WPB - 1) / WPB;
    auto* pp = packed4.data_ptr<uint8_t>(); auto* px = xrr.data_ptr<float>();
    auto* pl = lv.data_ptr<float>(); auto* pa = amax.data_ptr<float>(); auto* py = y.data_ptr<float>();
    #define LAUNCH_F0BT(BT) f0_gemv3_batch_kernel<BT><<<blocks, threads>>>(pp, px, pl, pa, py, out, in)
    switch (B) {
        case 1:  LAUNCH_F0BT(1);  break;
        case 2:  LAUNCH_F0BT(2);  break;
        case 4:  LAUNCH_F0BT(4);  break;
        case 8:  LAUNCH_F0BT(8);  break;
        case 16: LAUNCH_F0BT(16); break;
        case 32: LAUNCH_F0BT(32); break;
        default: TORCH_CHECK(false, "f0_gemv3_batch: B must be 1/2/4/8/16/32");
    }
    #undef LAUNCH_F0BT
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("f0_gemv3_batch", &f0_gemv3_batch, "batched 4-bit LUT decode-GEMM, weight-stationary (sm_61)");
    m.def("trellis_gemv3i_batch", &trellis_gemv3i_batch, "batched trellis decode-GEMM, weight-stationary (sm_61)");
    m.def("trellis_gemv", &trellis_gemv, "QTIP bitshift-trellis 2-bit decode-GEMV (sm_61)");
    m.def("trellis_gemv2", &trellis_gemv2, "trellis 2-bit decode-GEMV, coalesced shared staging (sm_61)");
    m.def("trellis_gemv3i", &trellis_gemv3i, "trellis 2-bit decode-GEMV, lookup-free 3INST computed code (sm_61)");
    m.def("f0_gemv", &f0_gemv, "fixed-4-bit decode-LUT GEMV (sm_61)");
    m.def("int8_gemv", &int8_gemv, "plain int8 dp4a GEMV for embed/head (sm_61)");
    m.def("f0_gemv2", &f0_gemv2, "fixed-4-bit decode-LUT GEMV v2 (MLP-unrolled, sm_61)");
    m.def("f0_gemv3", &f0_gemv3, "fixed-4-bit decode-LUT GEMV v3 (RPW rows/warp, sm_61)");
    m.def("f0_gemv4", &f0_gemv4, "fixed-4-bit decode-LUT GEMV v4 (conflict-free LUT, sm_61)");
    m.def("d0_gemv", &d0_gemv, "dp4a int8 GEMV (sm_61)");
    m.def("fwht_prep", &fwht_prep, "fused FWHT activation-prep (sm_61)");
    m.def("csr_outlier", &csr_outlier, "per-row CSR outlier sidecar add (sm_61)");
}
