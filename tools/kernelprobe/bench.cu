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
    float* d_y; CHECK(cudaMalloc(&d_y, (size_t)NMAT * ROWS * sizeof(float)));
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

    printf("\nllama.cpp, same card, measured through the runtime:\n");
    printf("  cuBLAS fp32 GEMV            161.3 GB/s\n");
    printf("  all-in-VRAM Q4_K (dense)     ~98   GB/s   (eta 0.51 of 192 spec)\n");
    printf("  split placement, GPU share    29.3 GB/s   (eta 0.15)\n");
    return 0;
}
