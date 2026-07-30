// U-33 follow-up / prereg #85: measure kernel-launch cost WITHOUT the instrumentation
// that made prereg #83 unusable for magnitude.
//
// #83 used an instrumented llama.cpp (GGML_GPU_PROFILE_OPS) that records a CUDA event
// PAIR AROUND EVERY OP. U-33 measured that harness cost at 3.80-8.32 us/op call by
// diffing its tg32 against the clean ladder - the SAME ORDER as the 5-16 us op times it
// reported. You cannot measure a 6 us cost with a 6 us ruler.
//
// This probe inverts the geometry: ONE event pair around N launches. The per-launch cost
// falls out of the slope, and the harness contributes exactly two events total no matter
// how large N gets.
//
// It also settles the question U-33 raised but could not answer: is the per-op cost a
// LAUNCH cost (paid once per kernel, independent of neighbours) or a PIPELINED cost
// (back-to-back kernels overlap, so the marginal cost of the Nth is below the first)?
// #83's us/call ANTI-correlated with ops-per-layer, which is the pipelining signature.
// Arm C measures the marginal cost directly.
//
// ---------------------------------------------------------------------------
// STAKED BEFORE RUNNING (prereg #85, kill rules below):
//
//   P1  Marginal launch cost of a back-to-back trivial kernel is BELOW 6.0 us -
//       i.e. materially below #83's reported 7.72-15.51 us/call for RMS_NORM.
//       KILL: if arm C's slope is >= 6.0 us, #83's magnitude was not instrumentation
//       inflation and L-21's shares stand roughly as published.
//
//   P2  Marginal cost is INDEPENDENT of the work each kernel does, across a 32x span
//       of vector length (896 -> 28672 floats, the hidden sizes on our ladder).
//       Predicted spread <= 1.5x, matching ROPE's measured 1.58x across a 32x model span.
//       KILL: if spread > 2.0x, these ops are work-bound after all and D-24's premise
//       ("latency-bound, tracks launches not bytes") is wrong.
//
//   P3  A DEPENDENT chain (each kernel reads the previous kernel's output, which is what
//       a real transformer layer is) costs MORE per kernel than an independent burst,
//       because the dependency blocks overlap. Predicted >= 1.3x arm C.
//       KILL: if dependent <= independent, then llama.cpp's serialization is free and
//       the per-layer framing in D-25 is wrong.
//
// P1+P2 both holding means: the overhead is real, launch-shaped, and SMALLER than #83
// implied. P3 holding means the right unit is the dependent chain (~layers), not the
// raw op count - which is exactly why D-24's op-count term over-taxed small models.
// ---------------------------------------------------------------------------
//
// build:  nvcc -O3 -arch=sm_61 launch.cu -o launch.exe
// run:    launch.exe
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

// An RMS-norm-shaped kernel: read n floats, reduce, write n floats. One block, because
// that is what a decode-time norm on a single token vector actually launches.
__global__ __launch_bounds__(256) void k_rmsnorm(const float* __restrict__ x,
                                                 float* __restrict__ y, int n)
{
    __shared__ float red[256];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) acc += x[i] * x[i];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / n + 1e-6f);
    for (int i = threadIdx.x; i < n; i += 256) y[i] = x[i] * inv;
}

// Empty kernel: isolates pure dispatch with zero memory traffic.
__global__ void k_empty() {}

static double time_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, a, b)); return ms;
}

int main() {
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device : %s (cc %d.%d, %d SMs)\n\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);

    const int NMAX = 28672;
    float *x, *y;
    CHECK(cudaMalloc(&x, (size_t)NMAX * 4));
    CHECK(cudaMalloc(&y, (size_t)NMAX * 4));
    std::vector<float> h(NMAX, 0.5f);
    CHECK(cudaMemcpy(x, h.data(), (size_t)NMAX * 4, cudaMemcpyHostToDevice));

    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    const int REPS = 50;

    // ---- arm A: pure dispatch floor, empty kernel, ONE event pair around N launches
    printf("=== arm A: empty kernel, one event pair around N launches ===\n");
    printf("%10s %12s %14s\n", "N", "total ms", "us/launch");
    for (int N : {1, 10, 100, 1000, 10000}) {
        for (int i = 0; i < N; i++) k_empty<<<1, 32>>>();
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(a));
        for (int r = 0; r < REPS; r++)
            for (int i = 0; i < N; i++) k_empty<<<1, 32>>>();
        CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
        const double ms = time_ms(a, b);
        printf("%10d %12.3f %14.3f\n", N, ms / REPS, ms * 1000.0 / REPS / N);
    }

    // ---- arm B: does the WORK change the per-launch cost? (P2)
    printf("\n=== arm B: rmsnorm kernel, N=1000 independent launches, vs vector length ===\n");
    printf("%10s %12s %14s   %s\n", "floats", "total ms", "us/launch", "model with this hidden");
    const int Ns[]  = {896, 1024, 2048, 2560, 3584, 3840, 8192, 28672};
    const char* who[] = {"Qwen2.5-0.5B", "Qwen3-0.6B", "-", "Qwen3.5-4B",
                         "Qwen2.5-7B", "gemma4-12B", "-", "(ffn width)"};
    double lo = 1e9, hi = 0;
    for (int i = 0; i < 8; i++) {
        const int n = Ns[i], N = 1000;
        for (int k = 0; k < N; k++) k_rmsnorm<<<1, 256>>>(x, y, n);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(a));
        for (int r = 0; r < REPS; r++)
            for (int k = 0; k < N; k++) k_rmsnorm<<<1, 256>>>(x, y, n);
        CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
        const double us = time_ms(a, b) * 1000.0 / REPS / N;
        if (us < lo) lo = us;
        if (us > hi) hi = us;
        printf("%10d %12.3f %14.3f   %s\n", n, time_ms(a, b) / REPS, us, who[i]);
    }
    printf("  spread across a %.0fx range of vector length: %.2fx  (P2 predicts <= 1.5x)\n",
           28672.0 / 896.0, hi / lo);

    // ---- arm C: marginal cost of a back-to-back burst (P1)
    printf("\n=== arm C: marginal cost, rmsnorm n=3584, burst length sweep ===\n");
    printf("%10s %12s %14s\n", "burst", "total ms", "us/launch");
    double prev_us = 0, prev_n = 0, slope = 0;
    for (int N : {1, 2, 4, 8, 16, 64, 256, 1024}) {
        for (int k = 0; k < N; k++) k_rmsnorm<<<1, 256>>>(x, y, 3584);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(a));
        for (int r = 0; r < REPS; r++)
            for (int k = 0; k < N; k++) k_rmsnorm<<<1, 256>>>(x, y, 3584);
        CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
        const double tot = time_ms(a, b) / REPS;
        printf("%10d %12.4f %14.3f\n", N, tot, tot * 1000.0 / N);
        if (N == 1024) slope = (tot - prev_us) * 1000.0 / (N - prev_n);
        prev_us = tot; prev_n = N;
    }
    printf("  MARGINAL us/launch (256->1024 segment): %.3f us   (P1 predicts < 6.0)\n", slope);

    // ---- arm D: dependent chain - the real transformer shape (P3)
    printf("\n=== arm D: DEPENDENT chain (each kernel reads the previous output) ===\n");
    printf("%10s %12s %14s\n", "chain", "total ms", "us/launch");
    float* bufs[2] = {x, y};
    double dep_us = 0;
    for (int N : {64, 256, 1024}) {
        for (int k = 0; k < N; k++) k_rmsnorm<<<1, 256>>>(bufs[k & 1], bufs[!(k & 1)], 3584);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(a));
        for (int r = 0; r < REPS; r++)
            for (int k = 0; k < N; k++)
                k_rmsnorm<<<1, 256>>>(bufs[k & 1], bufs[!(k & 1)], 3584);
        CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
        const double tot = time_ms(a, b) / REPS;
        dep_us = tot * 1000.0 / N;
        printf("%10d %12.4f %14.3f\n", N, tot, dep_us);
    }
    printf("  dependent vs independent marginal: %.2fx   (P3 predicts >= 1.3x)\n",
           slope > 0 ? dep_us / slope : 0.0);

    printf("\nreference: prereg #83 reported RMS_NORM at 7.72-15.51 us/call from inside\n");
    printf("llama.cpp with per-op event pairs; U-33 measured that harness at 3.80-8.32\n");
    printf("us/call. Everything above uses TWO events total per measurement.\n");
    return 0;
}
