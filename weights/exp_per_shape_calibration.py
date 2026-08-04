"""exp_per_shape_calibration.py -- pre-registration #92 (per-shape calibration).

Scores the two halves staked in preregistrations/2026-07-31-per-shape-calibration.md:

  Phase A (GPU, single session): the L-20 shape sweep rebuilt as a PRODUCT path -- CUDA source
      embedded here, compiled with the user's nvcc at run time, one output row per block, one
      launch per tensor (the llama.cpp decode geometry, #55). Gates P-1 (reproduces the
      prereg81_knee.log reference curve on this card), P-2 (monotone, >=2x span, rows-keyed
      knee at 4096 +/-1 step on BOTH widths), P-3 (two back-to-back sweeps agree within 5% --
      a calibration artifact may not store noise).

  Phase B (no GPU, deterministic): every readable GGUF in the eval dir x every shipped
      plan.MACHINES preset x ctx {0, 16384}, priced by the SHIPPED plan.evaluate(), then
      re-priced with per-shape factors derived from the Phase A curve and each model's own
      tensor geometry. Gate P-4: does the emitted WINNER (name+flags) change anywhere it
      arithmetically could? Gate P-5: bandwidth cannot move the split FRACTION in the shipped
      planner (capacity-determined) -- verified by perturbing vb/geta through evaluate().

THE PREDICTION HALF IS NOT SCORED HERE. U-32 already stakes it (LOO median < 8.7%, max
< 18.6% on a fresh ladder); this script neither runs that ladder nor touches its gate.

WHAT CANNOT VARY, GUARDED (the #85-arms-C/D signature):
  G-1  a cell whose top-2 rows sit further apart than the model's attn/exp factor differential
       can NEVER flip -- such cells cannot support a P-4 null. Zero bindable cells => exit 3
       (UNABLE TO BIND), never a null.
  G-2  a model whose factors are flat (differential < 1%) is excluded from the P-4 population.
       Dense models are flat BY CONSTRUCTION in the shipped planner (no row prices a
       shape-biased subset of them) and are reported but never scored.
  G-3  the re-pricer recomputes act_ne/act_ex/f from plan's own imported constants and REFUSES
       (exit 2) unless it reproduces every recognized row's `terms` decomposition to 1e-9.
       A re-pricer that scores a stale model of the planner is worse than none.
  --self-test constructs the failing inputs for all of these plus a non-monotone curve, and
  exits non-zero unless every guard demonstrably fires.

Outputs (all under weights/data/):
  exp_per_shape_curve.json          the calibration artifact prototype (both repeats, device id)
  exp_per_shape_sweep.log           raw probe output, prereg81-style tables
  exp_per_shape_calibration.json    machine-readable scoring, every cell
  exp_per_shape_calibration.log     human-readable transcript

Idempotent: re-running reuses the stored curve (pass --remeasure to sweep again) and
overwrites the same output files. Exit codes: 0 = all evaluable gates PASS; 1 = a kill rule
fired (INCLUDING the P-4 true null, which is a staked MISS); 2 = precondition missing, no
number produced; 3 = UNABLE TO BIND / INCOMPLETE (never citable as pass or null).

Run:
  python weights/exp_per_shape_calibration.py --self-test     # guards only, no GPU, run first
  python weights/exp_per_shape_calibration.py                 # full experiment
"""
from __future__ import annotations
import argparse, json, math, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
CURVE_PATH = os.path.join(DATA, "exp_per_shape_curve.json")
SWEEP_LOG = os.path.join(DATA, "exp_per_shape_sweep.log")
RESULT_JSON = os.path.join(DATA, "exp_per_shape_calibration.json")
LOG_PATH = os.path.join(DATA, "exp_per_shape_calibration.log")
REF_LOG = os.path.join(DATA, "prereg81_knee.log")
DEFAULT_GGUF_DIR = r"D:\evo-compress-data\gguf"

ROWS_SWEEP = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
WIDTHS = [2048, 4096]                  # K; bytes/row = 9K/16 (4.5-bit layout)
BUDGET_MB = 384                        # bytes touched per sweep point, as in #80/#81
REPS = 3

# Staked tolerances (prereg #92; renegotiating these after seeing data is the fraud this
# repo exists to prevent).
P1_PENALTY_TOL = 0.06                  # normalized-penalty band vs the reference log
P1_CEIL_TOL = 0.10                     # ceiling band vs the reference log
P2_NOISE = 0.03                        # monotonicity noise allowance
P2_SPAN = 2.0                          # min ceiling/floor ratio per width
P2_KNEE_SET = {2048, 4096, 8192}       # 4096 +/- one sweep step
P3_TOL = 0.05                          # repeat agreement
G2_MIN_DIFFERENTIAL = 0.01             # flat-factor exclusion
TERMS_RTOL = 1e-9                      # G-3 decomposition reproduction

PASS, KILL, UNABLE, REFUSE = "PASS", "KILL", "UNABLE", "REFUSE"


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.f.write(s + "\n")
        self.f.flush()


# --------------------------------------------------------------------------- Phase A: probe
CUDA_SRC = r'''
// prereg #92 rewrite of the #80/#81 shape probe (the original shape.cu was never committed;
// P-1 exists precisely because this is a rewrite scored against the surviving logs).
// 4.5-bit layout, byte-identical in size to Q4_K: per 256-weight superblock 128 B packed
// nibbles + 8 fp16 sub-block scales; row bytes = 9K/16. One output row per block, one kernel
// launch per tensor -- llama.cpp decode geometry (#55). Only rows-per-tensor and K vary.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(3); } } while (0)

extern __shared__ float xs[];

__global__ __launch_bounds__(128) void k_row(const uint8_t * __restrict__ mat,
        const int K, const int row_bytes,
        const float * __restrict__ x, float * __restrict__ y)
{
    for (int i = threadIdx.x; i < K; i += 128) xs[i] = x[i];
    __syncthreads();
    const uint8_t  * row = mat + (size_t)blockIdx.x * row_bytes;
    const uint32_t * q   = (const uint32_t *)row;
    const __half   * sc  = (const __half   *)(row + K / 2);
    const int nw = K / 8;                       // uint32 words of packed nibbles per row
    float acc = 0.0f;
    for (int i = threadIdx.x; i < nw; i += 128) {
        const int sb = i >> 5;                  // superblock: 32 words = 128 B = 256 weights
        const int b  = (i & 31) << 2;
        const uint32_t v = q[i];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint8_t byte = (uint8_t)(v >> (j * 8));
            const int wlo = sb * 256 + b + j;
            const int whi = wlo + 128;
            const float slo = __half2float(sc[sb * 8 + ((b + j)         >> 5)]);
            const float shi = __half2float(sc[sb * 8 + (((b + j) + 128) >> 5)]);
            acc += ((float)(byte & 0xF) - 8.0f) * slo * xs[wlo];
            acc += ((float)(byte >>  4) - 8.0f) * shi * xs[whi];
        }
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ float warp[4];
    if ((threadIdx.x & 31) == 0) warp[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) y[blockIdx.x] = warp[0] + warp[1] + warp[2] + warp[3];
}

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: probe K rows budget_mb reps\n"); return 2; }
    const int K = atoi(argv[1]);
    const int rows = atoi(argv[2]);
    const long long budget = (long long)atoll(argv[3]) * 1024LL * 1024LL;
    const int reps = atoi(argv[4]);
    if (K < 256 || (K % 256) || rows < 1 || reps < 1) { fprintf(stderr, "bad args\n"); return 2; }
    const int row_bytes = K * 9 / 16;
    const long long mat_bytes = (long long)rows * row_bytes;
    int n_t = (int)(budget / mat_bytes);
    if (n_t < 1) n_t = 1;

    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("DEVICE name=\"%s\" cc=%d.%d sms=%d\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);

    uint8_t *buf; float *x, *y;
    CHECK(cudaMalloc(&buf, (size_t)n_t * mat_bytes));
    CHECK(cudaMalloc(&x, (size_t)K * sizeof(float)));
    CHECK(cudaMalloc(&y, (size_t)rows * sizeof(float)));
    CHECK(cudaMemset(buf, 0x5A, (size_t)n_t * mat_bytes));   // deterministic non-zero weights
    float *hx = (float *)malloc((size_t)K * sizeof(float));
    for (int i = 0; i < K; i++) hx[i] = 1.0f;
    CHECK(cudaMemcpy(x, hx, (size_t)K * sizeof(float), cudaMemcpyHostToDevice));
    free(hx);

    const size_t shmem = (size_t)K * sizeof(float);
    cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    for (int t = 0; t < n_t; t++)                            // warmup pass
        k_row<<<rows, 128, shmem>>>(buf + (size_t)t * mat_bytes, K, row_bytes, x, y);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    double best = 0.0;
    for (int r = 0; r < reps; r++) {
        CHECK(cudaEventRecord(t0));
        for (int t = 0; t < n_t; t++)
            k_row<<<rows, 128, shmem>>>(buf + (size_t)t * mat_bytes, K, row_bytes, x, y);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, t0, t1));
        double gbs = (double)n_t * (double)mat_bytes / ((double)ms * 1e-3) / 1e9;
        if (gbs > best) best = gbs;
    }
    float ycheck = 0.0f;
    CHECK(cudaMemcpy(&ycheck, y, sizeof(float), cudaMemcpyDeviceToHost));
    if (!(ycheck == ycheck) || ycheck == 0.0f) {             // NaN or a no-op run: refuse
        fprintf(stderr, "output check failed (ycheck=%f) - the kernel did not do the work "
                        "whose bytes are being priced\n", (double)ycheck); return 3;
    }
    printf("POINT K=%d rows=%d row_bytes=%d tensors=%d gbs=%.2f\n",
           K, rows, row_bytes, n_t, best);
    return 0;
}
'''


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out.splitlines()[0].strip() if out else None
    except Exception:
        return None


def gpu_clocks():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return "n/a"


def compile_probe(log):
    nvcc = None
    for cand in ("nvcc", "nvcc.exe"):
        try:
            subprocess.run([cand, "--version"], capture_output=True, timeout=30)
            nvcc = cand
            break
        except Exception:
            continue
    if not nvcc:
        return None, "nvcc not found on PATH"
    src = os.path.join(DATA, "exp_per_shape_probe.cu")
    exe = os.path.join(DATA, "exp_per_shape_probe.exe")
    with open(src, "w", encoding="utf-8") as f:
        f.write(CUDA_SRC)
    for flags in (["-O3", "-arch=native"], ["-O3"]):
        r = subprocess.run([nvcc] + flags + ["-o", exe, src],
                           capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and os.path.isfile(exe):
            log(f"  probe compiled: nvcc {' '.join(flags)}")
            return exe, None
        err = (r.stderr or r.stdout or "").strip()
    return None, f"nvcc failed to compile the probe:\n{err[-2000:]}"


def run_sweep(exe, budget_mb, reps, log, raw):
    """One full two-width sweep. Returns ({K: {rows: gbs}}, device_name) or (None, err)."""
    curve, device = {}, None
    for K in WIDTHS:
        curve[K] = {}
        for rows in ROWS_SWEEP:
            r = subprocess.run([exe, str(K), str(rows), str(budget_mb), str(reps)],
                              capture_output=True, text=True, timeout=600)
            raw.write(r.stdout + (r.stderr or ""))
            if r.returncode != 0:
                return None, None, f"probe failed at K={K} rows={rows}: {r.stderr.strip()[:500]}"
            m = re.search(r'DEVICE name="([^"]+)"', r.stdout)
            if m:
                device = m.group(1)
            m = re.search(r"POINT K=\d+ rows=\d+ row_bytes=\d+ tensors=\d+ gbs=([\d.]+)",
                          r.stdout)
            if not m:
                return None, None, f"could not parse probe output at K={K} rows={rows}"
            curve[K][rows] = float(m.group(1))
            log(f"    K={K:5d} rows={rows:6d}  {curve[K][rows]:7.2f} GB/s")
    return curve, device, None


def parse_ref_log(path):
    """prereg81_knee.log -> {K: {rows: gbs}}."""
    out, cur = {}, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"=== K=(\d+)", line)
            if m:
                cur = int(m.group(1))
                out[cur] = {}
                continue
            m = re.match(r"\s*(\d+)\s+(\d+)\s+([\d.]+)\s+\d+%", line)
            if m and cur is not None:
                out[cur][int(m.group(1))] = float(m.group(3))
    return out


def knee90(points):
    """First rows value reaching 90% of the sweep's own max."""
    mx = max(points.values())
    for r in sorted(points):
        if points[r] >= 0.90 * mx:
            return r
    return None


def gate_p1(curve, ref):
    """(verdict, details): rewrite reproduces the reference curve on the same card."""
    bad = []
    for K in WIDTHS:
        if K not in ref or not ref[K]:
            return REFUSE, [f"reference log has no K={K} section"]
        rc, mc = ref[K], curve[K]
        ref_ceil, meas_ceil = rc[max(rc)], mc[max(mc)]
        if abs(meas_ceil - ref_ceil) / ref_ceil > P1_CEIL_TOL:
            bad.append(f"K={K} ceiling {meas_ceil:.1f} vs ref {ref_ceil:.1f} (> {P1_CEIL_TOL:.0%})")
        for rows in ROWS_SWEEP:
            if rows not in rc:
                continue
            pen_ref = 1 - rc[rows] / ref_ceil
            pen_meas = 1 - mc[rows] / meas_ceil
            if abs(pen_meas - pen_ref) > P1_PENALTY_TOL:
                bad.append(f"K={K} rows={rows} penalty {pen_meas:+.3f} vs ref {pen_ref:+.3f}")
    return (PASS if not bad else KILL), bad


def gate_p2(curve):
    """(verdict, details): monotone, >=2x span, rows-keyed knee at 4096 +/- 1 step, both widths."""
    bad, knees = [], {}
    for K in WIDTHS:
        pts = curve[K]
        rs = sorted(pts)
        for a, b in zip(rs, rs[1:]):
            if pts[b] < pts[a] * (1 - P2_NOISE):
                bad.append(f"K={K}: non-monotone {a}->{b} ({pts[a]:.1f} -> {pts[b]:.1f})")
        if pts[rs[-1]] / pts[rs[0]] < P2_SPAN:
            bad.append(f"K={K}: span {pts[rs[-1]] / pts[rs[0]]:.2f}x < {P2_SPAN}x")
        knees[K] = knee90(pts)
        if knees[K] not in P2_KNEE_SET:
            bad.append(f"K={K}: 90% knee at {knees[K]} rows, outside {sorted(P2_KNEE_SET)}")
    ks = [knees[K] for K in WIDTHS if knees[K]]
    if len(ks) == len(WIDTHS) and abs(math.log2(ks[0]) - math.log2(ks[1])) > 1:
        bad.append(f"knee moved between widths: {ks[0]} vs {ks[1]} rows (> 1 step; not rows-keyed)")
    return (PASS if not bad else KILL), bad, knees


def gate_p3(curve_a, curve_b):
    bad = []
    for K in WIDTHS:
        for rows in ROWS_SWEEP:
            a, b = curve_a[K][rows], curve_b[K][rows]
            if abs(a - b) / max(a, b) > P3_TOL:
                bad.append(f"K={K} rows={rows}: {a:.1f} vs {b:.1f} (> {P3_TOL:.0%})")
    return (PASS if not bad else KILL), bad


# ------------------------------------------------------------------- Phase B: pure helpers
def shape_factor(curve, rows, bpr):
    """Normalized factor in (0,1]: measured GB/s at (rows, bytes/row) over that width's own
    16384-row ceiling; log2-interpolated in rows, log2-interpolated across the two measured
    widths in bytes/row, clamped at the measured range (prereg #92 disclosed limit)."""
    def f_at(K):
        pts = curve[K]
        rs = sorted(pts)
        ceil = pts[rs[-1]]
        r = min(max(rows, rs[0]), rs[-1])
        lo = max(x for x in rs if x <= r)
        hi = min(x for x in rs if x >= r)
        if lo == hi:
            g = pts[lo]
        else:
            t = (math.log2(r) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
            g = pts[lo] + t * (pts[hi] - pts[lo])
        return g / ceil
    ws = sorted(curve)
    bprs = [K * 9 / 16.0 for K in ws]
    fs = [f_at(K) for K in ws]
    if len(ws) == 1 or bprs[-1] == bprs[0]:
        return fs[0]
    b = min(max(bpr, bprs[0]), bprs[-1])
    t = (math.log2(b) - math.log2(bprs[0])) / (math.log2(bprs[-1]) - math.log2(bprs[0]))
    return fs[0] + t * (fs[-1] - fs[0])


def model_factors(curve, tensors):
    """tensors: [(cls in {'ne','exp'}, rows, bytes_per_row, n_bytes)].
    Returns (F_ne, F_exp, clamped_byte_share, has_expert_set). Bytes-weighted HARMONIC means
    (time = bytes/BW), normalized so the whole-model mix factor is exactly 1 -- the pinned
    prereg #92 convention: eta keeps the level, shape only redistributes."""
    agg = {"ne": [0.0, 0.0], "exp": [0.0, 0.0]}
    clamped = tot = 0
    for cls, rows, bpr, nb in tensors:
        f = shape_factor(curve, rows, bpr)
        agg[cls][0] += nb
        agg[cls][1] += nb / f
        tot += nb
        ws = sorted(curve)
        if rows < 128 or rows > 16384 or bpr < ws[0] * 9 / 16 or bpr > ws[-1] * 9 / 16:
            clamped += nb
    f_ne = agg["ne"][0] / agg["ne"][1] if agg["ne"][1] else 1.0
    if not agg["exp"][1]:
        # Dense: the shipped planner never prices a shape-biased subset of a dense model
        # (hybrid/split rows are MoE-only), so the differential is 0 BY CONSTRUCTION (G-2).
        return 1.0, 1.0, (clamped / tot if tot else 0.0), False
    f_exp = agg["exp"][0] / agg["exp"][1]
    b_ne, b_exp = agg["ne"][0], agg["exp"][0]
    f_mix = (b_ne + b_exp) / (b_ne / f_ne + b_exp / f_exp)
    return f_ne / f_mix, f_exp / f_mix, (clamped / tot if tot else 0.0), True


def differential(f_ne, f_exp):
    return max(f_ne, f_exp) / min(f_ne, f_exp) - 1.0


def terms_match(actual, expected, rtol=TERMS_RTOL):
    """G-3: shipped row.terms must be reproduced exactly from imported plan constants."""
    if set(actual) != set(expected):
        return False
    for k in actual:
        a, e = actual[k], expected[k]
        if a == e == 0:
            continue
        if abs(a - e) / max(abs(a), abs(e)) > rtol:
            return False
    return True


def bindable(margin, diff):
    """G-1: can this cell's winner flip at all under a differential this size?"""
    return margin < diff


def score_p4(cells):
    """cells: [{scoreable, bind, flip, margin, diff}, ...]. Returns (verdict, detail).
    Staked: >=1 flip, every flip in a cell where margin < diff. Zero flips with >=1 bindable
    cell = TRUE NULL (a MISS, exit 1). Zero flips with zero bindable cells = UNABLE (exit 3)."""
    sc = [c for c in cells if c["scoreable"]]
    if not sc:
        return UNABLE, "no scoreable cells (every cell single-row, GPU-less, or flat-factor)"
    flips = [c for c in sc if c["flip"]]
    binds = [c for c in sc if c["bind"]]
    if flips:
        impossible = [c for c in flips if not c["bind"]]
        if impossible:
            return KILL, (f"{len(impossible)} flip(s) in cells whose top-2 margin exceeds the "
                          "factor differential - the re-pricing is broken, not informative")
        return PASS, f"{len(flips)} winner flip(s) across {len(sc)} scoreable cells ({len(binds)} bindable)"
    if binds:
        return KILL, (f"TRUE NULL (staked MISS): 0 flips although {len(binds)} of {len(sc)} "
                      "scoreable cells could have flipped - per-shape pricing changes no "
                      "emitted decision on this grid")
    return UNABLE, (f"0 flips but 0 of {len(sc)} scoreable cells could flip "
                    "(all margins exceed the differential) - the measurement cannot vary")


# ------------------------------------------------------------------------ Phase B: real data
def load_quantprobe():
    sys.path.insert(0, REPO)
    from quantprobe import plan, spec
    return plan, spec


def scan_geometry(path):
    """Per-tensor (cls, rows, bytes/row, bytes) from a GGUF; mirrors spec.from_gguf's
    expert-name rules and the U-26 untied-embedding exclusion."""
    from gguf import GGUFReader
    r = GGUFReader(path)
    has_output = any(t.name.startswith("output.") or "lm_head" in t.name for t in r.tensors)
    tensors, skipped, tot = [], 0, 0
    for t in r.tensors:
        nb = int(getattr(t, "n_bytes", 0) or 0)
        tot += nb
        shape = [int(d) for d in t.shape]
        if len(shape) < 2 or min(shape[0], shape[1]) <= 1:
            skipped += nb                      # 1D norms/bias: not matvec-shaped
            continue
        if "token_embd" in t.name and has_output:
            continue                           # gathered, not read (U-26 / prereg #76)
        experts = shape[2] if len(shape) >= 3 else 1
        rows = shape[1]
        bpr = nb / float(rows * experts)
        cls = "exp" if ("exps" in t.name or "_expert" in t.name) else "ne"
        tensors.append((cls, rows, bpr, nb))
    return tensors, (skipped / tot if tot else 0.0)


def recompute_terms(plan, m, hw, ctx):
    """Reproduce, from plan's own imported constants, the decomposition of every row this
    script re-prices. Used only through terms_match (G-3): if this drifts from the shipped
    evaluate(), the run REFUSES rather than scoring a stale model of the planner."""
    ab = max(m["bits"], 4.5)
    prot = m["ne"] if m["moe"] else min(m["ne"], m["t"] * plan.DENSE_PROTECTED_SHARE)
    act_ne = prot * ab / 8 * 1.15
    act_ex = (m["a"] - prot) * m["bits"] / 8 * 1.15
    kv_gb = ctx * m["kvp"] / 1e9 if ctx > 0 else 0.0
    eta_r = (plan.ETA_R_MOE if m["moe"] else plan.ETA_R_DENSE)
    eta_r /= (1.0 + m["cb"] * plan.IQ_CPU_TG_PENALTY)
    size = m["size_gb"]
    geta, vb, rb = hw["geta"], hw["vb"], hw["rb"]
    out = {"act_ne": act_ne, "act_ex": act_ex, "kv_gb": kv_gb}
    if vb > 0:
        out["all in VRAM"] = {"vram_bw": (act_ne + act_ex) / (geta * vb)
                                          + kv_gb / (plan.ETA_KV * vb)}
        v_need = m["ne"] * ab / 8 * 1.08 + 0.9 + kv_gb
        out["hybrid"] = {"vram_bw": act_ne / (geta * vb) + kv_gb / (plan.ETA_KV * vb),
                         "ram_bw": act_ex / (eta_r * rb)}
        experts_gb = size - m["ne"] * ab / 8 * 1.08
        v_free = hw["vc"] - v_need - plan.DESKTOP_VRAM_RESERVE
        if experts_gb > 0 and v_free > 0.3:
            f = min(1.0, v_free / experts_gb)
            out["split_f"] = f
            out["split"] = {"vram_bw": (act_ne + f * act_ex) / (geta * vb)
                                        + kv_gb / (plan.ETA_KV * vb),
                            "ram_bw": (1 - f) * act_ex / (eta_r * rb)}
    return out


def reprice_rows(plan, rows, rc, hw, f_ne, f_exp):
    """[(name, flags, tok_shipped, tok_repriced)] with G-3 verification per recognized row.
    Raises RuntimeError on a G-3 mismatch."""
    out = []
    geta, vb = hw["geta"], hw["vb"]
    for row in rows:
        name, tok_s, _w, flags = row[0], row[1], row[2], row[3]
        terms = dict(getattr(row, "terms", {}) or {})
        eff = getattr(row, "eff", 1.0)
        new_tok = tok_s
        key = None
        if name == "all in VRAM":
            key = "all in VRAM"
        elif name.startswith("hybrid: attention->VRAM"):
            key = "hybrid"
        elif name.startswith("split experts:"):
            key = "split"
        if key and vb > 0 and key in rc:
            if not terms_match(terms, rc[key]):
                raise RuntimeError(
                    f"G-3: recomputed decomposition for '{name}' does not reproduce the "
                    f"shipped row terms (got {terms}, expected {rc[key]}) - plan.py has "
                    "drifted from this script's model of it; refusing to score")
            kv_term = rc["kv_gb"] / (plan.ETA_KV * vb)
            t2 = dict(terms)
            if key == "all in VRAM":
                t2["vram_bw"] = (rc["act_ne"] / f_ne + rc["act_ex"] / f_exp) / (geta * vb) + kv_term
            elif key == "hybrid":
                t2["vram_bw"] = rc["act_ne"] / f_ne / (geta * vb) + kv_term
            else:
                f = rc["split_f"]
                t2["vram_bw"] = ((rc["act_ne"] / f_ne + f * rc["act_ex"] / f_exp)
                                 / (geta * vb) + kv_term)
            new_tok = eff / sum(t2.values())
        out.append((name, flags, tok_s, new_tok))
    return out


def p5_check(plan, models, machines, log):
    """Staked P-5: scaling vb / geta through the SHIPPED evaluate() moves tok/s but never a
    split fraction or an emitted flag (they are capacity-determined). Returns (verdict, bad)."""
    bad = []
    checked = moved = 0
    for m in models:
        for mac, hw in machines.items():
            base = plan.evaluate(m["t"], m["a"], m["ne"], m["moe"], m["bits"], hw["vc"],
                                 hw["vb"], hw["rc"], hw["rb"], hw["db"], hw["geta"],
                                 gl=hw.get("gl"), ctx=0, kvp=m["kvp"], n_layer=m["nl"],
                                 true_size_gb=m["size_gb"], codebook_share=m["cb"])[2]
            base_id = [(r[0], r[3]) for r in base]
            for vb_s, geta_s in ((0.5, 1.0), (2.0, 1.0), (1.0, 0.5), (1.0, 2.0)):
                alt = plan.evaluate(m["t"], m["a"], m["ne"], m["moe"], m["bits"], hw["vc"],
                                    hw["vb"] * vb_s, hw["rc"], hw["rb"], hw["db"],
                                    hw["geta"] * geta_s, gl=hw.get("gl"), ctx=0,
                                    kvp=m["kvp"], n_layer=m["nl"],
                                    true_size_gb=m["size_gb"], codebook_share=m["cb"])[2]
                checked += 1
                if [(r[0], r[3]) for r in alt] != base_id:
                    bad.append(f"{m['name']} x {mac} vb*{vb_s} geta*{geta_s}: "
                               "row names/flags changed with bandwidth")
                elif hw["vb"] > 0 and any(abs(a[1] - b[1]) > 1e-12
                                          for a, b in zip(alt, base)):
                    moved += 1
    if checked and moved == 0 and any(hw["vb"] > 0 for hw in machines.values()):
        bad.append("tok/s never moved under bandwidth scaling - the perturbation is not live "
                   "(a measurement that cannot vary)")
    log(f"  P-5: {checked} perturbed evaluations, tok/s moved in {moved}, "
        f"structure violations: {len(bad)}")
    return (PASS if not bad else UNABLE), bad


# ----------------------------------------------------------------------------- self-test
def self_test():
    """Constructs the prereg #92 failing inputs and verifies every guard fires (exits 1 if
    any guard can NOT fail -- an unfalsifiable guard is the failure signature)."""
    fails = []

    # 1. G-2: flat curve + uniform geometry must yield differential ~0 (excluded, never null)
    flat = {K: {r: 100.0 for r in ROWS_SWEEP} for K in WIDTHS}
    f_ne, f_exp, _, has_exp = model_factors(
        flat, [("ne", 8192, 1152.0, 10**9), ("exp", 8192, 1152.0, 10**9)])
    if not has_exp or differential(f_ne, f_exp) >= G2_MIN_DIFFERENTIAL:
        fails.append("G-2 did not fire on the flat-factor construction")

    # a REAL gradient must produce a nonzero differential (the guard must be able to not fire)
    grad = {K: dict(zip(ROWS_SWEEP, [30, 45, 61, 76, 88, 94, 97, 99])) for K in WIDTHS}
    f_ne2, f_exp2, _, _ = model_factors(
        grad, [("ne", 512, 1152.0, 10**9), ("exp", 8192, 1152.0, 10**9)])
    if differential(f_ne2, f_exp2) < G2_MIN_DIFFERENTIAL:
        fails.append("factor machinery is dead: a 512-vs-8192-row model shows no differential")

    # 2. G-1 + P-4: a grid of single-row and wide-margin cells must be UNABLE, never a null
    cells = [
        {"scoreable": False, "bind": False, "flip": False},                  # single-row cell
        {"scoreable": True, "bind": bindable(1.40, 0.20), "flip": False},    # 2.4x-apart pair
    ]
    v, _ = score_p4(cells)
    if v != UNABLE:
        fails.append(f"P-4 returned {v} on the cannot-vary grid (must be UNABLE, exit 3)")

    # ...and a bindable no-flip grid must be the TRUE NULL (KILL), not a pass
    v, _ = score_p4([{"scoreable": True, "bind": True, "flip": False}])
    if v != KILL:
        fails.append(f"P-4 returned {v} on the bindable no-flip grid (must be the staked MISS)")

    # ...and a flip in an un-bindable cell must KILL (broken re-pricing), not pass
    v, _ = score_p4([{"scoreable": True, "bind": False, "flip": True}])
    if v != KILL:
        fails.append(f"P-4 returned {v} on an impossible flip (must be KILL)")

    # 3. P-2 must fire on a non-monotone curve and on a knee outside 4096 +/- 1 step
    dip = {K: dict(zip(ROWS_SWEEP, [30, 45, 61, 40, 88, 94, 97, 99])) for K in WIDTHS}
    if gate_p2(dip)[0] != KILL:
        fails.append("P-2 did not fire on a non-monotone curve")
    early = {K: dict(zip(ROWS_SWEEP, [90, 95, 96, 97, 98, 98, 99, 100])) for K in WIDTHS}
    if gate_p2(early)[0] != KILL:
        fails.append("P-2 did not fire on a 128-row knee (span/knee gates both dead)")
    if gate_p2(grad)[0] != PASS:
        fails.append("P-2 rejects the reference-shaped curve - the gate cannot pass at all")

    # 4. G-3: a tampered decomposition must be refused
    if terms_match({"vram_bw": 1.0}, {"vram_bw": 1.0 + 1e-3}):
        fails.append("G-3 accepted a 0.1% decomposition mismatch")
    if not terms_match({"vram_bw": 1.0}, {"vram_bw": 1.0}):
        fails.append("G-3 rejects an exact match - it would refuse every run")

    # 5. P-1 must fire when the rewrite's curve disagrees with the reference
    ref = {K: dict(zip(ROWS_SWEEP, [30, 45, 61, 76, 88, 94, 97, 99])) for K in WIDTHS}
    off = {K: dict(zip(ROWS_SWEEP, [70, 80, 85, 90, 95, 97, 98, 99])) for K in WIDTHS}
    if gate_p1(off, ref)[0] != KILL:
        fails.append("P-1 did not fire on a curve with the wrong penalties")
    if gate_p1(ref, ref)[0] != PASS:
        fails.append("P-1 rejects the reference against itself - it can never pass")

    for f in fails:
        print("SELF-TEST FAIL:", f)
    if fails:
        print(f"SELF-TEST: {len(fails)} guard(s) cannot fail correctly - fix before running")
        return 1
    print("SELF-TEST PASS: every staked guard fires on its constructed failing input "
          "and passes on a sane one")
    return 0


# ---------------------------------------------------------------------------------- main
def phase_a(args, log):
    if os.path.isfile(CURVE_PATH) and not args.remeasure:
        with open(CURVE_PATH, encoding="utf-8") as f:
            stored = json.load(f)
        log(f"  Phase A: reusing stored curve ({stored.get('date')}, "
            f"device '{stored.get('device')}') - pass --remeasure to sweep again")
        cur_dev = gpu_name()
        if cur_dev and stored.get("device") and cur_dev != stored["device"]:
            log(f"REFUSE: stored curve is for '{stored['device']}' but this box has "
                f"'{cur_dev}' - a calibration is a property of ONE card")
            return None, REFUSE
        a = {int(K): {int(r): v for r, v in pts.items()}
             for K, pts in stored["sweep_1"].items()}
        b = {int(K): {int(r): v for r, v in pts.items()}
             for K, pts in stored["sweep_2"].items()}
        return {"curve": a, "repeat": b, "device": stored.get("device")}, None
    if not os.path.isfile(REF_LOG):
        log(f"REFUSE: reference log missing: {REF_LOG} (P-1 cannot be scored)")
        return None, REFUSE
    if not gpu_name():
        log("REFUSE: no NVIDIA GPU visible to nvidia-smi - Phase A needs the card")
        return None, REFUSE
    exe, err = compile_probe(log)
    if not exe:
        log(f"REFUSE: {err}")
        return None, REFUSE
    log(f"  clocks before: {gpu_clocks()}")
    with open(SWEEP_LOG, "w", encoding="utf-8") as raw:
        raw.write(f"# prereg #92 shape sweep, {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log("  sweep 1/2:")
        c1, dev, err = run_sweep(exe, args.budget_mb, args.reps, log, raw)
        if err:
            log(f"REFUSE: {err}")
            return None, REFUSE
        log("  sweep 2/2 (P-3 repeat, same session):")
        c2, _, err = run_sweep(exe, args.budget_mb, args.reps, log, raw)
        if err:
            log(f"REFUSE: {err}")
            return None, REFUSE
    log(f"  clocks after:  {gpu_clocks()}")
    artifact = dict(prereg=92, date=time.strftime("%Y-%m-%d"), device=dev,
                    budget_mb=args.budget_mb, reps=args.reps,
                    note="per-shape bandwidth curve; the calibrate --shapes artifact prototype",
                    sweep_1={str(K): c1[K] for K in c1},
                    sweep_2={str(K): c2[K] for K in c2})
    with open(CURVE_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=1)
    log(f"  curve artifact written: {CURVE_PATH}")
    return {"curve": c1, "repeat": c2, "device": dev}, None


def phase_b(args, curve, log, results):
    plan, spec = load_quantprobe()
    gdir = args.gguf_dir
    if not os.path.isdir(gdir):
        log(f"REFUSE: GGUF dir not found: {gdir}")
        return REFUSE
    files = sorted(f for f in os.listdir(gdir) if f.lower().endswith(".gguf"))
    if len(files) < 3:
        log(f"REFUSE: need >=3 GGUFs in {gdir}, found {len(files)}")
        return REFUSE
    models = []
    for fn in files:
        path = os.path.join(gdir, fn)
        try:
            s = spec.from_gguf(path)
            tensors, skipped_share = scan_geometry(path)
        except Exception as e:
            log(f"  skip {fn}: unreadable ({e})")
            continue
        if skipped_share > 0.20:
            log(f"  skip {fn}: {skipped_share:.0%} of bytes not matvec-classifiable")
            continue
        f_ne, f_exp, clamp_share, has_exp = model_factors(curve, tensors)
        models.append(dict(name=fn, t=s["t"], a=s["a"], ne=s["ne"], moe=s["moe"],
                           bits=s["bits"], kvp=s["kvp"], nl=s["n_layer"],
                           cb=s["codebook_share"],
                           size_gb=os.path.getsize(path) / 1e9,
                           f_ne=f_ne, f_exp=f_exp, has_exp=has_exp,
                           diff=differential(f_ne, f_exp), clamp=clamp_share))
        log(f"  {fn}: F_ne={f_ne:.3f} F_exp={f_exp:.3f} diff={differential(f_ne, f_exp):.1%} "
            f"clamped-width bytes={clamp_share:.0%}{'' if has_exp else '  [dense: G-2 flat]'}")
    if not models:
        log("REFUSE: no scoreable GGUFs")
        return REFUSE
    if not any(m["has_exp"] and m["diff"] >= G2_MIN_DIFFERENTIAL for m in models):
        log("REFUSE: no model in the eval set carries a shape differential the shipped "
            "planner can act on (all dense or flat) - P-4 could never bind; this grid is "
            "the constructed failing input, not an experiment")
        return REFUSE

    machines = plan.MACHINES
    cells, cell_log = [], []
    for m in models:
        for mac, hw in machines.items():
            for ctx in (0, 16384):
                rows = plan.evaluate(m["t"], m["a"], m["ne"], m["moe"], m["bits"],
                                     hw["vc"], hw["vb"], hw["rc"], hw["rb"], hw["db"],
                                     hw["geta"], gl=hw.get("gl"), ctx=ctx, kvp=m["kvp"],
                                     n_layer=m["nl"], true_size_gb=m["size_gb"],
                                     codebook_share=m["cb"])[2]
                if not rows:
                    continue
                rc = recompute_terms(plan, m, hw, ctx)
                try:
                    priced = reprice_rows(plan, rows, rc, hw, m["f_ne"], m["f_exp"])
                except RuntimeError as e:
                    log(str(e))
                    return REFUSE
                shipped = max(priced, key=lambda r: r[2])
                repriced = max(priced, key=lambda r: r[3])
                margin = None
                if len(priced) >= 2:
                    top2 = sorted((r[2] for r in priced), reverse=True)[:2]
                    margin = (top2[0] - top2[1]) / top2[1]
                gpu_row = any(r[0] in ("all in VRAM",) or r[0].startswith(("hybrid:", "split "))
                              for r in priced) and hw["vb"] > 0
                scoreable = (len(priced) >= 2 and gpu_row
                             and m["diff"] >= G2_MIN_DIFFERENTIAL)
                cell = dict(model=m["name"], machine=mac, ctx=ctx,
                            scoreable=scoreable,
                            bind=(scoreable and margin is not None
                                  and bindable(margin, m["diff"])),
                            flip=(shipped[0], shipped[1]) != (repriced[0], repriced[1]),
                            margin=margin, diff=m["diff"],
                            shipped=[shipped[0], shipped[2]],
                            repriced=[repriced[0], repriced[3]])
                cells.append(cell)
                if cell["flip"]:
                    cell_log.append(f"    FLIP {m['name']} x {mac} ctx={ctx}: "
                                    f"'{shipped[0]}' -> '{repriced[0]}' "
                                    f"(margin {margin:.1%} < diff {m['diff']:.1%}?)")
    for line in cell_log:
        log(line)
    v4, d4 = score_p4(cells)
    n_sc = sum(c["scoreable"] for c in cells)
    n_b = sum(c["bind"] for c in cells)
    log(f"  grid: {len(cells)} cells, {n_sc} scoreable, {n_b} bindable, "
        f"{sum(c['flip'] for c in cells)} flips")
    if n_sc < 10:
        log(f"REFUSE: only {n_sc} scoreable cells (< 10) - the grid is too degenerate to "
            "support any P-4 verdict")
        return REFUSE
    log(f"  P-4 [{v4}] {d4}")
    v5, bad5 = p5_check(plan, models, machines, log)
    for b in bad5:
        log(f"    P-5: {b}")
    log(f"  P-5 [{v5}] split fractions/flags are capacity-determined "
        f"{'(as staked)' if v5 == PASS else '- STRUCTURAL READING WRONG, run INCOMPLETE'}")
    results["phase_b"] = dict(cells=cells, p4=[v4, d4], p5=[v5, bad5],
                              models=[{k: m[k] for k in
                                       ("name", "moe", "f_ne", "f_exp", "diff", "clamp")}
                                      for m in models])
    if v5 != PASS:
        return UNABLE
    return v4


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--phase", choices=("a", "b", "all"), default="all")
    ap.add_argument("--gguf-dir", default=DEFAULT_GGUF_DIR)
    ap.add_argument("--remeasure", action="store_true")
    ap.add_argument("--budget-mb", type=int, default=BUDGET_MB)
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    os.makedirs(DATA, exist_ok=True)
    log = Tee(LOG_PATH)
    log(f"prereg #92 - per-shape calibration - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        load_quantprobe()
        import gguf  # noqa: F401
    except Exception as e:
        log(f"REFUSE: cannot import quantprobe/gguf from {REPO}: {e}")
        return 2
    if self_test() != 0:
        log("REFUSE: self-test failed - guards are not falsifiable, no measurement is run")
        return 2

    results = dict(prereg=92, date=time.strftime("%Y-%m-%d"))
    verdicts = {}

    a, err = phase_a(args, log)
    if err:
        return 2
    curve, repeat = a["curve"], a["repeat"]
    ref = parse_ref_log(REF_LOG)
    v1, bad1 = gate_p1(curve, ref)
    if v1 == REFUSE:
        log(f"REFUSE: {bad1}")
        return 2
    v2, bad2, knees = gate_p2(curve)
    v3, bad3 = gate_p3(curve, repeat)
    for name, v, bad in (("P-1", v1, bad1), ("P-2", v2, bad2), ("P-3", v3, bad3)):
        log(f"  {name} [{v}]" + (f" {len(bad)} violation(s)" if bad else ""))
        for b in bad:
            log(f"    {b}")
    log(f"  knees (90% of own max): {knees}")
    verdicts.update(p1=v1, p2=v2, p3=v3)
    results["phase_a"] = dict(device=a.get("device"),
                              sweep_1={str(K): curve[K] for K in curve},
                              sweep_2={str(K): repeat[K] for K in repeat},
                              knees={str(k): v for k, v in knees.items()},
                              p1=[v1, bad1], p2=[v2, bad2], p3=[v3, bad3])

    if args.phase in ("b", "all"):
        if all(v == PASS for v in (v1, v2, v3)):
            vb_ = phase_b(args, curve, log, results)
            if vb_ == REFUSE:
                return 2
            verdicts["p4_p5"] = vb_
        else:
            log("  Phase B NOT EVALUATED: the curve failed its own gates; scoring placements "
                "against a refuted curve would be theatre")
            verdicts["p4_p5"] = "NOT_EVALUATED"

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(dict(results, verdicts=verdicts), f, indent=1, default=str)
    log(f"  results: {RESULT_JSON}")

    vs = list(verdicts.values())
    if any(v == KILL for v in vs):
        log("VERDICT: FAIL - a staked kill rule fired (see above; a P-4 true null is a MISS "
            "and is published as one)")
        return 1
    if any(v in (UNABLE, "NOT_EVALUATED") for v in vs):
        log("VERDICT: INCOMPLETE - a gate could not be evaluated; this run must not be cited")
        return 3
    log("VERDICT: PASS - all staked gates hit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
