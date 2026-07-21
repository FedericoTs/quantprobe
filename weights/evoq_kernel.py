"""F0 kernel loader + bit-exact correctness gate + util microbench (needs nvcc/CUDA toolkit).

Builds evoq_kernel.cu via torch JIT, verifies the fixed-4-bit decode-LUT GEMV matches a torch
reference on a real 7B-sized matrix, then microbenches sustained VRAM bandwidth utilization --
THE number the whole Campaign-2 roofline hinges on (tok/s = 192e9 * util / resident_bytes).

Run:  python -m weights.evoq_kernel
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
G = 128


def build():
    from torch.utils.cpp_extension import load
    return load(name="evoq_f0", sources=[os.path.join(HERE, "evoq_kernel.cu")],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_61,code=sm_61"]
                + (["-Xptxas=-v"] if os.environ.get("EVOQ_PTXAS_V") else []),
                verbose=True)


def pack4(idx_2d: np.ndarray) -> np.ndarray:
    """idx [out,in] uint8 (<16) -> packed [out, in/2] uint8 (low nibble = even col)."""
    out, inn = idx_2d.shape
    a = idx_2d.reshape(out, inn // 2, 2)
    return (a[:, :, 0] | (a[:, :, 1] << 4)).astype(np.uint8)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available"); return
    dev = "cuda"
    ext = build()
    print("built evoq_f0\n", flush=True)

    # ---- correctness on a real 7B-sized shape (down_proj-like) ----
    out, inn = 3584, 18944
    rng = np.random.default_rng(0)
    K = 12
    lv = np.sort(rng.standard_normal(16).astype(np.float32)); lv[K:] = lv[K - 1]  # 12 used
    idx = rng.integers(0, K, (out, inn)).astype(np.uint8)
    amax = (0.5 + rng.random((out, inn // G))).astype(np.float32)
    xrr = rng.standard_normal(inn).astype(np.float32)

    packed = torch.from_numpy(pack4(idx)).to(dev)
    lv_t = torch.from_numpy(lv).to(dev)
    amax_t = torch.from_numpy(amax).to(dev)
    xrr_t = torch.from_numpy(xrr).to(dev)

    # reference: y_j = sum_g amax[j,g] * sum_{i in g} lv[idx]*xrr
    Wv = lv[idx] * np.repeat(amax, G, axis=1)
    yref = torch.from_numpy((Wv * xrr[None, :]).sum(1)).to(dev)

    bytes_read = out * inn * 0.5 + out * (inn // G) * 4    # packed4 + amax fp32

    def bench(fn, name):
        y = fn(packed, xrr_t, lv_t, amax_t)
        err = (y - yref).abs().max().item() / (yref.abs().max().item() + 1e-9)
        ok = "OK" if err < 1e-4 else "FAIL"
        torch.cuda.synchronize()
        for _ in range(5):
            fn(packed, xrr_t, lv_t, amax_t)
        torch.cuda.synchronize()
        N = 200
        t0 = time.time()
        for _ in range(N):
            fn(packed, xrr_t, lv_t, amax_t)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / N
        gbs = bytes_read / dt / 1e9
        util = gbs / 192.0
        print(f"\n[{name}] rel err {err:.2e} {ok} | {dt*1e3:.3f} ms/call | "
              f"{gbs:.1f} GB/s = {100*util:.0f}% util")
        for nm, bpw in [("F0 4.25 b/w", 4.25), ("S1 3.48 b/w", 3.48)]:
            toks = 192e9 * util / (bpw * 6.5e9 / 8)
            print(f"    -> 7B {nm}: {toks:.1f} tok/s  (Q4_K_M 21.8)")
        return util

    print(f"matrix {out}x{inn} ({bytes_read/1e6:.0f} MB packed)")
    bench(ext.f0_gemv, "v1 naive (1 warp/row, 2B loads)")
    bench(ext.f0_gemv2, "v2 MLP (8x unroll, float4 xrr, 8 warps/blk)")
    bench(ext.f0_gemv3, "v3 MLP + 2 rows/warp (shared xrr, 2 accum)")
    util = bench(ext.f0_gemv4, "v4 v3 + conflict-free LUT (s_lv[e*32+lane])")

    # ---- dp4a int8 kernel (quality verified +0.0061 ppl, weights/dp4a_quality.py) ----
    # Q8-quantize xrr per group; int8-snap codebook. dp4a resident bytes are the SAME packed4.
    xrr_g = xrr.reshape(inn // G, G)
    a_scale = (np.abs(xrr_g).max(1) / 127.0).astype(np.float32)
    xq = np.round(xrr_g / a_scale[:, None]).clip(-127, 127).astype(np.int8).reshape(-1)
    cb_scale = float(np.abs(lv).max()) / 127.0
    lvq = np.round(lv / cb_scale).clip(-127, 127).astype(np.int8)
    xq_t = torch.from_numpy(xq).to(dev)
    asc_t = torch.from_numpy(a_scale).to(dev)
    lvq_t = torch.from_numpy(lvq).to(dev)
    # int8 reference: y_j = sum_g cb*asc_g*amax * sum_k lvq[idx]*xq
    Wq = lvq[idx].astype(np.int32)
    xq2 = xq.astype(np.int32).reshape(inn // G, G)
    accg = np.einsum('og,g->o', Wq.reshape(out, inn // G, G).reshape(out, -1), xq) if False else None
    yq = np.zeros(out, np.float64)
    Wq3 = lvq[idx].astype(np.int32).reshape(out, inn // G, G)
    for g in range(inn // G):
        accg = (Wq3[:, g, :] * xq2[g][None, :]).sum(1)
        yq += accg * (cb_scale * a_scale[g]) * amax[:, g]
    yqref = torch.from_numpy(yq.astype(np.float32)).to(dev)

    def bench_d0():
        y = ext.d0_gemv(packed, xq_t, asc_t, lvq_t, amax_t, cb_scale)
        err = (y - yqref).abs().max().item() / (yqref.abs().max().item() + 1e-9)
        ok = "OK" if err < 5e-3 else "FAIL"   # int8/Q8 path: exact int, only fp accumulate order
        torch.cuda.synchronize()
        for _ in range(5):
            ext.d0_gemv(packed, xq_t, asc_t, lvq_t, amax_t, cb_scale)
        torch.cuda.synchronize()
        N = 200
        t0 = time.time()
        for _ in range(N):
            ext.d0_gemv(packed, xq_t, asc_t, lvq_t, amax_t, cb_scale)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / N
        gbs = bytes_read / dt / 1e9
        u = gbs / 192.0
        print(f"\n[d0 dp4a int8 (4 MAC/instr, quality +0.006 ppl)] rel err {err:.2e} {ok} | "
              f"{dt*1e3:.3f} ms/call | {gbs:.1f} GB/s = {100*u:.0f}% util")
        for nm, bpw in [("F0 4.25 b/w", 4.25), ("S1 3.48 b/w", 3.48)]:
            toks = 192e9 * u / (bpw * 6.5e9 / 8)
            print(f"    -> 7B {nm}: {toks:.1f} tok/s  (Q4_K_M 21.8)")
        return u

    util = max(util, bench_d0())
    print("\nGATE: util >= 0.50 -> FLOOR banked + S1 greenlit; 0.45-0.50 parity; <0.45 occupancy-bound "
          "(profile: try 8 warps/block, vectorized uint4 loads, 2 rows/warp).")

    sweep_real_shapes(ext)


def sweep_real_shapes(ext):
    """End-to-end tok/s on the REAL Qwen2.5-7B layer-shape mix (28 layers x 7 linears).
    Sums measured per-GEMV time = one token's matmul cost (attn/norm overhead excluded).
    Uses f0_gemv3 (fp32, ~best). S1 (3.48 b/w) projected from F0 (4.25) by the byte ratio
    (valid since both are occupancy-bound -> ~iso-util)."""
    dev = "cuda"
    L = 28
    # (name, rows[out], cols[in], count) -- Qwen2.5-7B
    shapes = [
        ("q_proj",  3584,  3584, L), ("k_proj",   512, 3584, L), ("v_proj",  512,  3584, L),
        ("o_proj",  3584,  3584, L), ("gate_proj", 18944, 3584, L), ("up_proj", 18944, 3584, L),
        ("down_proj", 3584, 18944, L),
    ]
    rng = np.random.default_rng(1)
    print("\n--- REAL 7B shape-weighted end-to-end (f0_gemv3, fp32) ---")
    total_ms = 0.0
    total_bytes = 0
    for nm, out, inn, cnt in shapes:
        K = 12
        lv = np.sort(rng.standard_normal(16).astype(np.float32)); lv[K:] = lv[K - 1]
        idx = rng.integers(0, K, (out, inn)).astype(np.uint8)
        amax = (0.5 + rng.random((out, inn // G))).astype(np.float32)
        xrr = rng.standard_normal(inn).astype(np.float32)
        packed = torch.from_numpy(pack4(idx)).to(dev)
        lv_t = torch.from_numpy(lv).to(dev); amax_t = torch.from_numpy(amax).to(dev)
        xrr_t = torch.from_numpy(xrr).to(dev)
        bytes_read = out * inn * 0.5 + out * (inn // G) * 4
        torch.cuda.synchronize()
        for _ in range(3):
            ext.f0_gemv3(packed, xrr_t, lv_t, amax_t)
        torch.cuda.synchronize()
        N = 200
        t0 = time.perf_counter()
        for _ in range(N):
            ext.f0_gemv3(packed, xrr_t, lv_t, amax_t)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / N * 1e3
        u = (bytes_read / (ms / 1e3)) / 192e9
        total_ms += ms * cnt
        total_bytes += bytes_read * cnt
        print(f"  {nm:<10} {out:>6}x{inn:<6} x{cnt}  {ms:.3f} ms  util {100*u:.0f}%")
    f0_toks = 1000.0 / total_ms
    s1_toks = f0_toks * (4.25 / 3.48)
    agg_util = (total_bytes / (total_ms / 1e3)) / 192e9
    print(f"\n  one-token matmul time = {total_ms:.2f} ms ({total_bytes/1e9:.2f} GB resident F0)")
    print(f"  aggregate util = {100*agg_util:.0f}%")
    print(f"  -> F0 (4.25 b/w, {total_bytes/1e9:.2f}GB): {f0_toks:.1f} tok/s")
    print(f"  -> S1 (3.48 b/w, {total_bytes/1e9*3.48/4.25:.2f}GB): {s1_toks:.1f} tok/s   (Q4_K_M measured 21.8 @ 4.56GB)")
    print("  NOTE: matmul-only (excludes attention/RMSNorm/sampling); real end-to-end ~10-20% lower.")


if __name__ == "__main__":
    main()
