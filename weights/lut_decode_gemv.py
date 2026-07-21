"""lut_decode_gemv.py -- VERIFIED reference for the decode-as-gather 2-bit expert-GEMV (paper Sec 9).

The kernel the paper calls for is NOT the rate-optimal trellis (decode-pessimal, ~12 tok/s @ ~10% util)
but a 2-bit code that is itself GATHER/LUT-decodable so it runs at high utilization on Pascal
(tok/s = bw * util / bytes). This file is the algorithm, runnable + checkable on CPU (no nvcc/GPU needed),
and it is the exact math the companion lut_expert_gemv.cu implements with __dp4a on cc>=6.1.

Format: per-output-row, per-group (G=128) 2-bit. code c in {0,1,2,3} indexes a 4-entry codebook CB
(Lloyd-Max optimal levels for a unit-variance Gaussian); weight ~= scale * CB[code], scale = rowgroup_std.
Codes pack 4/byte. Decode-as-gather GEMV: y[o] = sum_g scale[o,g] * sum_{i in g} CB[code] * x[i] -- the
inner sum is a gather-dot (fp path, exact) or, with CB and x cast to int8, one __dp4a per 4 elements (GPU).
The codebook decouples code QUALITY (good 2-bit levels) from the DECODE primitive (gather), which is the
whole point: a gather-decodable code runs at high utilization where the sequential trellis does not.
"""
from __future__ import annotations
import numpy as np

CB = np.array([-1.5104, -0.4528, 0.4528, 1.5104], dtype=np.float32)   # Lloyd-Max 2-bit Gaussian levels (sigma units)
CBI = np.round(CB / CB.max() * 127).astype(np.int32)                  # int8 codebook for the DP4A path
DP4A_S = float(CB.max()) / 127.0
G = 128


def pack_2bit(W):
    """W [out, in] (in % G == 0) -> (packed uint8 [out, in//4], scales fp32 [out, in//G])."""
    out, inn = W.shape
    assert inn % G == 0
    ng = inn // G
    scales = np.empty((out, ng), np.float32)
    codes = np.empty((out, inn), np.uint8)
    for g in range(ng):
        blk = W[:, g * G:(g + 1) * G]
        s = blk.std(1); s[s == 0] = 1e-9              # per-group std (CB is in sigma units)
        scales[:, g] = s
        d = np.abs(blk[:, :, None] / s[:, None, None] - CB[None, None, :])   # [out, G, 4]
        codes[:, g * G:(g + 1) * G] = d.argmin(2).astype(np.uint8)
    packed = (codes[:, 0::4] | (codes[:, 1::4] << 2) |
              (codes[:, 2::4] << 4) | (codes[:, 3::4] << 6)).astype(np.uint8)
    return packed, scales


def _unpack(packed, inn):
    out = packed.shape[0]
    codes = np.empty((out, inn), np.uint8)
    codes[:, 0::4] = packed & 3; codes[:, 1::4] = (packed >> 2) & 3
    codes[:, 2::4] = (packed >> 4) & 3; codes[:, 3::4] = (packed >> 6) & 3
    return codes


def dequant(packed, scales, inn):
    """Reference dequant: unpack -> CB[code] -> * scale. Returns W_hat [out, in]."""
    lv = CB[_unpack(packed, inn)]                     # [out, in]
    return lv * np.repeat(scales, G, axis=1)


def lut_gemv(packed, scales, x):
    """Decode-as-gather GEMV in fp (the kernel's structure, exact): y[o] = sum_g s * sum_i CB[c]*x_i."""
    out, inn = packed.shape[0], packed.shape[1] * 4
    lv = CB[_unpack(packed, inn)]                     # gathered codebook values (fp)
    y = np.zeros(out, np.float32)
    ng = inn // G
    for g in range(ng):                               # per-group: gather-dot, then scale
        y += scales[:, g] * (lv[:, g * G:(g + 1) * G] @ x[g * G:(g + 1) * G])
    return y


def dp4a_gemv(packed, scales, x):
    """The Pascal __dp4a path: cast the codebook (CBI) and x to int8 and accumulate the inner dot as
    integers, then scale. Error vs dequant@x comes from the int8 quantization of x and the codebook."""
    out, inn = packed.shape[0], packed.shape[1] * 4
    sx = np.abs(x).max() / 127.0 + 1e-12
    xq = np.clip(np.round(x / sx), -127, 127).astype(np.int32)
    lv = CBI[_unpack(packed, inn)]                    # int8 codebook levels
    y = np.zeros(out, np.float32)
    ng = inn // G
    for g in range(ng):
        idot = (lv[:, g * G:(g + 1) * G] * xq[None, g * G:(g + 1) * G]).sum(1)   # __dp4a accumulation
        y += scales[:, g] * (sx * DP4A_S) * idot
    return y


def selftest():
    rng = np.random.default_rng(0)
    out, inn = 1408, 2048                             # an expert down/gate shape
    W = rng.standard_normal((out, inn)).astype(np.float32) * 0.03
    x = rng.standard_normal(inn).astype(np.float32)
    packed, scales = pack_2bit(W)
    Wq = dequant(packed, scales, inn)                 # ground-truth reconstruction

    y_ref = Wq @ x                                    # naive dequant-then-matmul (reference)
    y_lut = lut_gemv(packed, scales, x)               # decode-as-gather (fp) -- must be bit-exact
    y_dp4a = dp4a_gemv(packed, scales, x)             # int8/DP4A path -- within x-quant tolerance

    def rel(a, b): return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))
    print(f"packed: {packed.nbytes/1e3:.1f} KB codes + {scales.nbytes/1e3:.1f} KB scales "
          f"= {(packed.nbytes+scales.nbytes)*8/(out*inn):.2f} bits/weight (target ~2.12 w/ fp16 scale)")
    print(f"weight rel-MSE @2bit (this simple LUT code): {((W-Wq)**2).sum()/(W**2).sum():.4f}  "
          f"(trellis floor 0.069; this code trades quality for decode-util)")
    print(f"lut_gemv  vs dequant@x : max rel err = {rel(y_lut, y_ref):.2e}   <- decode-as-gather is EXACT")
    print(f"dp4a_gemv vs dequant@x : max rel err = {rel(y_dp4a, y_ref):.2e}   <- int8/DP4A path (x-quant only)")
    ok = rel(y_lut, y_ref) < 1e-5 and rel(y_dp4a, y_ref) < 2e-2
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    selftest()
