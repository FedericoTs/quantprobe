"""Potential of a LOW-RANK residual mode on the (rank-<=4) abliteration delta.

For each changed 2D tensor: dW = ft - base (float); take rank-r SVD; quantize factors
A,B to bf16; reconstruct L = Af@Bf; ref' = round_bf16(base + L); store factors + the
arith residual between ft and ref'. If dW is low-rank, ref' ~= ft so the residual is
tiny -> the changed tensors (currently ~55%) compress to ~95%+, lifting the whole model.
Measures size + verifies byte-exact reconstruction (this machine).
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evocompress import backends  # noqa: E402
from weights import wcodec as wc  # noqa: E402

ZSTD = backends.get_backend("zstd")
BASE = os.path.join(_ROOT, "weights", "data", "qwen", "base.safetensors")
ABL = os.path.join(_ROOT, "weights", "data", "qwen", "ablit.safetensors")


def to_bf16(f32):
    u = f32.view(np.uint32)
    return (((u.astype(np.uint64) + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16))


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def lowrank_blob(ft_u16, base_u16, shape, r=8):
    m, n = shape
    base_f = bf16_to_f32(base_u16).reshape(m, n)
    dW = (bf16_to_f32(ft_u16).reshape(m, n) - base_f).astype(np.float64)
    U, S, Vt = np.linalg.svd(dW, full_matrices=False)
    r = min(r, S.size)
    A = (U[:, :r] * S[:r]).astype(np.float32)
    B = Vt[:r].astype(np.float32)
    Aq, Bq = to_bf16(A.copy()), to_bf16(B.copy())          # store factors in bf16
    L = (bf16_to_f32(Aq).reshape(m, r) @ bf16_to_f32(Bq).reshape(r, n)).astype(np.float32)
    refp = to_bf16((base_f + L).ravel().copy())            # ref' = round_bf16(base + L), flat
    # arith residual between ft and ref'
    d = wc._mono16(ft_u16).astype(np.int64) - wc._mono16(refp).astype(np.int64)
    zz = ((d << 1) ^ (d >> 63)).astype(np.uint32)
    perplane = wc._codecs(19)[1]
    res_blob = perplane.compress(zz.astype("<u4").tobytes(), "fp32")
    fac_blob = ZSTD.compress(Aq.tobytes(), 19) + b"||" + ZSTD.compress(Bq.tobytes(), 19)
    size = len(res_blob) + len(fac_blob) + 8
    # round-trip
    d2 = (zz >> 1).astype(np.int64) ^ -(zz & 1).astype(np.int64)
    k = (wc._mono16(refp).astype(np.int64) + d2).astype(np.uint16)
    rec = wc._inv_mono16(k)
    ok = rec.tobytes() == ft_u16.tobytes()
    return size, ok, r


def main():
    braw = open(BASE, "rb").read()
    araw = open(ABL, "rb").read()
    _, _, hb, ob = wc.parse(BASE)
    _, _, ha, oa = wc.parse(ABL)
    bt = {n: (dt, b, e) for n, dt, b, e in wc._tensors_in_order(hb)}
    ct, cp = wc._codecs(19), wc._codecs(3)

    total_raw = best_total = lr_total = 0
    n_lr_used = lr_ok_all = 0
    n_lr = 0
    for name, dt, b, e in wc._tensors_in_order(ha):
        fbuf = araw[oa + b:oa + e]
        total_raw += len(fbuf)
        if name not in bt or bt[name][0] != dt:
            best_total += len(wc._enc_tensor(fbuf, dt, None, ct, cp, 19, 3)[1])
            lr_total += len(wc._enc_tensor(fbuf, dt, None, ct, cp, 19, 3)[1])
            continue
        rbuf = braw[ob + bt[name][1]:ob + bt[name][2]]
        m, blob = wc._enc_tensor(fbuf, dt, rbuf, ct, cp, 19, 3)
        best_total += len(blob)
        best_sz = len(blob)
        shape = ha[name]["shape"]
        lr_sz = best_sz
        if dt == "BF16" and len(shape) == 2 and min(shape) >= 16 and rbuf != fbuf:
            n_lr += 1
            ft_u16 = np.frombuffer(fbuf, "<u2")
            base_u16 = np.frombuffer(rbuf, "<u2")
            sz, ok, r = lowrank_blob(ft_u16, base_u16, shape)
            lr_ok_all += int(ok)
            if sz < best_sz:
                lr_sz = sz
                n_lr_used += 1
        lr_total += lr_sz

    print("LOW-RANK residual potential on Qwen abliteration (rank<=4 delta)")
    print(f"  raw:               {total_raw/1e6:>8.1f} MB")
    print(f"  best-of (current): {best_total/1e6:>8.1f} MB  save {(1-best_total/total_raw)*100:5.1f}%")
    print(f"  + low-rank mode:   {lr_total/1e6:>8.1f} MB  save {(1-lr_total/total_raw)*100:5.1f}%")
    print(f"  low-rank used on {n_lr_used}/{n_lr} changed 2D tensors; round-trip ok {lr_ok_all}/{n_lr}")


if __name__ == "__main__":
    main()
