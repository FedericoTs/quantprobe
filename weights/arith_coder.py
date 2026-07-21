"""How much headroom is left above plain-zstd on the arithmetic delta?

The zig-zag ULP-move values follow a peaked, heavy-tailed distribution (median ~12).
zstd treats them as bytes. A magnitude-class split -- store bit_length(zz) (the
"class", tiny entropy) separately from the (class-1) low "remainder" bits (near
uniform) -- is what a range/Golomb coder effectively does. We measure the achievable
size of that split vs zstd-per-plane and brotli, to decide whether a custom coder is
worth building into wcodec.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evocompress import backends  # noqa: E402
from weights import codecs as cd  # noqa: E402
from weights import wcodec as wc  # noqa: E402

BB = os.path.join(_ROOT, "weights", "data", "smollm", "base_bf16.safetensors")
INST = os.path.join(_ROOT, "weights", "data", "smollm", "instruct.safetensors")


def main():
    braw = open(BB, "rb").read()
    iraw = open(INST, "rb").read()
    _, _, hb, ob = wc.parse(BB)
    _, _, hi, oi = wc.parse(INST)
    bt = {n: (dt, b, e) for n, dt, b, e in wc._tensors_in_order(hb)}

    zzs = []
    for name, dt, b, e in wc._tensors_in_order(hi):
        if name not in bt:
            continue
        u = np.frombuffer(iraw[oi + b:oi + e], "<u2")
        r = np.frombuffer(braw[ob + bt[name][1]:ob + bt[name][2]], "<u2")
        if u.size != r.size:
            continue
        d = wc._mono16(u).astype(np.int64) - wc._mono16(r).astype(np.int64)
        zz = ((d << 1) ^ (d >> 63)).astype(np.uint32)
        zzs.append(zz)
    zz = np.concatenate(zzs)
    if zz.size > 30_000_000:
        zz = zz[:30_000_000]
    n = zz.size
    raw = n * 2  # bf16 raw bytes
    print(f"arith zig-zag values: {n:,}  (raw bf16 {raw/1e6:.1f} MB)")

    zstd = backends.get_backend("zstd")
    brotli = backends.get_backend("brotli")

    def sz(x, lvl=19, be=zstd):
        return len(be.compress(x, lvl))

    # baseline: per-plane zstd of zz as u32
    planes = zz.astype("<u4").view(np.uint8).reshape(-1, 4)
    pp = sum(sz(planes[:, p].tobytes()) for p in range(4))
    print(f"  zstd per-plane (current):   {pp/1e6:>7.2f} MB  save {(1-pp/raw)*100:5.1f}%")

    # brotli per-plane (lower planes only; high planes ~0)
    bp = sum(sz(planes[:, p].tobytes(), 11, brotli) for p in range(2))
    bp += sum(sz(planes[:, p].tobytes()) for p in range(2, 4))
    print(f"  brotli lo + zstd hi:        {bp/1e6:>7.2f} MB  save {(1-bp/raw)*100:5.1f}%")

    # magnitude-class split
    cls = np.zeros(n, dtype=np.uint8)
    nz = zz > 0
    cls[nz] = (np.floor(np.log2(zz[nz].astype(np.float64))).astype(np.uint8) + 1)
    rem_bits = np.where(cls > 0, cls - 1, 0).astype(np.int64)
    total_rem_bits = int(rem_bits.sum())
    cls_comp = sz(cls.tobytes())
    rem_bytes = (total_rem_bits + 7) // 8
    total = cls_comp + rem_bytes
    print(f"  class dist: mean_bitlen={cls.mean():.2f}  "
          f"remainder={rem_bytes/1e6:.2f}MB  class_comp={cls_comp/1e6:.2f}MB")
    print(f"  magnitude-class split:      {total/1e6:>7.2f} MB  save {(1-total/raw)*100:5.1f}%")

    # order-0 entropy floor of the split (class entropy + remainder bits)
    p = np.bincount(cls, minlength=256).astype(np.float64)
    p = p[p > 0] / n
    h_cls = float(-(p * np.log2(p)).sum())
    floor = (h_cls * n / 8) + rem_bytes
    print(f"  -> order-0 floor of split:  {floor/1e6:>7.2f} MB  save {(1-floor/raw)*100:5.1f}%  "
          f"(H(class)={h_cls:.2f} bits)")


if __name__ == "__main__":
    main()
