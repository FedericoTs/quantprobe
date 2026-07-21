"""Arithmetic (float-ordered) delta vs XOR delta on the real bf16 fine-tune.

XOR over-counts small magnitude moves: a 1-ULP change that crosses an exponent
boundary flips many bits. If we map bf16 to a monotonic integer key (so adjacent
representable values have adjacent keys), then (key_ft - key_base) equals the number
of ULP steps the weight moved -- small for gentle fine-tuning -- and zig-zag + byte
split compresses far better. Exactly reversible.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from weights import codecs as cd  # noqa: E402
from weights import wcodec as wc  # noqa: E402

BASE = os.path.join(_ROOT, "weights", "data", "smollm", "base.safetensors")
INST = os.path.join(_ROOT, "weights", "data", "smollm", "instruct.safetensors")


def to_bf16(f32_bytes):
    u32 = np.frombuffer(f32_bytes, "<u4")
    return (((u32.astype(np.uint64) + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16))


def mono(u16):
    # total-order key: positives set top bit; negatives flip all bits
    u16 = u16.astype(np.uint16)
    sign = (u16 >> 15) & 1
    return np.where(sign == 1, (~u16) & 0xFFFF, u16 | 0x8000).astype(np.uint16)


def zigzag16(d):  # int32 array -> uint16 (assumes |d| fits; we clip-check)
    return ((d << 1) ^ (d >> 31)).astype(np.uint32)


def main():
    braw = open(BASE, "rb").read()
    iraw = open(INST, "rb").read()
    _, _, hb, ob = wc.parse(BASE)
    _, _, hi, oi = wc.parse(INST)
    bt = {n: (dt, b, e) for n, dt, b, e in wc._tensors_in_order(hb)}
    perplane = cd.SplitPerPlaneCodec(["zstd"], 19, "pp")
    zstd = wc._ZSTD

    tot_raw = tot_xor = tot_arith = 0
    absd = []
    big = 0
    ok = True
    for name, dt, b, e in wc._tensors_in_order(hi):
        ibuf = iraw[oi + b:oi + e]
        bdt, bb, be = bt[name]
        ref = to_bf16(braw[ob + bb:ob + be])
        u = np.frombuffer(ibuf, "<u2")
        if ref.size != u.size:
            continue
        tot_raw += len(ibuf)
        # XOR per-plane
        x = (u ^ ref).astype("<u2")
        tot_xor += len(perplane.compress(x.tobytes(), "bf16"))
        # arithmetic delta in monotonic key space
        ku = mono(u).astype(np.int32)
        kr = mono(ref).astype(np.int32)
        d = ku - kr
        zz = zigzag16(d)                      # uint32, usually small
        if zz.max() > 0xFFFF:
            big += int((zz > 0xFFFF).sum())
        lo = (zz & 0xFF).astype(np.uint8)     # low byte (most entropy)
        hi_b = ((zz >> 8) & 0xFF).astype(np.uint8)  # high byte (mostly 0)
        ov = (zz >> 16).astype(np.uint16)     # overflow (rare); store sparse
        ovmask = zz > 0xFFFF
        blob = (zstd.compress(lo.tobytes(), 19) + b"||" + zstd.compress(hi_b.tobytes(), 19)
                + b"||" + zstd.compress(np.packbits(ovmask).tobytes(), 19)
                + zstd.compress(ov[ovmask].tobytes(), 19))
        tot_arith += len(blob)
        # round-trip: reconstruct u from ref + zz
        d2 = (zz >> 1).astype(np.int64) ^ -(zz & 1).astype(np.int64)
        k2 = (kr.astype(np.int64) + d2).astype(np.uint16)
        u2 = inv_mono(k2)
        if u2.tobytes() != ibuf:
            ok = False
        absd.append(np.abs(d).astype(np.int64))

    absd = np.concatenate(absd)
    s_xor = (1 - tot_xor / tot_raw) * 100
    s_arith = (1 - tot_arith / tot_raw) * 100
    print("ARITHMETIC vs XOR delta -- real bf16 fine-tune (SmolLM-135M)")
    print(f"  |ULP move| distribution: median={np.median(absd):.0f}  "
          f"mean={absd.mean():.1f}  p90={np.percentile(absd,90):.0f}  max={absd.max():.0f}")
    for thr in (0, 1, 2, 4, 16):
        print(f"    <= {thr:>3} ULP: {(absd<=thr).mean()*100:5.1f}%")
    print(f"  raw:            {tot_raw/1e6:>8.1f} MB")
    print(f"  XOR per-plane:  {tot_xor/1e6:>8.1f} MB   save {s_xor:5.1f}%")
    print(f"  ARITH zigzag:   {tot_arith/1e6:>8.1f} MB   save {s_arith:5.1f}%   "
          f"({'WIN +%.1f' % (s_arith-s_xor) if s_arith>s_xor else 'lose'})")
    print(f"  round-trip: {'OK (byte-exact)' if ok else 'FAIL'}   overflow elems: {big}")


def inv_mono(k16):
    k16 = k16.astype(np.uint16)
    top = (k16 >> 15) & 1            # 1 => was positive
    return np.where(top == 1, k16 & 0x7FFF, (~k16) & 0xFFFF).astype(np.uint16)


if __name__ == "__main__":
    main()
