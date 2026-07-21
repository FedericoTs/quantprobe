"""Can the BASE model be compressed past ~33%? Measure its true lossless entropy floor.

bf16 = 1 sign + 8 exponent + 7 mantissa bits. ZipNN/DFloat11/us reach ~33% by coding the
skewed exponent; the claim is the mantissa is ~uniform (incompressible). We MEASURE it: the
per-bit entropy, the order-0 value entropy (the hard floor for any memoryless coder), and the
order-1 conditional entropy (is there spatial structure a stronger coder could exploit?).
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from weights import wcodec as wc  # noqa: E402

BASE = os.path.join(_ROOT, "weights", "data", "qwen", "base.safetensors")


def entropy_bits(counts):
    p = counts[counts > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    raw, _, h, off = wc.parse(BASE)
    # concatenate all bf16 weights into one u16 stream (sample if huge)
    parts = []
    for name, dt, b, e in wc._tensors_in_order(h):
        if dt == "BF16":
            parts.append(np.frombuffer(raw[off + b:off + e], "<u2"))
    u = np.concatenate(parts)
    if u.size > 60_000_000:
        u = u[:60_000_000]
    n = u.size
    print(f"Qwen2.5-0.5B base: {n:,} bf16 weights ({n*2/1e6:.0f} MB sampled)\n")

    # per-bit fraction of 1s and entropy (bit 15=sign, 14..7=exp, 6..0=mantissa)
    bits = np.unpackbits(u.view(np.uint8).reshape(-1, 2)[:, ::-1], axis=1)  # MSB..LSB
    print("per-bit P(1) and entropy (bit 15=sign, 14-7=exp, 6-0=mantissa):")
    labels = ["S"] + [f"e{i}" for i in range(8)] + [f"m{i}" for i in range(7)]
    h_bits = 0.0
    for i in range(16):
        p1 = float(bits[:, i].mean())
        hb = entropy_bits(np.array([1 - p1, p1]) * n) if 0 < p1 < 1 else 0.0
        h_bits += hb
        print(f"  bit{15-i:>2} {labels[i]:<3} P(1)={p1:5.3f}  H={hb:4.2f}")

    # order-0 value entropy (hard floor for a memoryless coder) and order-1
    h0 = entropy_bits(np.bincount(u, minlength=65536))
    # order-1: H(u_t | u_{t-1}) approximated via byte0 conditional (cheap proxy for spatial structure)
    lo = u.astype(np.uint8)  # low byte (mantissa-heavy)
    pair = (lo[:-1].astype(np.int32) << 8) | lo[1:]
    h1_lo = entropy_bits(np.bincount(pair, minlength=65536)) - entropy_bits(np.bincount(lo[:-1], minlength=256))

    print(f"\n  sum of independent per-bit entropy:   {h_bits:5.2f} / 16 bits")
    print(f"  order-0 value entropy H(u):            {h0:5.2f} / 16 bits  -> floor = save {(1-h0/16)*100:4.1f}%")
    print(f"  low-byte order-1 H(b_t|b_t-1):         {h1_lo:5.2f} / 8 bits (vs 8 = no spatial structure)")
    print(f"\n  our smart codec achieves ~33% (= save). Floor says max lossless save ~ {(1-h0/16)*100:.0f}%.")
    print("  If H(u)/16 ~ 0.67 and low-byte order-1 ~ 8.0, the mantissa is genuinely random:")
    print("  the base is at the information wall; pushing further needs lossy or amortization.")


if __name__ == "__main__":
    main()
