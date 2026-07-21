"""Proof-of-concept: reference/delta compression of related models.

The weights are random (mantissa wall), but the *change* between a base model and a
fine-tune / next checkpoint is NOT. We XOR the bit-patterns (exactly reversible given
the reference) and compress that. For a small or sparse change, the XOR is mostly
zeros / low-bit differences -> compresses far better than the model itself.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from weights import codecs as cd  # noqa: E402
from weights import evaluate as ev  # noqa: E402


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f32: np.ndarray) -> np.ndarray:
    u32 = f32.view(np.uint32)
    return ((u32.astype(np.uint64) + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)


def main():
    items = ev.load_manifest()
    base_items = sorted([it for it in items if it["dtype"] == "bf16" and "real" in it["kind"]],
                        key=lambda it: it["nbytes"], reverse=True)
    if not base_items:
        base_items = sorted([it for it in items if it["dtype"] == "bf16"],
                            key=lambda it: it["nbytes"], reverse=True)
    it = base_items[0]
    base = np.frombuffer(ev._read(it), dtype="<u2").copy()
    n = base.size
    bf = bf16_to_f32(base)
    std = float(np.std(bf)) or 1.0
    codec = cd.SplitCodec("brotli", 11)  # strongest baseline coder

    def csize(u16: np.ndarray) -> int:
        data = u16.tobytes()
        blob = codec.compress(data, "bf16")
        assert codec.decompress(blob) == data, "round-trip failed"
        return len(blob)

    raw = n * 2
    base_c = csize(base)
    print(f"base tensor ({it['source']}): {raw:,} bytes, {n:,} bf16 params")
    print(f"standalone compress(base): {base_c:,} bytes  (save {(1-base_c/raw)*100:.1f}%)\n")
    print(f"{'scenario':<26}{'compress(ft)':>14}{'compress(delta)':>16}{'delta/ft':>10}{'changed':>9}")
    print("-" * 75)
    rs = np.random.RandomState(0)
    for frac, eps in [(1.0, 0.01), (1.0, 0.05), (1.0, 0.2), (0.10, 0.3), (0.01, 0.5)]:
        ft = bf.copy()
        mask = rs.rand(n) < frac
        ft[mask] = ft[mask] + rs.randn(int(mask.sum())).astype(np.float32) * std * eps
        ftq = f32_to_bf16(ft)
        delta = ftq ^ base  # exact, reversible given base
        s_ft = csize(ftq)
        s_dl = csize(delta)
        changed = float((delta != 0).mean()) * 100
        scen = f"full,eps={eps}" if frac == 1.0 else f"sparse {frac*100:.0f}%,eps={eps}"
        print(f"{scen:<26}{s_ft:>14,}{s_dl:>16,}{s_dl/s_ft*100:>9.0f}%{changed:>8.0f}%")
    print("\nNote: compress(delta) is what you store per checkpoint/fine-tune; the base "
          "is stored once. For real fine-tunes/LoRAs/checkpoints the change is small.")


if __name__ == "__main__":
    main()
