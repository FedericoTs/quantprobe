"""Quick exploration: lossless compression of LLM weight tensors (bf16).

Establishes where we stand vs the ZipNN approach (byte-split exponent/mantissa, then
a backend). Uses the existing transforms + backends. Synthetic but realistic bf16
weights (per-tensor Gaussian scales + outliers, round-to-nearest-even bf16)."""

from __future__ import annotations

import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evocompress import transforms  # noqa: E402
from evocompress.pipeline import Pipeline  # noqa: E402


def make_bf16_weights(total=4_000_000, n_tensors=12, seed=0) -> bytes:
    rs = np.random.RandomState(seed)
    chunks = []
    per = total // 2 // n_tensors  # total/2 elements (2 bytes each)
    for _ in range(n_tensors):
        std = float(10 ** rs.uniform(-2.2, -0.6))  # per-tensor magnitude scale
        w = rs.standard_normal(per).astype(np.float32) * std
        nout = per // 200
        idx = rs.randint(0, per, nout)
        w[idx] *= rs.uniform(4, 15, nout).astype(np.float32)  # heavy-tail outliers
        chunks.append(w)
    fp32 = np.concatenate(chunks)
    u32 = fp32.view(np.uint32)
    # round-to-nearest-even truncation to bf16 (top 16 bits)
    bf16 = ((u32.astype(np.uint64) + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)
    return bf16.tobytes()


def main():
    data = make_bf16_weights()
    n = len(data)
    print(f"synthetic bf16 weights: {n:,} bytes ({n // 2:,} params)\n")
    print(f"{'method':<24}{'bytes':>12}{'ratio':>8}{'save%':>8}{'enc MB/s':>10}  rt")
    print("-" * 74)

    def bench(name, pipe):
        t0 = time.perf_counter()
        blob = pipe.encode(data)
        t1 = time.perf_counter()
        out = Pipeline.decode_blob(blob)
        ok = out == data
        print(f"{name:<24}{len(blob):>12,}{n/len(blob):>8.3f}{(1-len(blob)/n)*100:>8.1f}"
              f"{n/1e6/(t1-t0):>10.0f}  {'ok' if ok else 'FAIL'}")

    split = transforms.build("transpose", {"stride": 2})
    bench("raw gzip-9", Pipeline([], "gzip", 9))
    bench("raw zstd-19", Pipeline([], "zstd", 19))
    bench("raw lzma-9", Pipeline([], "lzma", 9))
    print("  -- byte-split (ZipNN-style: exponent/mantissa planes) --")
    bench("split+zstd-19", Pipeline([split], "zstd", 19))
    bench("split+lzma-9", Pipeline([split], "lzma", 9))
    bench("split+brotli-11", Pipeline([split], "brotli", 11))
    bench("split+bz2-9", Pipeline([split], "bz2", 9))

    # plane-level analysis: how compressible is each byte plane alone?
    arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, 2)
    lo = arr[:, 0].tobytes()  # mantissa-dominated
    hi = arr[:, 1].tobytes()  # sign + top-7 exponent
    import lzma
    print("\n  -- per-plane (lzma-9) --")
    for name, plane in [("low byte (mantissa)", lo), ("high byte (sign+exp)", hi)]:
        c = lzma.compress(plane, preset=9)
        print(f"  {name:<22} {len(plane):>10,} -> {len(c):>10,}  save={ (1-len(c)/len(plane))*100:5.1f}%")


if __name__ == "__main__":
    main()
