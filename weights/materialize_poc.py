"""Lossless on-demand variant materialization PoC (the serving frontier).

Claim to stake: keep ONE base resident; materialize any EXACT variant by applying its tiny
compressed delta. We measure (1) the time to reconstruct the exact variant from base + delta,
(2) bit-exactness (integrity-checked), and (3) the multi-tenant memory math: how many exact
variants fit in the memory of a few base models. This is "LoRA serving, but lossless."
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from weights import wcodec as wc  # noqa: E402

BASE = os.path.join(_ROOT, "weights", "data", "qwen", "base.safetensors")
ABL = os.path.join(_ROOT, "weights", "data", "qwen", "ablit.safetensors")


def main():
    model_mb = os.path.getsize(ABL) / 1e6
    t0 = time.perf_counter()
    comp = wc.compress_file(ABL, BASE, level=19)
    enc_t = time.perf_counter() - t0
    delta_mb = len(comp) / 1e6

    tmp = tempfile.mkdtemp(prefix="mat_")
    out = os.path.join(tmp, "variant.safetensors")
    t1 = time.perf_counter()
    # materialize the EXACT variant from base + delta (integrity-checked => bit-exact)
    wc.decompress_file(comp, ref_path=BASE, out_path=out)
    mat_t = time.perf_counter() - t1
    exact = os.path.getsize(out) == os.path.getsize(ABL) and \
        open(out, "rb").read() == open(ABL, "rb").read()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    print("Lossless on-demand variant materialization (Qwen2.5-0.5B abliteration)")
    print(f"  full model:        {model_mb:>8.1f} MB")
    print(f"  delta (stored):    {delta_mb:>8.2f} MB   ({delta_mb/model_mb*100:.1f}% of the model)")
    print(f"  materialize exact variant from base+delta: {mat_t:.1f} s "
          f"({model_mb/mat_t:.0f} MB/s)   bit-exact: {exact}")
    print(f"  (compress/build the delta once: {enc_t:.1f} s)")

    print("\n  Multi-tenant memory math (hold base + N exact variant-deltas):")
    print(f"  {'N variants':>11}{'full copies':>14}{'base + deltas':>16}{'reduction':>11}")
    for N in (10, 100, 1000):
        full = (N + 1) * model_mb
        ours = model_mb + N * delta_mb
        print(f"  {N:>11}{full/1000:>12.1f} GB{ours/1000:>14.2f} GB{full/ours:>10.0f}x")
    print("  => in the memory of ~1.1 base models you can hold a base + ~100 EXACT variants;")
    print("     materialize any of them on demand, byte-exact. Lossy methods (BitDelta) can't")
    print("     give exactness; single-model coders can't exploit the shared base at all.")


if __name__ == "__main__":
    main()
