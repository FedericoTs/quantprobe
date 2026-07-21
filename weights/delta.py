"""Reference/delta codec + benchmark -- the breakthrough direction.

A DeltaCodec stores a model as the XOR vs a reference (base model / previous
checkpoint), which is exactly reversible given the reference and far lower entropy
for related models. We evaluate on realistic fine-tune scenarios derived from real
bf16 bases, and record results to the same tracker.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from weights import codecs as cd  # noqa: E402
from weights import evaluate as ev  # noqa: E402
from weights import track  # noqa: E402

UVIEW = {"bf16": "<u2", "fp16": "<u2", "fp32": "<u4", "fp64": "<u8"}


class DeltaCodec:
    """Wrap an inner WeightCodec to compress the XOR-delta vs a reference."""

    def __init__(self, inner: cd.WeightCodec, name: str):
        self.inner = inner
        self.name = name

    def compress(self, data: bytes, dtype: str, ref: bytes) -> bytes:
        u = np.frombuffer(data, UVIEW[dtype]).copy()
        r = np.frombuffer(ref, UVIEW[dtype])
        m = min(u.size, r.size)
        u[:m] ^= r[:m]
        return self.inner.compress(u.tobytes(), dtype)

    def decompress(self, blob: bytes, dtype: str, ref: bytes) -> bytes:
        u = np.frombuffer(self.inner.decompress(blob), UVIEW[dtype]).copy()
        r = np.frombuffer(ref, UVIEW[dtype])
        m = min(u.size, r.size)
        u[:m] ^= r[:m]
        return u.tobytes()


def _bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _f32_to_bf16(f32):
    u32 = f32.view(np.uint32)
    return ((u32.astype(np.uint64) + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)


def make_variant(base_u16, std, frac, eps, seed):
    rs = np.random.RandomState(seed)
    f = _bf16_to_f32(base_u16).copy()
    mask = rs.rand(f.size) < frac
    f[mask] = f[mask] + rs.randn(int(mask.sum())).astype(np.float32) * std * eps
    return _f32_to_bf16(f)


def main():
    items = ev.load_manifest()
    bases = sorted([it for it in items if it["dtype"] == "bf16" and "real" in it["kind"]],
                   key=lambda it: it["nbytes"], reverse=True)
    if not bases:
        bases = sorted([it for it in items if it["dtype"] == "bf16"],
                       key=lambda it: it["nbytes"], reverse=True)
    # use up to 3 largest real bf16 tensors as bases
    bases = bases[:3]

    scenarios = [
        ("full eps=1%", 1.0, 0.01),
        ("full eps=5%", 1.0, 0.05),
        ("sparse 10%", 0.10, 0.3),
        ("sparse 1%", 0.01, 0.5),
    ]
    inner = cd.SplitCodec("brotli", 11)  # good general coder for the (sparse) delta
    inner_fast = cd.SplitCodec("zstd", 19)
    delta_codecs = [
        ("delta-brotli11", DeltaCodec(inner, "delta-brotli11")),
        ("delta-zstd19", DeltaCodec(inner_fast, "delta-zstd19")),
    ]
    standalone = cd.SplitSmartCodec("zstd", 19, "smart-zstd19")  # best single-tensor

    print(f"{'scenario':<16}{'codec':<16}{'variant raw':>12}{'standalone':>12}"
          f"{'delta':>12}{'save%':>8}{'vs standalone':>14}  rt")
    print("-" * 96)
    for scen, frac, eps in scenarios:
        agg_raw = agg_std = {}
        # accumulate over bases
        for cname, dc in delta_codecs:
            t_raw = t_std = t_delta = 0
            ok_all = True
            dec_t = 0.0
            for it in bases:
                base = np.frombuffer(ev._read(it), dtype="<u2").copy()
                std = float(np.std(_bf16_to_f32(base))) or 1.0
                var = make_variant(base, std, frac, eps, seed=hash((it["id"], scen)) & 0xFFFF)
                var_bytes = var.tobytes()
                base_bytes = base.tobytes()
                blob = dc.compress(var_bytes, "bf16", base_bytes)
                t0 = time.perf_counter()
                rec = dc.decompress(blob, "bf16", base_bytes)
                dec_t += time.perf_counter() - t0
                ok = rec == var_bytes
                ok_all = ok_all and ok
                std_blob = standalone.compress(var_bytes, "bf16")
                t_raw += len(var_bytes)
                t_std += len(std_blob)
                t_delta += len(blob)
            save = (1 - t_delta / t_raw) * 100
            vs = t_delta / t_std * 100
            dec_mbps = t_raw / 1e6 / dec_t if dec_t > 0 else 0
            print(f"{scen:<16}{cname:<16}{t_raw:>12,}{t_std:>12,}{t_delta:>12,}"
                  f"{save:>7.1f}%{vs:>13.0f}%  {'ok' if ok_all else 'FAIL'}")
            # record to tracker
            track.record({
                "codec": f"{cname}@{scen}",
                "config": {"type": "delta", "inner": dc.inner.config(), "scenario": scen},
                "overall": {"in_bytes": t_raw, "out_bytes": t_delta,
                            "ratio": round(t_raw / t_delta, 4), "save_pct": round(save, 2),
                            "enc_MBps": 0.0, "dec_MBps": round(dec_mbps, 1), "rt_ok": ok_all},
                "by_dtype": {"bf16": {"in_bytes": t_raw, "out_bytes": t_delta,
                                      "ratio": round(t_raw / t_delta, 4), "save_pct": round(save, 2),
                                      "dec_MBps": round(dec_mbps, 1), "rt_ok": ok_all}},
                "n_tensors": len(bases),
            }, note=f"delta vs standalone {vs:.0f}%")


if __name__ == "__main__":
    main()
