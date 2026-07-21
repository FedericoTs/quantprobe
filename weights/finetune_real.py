"""REAL base->fine-tune delta validation (SmolLM-135M base vs -Instruct).

Base ships fp32, the Instruct fine-tune ships bf16. The realistic deployment keeps a
bf16 copy of the base as the shared reference and stores the bf16 fine-tune as a delta
vs it. We convert base fp32 -> bf16 (round-to-nearest-even), then for each tensor pick
the best mode (sparse bitmap if <50% of bf16 elements changed, else per-plane XOR),
exactly like wcodec. Reports standalone vs delta, round-trip gated.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from weights import codecs as cd  # noqa: E402
from weights import track  # noqa: E402
from weights import wcodec as wc  # noqa: E402

BASE = os.path.join(_ROOT, "weights", "data", "smollm", "base.safetensors")
INST = os.path.join(_ROOT, "weights", "data", "smollm", "instruct.safetensors")


def to_bf16(f32_bytes):
    u32 = np.frombuffer(f32_bytes, "<u4")
    return (((u32.astype(np.uint64) + 0x7FFF + ((u32 >> 16) & 1)) >> 16)
            .astype(np.uint16))


def main():
    braw = open(BASE, "rb").read()
    iraw = open(INST, "rb").read()
    _, _, hb, ob = wc.parse(BASE)
    _, _, hi, oi = wc.parse(INST)
    bt = {n: (dt, b, e) for n, dt, b, e in wc._tensors_in_order(hb)}

    smart = cd.SplitSmartCodec("zstd", 19)
    perplane = cd.SplitPerPlaneCodec(["zstd"], 19, "pp")

    tot_raw = tot_std = tot_delta = 0
    tot_changed = tot_elems = 0
    n_sparse = n_dense = n_identical = 0
    ok = True
    for name, dt, b, e in wc._tensors_in_order(hi):
        ibuf = iraw[oi + b:oi + e]                 # instruct bf16 bytes
        bdt, bb, be = bt[name]
        ref = to_bf16(braw[ob + bb:ob + be])       # base fp32 -> bf16 (reference)
        u = np.frombuffer(ibuf, "<u2")
        if ref.size != u.size:
            continue
        changed = u != ref
        nch = int(changed.sum())
        frac = nch / max(u.size, 1)
        tot_raw += len(ibuf)
        tot_elems += u.size
        tot_changed += nch
        tot_std += len(smart.compress(ibuf, "bf16"))

        if nch == 0:
            n_identical += 1
            blob = b""  # M_COPY
            dsz = 0
            rec = ref.tobytes()
        elif frac < 0.5:
            n_sparse += 1
            cbits = wc._ZSTD.compress(np.packbits(changed).tobytes(), 19)
            vals = u[changed].tobytes()
            dsz = len(cbits) + len(vals) + 8
            # round-trip
            ch = np.unpackbits(np.frombuffer(wc._ZSTD.decompress(cbits), np.uint8))[:u.size].astype(bool)
            r2 = ref.copy(); r2[ch] = np.frombuffer(vals, "<u2")
            rec = r2.tobytes()
        else:
            n_dense += 1
            x = (u ^ ref).astype("<u2")
            blob = perplane.compress(x.tobytes(), "bf16")
            dsz = len(blob)
            xr = np.frombuffer(perplane.decompress(blob), "<u2")
            rec = (xr ^ ref).tobytes()
        if rec != ibuf:
            ok = False
        tot_delta += dsz

    s_std = (1 - tot_std / tot_raw) * 100
    s_delta = (1 - tot_delta / tot_raw) * 100
    print("REAL base->fine-tune delta: SmolLM-135M (fp32 base, bf16 instruct)")
    print(f"  tensors: {n_identical} identical, {n_sparse} sparse(<50%), {n_dense} dense")
    print(f"  elements changed at bf16: {tot_changed/max(tot_elems,1)*100:.1f}%")
    print(f"  raw instruct:   {tot_raw/1e6:>8.1f} MB")
    print(f"  standalone:     {tot_std/1e6:>8.1f} MB   save {s_std:5.1f}%  (bf16 floor)")
    print(f"  DELTA vs base:  {tot_delta/1e6:>8.1f} MB   save {s_delta:5.1f}%   "
          f"({tot_delta/tot_std*100:.0f}% of standalone)")
    print(f"  round-trip: {'OK (byte-exact)' if ok else 'FAIL'}")

    track.record({
        "codec": "finetune-delta@SmolLM-135M",
        "config": {"type": "finetune-delta", "base": "SmolLM-135M",
                   "ft": "SmolLM-135M-Instruct", "ref_dtype": "bf16"},
        "overall": {"in_bytes": tot_raw, "out_bytes": tot_delta,
                    "ratio": round(tot_raw / tot_delta, 4), "save_pct": round(s_delta, 2),
                    "enc_MBps": 0.0, "dec_MBps": 0.0, "rt_ok": ok},
        "by_dtype": {"bf16": {"in_bytes": tot_raw, "out_bytes": tot_delta,
                              "ratio": round(tot_raw / tot_delta, 4),
                              "save_pct": round(s_delta, 2), "dec_MBps": 0.0, "rt_ok": ok}},
        "n_tensors": n_identical + n_sparse + n_dense,
    }, note=f"REAL fine-tune delta; standalone={s_std:.1f}%; {tot_changed/max(tot_elems,1)*100:.0f}% elems changed")


if __name__ == "__main__":
    main()
