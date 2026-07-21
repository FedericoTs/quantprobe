"""fp8 (E4M3) variant delta -- the newest deployment format. We quantize the real Qwen
abliteration pair to torch float8_e4m3fn (per-tensor scale from the base) and compress the
abliterated fp8 model as a delta vs the base fp8, with wcodec (now fp8-aware)."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from weights import wcodec as wc  # noqa: E402

BASE = os.path.join(_ROOT, "weights", "data", "qwen", "base.safetensors")
ABL = os.path.join(_ROOT, "weights", "data", "qwen", "ablit.safetensors")


def to_fp8_bytes(f32, scale):
    t = torch.from_numpy((f32 / scale).astype(np.float32))
    return t.to(torch.float8_e4m3fn).view(torch.uint8).numpy().tobytes()


def main():
    braw, _, hb, ob = wc.parse(BASE)
    araw, _, ha, oa = wc.parse(ABL)
    bt = {n: (dt, b, e) for n, dt, b, e in wc._tensors_in_order(hb)}
    ct, cp = wc._codecs(19), wc._codecs(3)
    tot_raw = tot_std = tot_delta = 0
    n_copy = n_changed = 0
    ok = True
    for name, dt, b, e in wc._tensors_in_order(ha):
        if name not in bt or bt[name][0] != dt or dt != "BF16":
            continue
        bf = wc._bf16_to_f32(np.frombuffer(braw[ob + bt[name][1]:ob + bt[name][2]], "<u2"))
        af = wc._bf16_to_f32(np.frombuffer(araw[oa + b:oa + e], "<u2"))
        scale = (float(np.abs(bf).max()) or 1.0) / 448.0
        bb = to_fp8_bytes(bf, scale)
        ab = to_fp8_bytes(af, scale)
        tot_raw += len(ab)
        _, sblob = wc._enc_tensor(ab, "F8_E4M3", None, ct, cp, 19, 3, None)
        m, dblob = wc._enc_tensor(ab, "F8_E4M3", bb, ct, cp, 19, 3, None)
        if wc._dec_tensor(m, dblob, "F8_E4M3", len(ab), bb) != ab:
            ok = False
        tot_std += len(sblob)
        tot_delta += len(dblob)
        n_copy += (ab == bb)
        n_changed += (ab != bb)

    print("fp8 (E4M3) variant delta -- Qwen2.5-0.5B abliteration")
    print(f"  tensors: {n_copy} identical, {n_changed} changed")
    print(f"  raw fp8:         {tot_raw/1e6:>8.1f} MB")
    print(f"  standalone:      {tot_std/1e6:>8.1f} MB   save {(1-tot_std/tot_raw)*100:5.1f}%")
    print(f"  DELTA vs base:   {tot_delta/1e6:>8.1f} MB   save {(1-tot_delta/tot_raw)*100:5.1f}%")
    print(f"  round-trip: {'OK (byte-exact)' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
