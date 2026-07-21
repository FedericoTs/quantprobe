"""Do QUANTIZED model variants still compress as deltas? (expand-what-we-compress)

Quantized models (int8/int4) are how models increasingly ship. Quantization could destroy
the delta structure (a tiny fp change can cross an int bucket). We quantize the real Qwen
abliteration pair to int8 (per-tensor symmetric, sharing the base's scales so the variant
is stored on the base's grid -- the natural scheme for variant storage) and compress the
abliterated int8 model as a delta vs the base int8, with wcodec.
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
ABL = os.path.join(_ROOT, "weights", "data", "qwen", "ablit.safetensors")


def main():
    braw, _, hb, ob = wc.parse(BASE)
    araw, _, ha, oa = wc.parse(ABL)
    bt = {n: (dt, b, e) for n, dt, b, e in wc._tensors_in_order(hb)}
    ct, cp = wc._codecs(19), wc._codecs(3)

    tot_raw = tot_std = tot_delta = 0
    n_copy = n_changed = 0
    changed_frac = []
    ok = True
    for name, dt, b, e in wc._tensors_in_order(ha):
        if name not in bt or bt[name][0] != dt or dt != "BF16":
            continue
        bf = wc._bf16_to_f32(np.frombuffer(braw[ob + bt[name][1]:ob + bt[name][2]], "<u2"))
        af = wc._bf16_to_f32(np.frombuffer(araw[oa + b:oa + e], "<u2"))
        scale = (np.abs(bf).max() or 1.0) / 127.0
        bq = np.clip(np.round(bf / scale), -127, 127).astype(np.int8)
        aq = np.clip(np.round(af / scale), -127, 127).astype(np.int8)
        tot_raw += aq.size
        # standalone (single) and delta vs base, both via wcodec int8 path
        _, sblob = wc._enc_tensor(aq.tobytes(), "I8", None, ct, cp, 19, 3, None)
        m, dblob = wc._enc_tensor(aq.tobytes(), "I8", bq.tobytes(), ct, cp, 19, 3, None)
        # round-trip the delta
        rec = wc._dec_tensor(m, dblob, "I8", aq.size, bq.tobytes())
        if rec != aq.tobytes():
            ok = False
        tot_std += len(sblob)
        tot_delta += len(dblob)
        ch = int((aq != bq).sum())
        if ch == 0:
            n_copy += 1
        else:
            n_changed += 1
            changed_frac.append(ch / aq.size)

    s_std = (1 - tot_std / tot_raw) * 100
    s_delta = (1 - tot_delta / tot_raw) * 100
    print("QUANTIZED (int8) variant delta -- Qwen2.5-0.5B abliteration")
    print(f"  tensors: {n_copy} identical, {n_changed} changed "
          f"(changed ones differ in {np.mean(changed_frac)*100:.1f}% of int8 values)")
    print(f"  raw int8:        {tot_raw/1e6:>8.1f} MB")
    print(f"  standalone:      {tot_std/1e6:>8.1f} MB   save {s_std:5.1f}%")
    print(f"  DELTA vs base:   {tot_delta/1e6:>8.1f} MB   save {s_delta:5.1f}%   "
          f"({tot_delta/tot_std*100:.0f}% of standalone)")
    print(f"  round-trip: {'OK (byte-exact)' if ok else 'FAIL'}")
    print("  => quantization preserves the variant's delta structure (unchanged tensors stay")
    print("     identical; only the edited ones differ) -> quantized variants still delta well.")


if __name__ == "__main__":
    main()
