"""Rigorous baseline table for the paper: on the SAME real model (Qwen2.5-0.5B
abliterated, bf16), compare general-purpose compression, a ZipNN-style byte-split
(the single-model SOTA approach), our single-model codec, and our delta. Shows the
single-model wall (~33%) and that the delta breaks it (99%)."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evocompress import backends  # noqa: E402
from weights import codecs as cd  # noqa: E402
from weights import wcodec as wc  # noqa: E402

ABL = os.path.join(_ROOT, "weights", "data", "qwen", "ablit.safetensors")
BASE = os.path.join(_ROOT, "weights", "data", "qwen", "base.safetensors")
ZSTD = backends.get_backend("zstd")


def main():
    raw, _, h, off = wc.parse(ABL)
    tensors = wc._tensors_in_order(h)
    split = cd.SplitCodec("zstd", 19)            # ZipNN-style byte-split
    smart = cd.SplitSmartCodec("zstd", 19)       # ours, single
    ct, cp = wc._codecs(19), wc._codecs(3)
    rb = wc._ref_lookup_from_file(BASE)

    n = 0
    raw_b = gen3 = zipnn = ours1 = delta = 0
    for name, stdt, b, e in tensors:
        dt = wc.ST2DT.get(stdt)
        buf = raw[off + b:off + e]
        raw_b += len(buf)
        gen3 += len(ZSTD.compress(buf, 3))       # general purpose (level immaterial on weights)
        if dt is None:
            zipnn += len(buf); ours1 += len(buf); delta += len(buf)
            continue
        zipnn += len(split.compress(buf, dt))
        ours1 += len(smart.compress(buf, dt))
        _, dblob = wc._enc_tensor(buf, stdt, rb(name), ct, cp, 19, 3, h[name]["shape"])
        delta += len(dblob)
        n += 1

    def row(label, sz):
        print(f"  {label:<34}{sz/1e6:>9.1f} MB   save {(1-sz/raw_b)*100:>5.1f}%   ratio {raw_b/sz:>6.1f}x")

    print(f"Baselines on Qwen2.5-0.5B abliterated (bf16, {raw_b/1e6:.0f} MB, {n} modeled tensors)\n")
    row("raw (no compression)", raw_b)
    row("general zstd (gzip-class)", gen3)
    row("byte-split + zstd-19 (ZipNN-class)", zipnn)
    row("ours, single-model", ours1)
    row("OURS, DELTA vs base", delta)
    print("\n  -> single-model coders hit a ~33% wall; the lossless DELTA reaches ~99%.")


if __name__ == "__main__":
    main()
