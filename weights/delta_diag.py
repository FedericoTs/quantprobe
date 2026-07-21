"""Diagnose the structure of a REAL training delta, to design a SOTA delta coder.

Plain zstd on the XOR gets ~68%. The delta has more structure: for small weight
changes, high bits (sign/exponent/high-mantissa) rarely flip while low mantissa bits
do. We measure per-byte-plane entropy and per-bit-position flip rates on the real
Pythia 1000-step delta, then compare candidate delta coders.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evocompress import backends  # noqa: E402
from weights import evaluate as ev  # noqa: E402
from weights.fetch_weights import parse_safetensors  # noqa: E402


def load(rev):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("EleutherAI/pythia-70m", "model.safetensors", revision=rev,
                        local_files_only=True)
    return {n: (dt, sh, raw) for n, dt, sh, raw in parse_safetensors(p)}


def main():
    Tn, To = load("step143000"), load("step142000")
    common = [k for k in Tn if k in To and Tn[k][0] == "F32"
              and len(Tn[k][2]) == len(To[k][2])]
    parts = []
    for k in common:
        a = np.frombuffer(Tn[k][2], "<u4")
        b = np.frombuffer(To[k][2], "<u4")
        parts.append(a ^ b)
    delta = np.concatenate(parts)
    # sample to keep the diagnostic fast/light
    if delta.size > 20_000_000:
        delta = delta[:20_000_000]
    db = delta.view(np.uint8)
    arr = db.reshape(-1, 4)
    n = delta.size
    raw = n * 4
    print(f"real 1000-step delta: {n:,} fp32 elems, {raw/1e6:.1f} MB (sampled)\n")

    print("per-byte-plane entropy of the XOR delta (byte3=sign+exp, byte0=low mantissa):")
    for p in range(4):
        frac_nz = float((arr[:, p] != 0).mean()) * 100
        print(f"  byte{p}: h0={ev._h0(arr[:,p]):>5.2f}  h1={ev._h1(arr[:,p]):>5.2f} bits/byte"
              f"   nonzero={frac_nz:>5.1f}%")

    # per-bit-position flip rate (significance gradient)
    bits = np.unpackbits(arr, axis=1, bitorder="little")  # (n,32), bit i of byte = significance
    flip = bits.mean(axis=0)
    print("\nbit flip-rate by significance (low->high within each byte group):")
    for byte in range(4):
        seg = flip[byte * 8:(byte + 1) * 8]
        print(f"  byte{byte}: " + " ".join(f"{x*100:4.1f}" for x in seg))

    zstd = backends.get_backend("zstd")

    def sz(x):
        return len(zstd.compress(x, 19))

    print("\ncandidate delta coders (zstd-19):")
    whole = sz(db.tobytes())
    print(f"  zstd whole:           {whole:>12,}  save {(1-whole/raw)*100:5.1f}%")
    planes = arr.T.tobytes()
    sp = sz(planes)
    print(f"  byte-split + zstd:    {sp:>12,}  save {(1-sp/raw)*100:5.1f}%")
    perplane = sum(sz(arr[:, p].tobytes()) for p in range(4))
    print(f"  per-plane zstd:       {perplane:>12,}  save {(1-perplane/raw)*100:5.1f}%")
    # bit-plane: pack each of 32 bit positions into bytes, concat, zstd
    bp = np.packbits(bits, axis=0)  # pack along elements: (ceil(n/8), 32)
    bpb = bp.T.tobytes()  # plane-major
    bpz = sz(bpb)
    print(f"  bit-plane + zstd:     {bpz:>12,}  save {(1-bpz/raw)*100:5.1f}%")


if __name__ == "__main__":
    main()
