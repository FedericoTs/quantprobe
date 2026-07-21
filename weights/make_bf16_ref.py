"""Convert an fp32 .safetensors to a bf16 .safetensors (round-to-nearest-even).

Used to produce a bf16 reference (base) so a bf16 fine-tune can be stored as a delta
vs it -- the realistic deployment (ship the base once in bf16, fine-tunes as deltas).

  python -m weights.make_bf16_ref base_fp32.safetensors base_bf16.safetensors
"""

from __future__ import annotations

import json
import struct
import sys

import numpy as np

from weights import wcodec as wc


def to_bf16(f32_bytes):
    u32 = np.frombuffer(f32_bytes, "<u4")
    return (((u32.astype(np.uint64) + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16))


def convert(src, dst):
    raw, _, header, doff = wc.parse(src)
    new_header = {}
    if "__metadata__" in header:
        new_header["__metadata__"] = header["__metadata__"]
    data = bytearray()
    for name, dt, b, e in wc._tensors_in_order(header):
        buf = raw[doff + b:doff + e]
        if dt == "F32":
            out = to_bf16(buf).tobytes()
            ndt = "BF16"
        else:
            out, ndt = buf, dt
        nb = len(data)
        data += out
        new_header[name] = {"dtype": ndt, "shape": header[name]["shape"],
                            "data_offsets": [nb, len(data)]}
    hb = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(bytes(data))
    print(f"{src} -> {dst}  ({len(data)/1e6:.1f} MB bf16, {len(new_header)} entries)")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
