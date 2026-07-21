"""Split a single .safetensors into N shards + index.json (utility + test fixture).

  python -m weights.shard_model IN.safetensors OUT_DIR [N]
"""

from __future__ import annotations

import json
import os
import struct
import sys

from weights import wcodec as wc


def shard(src, out_dir, n=2):
    raw, hb, header, doff = wc.parse(src)
    tensors = wc._tensors_in_order(header)
    groups = [[] for _ in range(n)]
    sizes = [0] * n
    for t in tensors:
        i = min(range(n), key=lambda j: sizes[j])
        groups[i].append(t)
        sizes[i] += t[3] - t[2]
    os.makedirs(out_dir, exist_ok=True)
    weight_map = {}
    total = 0
    for gi, g in enumerate(groups):
        fn = f"model-{gi+1:05d}-of-{n:05d}.safetensors"
        nh = {}
        data = bytearray()
        for name, dt, b, e in g:
            nb = len(data)
            data += raw[doff + b:doff + e]
            nh[name] = {"dtype": dt, "shape": header[name]["shape"],
                        "data_offsets": [nb, len(data)]}
            weight_map[name] = fn
        hbytes = json.dumps(nh, separators=(",", ":")).encode("utf-8")
        with open(os.path.join(out_dir, fn), "wb") as f:
            f.write(struct.pack("<Q", len(hbytes)))
            f.write(hbytes)
            f.write(bytes(data))
        total += len(data)
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f)
    print(f"sharded {src} -> {out_dir} ({n} shards, {total/1e6:.1f} MB)")


if __name__ == "__main__":
    shard(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2)
