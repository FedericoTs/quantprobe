"""Round-trip property tests: inverse(forward(x)) == x for EVERY transform.

This is the un-gameable core of the project, so we hit each transform with a wide
spread of inputs: empty, single byte, random of many lengths (including lengths
that are NOT multiples of the element size), structured ramps/runs, and packed
int16 / float32 buffers.
"""

from __future__ import annotations

import math
import random
import struct

import pytest

from evocompress import transforms as T


def make_inputs():
    rng = random.Random(12345)
    inputs = [b"", b"\x00", b"\xff", bytes([7]), b"AAAA", b"ABABABAB", b"\x00" * 64]
    for length in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 255, 256, 257, 1000]:
        inputs.append(bytes(rng.randrange(256) for _ in range(length)))
    inputs.append(bytes(i % 256 for i in range(500)))            # ramp
    inputs.append(bytes((i * 7) % 256 for i in range(300)))      # strided ramp
    inputs.append(b"\x05" * 200 + b"\x09" * 123 + bytes(rng.randrange(256) for _ in range(77)))
    inputs.append(b"".join(struct.pack("<H", (i * 3) % 65536) for i in range(200)))   # int16
    inputs.append(b"".join(struct.pack("<f", math.sin(i * 0.1) * 100) for i in range(200)))  # f32
    inputs.append(b"".join(struct.pack("<d", math.cos(i * 0.05) * 7) for i in range(120)))   # f64
    return inputs


INPUTS = make_inputs()

# (transform name, params) instances to exercise.
TRANSFORM_CASES = [
    ("identity", {}),
    *[("delta", {"size": s}) for s in (1, 2, 4, 8)],
    *[("double_delta", {"size": s}) for s in (1, 2, 4, 8)],
    *[("zigzag", {"size": s}) for s in (1, 2, 4, 8)],
    *[("xor_prev", {"size": s}) for s in (1, 2, 4, 8)],
    *[("transpose", {"stride": s}) for s in (1, 2, 3, 4, 8, 16)],
    ("float_split", {"dtype": "f4"}),
    ("float_split", {"dtype": "f8"}),
    ("rle", {}),
    ("mtf", {}),
    *[("bwt", {"block": b}) for b in (16, 64, 256)],
    *[("bitpack", {"block": b}) for b in (16, 64, 256)],
    *[("lz77", {"window": w}) for w in (64, 256, 4096)],
]


@pytest.mark.parametrize("name,params", TRANSFORM_CASES, ids=lambda v: str(v))
def test_roundtrip_every_transform(name, params):
    t = T.build(name, params)
    for data in INPUTS:
        out = t.forward(data)
        back = t.inverse(out)
        assert back == data, f"{name}{params} failed round-trip on len={len(data)}"


def test_registry_has_required_transforms():
    required = {
        "delta", "double_delta", "zigzag", "transpose", "rle", "mtf",
        "bwt", "bitpack", "lz77", "float_split", "xor_prev",
    }
    assert required.issubset(set(T.available_transforms()))


def test_delta_then_inverse_on_monotonic():
    # a monotonically increasing uint32 sequence becomes near-constant after delta
    raw = b"".join(struct.pack("<I", 1000 + 3 * i) for i in range(256))
    d = T.build("delta", {"size": 4})
    fwd = d.forward(raw)
    # the body (minus first element) should be a single repeated 4-byte value
    body = fwd[4:]
    assert len(set(body[i : i + 4] for i in range(0, len(body), 4))) == 1
    assert d.inverse(fwd) == raw
