"""A Pipeline = ordered reversible transforms + a backend codec + a level.

The encoded blob is fully self-describing:

    MAGIC(4) | VERSION(1) | HEADER_LEN(uint32 BE) | HEADER_JSON | PAYLOAD

``HEADER_JSON`` records the transform sequence (names + params), the backend name
and the level, so ``Pipeline.decode_blob`` can reconstruct and invert the pipeline
from the bytes alone -- nothing external is required for a correct round-trip.
"""

from __future__ import annotations

import json
import struct
from typing import List

from . import backends, transforms

MAGIC = b"EVC1"
VERSION = 1
_HEADER_STRUCT = struct.Struct(">I")  # header length prefix


class Pipeline:
    def __init__(self, transform_list: List[transforms.Transform], backend: str, level: int):
        self.transforms = list(transform_list)
        self.backend_name = backend
        self.backend = backends.get_backend(backend)
        self.level = self.backend.clamp(level)

    # -- construction --------------------------------------------------------
    @classmethod
    def from_spec(cls, spec: dict) -> "Pipeline":
        tlist = [transforms.build(t["name"], t.get("params")) for t in spec["transforms"]]
        return cls(tlist, spec["backend"], int(spec["level"]))

    def spec(self) -> dict:
        return {
            "transforms": [t.spec() for t in self.transforms],
            "backend": self.backend_name,
            "level": self.level,
        }

    def describe(self) -> str:
        chain = " -> ".join(repr(t) for t in self.transforms) or "(none)"
        return f"[{chain}] => {self.backend_name}-{self.level}"

    # -- encode / decode -----------------------------------------------------
    def encode(self, data: bytes) -> bytes:
        x = data
        for t in self.transforms:
            x = t.forward(x)
        payload = self.backend.compress(x, self.level)
        header = json.dumps(self.spec(), separators=(",", ":"), sort_keys=True).encode("utf-8")
        return MAGIC + bytes([VERSION]) + _HEADER_STRUCT.pack(len(header)) + header + payload

    def decode(self, blob: bytes) -> bytes:
        """Decode using *this* pipeline's transform objects (header still parsed
        for the payload offset).  Equivalent to :meth:`decode_blob`."""
        return self.decode_blob(blob)

    @staticmethod
    def _parse(blob: bytes):
        if blob[:4] != MAGIC:
            raise ValueError("bad magic: not an evo-compress blob")
        version = blob[4]
        if version != VERSION:
            raise ValueError(f"unsupported version {version}")
        (hlen,) = _HEADER_STRUCT.unpack_from(blob, 5)
        start = 5 + _HEADER_STRUCT.size
        header = blob[start : start + hlen]
        spec = json.loads(header.decode("utf-8"))
        payload = blob[start + hlen :]
        return spec, payload

    @classmethod
    def decode_blob(cls, blob: bytes) -> bytes:
        """Reconstruct the pipeline purely from the blob header and invert it."""
        spec, payload = cls._parse(blob)
        backend = backends.get_backend(spec["backend"])
        transformed = backend.decompress(payload)
        tlist = [transforms.build(t["name"], t.get("params")) for t in spec["transforms"]]
        x = transformed
        for t in reversed(tlist):
            x = t.inverse(x)
        return x

    # -- serialization -------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(self.spec(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Pipeline":
        return cls.from_spec(json.loads(text))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Pipeline({self.describe()})"
