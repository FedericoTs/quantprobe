"""Weight-codec interface + baselines.

A WeightCodec maps (raw tensor bytes, dtype) -> self-describing blob, and blob ->
raw bytes (byte-exact). Baselines include raw backends and the ZipNN approach
(byte-plane split of exponent/mantissa, then a backend). Our evolved codecs plug
into the same interface so the evaluator scores everything uniformly.
"""

from __future__ import annotations

import os
import struct
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evocompress import backends  # noqa: E402

# bytes per element by dtype label
DTYPE_ELEM = {"bf16": 2, "fp16": 2, "fp32": 4, "fp64": 8, "int8": 1,
              "f8e4m3": 1, "f8e5m2": 1, "uint8": 1, "int16": 2, "int32": 4, "int64": 8}
_DT_CODE = {"bf16": 0, "fp16": 1, "fp32": 2, "fp64": 3, "int8": 4,
            "f8e4m3": 5, "f8e5m2": 6, "uint8": 7, "int16": 8, "int32": 9, "int64": 10}
_DT_NAME = {v: k for k, v in _DT_CODE.items()}


class WeightCodec:
    name: str = "base"

    def compress(self, data: bytes, dtype: str) -> bytes:  # pragma: no cover
        raise NotImplementedError

    def decompress(self, blob: bytes) -> bytes:  # pragma: no cover
        raise NotImplementedError

    def config(self) -> dict:
        return {"name": self.name}


class RawCodec(WeightCodec):
    """Apply a backend directly (no transform). dtype byte kept for uniformity."""

    def __init__(self, backend: str, level: int):
        self.b = backends.get_backend(backend)
        self.level = level
        self.name = f"raw-{backend}{level}"

    def compress(self, data: bytes, dtype: str) -> bytes:
        return bytes([_DT_CODE[dtype]]) + self.b.compress(data, self.level)

    def decompress(self, blob: bytes) -> bytes:
        return self.b.decompress(blob[1:])

    def config(self) -> dict:
        return {"type": "raw", "backend": self.b.name, "level": self.level}


class SplitCodec(WeightCodec):
    """ZipNN-style: split each element into byte planes (grouping exponent bytes
    away from mantissa bytes), then one backend over the concatenated planes."""

    def __init__(self, backend: str, level: int, name: str | None = None):
        self.b = backends.get_backend(backend)
        self.level = level
        self.name = name or f"split-{backend}{level}"

    def compress(self, data: bytes, dtype: str) -> bytes:
        e = DTYPE_ELEM[dtype]
        k = len(data) // e
        body = np.frombuffer(data[: k * e], dtype=np.uint8).reshape(k, e)
        rem = data[k * e :]
        planes = body.T.tobytes() + rem
        payload = self.b.compress(planes, self.level)
        return bytes([_DT_CODE[dtype]]) + struct.pack("<I", len(rem)) + payload

    def decompress(self, blob: bytes) -> bytes:
        dtype = _DT_NAME[blob[0]]
        e = DTYPE_ELEM[dtype]
        (remlen,) = struct.unpack("<I", blob[1:5])
        planes = self.b.decompress(blob[5:])
        body, rem = planes[: len(planes) - remlen], planes[len(planes) - remlen :]
        k = len(body) // e
        arr = np.frombuffer(body, dtype=np.uint8).reshape(e, k).T
        return arr.tobytes() + rem

    def config(self) -> dict:
        return {"type": "byte-split", "backend": self.b.name, "level": self.level}


class SplitPerPlaneCodec(WeightCodec):
    """Like SplitCodec but compresses each byte plane independently with a chosen
    backend per plane (e.g. strong coder on the exponent plane, store on mantissa).
    This is the natural home for our 'CM-on-exponent' edge."""

    def __init__(self, plane_backends, level: int, name: str):
        # plane_backends: list of backend names, one per byte position (low..high)
        self.pb = [backends.get_backend(b) for b in plane_backends]
        self.level = level
        self.name = name

    def compress(self, data: bytes, dtype: str) -> bytes:
        e = DTYPE_ELEM[dtype]
        k = len(data) // e
        body = np.frombuffer(data[: k * e], dtype=np.uint8).reshape(k, e)
        rem = data[k * e :]
        out = bytes([_DT_CODE[dtype]]) + struct.pack("<I", len(rem)) + struct.pack("<I", k)
        for p in range(e):
            plane = body[:, p].tobytes()
            be = self.pb[p % len(self.pb)]
            cp = be.compress(plane, self.level)
            out += struct.pack("<I", len(cp)) + cp
        out += rem
        return out

    def decompress(self, blob: bytes) -> bytes:
        dtype = _DT_NAME[blob[0]]
        e = DTYPE_ELEM[dtype]
        (remlen,) = struct.unpack("<I", blob[1:5])
        (k,) = struct.unpack("<I", blob[5:9])
        pos = 9
        planes = []
        for p in range(e):
            (clen,) = struct.unpack("<I", blob[pos : pos + 4])
            pos += 4
            be = self.pb[p % len(self.pb)]
            planes.append(np.frombuffer(be.decompress(blob[pos : pos + clen]), dtype=np.uint8))
            pos += clen
        rem = blob[pos : pos + remlen]
        arr = np.stack(planes, axis=1)  # (k, e)
        return arr.tobytes() + rem

    def config(self) -> dict:
        return {"type": "per-plane", "plane_backends": [b.name for b in self.pb], "level": self.level}


# Exponent-bearing (high) byte plane index per dtype; the rest is ~random mantissa.
EXP_PLANE = {"bf16": [1], "fp16": [1], "fp32": [3], "fp64": [7]}


class SplitSmartCodec(WeightCodec):
    """Lightweight single-tensor codec: byte-split, compress only the exponent
    plane(s) (the structured part), and STORE the mantissa plane(s) raw (they are
    near-random, so compressing them only wastes time). Matches the best ratio at
    a fraction of the work -> faster + lighter than 'split + brotli everything'."""

    def __init__(self, backend: str, level: int, name: str | None = None):
        self.b = backends.get_backend(backend)
        self.level = level
        self.name = name or f"smart-{backend}{level}"

    def compress(self, data: bytes, dtype: str) -> bytes:
        e = DTYPE_ELEM[dtype]
        k = len(data) // e
        body = np.frombuffer(data[: k * e], dtype=np.uint8).reshape(k, e)
        rem = data[k * e :]
        exp = set(EXP_PLANE.get(dtype, [e - 1]))
        out = bytes([_DT_CODE[dtype]]) + struct.pack("<II", len(rem), k)
        for p in range(e):
            plane = body[:, p].tobytes()
            if p in exp:
                cp = self.b.compress(plane, self.level)
                out += b"\x01" + struct.pack("<I", len(cp)) + cp
            else:
                out += b"\x00" + struct.pack("<I", len(plane)) + plane  # stored raw
        return out + rem

    def decompress(self, blob: bytes) -> bytes:
        dtype = _DT_NAME[blob[0]]
        e = DTYPE_ELEM[dtype]
        remlen, k = struct.unpack("<II", blob[1:9])
        pos = 9
        planes = []
        for _ in range(e):
            mode = blob[pos]
            (clen,) = struct.unpack("<I", blob[pos + 1 : pos + 5])
            pos += 5
            chunk = blob[pos : pos + clen]
            pos += clen
            raw = self.b.decompress(chunk) if mode == 1 else chunk
            planes.append(np.frombuffer(raw, dtype=np.uint8))
        rem = blob[pos : pos + remlen]
        return np.stack(planes, axis=1).tobytes() + rem

    def config(self) -> dict:
        return {"type": "smart-split", "exp_backend": self.b.name, "level": self.level,
                "mantissa": "stored-raw"}


def baseline_codecs() -> list[WeightCodec]:
    avail = set(backends.available_backends())
    codecs: list[WeightCodec] = [RawCodec("gzip", 9), RawCodec("lzma", 9)]
    if "zstd" in avail:
        codecs.append(RawCodec("zstd", 19))
    # ZipNN-style byte split
    if "zstd" in avail:
        codecs.append(SplitCodec("zstd", 19, "zipnn-zstd19"))
    codecs.append(SplitCodec("lzma", 9, "split-lzma9"))
    if "brotli" in avail:
        codecs.append(SplitCodec("brotli", 11, "split-brotli11"))
    # per-plane: strong coder (lzma) only on the high (exponent) plane idea is
    # generalized in evaluate via the per-plane diagnostic; keep a concrete one:
    if "zstd" in avail:
        codecs.append(SplitPerPlaneCodec(["zstd", "lzma"], 9, "perplane-zstd-lzma"))
    # our lightweight single-tensor codec: compress exponent plane, store mantissa
    if "zstd" in avail:
        codecs.append(SplitSmartCodec("zstd", 19, "smart-zstd19"))
    if "brotli" in avail:
        codecs.append(SplitSmartCodec("brotli", 11, "smart-brotli11"))
    codecs.append(SplitSmartCodec("lzma", 9, "smart-lzma9"))
    return codecs
