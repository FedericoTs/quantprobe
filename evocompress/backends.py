"""Backend entropy coders / codecs.

Each backend wraps a real library (stdlib or third-party) -- we are testing the
*evolutionary pipeline search*, not re-implementing a codec.  Backends that are
not installed are marked unavailable and excluded from the search space; the rest
of the system degrades gracefully.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import zlib
from typing import Dict

# Optional third-party codecs ------------------------------------------------
try:
    import zstandard as _zstd  # type: ignore

    _HAVE_ZSTD = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_ZSTD = False

try:
    import brotli as _brotli  # type: ignore

    _HAVE_BROTLI = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_BROTLI = False


class Backend:
    name: str = "base"
    available: bool = True
    # (min_level, max_level) inclusive; level is the single tunable knob.
    level_range: tuple[int, int] = (0, 0)
    default_level: int = 0

    def compress(self, data: bytes, level: int) -> bytes:  # pragma: no cover
        raise NotImplementedError

    def decompress(self, data: bytes) -> bytes:  # pragma: no cover
        raise NotImplementedError

    def clamp(self, level: int) -> int:
        lo, hi = self.level_range
        return max(lo, min(hi, int(level)))


class Store(Backend):
    name = "store"
    level_range = (0, 0)

    def compress(self, data: bytes, level: int = 0) -> bytes:
        return data

    def decompress(self, data: bytes) -> bytes:
        return data


class Zlib(Backend):
    name = "zlib"
    level_range = (0, 9)
    default_level = 6

    def compress(self, data: bytes, level: int = 6) -> bytes:
        return zlib.compress(data, self.clamp(level))

    def decompress(self, data: bytes) -> bytes:
        return zlib.decompress(data)


class Gzip(Backend):
    name = "gzip"
    level_range = (0, 9)
    default_level = 9

    def compress(self, data: bytes, level: int = 9) -> bytes:
        # mtime=0 keeps the output deterministic across runs.
        return gzip.compress(data, compresslevel=self.clamp(level), mtime=0)

    def decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)


class Bzip2(Backend):
    name = "bz2"
    level_range = (1, 9)
    default_level = 9

    def compress(self, data: bytes, level: int = 9) -> bytes:
        return bz2.compress(data, compresslevel=self.clamp(level))

    def decompress(self, data: bytes) -> bytes:
        return bz2.decompress(data)


class Lzma(Backend):
    name = "lzma"
    level_range = (0, 9)
    default_level = 6

    def compress(self, data: bytes, level: int = 6) -> bytes:
        return lzma.compress(data, preset=self.clamp(level))

    def decompress(self, data: bytes) -> bytes:
        return lzma.decompress(data)


class Zstd(Backend):
    name = "zstd"
    available = _HAVE_ZSTD
    level_range = (1, 22)
    default_level = 19

    def compress(self, data: bytes, level: int = 19) -> bytes:
        return _zstd.ZstdCompressor(level=self.clamp(level)).compress(data)

    def decompress(self, data: bytes) -> bytes:
        # content-size may be unknown for very small inputs; allow streaming read.
        return _zstd.ZstdDecompressor().decompress(data, max_output_size=1 << 31)


class Brotli(Backend):
    name = "brotli"
    available = _HAVE_BROTLI
    level_range = (0, 11)
    default_level = 11

    def compress(self, data: bytes, level: int = 11) -> bytes:
        return _brotli.compress(data, quality=self.clamp(level))

    def decompress(self, data: bytes) -> bytes:
        return _brotli.decompress(data)


# Registry -------------------------------------------------------------------
_ALL = [Store(), Zlib(), Gzip(), Bzip2(), Lzma(), Zstd(), Brotli()]
_REGISTRY: Dict[str, Backend] = {b.name: b for b in _ALL}


def get_backend(name: str) -> Backend:
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(_REGISTRY)}")
    b = _REGISTRY[name]
    if not b.available:
        raise RuntimeError(f"backend {name!r} is not available in this environment")
    return b


def available_backends() -> list[str]:
    return [b.name for b in _ALL if b.available]


def backend_info() -> dict:
    return {b.name: {"available": b.available, "levels": b.level_range} for b in _ALL}
