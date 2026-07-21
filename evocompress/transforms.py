"""Reversible preprocessing transforms.

Every transform exposes ``forward(bytes) -> bytes`` and ``inverse(bytes) -> bytes``
such that ``inverse(forward(x)) == x`` for *any* byte string ``x`` (verified by the
property tests in ``tests/test_transforms.py``).  Transforms are the evolvable
building blocks: a Pipeline chains several of them ahead of a backend codec.

Design rules that keep round-trips exact:

* Element-wise transforms (delta, zigzag, ...) interpret the stream as an array of
  little-endian unsigned integers of width ``size`` bytes.  Any trailing bytes that
  do not fill a whole element are passed through unchanged.  Because these
  transforms are *length-preserving*, the inverse recomputes the same split.
* Transforms that change length (rle, bwt, bitpack, lz77) embed a small,
  self-describing header in their own output so the inverse needs nothing external.
"""

from __future__ import annotations

from typing import Dict, Type

import numpy as np

# Little-endian unsigned numpy dtypes keyed by element width in bytes.
_UDTYPE = {1: np.dtype("u1"), 2: np.dtype("<u2"), 4: np.dtype("<u4"), 8: np.dtype("<u8")}

_REGISTRY: Dict[str, Type["Transform"]] = {}


def register(cls: Type["Transform"]) -> Type["Transform"]:
    _REGISTRY[cls.name] = cls
    return cls


def build(name: str, params: dict | None = None) -> "Transform":
    """Construct a transform from its registered name and params dict."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown transform {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**(params or {}))


def available_transforms() -> list[str]:
    return sorted(_REGISTRY)


class Transform:
    """Base class.  Subclasses set ``name`` and implement forward/inverse."""

    name: str = "base"

    def __init__(self, **params):
        self.params: dict = dict(params)

    # -- interface -----------------------------------------------------------
    def forward(self, data: bytes) -> bytes:  # pragma: no cover - abstract
        raise NotImplementedError

    def inverse(self, data: bytes) -> bytes:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- serialization -------------------------------------------------------
    def spec(self) -> dict:
        return {"name": self.name, "params": dict(self.params)}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.params:
            ps = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
            return f"{self.name}({ps})"
        return self.name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _split_elements(data: bytes, size: int):
    """Return (head_array, remainder_bytes) for an element width of ``size``."""
    n = len(data)
    k = n // size
    head = np.frombuffer(data[: k * size], dtype=_UDTYPE[size]).copy()
    remainder = data[k * size :]
    return head, remainder


# ---------------------------------------------------------------------------
# Length-preserving element-wise transforms
# ---------------------------------------------------------------------------
@register
class Identity(Transform):
    name = "identity"

    def forward(self, data: bytes) -> bytes:
        return data

    def inverse(self, data: bytes) -> bytes:
        return data


@register
class Delta(Transform):
    """First difference of width-``size`` little-endian unsigned integers."""

    name = "delta"

    def __init__(self, size: int = 1):
        if size not in _UDTYPE:
            raise ValueError(f"delta size must be one of {sorted(_UDTYPE)}")
        super().__init__(size=size)
        self.size = size

    def forward(self, data: bytes) -> bytes:
        a, rem = _split_elements(data, self.size)
        if a.size:
            out = a.copy()
            out[1:] = a[1:] - a[:-1]  # modular subtraction (dtype wraps)
            data_head = out.tobytes()
        else:
            data_head = b""
        return data_head + rem

    def inverse(self, data: bytes) -> bytes:
        a, rem = _split_elements(data, self.size)
        if a.size:
            recovered = np.cumsum(a, dtype=a.dtype)  # modular prefix sum
            data_head = recovered.tobytes()
        else:
            data_head = b""
        return data_head + rem


@register
class DoubleDelta(Transform):
    """Delta applied twice -- good for signals with near-constant slope."""

    name = "double_delta"

    def __init__(self, size: int = 1):
        if size not in _UDTYPE:
            raise ValueError(f"double_delta size must be one of {sorted(_UDTYPE)}")
        super().__init__(size=size)
        self._d = Delta(size=size)

    def forward(self, data: bytes) -> bytes:
        return self._d.forward(self._d.forward(data))

    def inverse(self, data: bytes) -> bytes:
        return self._d.inverse(self._d.inverse(data))


@register
class ZigZag(Transform):
    """Map two's-complement signed integers to small unsigned magnitudes.

    Pairs well with delta: small +/- residuals become small non-negative ints,
    which entropy coders and bit-packing handle better.
    """

    name = "zigzag"

    def __init__(self, size: int = 1):
        if size not in _UDTYPE:
            raise ValueError(f"zigzag size must be one of {sorted(_UDTYPE)}")
        super().__init__(size=size)
        self.size = size
        self.bits = size * 8

    def forward(self, data: bytes) -> bytes:
        a, rem = _split_elements(data, self.size)
        if a.size:
            shifted = a << np.array(1, dtype=a.dtype)  # (a*2) mod 2^bits
            mask = np.zeros_like(a) - (a >> np.array(self.bits - 1, dtype=a.dtype))
            zz = shifted ^ mask
            head = zz.tobytes()
        else:
            head = b""
        return head + rem

    def inverse(self, data: bytes) -> bytes:
        a, rem = _split_elements(data, self.size)
        if a.size:
            lowbit = a & np.array(1, dtype=a.dtype)
            mask = np.zeros_like(a) - lowbit
            u = (a >> np.array(1, dtype=a.dtype)) ^ mask
            head = u.tobytes()
        else:
            head = b""
        return head + rem


@register
class XorPrev(Transform):
    """XOR each width-``size`` element with the previous one."""

    name = "xor_prev"

    def __init__(self, size: int = 1):
        if size not in _UDTYPE:
            raise ValueError(f"xor_prev size must be one of {sorted(_UDTYPE)}")
        super().__init__(size=size)
        self.size = size

    def forward(self, data: bytes) -> bytes:
        a, rem = _split_elements(data, self.size)
        if a.size:
            out = a.copy()
            out[1:] = a[1:] ^ a[:-1]
            head = out.tobytes()
        else:
            head = b""
        return head + rem

    def inverse(self, data: bytes) -> bytes:
        a, rem = _split_elements(data, self.size)
        if a.size:
            recovered = np.bitwise_xor.accumulate(a)
            head = recovered.tobytes()
        else:
            head = b""
        return head + rem


@register
class Transpose(Transform):
    """Structure-of-arrays / byte-shuffle: group byte k of every ``stride``-byte
    record together.  The classic HDF5 "shuffle" filter; helps when records have
    columns of differing entropy (e.g. interleaved sensor channels)."""

    name = "transpose"

    def __init__(self, stride: int = 4):
        if stride < 1 or stride > 1024:
            raise ValueError("transpose stride must be in 1..1024")
        super().__init__(stride=stride)
        self.stride = stride

    def forward(self, data: bytes) -> bytes:
        s = self.stride
        k = len(data) // s
        if k == 0:
            return data
        body = np.frombuffer(data[: k * s], dtype=np.uint8).reshape(k, s)
        return body.T.tobytes() + data[k * s :]

    def inverse(self, data: bytes) -> bytes:
        s = self.stride
        k = len(data) // s
        if k == 0:
            return data
        body = np.frombuffer(data[: k * s], dtype=np.uint8).reshape(s, k)
        return body.T.tobytes() + data[k * s :]


@register
class FloatSplit(Transform):
    """Split IEEE-754 floats into a high-half plane (sign+exponent+high mantissa)
    and a low-half plane (noisy low mantissa).  Concentrates the compressible
    exponent bytes; a permutation of bytes, so exactly reversible."""

    name = "float_split"

    def __init__(self, dtype: str = "f4"):
        if dtype not in ("f4", "f8"):
            raise ValueError("float_split dtype must be 'f4' or 'f8'")
        super().__init__(dtype=dtype)
        self.elem = 4 if dtype == "f4" else 8
        self.half = self.elem // 2

    def forward(self, data: bytes) -> bytes:
        e, h = self.elem, self.half
        k = len(data) // e
        if k == 0:
            return data
        body = np.frombuffer(data[: k * e], dtype=np.uint8).reshape(k, e)
        high = body[:, h:]  # most-significant half (little-endian -> high indices)
        low = body[:, :h]
        return high.tobytes() + low.tobytes() + data[k * e :]

    def inverse(self, data: bytes) -> bytes:
        e, h = self.elem, self.half
        k = len(data) // e
        if k == 0:
            return data
        planes = np.frombuffer(data[: k * e], dtype=np.uint8)
        high = planes[: k * h].reshape(k, h)
        low = planes[k * h : k * e].reshape(k, h)
        body = np.empty((k, e), dtype=np.uint8)
        body[:, h:] = high
        body[:, :h] = low
        return body.tobytes() + data[k * e :]


# ---------------------------------------------------------------------------
# Length-changing transforms (self-describing headers)
# ---------------------------------------------------------------------------
@register
class RLE(Transform):
    """PackBits run-length encoding -- reversible for arbitrary byte values."""

    name = "rle"

    def forward(self, data: bytes) -> bytes:
        out = bytearray()
        n = len(data)
        i = 0
        while i < n:
            # detect a run of >= 3 identical bytes
            run = 1
            while i + run < n and data[i + run] == data[i] and run < 128:
                run += 1
            if run >= 3:
                out.append(257 - run)  # 129..254 -> repeat (257-h) times
                out.append(data[i])
                i += run
            else:
                # gather a literal block up to 128 bytes, stopping before a >=3 run
                start = i
                lit = 0
                while i < n and lit < 128:
                    # peek for an upcoming 3-run to break the literal block
                    if (
                        i + 2 < n
                        and data[i] == data[i + 1] == data[i + 2]
                    ):
                        break
                    i += 1
                    lit += 1
                out.append(lit - 1)  # 0..127 -> copy (h+1) literals
                out += data[start:i]
        return bytes(out)

    def inverse(self, data: bytes) -> bytes:
        out = bytearray()
        n = len(data)
        i = 0
        while i < n:
            h = data[i]
            i += 1
            if h < 128:
                count = h + 1
                out += data[i : i + count]
                i += count
            elif h > 128:
                count = 257 - h
                out += bytes([data[i]]) * count
                i += 1
            # h == 128 is a no-op
        return bytes(out)


@register
class MTF(Transform):
    """Move-to-front coding over the 256-symbol byte alphabet."""

    name = "mtf"

    def forward(self, data: bytes) -> bytes:
        table = bytearray(range(256))
        out = bytearray(len(data))
        for idx, b in enumerate(data):
            j = table.index(b)
            out[idx] = j
            if j:
                del table[j]
                table.insert(0, b)
        return bytes(out)

    def inverse(self, data: bytes) -> bytes:
        table = bytearray(range(256))
        out = bytearray(len(data))
        for idx, j in enumerate(data):
            b = table[j]
            out[idx] = b
            if j:
                del table[j]
                table.insert(0, b)
        return bytes(out)


def _bwt_block(s: np.ndarray):
    """Cyclic BWT of a uint8 block via prefix-doubling suffix array.

    Returns (primary_index, L_column_bytes)."""
    m = s.size
    if m == 0:
        return 0, b""
    if m == 1:
        return 0, s.tobytes()
    idx = np.arange(m)
    rank = s.astype(np.int64)
    tmp = np.empty(m, dtype=np.int64)
    k = 1
    while True:
        second = rank[(idx + k) % m]
        order = np.lexsort((second, rank))
        srank = rank[order]
        ssec = second[order]
        tmp[order[0]] = 0
        neq = (srank[1:] != srank[:-1]) | (ssec[1:] != ssec[:-1])
        tmp[order[1:]] = np.cumsum(neq)
        rank = tmp.copy()
        if rank[order[-1]] == m - 1 or k >= m:
            break
        k <<= 1
    sa = np.argsort(rank, kind="stable")
    last = s[(sa - 1) % m]
    primary = int(np.flatnonzero(sa == 0)[0])
    return primary, last.tobytes()


def _ibwt_block(primary: int, last: bytes) -> bytes:
    L = np.frombuffer(last, dtype=np.uint8)
    m = L.size
    if m == 0:
        return b""
    if m == 1:
        return last
    sorted_pos = np.argsort(L, kind="stable")  # F-order -> source position in L
    lf = np.empty(m, dtype=np.int64)
    lf[sorted_pos] = np.arange(m)
    out = np.empty(m, dtype=np.uint8)
    p = primary
    for i in range(m - 1, -1, -1):
        out[i] = L[p]
        p = lf[p]
    return out.tobytes()


@register
class BWT(Transform):
    """Block Burrows-Wheeler transform.  Output is self-describing: an 8-byte
    total length, then per block a 4-byte primary index and the L column."""

    name = "bwt"

    def __init__(self, block: int = 4096):
        if block < 1 or block > (1 << 22):
            raise ValueError("bwt block must be in 1..4194304")
        super().__init__(block=block)
        self.block = block

    def forward(self, data: bytes) -> bytes:
        out = bytearray()
        out += len(data).to_bytes(8, "big")
        arr = np.frombuffer(data, dtype=np.uint8)
        for off in range(0, len(data), self.block):
            primary, last = _bwt_block(arr[off : off + self.block])
            out += primary.to_bytes(4, "big")
            out += last
        return bytes(out)

    def inverse(self, data: bytes) -> bytes:
        total = int.from_bytes(data[:8], "big")
        pos = 8
        remaining = total
        out = bytearray()
        while remaining > 0:
            blen = min(self.block, remaining)
            primary = int.from_bytes(data[pos : pos + 4], "big")
            pos += 4
            last = data[pos : pos + blen]
            pos += blen
            out += _ibwt_block(primary, last)
            remaining -= blen
        return bytes(out)


@register
class BitPack(Transform):
    """Per-block minimal-bit-width packing of bytes.  Output is self-describing:
    8-byte total length, then per block a 1-byte width and the packed bits."""

    name = "bitpack"

    def __init__(self, block: int = 4096):
        if block < 1 or block > (1 << 22):
            raise ValueError("bitpack block must be in 1..4194304")
        super().__init__(block=block)
        self.block = block

    def forward(self, data: bytes) -> bytes:
        out = bytearray()
        out += len(data).to_bytes(8, "big")
        arr = np.frombuffer(data, dtype=np.uint8)
        for off in range(0, len(arr), self.block):
            blk = arr[off : off + self.block]
            w = int(blk.max()).bit_length() if blk.size else 0
            out.append(w)
            if w > 0:
                bits = np.unpackbits(blk[:, None], axis=1)[:, 8 - w :]
                out += np.packbits(bits.reshape(-1)).tobytes()
        return bytes(out)

    def inverse(self, data: bytes) -> bytes:
        total = int.from_bytes(data[:8], "big")
        pos = 8
        remaining = total
        out = bytearray()
        while remaining > 0:
            blen = min(self.block, remaining)
            w = data[pos]
            pos += 1
            if w == 0:
                out += b"\x00" * blen
            else:
                nbits = blen * w
                nbytes = (nbits + 7) // 8
                packed = np.frombuffer(data[pos : pos + nbytes], dtype=np.uint8)
                pos += nbytes
                bits = np.unpackbits(packed)[:nbits].reshape(blen, w)
                full = np.zeros((blen, 8), dtype=np.uint8)
                full[:, 8 - w :] = bits
                out += np.packbits(full, axis=1).reshape(-1).tobytes()
            remaining -= blen
        return bytes(out)


@register
class LZ77(Transform):
    """Compact, exactly-reversible LZSS pre-pass.  Mostly redundant with the LZ
    backends but exposed so the search can stack it; bounded window keeps it
    deterministic.  Format: groups of 8 tokens preceded by a flag byte; a clear
    bit = literal, a set bit = (offset:2 bytes, length:1 byte) back-reference."""

    name = "lz77"
    MIN_MATCH = 4
    MAX_MATCH = 255 + MIN_MATCH

    def __init__(self, window: int = 4096):
        if window < 16 or window > 65535:
            raise ValueError("lz77 window must be in 16..65535")
        super().__init__(window=window)
        self.window = window

    def forward(self, data: bytes) -> bytes:
        n = len(data)
        out = bytearray()
        heads: Dict[bytes, list] = {}
        i = 0
        tokens = []  # (is_match, payload)
        mm, xm, win = self.MIN_MATCH, self.MAX_MATCH, self.window
        while i < n:
            best_len = 0
            best_off = 0
            if i + mm <= n:
                key = data[i : i + mm]
                cand = heads.get(key)
                if cand:
                    lo = i - win
                    for p in reversed(cand):
                        if p < lo:
                            break
                        length = mm
                        maxl = min(xm, n - i)
                        while length < maxl and data[p + length] == data[i + length]:
                            length += 1
                        if length > best_len:
                            best_len = length
                            best_off = i - p
                            if length == maxl:
                                break
            if best_len >= mm:
                tokens.append((True, (best_off, best_len)))
                end = i + best_len
                while i < end:
                    if i + mm <= n:
                        heads.setdefault(data[i : i + mm], []).append(i)
                    i += 1
            else:
                tokens.append((False, data[i]))
                if i + mm <= n:
                    heads.setdefault(data[i : i + mm], []).append(i)
                i += 1
        # serialize: 8-byte length, then flag-grouped tokens
        out += n.to_bytes(8, "big")
        for g in range(0, len(tokens), 8):
            group = tokens[g : g + 8]
            flag = 0
            for bit, (is_match, _) in enumerate(group):
                if is_match:
                    flag |= 1 << bit
            out.append(flag)
            for is_match, payload in group:
                if is_match:
                    off, length = payload
                    out += off.to_bytes(2, "big")
                    out.append(length - mm)
                else:
                    out.append(payload)
        return bytes(out)

    def inverse(self, data: bytes) -> bytes:
        n = int.from_bytes(data[:8], "big")
        pos = 8
        out = bytearray()
        mm = self.MIN_MATCH
        size = len(data)
        while len(out) < n and pos < size:
            flag = data[pos]
            pos += 1
            for bit in range(8):
                if len(out) >= n:
                    break
                if flag & (1 << bit):
                    off = int.from_bytes(data[pos : pos + 2], "big")
                    length = data[pos + 2] + mm
                    pos += 3
                    start = len(out) - off
                    for j in range(length):
                        out.append(out[start + j])
                else:
                    out.append(data[pos])
                    pos += 1
        return bytes(out)
