"""The honest scorer.

``score`` runs a Pipeline over a set of files and returns a metrics dict.  The
hard gate is byte-exact round-trip: ``decode(encode(x)) == x`` for *every* file,
double-checked with a SHA-256 hash.  If any file fails, ``roundtrip_ok`` is False
and the candidate must be treated as invalid (fitness ``-inf``).  This is what
makes the objective un-gameable: you cannot win by silently losing bytes.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Sequence

from .pipeline import Pipeline


@dataclass
class Metrics:
    ratio: float = 0.0
    encode_MBps: float = 0.0
    decode_MBps: float = 0.0
    roundtrip_ok: bool = False
    total_in: int = 0
    total_out: int = 0
    n_files: int = 0
    error: str = ""
    per_file: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ratio": self.ratio,
            "encode_MBps": self.encode_MBps,
            "decode_MBps": self.decode_MBps,
            "roundtrip_ok": self.roundtrip_ok,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "n_files": self.n_files,
            "error": self.error,
        }


def score(pipeline: Pipeline, files: Sequence[bytes]) -> Metrics:
    """Encode/decode every file, verify exact round-trip, and measure size+speed.

    ``files`` is a sequence of raw byte strings (the corpus, already loaded).
    """
    m = Metrics(n_files=len(files))
    enc_time = 0.0
    dec_time = 0.0
    try:
        for data in files:
            t0 = time.perf_counter()
            blob = pipeline.encode(data)
            t1 = time.perf_counter()
            restored = Pipeline.decode_blob(blob)
            t2 = time.perf_counter()

            if len(restored) != len(data) or restored != data or (
                hashlib.sha256(restored).digest() != hashlib.sha256(data).digest()
            ):
                m.roundtrip_ok = False
                m.error = "roundtrip mismatch"
                return m

            enc_time += t1 - t0
            dec_time += t2 - t1
            m.total_in += len(data)
            m.total_out += len(blob)
            m.per_file.append((len(data), len(blob)))
    except Exception as exc:  # any codec/transform failure invalidates the candidate
        m.roundtrip_ok = False
        m.error = f"{type(exc).__name__}: {exc}"
        return m

    m.roundtrip_ok = True
    m.ratio = (m.total_in / m.total_out) if m.total_out else 0.0
    if enc_time > 0:
        m.encode_MBps = (m.total_in / 1e6) / enc_time
    if dec_time > 0:
        m.decode_MBps = (m.total_in / 1e6) / dec_time
    return m


def score_named(pipeline: Pipeline, named_files: Sequence[tuple[str, bytes]]) -> Metrics:
    """Like :func:`score` but ``named_files`` is a list of (name, bytes); useful
    when a caller wants names preserved.  Names are ignored for scoring."""
    return score(pipeline, [data for _, data in named_files])
