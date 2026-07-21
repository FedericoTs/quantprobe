"""Evaluator tests: it accepts honest pipelines and REJECTS any pipeline that
fails byte-exact round-trip (the un-gameable gate)."""

from __future__ import annotations

import struct

from evocompress import transforms as T
from evocompress.evaluator import score
from evocompress.genome import Gene, Genome
from evocompress.pipeline import Pipeline


# A deliberately broken transform: its inverse corrupts the data.  Registered so
# Pipeline.decode_blob can rebuild it by name during scoring.
@T.register
class _Broken(T.Transform):
    name = "broken_test"

    def forward(self, data: bytes) -> bytes:
        return data

    def inverse(self, data: bytes) -> bytes:
        return data + b"\x00"  # changes length -> guaranteed mismatch


def compressible_files():
    return [
        b"".join(struct.pack("<I", 1000 + i) for i in range(512)),  # monotonic -> very compressible
        b"the quick brown fox " * 200,
    ]


def test_accepts_honest_pipeline():
    pipe = Genome([Gene("delta", {"size": 4})], "zlib", 9).to_pipeline()
    m = score(pipe, compressible_files())
    assert m.roundtrip_ok is True
    assert m.ratio > 1.0
    assert m.total_in > m.total_out
    assert m.encode_MBps >= 0.0 and m.decode_MBps >= 0.0
    assert m.n_files == 2


def test_rejects_non_roundtripping_pipeline():
    pipe = Pipeline([T.build("broken_test")], "store", 0)
    m = score(pipe, compressible_files())
    assert m.roundtrip_ok is False
    assert m.ratio == 0.0


def test_store_pipeline_ratio_near_one():
    # store + header overhead -> ratio slightly below 1, still valid round-trip
    pipe = Pipeline([], "store", 0)
    m = score(pipe, compressible_files())
    assert m.roundtrip_ok is True
    assert 0.9 < m.ratio <= 1.0


def test_metrics_finite_and_sane():
    pipe = Genome([], "lzma", 6).to_pipeline()
    m = score(pipe, compressible_files())
    d = m.as_dict()
    for k in ("ratio", "encode_MBps", "decode_MBps"):
        assert d[k] == d[k] and d[k] >= 0.0  # not NaN, non-negative
    assert d["roundtrip_ok"] is True
