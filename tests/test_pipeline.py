"""Pipeline round-trip + self-describing-header tests."""

from __future__ import annotations

import math
import random
import struct

import pytest

from evocompress import backends, transforms
from evocompress.genome import Gene, Genome
from evocompress.pipeline import Pipeline


def sample_data():
    rng = random.Random(7)
    return [
        b"",
        b"hello world " * 50,
        bytes(rng.randrange(256) for _ in range(2000)),
        b"".join(struct.pack("<f", math.sin(i * 0.05) * 10) for i in range(500)),
        bytes(i % 256 for i in range(1000)),
    ]


def candidate_pipelines():
    avail = set(backends.available_backends())
    specs = [
        Pipeline([], "store", 0),
        Pipeline([], "zlib", 6),
        Pipeline([transforms.build("delta", {"size": 1})], "zlib", 9),
    ]
    # build a few via Genome -> Pipeline to exercise that path too
    genome_specs = [
        Genome([Gene("transpose", {"stride": 4}), Gene("delta", {"size": 1})], "zlib", 6),
        Genome([Gene("delta", {"size": 4}), Gene("zigzag", {"size": 4})], "lzma", 6),
        Genome([Gene("rle", {})], "gzip", 9),
        Genome([Gene("bwt", {"block": 256}), Gene("mtf", {})], "bz2", 9),
    ]
    if "zstd" in avail:
        genome_specs.append(Genome([Gene("float_split", {"dtype": "f4"})], "zstd", 19))
    if "brotli" in avail:
        genome_specs.append(Genome([Gene("delta", {"size": 2})], "brotli", 11))
    return specs + [g.to_pipeline() for g in genome_specs]


@pytest.mark.parametrize("pipe", candidate_pipelines(), ids=lambda p: p.describe())
def test_pipeline_roundtrip(pipe):
    for data in sample_data():
        blob = pipe.encode(data)
        assert pipe.decode(blob) == data


def test_header_self_description():
    """decode_blob must reconstruct the pipeline from the blob ALONE -- no access
    to the original Pipeline object."""
    original = Genome(
        [Gene("transpose", {"stride": 4}), Gene("delta", {"size": 1}), Gene("zigzag", {"size": 1})],
        "zlib", 7,
    ).to_pipeline()
    data = b"".join(struct.pack("<f", math.cos(i * 0.02)) for i in range(400))
    blob = original.encode(data)

    # reconstruct purely from bytes
    assert Pipeline.decode_blob(blob) == data

    # the parsed spec round-trips through from_spec into an equivalent pipeline
    spec, _ = Pipeline._parse(blob)
    rebuilt = Pipeline.from_spec(spec)
    assert rebuilt.decode(blob) == data
    assert rebuilt.spec() == original.spec()


def test_bad_magic_rejected():
    with pytest.raises(ValueError):
        Pipeline.decode_blob(b"NOPE" + b"\x00" * 20)
