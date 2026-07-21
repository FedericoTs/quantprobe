"""evo-compress: evolutionary search for lossless compression pipelines.

A pipeline is an ordered list of reversible preprocessing transforms followed by a
backend codec (zstd/brotli/lzma/...).  A genetic algorithm searches the pipeline
space on a TRAIN split; the headline number is the byte-exact compression ratio on
a disjoint HELD-OUT TEST split of the same domain.  Round-trip is always verified.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import backends, evaluator, genome, transforms  # noqa: F401
from .pipeline import Pipeline  # noqa: F401

__all__ = ["Pipeline", "transforms", "backends", "evaluator", "genome", "__version__"]
