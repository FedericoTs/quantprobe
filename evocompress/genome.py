"""Genome representation and the search space.

A :class:`Genome` is an ordered list of (transform_name, params) genes plus a
backend choice and a level.  :class:`SearchSpace` defines what is sampleable and
how params/levels are jittered, so the GA stays decoupled from transform details.
A genome maps deterministically to a :class:`~evocompress.pipeline.Pipeline`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from . import backends
from .pipeline import Pipeline


@dataclass
class Gene:
    name: str
    params: dict = field(default_factory=dict)

    def copy(self) -> "Gene":
        return Gene(self.name, dict(self.params))

    def key(self) -> tuple:
        return (self.name, tuple(sorted(self.params.items())))


@dataclass
class Genome:
    genes: List[Gene] = field(default_factory=list)
    backend: str = "zstd"
    level: int = 19

    def copy(self) -> "Genome":
        return Genome([g.copy() for g in self.genes], self.backend, self.level)

    def key(self) -> tuple:
        return (tuple(g.key() for g in self.genes), self.backend, self.level)

    def to_pipeline(self) -> Pipeline:
        from . import transforms

        tlist = [transforms.build(g.name, g.params) for g in self.genes]
        return Pipeline(tlist, self.backend, self.level)

    def spec(self) -> dict:
        return {
            "transforms": [{"name": g.name, "params": g.params} for g in self.genes],
            "backend": self.backend,
            "level": self.level,
        }

    def to_json(self) -> str:
        return json.dumps(self.spec(), sort_keys=True)

    @classmethod
    def from_spec(cls, spec: dict) -> "Genome":
        genes = [Gene(t["name"], dict(t.get("params", {}))) for t in spec["transforms"]]
        return cls(genes, spec["backend"], int(spec["level"]))

    def describe(self) -> str:
        return self.to_pipeline().describe()


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
# A param sampler takes an rng and returns a params dict for a transform.
ParamSampler = Callable[[random.Random], dict]


def _size_sampler(choices: Tuple[int, ...]) -> ParamSampler:
    return lambda rng: {"size": rng.choice(choices)}


class SearchSpace:
    """Defines the catalog of transforms, their param samplers, and the set of
    backends + level ranges available to the GA."""

    def __init__(
        self,
        transform_catalog: Dict[str, ParamSampler],
        backend_levels: Dict[str, Tuple[int, int]],
        max_len: int = 5,
    ):
        self.transform_catalog = transform_catalog
        self.transform_names = list(transform_catalog)
        self.backend_levels = backend_levels
        self.backend_names = list(backend_levels)
        self.max_len = max_len

    # -- sampling ------------------------------------------------------------
    def random_gene(self, rng: random.Random) -> Gene:
        name = rng.choice(self.transform_names)
        params = self.transform_catalog[name](rng)
        return Gene(name, params)

    def random_backend(self, rng: random.Random) -> Tuple[str, int]:
        name = rng.choice(self.backend_names)
        lo, hi = self.backend_levels[name]
        return name, rng.randint(lo, hi)

    def random_genome(self, rng: random.Random) -> Genome:
        n = rng.randint(0, self.max_len)
        genes = [self.random_gene(rng) for _ in range(n)]
        backend, level = self.random_backend(rng)
        return Genome(genes, backend, level)

    def jitter_level(self, backend: str, level: int, rng: random.Random) -> int:
        lo, hi = self.backend_levels[backend]
        step = rng.choice((-2, -1, 1, 2))
        return max(lo, min(hi, level + step))

    def clamp_backend(self, genome: Genome) -> Genome:
        """Ensure backend/level lie in the available space (e.g. after loading)."""
        if genome.backend not in self.backend_levels:
            genome.backend = self.backend_names[0]
        lo, hi = self.backend_levels[genome.backend]
        genome.level = max(lo, min(hi, genome.level))
        return genome


def default_catalog(include_text: bool = True, include_float: bool = True) -> Dict[str, ParamSampler]:
    """Catalog of transforms with sensible param samplers."""
    catalog: Dict[str, ParamSampler] = {
        "identity": lambda rng: {},
        "delta": _size_sampler((1, 2, 4, 8)),
        "double_delta": _size_sampler((1, 2, 4, 8)),
        "zigzag": _size_sampler((1, 2, 4, 8)),
        "xor_prev": _size_sampler((1, 2, 4, 8)),
        "transpose": lambda rng: {"stride": rng.choice((2, 3, 4, 6, 8, 12, 16))},
        "rle": lambda rng: {},
        "bitpack": lambda rng: {"block": rng.choice((1024, 2048, 4096, 8192))},
    }
    if include_float:
        catalog["float_split"] = lambda rng: {"dtype": rng.choice(("f4", "f8"))}
    if include_text:
        catalog["mtf"] = lambda rng: {}
        catalog["bwt"] = lambda rng: {"block": rng.choice((1024, 2048, 4096, 8192))}
        catalog["lz77"] = lambda rng: {"window": rng.choice((1024, 4096, 16384))}
    return catalog


def default_backend_levels() -> Dict[str, Tuple[int, int]]:
    """Available backends mapped to (min, max) level, filtered by availability."""
    levels = {
        "store": (0, 0),
        "zlib": (1, 9),
        "gzip": (1, 9),
        "bz2": (1, 9),
        "lzma": (0, 9),
        "zstd": (1, 22),
        "brotli": (0, 11),
    }
    return {name: rng for name, rng in levels.items() if name in set(backends.available_backends())}


def make_search_space(domain: str = "generic", max_len: int = 5) -> SearchSpace:
    """Build a domain-aware search space.  Numeric domains favour float/byte
    transforms; text/binary domains add BWT/MTF/LZ pre-passes."""
    numeric = domain in ("time-series", "ml-weights")
    catalog = default_catalog(include_text=not numeric or True, include_float=True)
    # For numeric domains, drop the slow text-only transforms to keep search fast,
    # unless explicitly generic.
    if numeric:
        for slow in ("mtf", "bwt", "lz77"):
            catalog.pop(slow, None)
    return SearchSpace(catalog, default_backend_levels(), max_len=max_len)
