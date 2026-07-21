"""GA engine tests: it runs deterministically, improves fitness on a dataset with
planted redundancy, populates a hall-of-fame, and produces a valid champion."""

from __future__ import annotations

import struct

from evocompress.engine import Engine, GAConfig
from evocompress.fitness import FitnessConfig
from evocompress.genome import Gene, Genome, SearchSpace


def planted_redundancy_files():
    """Monotonic uint32 counters: nearly incompressible byte-wise, but a delta
    pre-pass turns them near-constant -> huge ratio.  The GA should discover this."""
    files = []
    for f in range(4):
        base = 100000 + f * 777
        files.append(b"".join(struct.pack("<I", base + 5 * i) for i in range(1024)))
    return files


def fast_search_space():
    # only fast, fully-deterministic transforms + the always-available zlib backend
    catalog = {
        "identity": lambda rng: {},
        "delta": lambda rng: {"size": rng.choice((1, 2, 4))},
        "double_delta": lambda rng: {"size": rng.choice((1, 2, 4))},
        "transpose": lambda rng: {"stride": rng.choice((2, 4))},
        "zigzag": lambda rng: {"size": rng.choice((1, 2, 4))},
    }
    backend_levels = {"store": (0, 0), "zlib": (1, 9)}
    return SearchSpace(catalog, backend_levels, max_len=4)


def run_small_ga(seed=0):
    space = fast_search_space()
    files = planted_redundancy_files()
    fcfg = FitnessConfig(objective="max_ratio")
    gacfg = GAConfig(population=20, generations=10, seed=seed, n_islands=2, patience=20)
    seeds = [Genome([], "store", 0), Genome([], "zlib", 9),
             Genome([Gene("delta", {"size": 4})], "zlib", 9)]
    engine = Engine(space, files, fcfg, gacfg, seed_genomes=seeds)
    return engine.run()


def test_ga_runs_and_improves():
    result = run_small_ga(seed=0)
    assert result.history, "no generations recorded"
    assert result.history[-1]["best_fitness"] >= result.history[0]["best_fitness"]
    assert result.champion_metrics.roundtrip_ok is True


def test_hall_of_fame_populated():
    result = run_small_ga(seed=0)
    assert len(result.hall_of_fame) >= 1
    # HoF sorted by fitness descending
    fits = [ind.fit for ind in result.hall_of_fame]
    assert fits == sorted(fits, reverse=True)


def test_champion_beats_store_on_planted_redundancy():
    result = run_small_ga(seed=0)
    # delta on monotonic uint32 -> near-constant -> very high ratio
    assert result.champion_metrics.ratio > 5.0, result.champion.describe()


def test_determinism_same_seed():
    a = run_small_ga(seed=1)
    b = run_small_ga(seed=1)
    assert a.champion.key() == b.champion.key()
    assert abs(a.champion_fitness - b.champion_fitness) < 1e-9
