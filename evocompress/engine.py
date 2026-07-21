"""Classical genetic algorithm over compression pipelines.

Operators:
  * tournament selection
  * crossover: one-point splice of transform chains (backend/level from a parent)
  * mutation: insert / delete / swap a transform, jitter a param, or change the
    backend/level
  * elitism, island model with periodic migration, hall-of-fame, early stopping

Fully seeded and deterministic for the default ``max_ratio`` objective (which has
no timing term).  Evaluation results are cached by genome key, so re-evaluating an
identical genome is free -- important because GA populations rediscover genomes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from . import evaluator
from .evaluator import Metrics
from .fitness import FitnessConfig, fitness
from .genome import Gene, Genome, SearchSpace


@dataclass
class GAConfig:
    population: int = 60
    generations: int = 30
    seed: int = 0
    tournament_size: int = 3
    crossover_rate: float = 0.6
    mutation_rate: float = 0.8
    elitism: int = 2
    n_islands: int = 2
    migration_interval: int = 5
    migrants: int = 1
    patience: int = 10          # early stop after this many gens with no global improvement
    hof_size: int = 12


@dataclass
class Individual:
    genome: Genome
    fit: float
    metrics: Metrics


@dataclass
class GAResult:
    champion: Genome
    champion_metrics: Metrics
    champion_fitness: float
    hall_of_fame: List[Individual]
    history: List[dict]
    n_evaluated: int


class Engine:
    def __init__(
        self,
        space: SearchSpace,
        train_files: Sequence[bytes],
        fitness_cfg: FitnessConfig,
        ga_cfg: GAConfig,
        seed_genomes: Sequence[Genome] | None = None,
    ):
        self.space = space
        self.files = list(train_files)
        self.fcfg = fitness_cfg
        self.cfg = ga_cfg
        self.seed_genomes = list(seed_genomes or [])
        self._cache: dict = {}
        self.n_evaluated = 0

    # -- evaluation ----------------------------------------------------------
    def evaluate(self, genome: Genome) -> Tuple[float, Metrics]:
        key = genome.key()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        metrics = evaluator.score(genome.to_pipeline(), self.files)
        fit = fitness(metrics, len(genome.genes), self.fcfg)
        self._cache[key] = (fit, metrics)
        self.n_evaluated += 1
        return fit, metrics

    # -- genetic operators ---------------------------------------------------
    def _tournament(self, pop: List[Individual], rng: random.Random) -> Genome:
        k = min(self.cfg.tournament_size, len(pop))
        contenders = rng.sample(pop, k)
        winner = max(contenders, key=lambda ind: ind.fit)
        return winner.genome

    def _crossover(self, a: Genome, b: Genome, rng: random.Random) -> Genome:
        if rng.random() >= self.cfg.crossover_rate or not a.genes or not b.genes:
            child = a.copy()
        else:
            ca = rng.randint(0, len(a.genes))
            cb = rng.randint(0, len(b.genes))
            genes = [g.copy() for g in a.genes[:ca]] + [g.copy() for g in b.genes[cb:]]
            genes = genes[: self.space.max_len]
            backend, level = (a.backend, a.level) if rng.random() < 0.5 else (b.backend, b.level)
            child = Genome(genes, backend, level)
        return self.space.clamp_backend(child)

    def _mutate(self, g: Genome, rng: random.Random) -> Genome:
        if rng.random() >= self.cfg.mutation_rate:
            return g
        g = g.copy()
        ops = ["insert", "delete", "swap", "param", "backend", "level"]
        op = rng.choice(ops)
        if op == "insert" and len(g.genes) < self.space.max_len:
            pos = rng.randint(0, len(g.genes))
            g.genes.insert(pos, self.space.random_gene(rng))
        elif op == "delete" and g.genes:
            del g.genes[rng.randrange(len(g.genes))]
        elif op == "swap" and len(g.genes) >= 2:
            i, j = rng.sample(range(len(g.genes)), 2)
            g.genes[i], g.genes[j] = g.genes[j], g.genes[i]
        elif op == "param" and g.genes:
            i = rng.randrange(len(g.genes))
            g.genes[i] = self.space.random_gene(rng) if rng.random() < 0.3 else _reparam(
                g.genes[i], self.space, rng
            )
        elif op == "backend":
            g.backend, g.level = self.space.random_backend(rng)
        elif op == "level":
            g.level = self.space.jitter_level(g.backend, g.level, rng)
        else:
            # fallback when the chosen op is not applicable: re-roll the backend
            g.backend, g.level = self.space.random_backend(rng)
        return self.space.clamp_backend(g)

    # -- main loop -----------------------------------------------------------
    def run(self, log=lambda *_: None) -> GAResult:
        cfg = self.cfg
        pop_per_island = max(2, cfg.population // max(1, cfg.n_islands))
        islands: List[List[Individual]] = []
        for i in range(cfg.n_islands):
            irng = random.Random(cfg.seed * 7919 + i * 104729 + 1)
            genomes = [self.space.random_genome(irng) for _ in range(pop_per_island)]
            if i == 0:
                # inject seed genomes for a sane starting champion
                for sg in self.seed_genomes:
                    if genomes:
                        genomes[irng.randrange(len(genomes))] = sg.copy()
            islands.append([Individual(g, *self.evaluate(g)) for g in genomes])

        island_rngs = [random.Random(cfg.seed * 31 + i * 17 + 3) for i in range(cfg.n_islands)]

        hof: List[Individual] = []
        history: List[dict] = []
        best_fit = float("-inf")
        no_improve = 0

        for gen in range(cfg.generations):
            for idx in range(cfg.n_islands):
                islands[idx] = self._step_island(islands[idx], island_rngs[idx])

            # migration: copy best migrants from island i to island i+1
            if cfg.n_islands > 1 and cfg.migration_interval > 0 and gen > 0 and (
                gen % cfg.migration_interval == 0
            ):
                self._migrate(islands)

            # global bookkeeping
            all_inds = [ind for isl in islands for ind in isl]
            self._update_hof(hof, all_inds)
            gen_best = max(all_inds, key=lambda ind: ind.fit)
            valid = [ind for ind in all_inds if ind.metrics.roundtrip_ok]
            best_ratio = max((ind.metrics.ratio for ind in valid), default=0.0)
            history.append(
                {
                    "generation": gen,
                    "best_fitness": gen_best.fit,
                    "best_ratio": best_ratio,
                    "evaluated": self.n_evaluated,
                }
            )
            log(
                f"gen {gen:3d}  best_fit={gen_best.fit:.4f}  best_ratio={best_ratio:.4f}  "
                f"evals={self.n_evaluated}  champ={hof[0].genome.describe() if hof else '-'}"
            )

            if hof and hof[0].fit > best_fit + 1e-9:
                best_fit = hof[0].fit
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    log(f"early stop at gen {gen} (no improvement for {cfg.patience} gens)")
                    break

        champion = hof[0]
        return GAResult(
            champion=champion.genome,
            champion_metrics=champion.metrics,
            champion_fitness=champion.fit,
            hall_of_fame=hof,
            history=history,
            n_evaluated=self.n_evaluated,
        )

    # -- helpers -------------------------------------------------------------
    def _step_island(self, pop: List[Individual], rng: random.Random) -> List[Individual]:
        cfg = self.cfg
        pop_sorted = sorted(pop, key=lambda ind: ind.fit, reverse=True)
        next_genomes: List[Genome] = [ind.genome.copy() for ind in pop_sorted[: cfg.elitism]]
        while len(next_genomes) < len(pop):
            p1 = self._tournament(pop, rng)
            p2 = self._tournament(pop, rng)
            child = self._crossover(p1, p2, rng)
            child = self._mutate(child, rng)
            next_genomes.append(child)
        return [Individual(g, *self.evaluate(g)) for g in next_genomes]

    def _migrate(self, islands: List[List[Individual]]) -> None:
        n = len(islands)
        bests = []
        for isl in islands:
            isl_sorted = sorted(isl, key=lambda ind: ind.fit, reverse=True)
            bests.append([ind.genome.copy() for ind in isl_sorted[: self.cfg.migrants]])
        for i in range(n):
            dst = (i + 1) % n
            target = islands[dst]
            target.sort(key=lambda ind: ind.fit)  # worst first
            for j, mg in enumerate(bests[i]):
                if j < len(target):
                    target[j] = Individual(mg, *self.evaluate(mg))

    def _update_hof(self, hof: List[Individual], inds: List[Individual]) -> None:
        seen = {ind.genome.key() for ind in hof}
        for ind in inds:
            if not ind.metrics.roundtrip_ok:
                continue
            k = ind.genome.key()
            if k in seen:
                continue
            hof.append(ind)
            seen.add(k)
        hof.sort(key=lambda ind: ind.fit, reverse=True)
        del hof[self.cfg.hof_size :]


def _reparam(gene: Gene, space: SearchSpace, rng: random.Random) -> Gene:
    """Re-sample only the params of a gene, keeping its transform name."""
    if gene.name in space.transform_catalog:
        return Gene(gene.name, space.transform_catalog[gene.name](rng))
    return gene.copy()
