"""Fitness functions over TRAIN metrics.

A pipeline that fails round-trip is always ``-inf`` (invalid).  Otherwise the
scalar fitness is the chosen objective minus penalties for violating a decode
speed floor and for pipeline length (complexity / overfitting pressure).

Objectives (pluggable):
  * ``max_ratio``      -- maximize compression ratio (default).
  * ``ratio_at_speed`` -- maximize ratio but hard-require decode >= speed floor.
  * ``pareto``         -- weighted blend of ratio and decode throughput.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .evaluator import Metrics

NEG_INF = float("-inf")


@dataclass
class FitnessConfig:
    objective: str = "max_ratio"
    speed_floor_MBps: float = 0.0  # minimum decode throughput before penalty
    speed_penalty: float = 0.5      # ratio-points subtracted per "speed deficit" unit
    length_penalty: float = 0.01    # ratio-points subtracted per transform in chain
    pareto_speed_weight: float = 0.02  # ratio-points awarded per decode MB/s


def fitness(metrics: Metrics, n_transforms: int, cfg: FitnessConfig) -> float:
    if not metrics.roundtrip_ok or not math.isfinite(metrics.ratio) or metrics.ratio <= 0:
        return NEG_INF

    ratio = metrics.ratio
    dec = metrics.decode_MBps

    if cfg.objective == "max_ratio":
        base = ratio
        if cfg.speed_floor_MBps > 0 and dec < cfg.speed_floor_MBps:
            deficit = (cfg.speed_floor_MBps - dec) / cfg.speed_floor_MBps
            base -= cfg.speed_penalty * deficit

    elif cfg.objective == "ratio_at_speed":
        if cfg.speed_floor_MBps > 0 and dec < cfg.speed_floor_MBps:
            # hard requirement: heavily penalize but keep ordering by how close.
            return -1.0 + (dec / max(cfg.speed_floor_MBps, 1e-9))
        base = ratio

    elif cfg.objective == "pareto":
        base = ratio + cfg.pareto_speed_weight * dec

    else:
        raise ValueError(f"unknown objective {cfg.objective!r}")

    base -= cfg.length_penalty * n_transforms
    return base


def dominates(a: Metrics, b: Metrics) -> bool:
    """Pareto domination on (ratio up, decode_MBps up) for valid candidates."""
    if not a.roundtrip_ok:
        return False
    if not b.roundtrip_ok:
        return True
    better_or_equal = a.ratio >= b.ratio and a.decode_MBps >= b.decode_MBps
    strictly_better = a.ratio > b.ratio or a.decode_MBps > b.decode_MBps
    return better_or_equal and strictly_better
