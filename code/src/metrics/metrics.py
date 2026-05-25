"""Metric computations for UAV swarm experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class EpisodeStats:
    """Container tracking per-episode statistics."""

    steps: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    new_cell_actions: int = 0
    total_actions: int = 0
    visited_cells: int = 0
    total_cells: int = 0
    collisions: int = 0
    obstacle_hits: int = 0
    energy_consumed: float = 0.0
    per_uav_new_cells: List[int] = field(default_factory=list)
    success: bool = False

    def to_dict(self) -> Dict[str, float]:
        return {
            "steps": self.steps,
            "execution_time": max(0.0, self.end_time - self.start_time),
            "pa": self.valid_action_rate,
            "coverage": self.coverage,
            "collisions": self.collisions,
            "obstacle_hits": self.obstacle_hits,
            "energy": self.energy_consumed,
            "workload_balance": self.workload_balance,
            "success": float(self.success),
        }

    @property
    def execution_time(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def valid_action_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.new_cell_actions / self.total_actions

    @property
    def coverage(self) -> float:
        if self.total_cells == 0:
            return 0.0
        return self.visited_cells / self.total_cells

    @property
    def workload_balance(self) -> float:
        if not self.per_uav_new_cells:
            return 0.0
        values = np.array(self.per_uav_new_cells, dtype=float)
        if np.all(values == 0):
            return 1.0
        mean_val = np.mean(values)
        if mean_val == 0:
            return 0.0
        cv = np.std(values) / mean_val
        return 1.0 / (1.0 + cv)


def aggregate_episode_stats(stats: List[EpisodeStats]) -> Dict[str, float]:
    """Aggregate statistics across multiple episodes."""

    if not stats:
        return {}

    stacked = {key: [] for key in stats[0].to_dict().keys()}
    for ep in stats:
        for key, value in ep.to_dict().items():
            stacked[key].append(value)

    summary = {}
    for key, values in stacked.items():
        arr = np.array(values, dtype=float)
        summary[f"{key}_mean"] = float(np.mean(arr))
        summary[f"{key}_std"] = float(np.std(arr))
        summary[f"{key}_min"] = float(np.min(arr))
        summary[f"{key}_max"] = float(np.max(arr))

    return summary
