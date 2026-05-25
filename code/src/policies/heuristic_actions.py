"""Heuristic policies for GridWorld3DEnv (random / greedy toward nearest unvisited free cell)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.envs.grid_world_3d import Action3D, GridWorld3DEnv


def random_actions(env: GridWorld3DEnv, rng: np.random.Generator) -> np.ndarray:
    """Independent uniform random discrete action per UAV (factorized uniform joint)."""
    n_act = int(env.action_space.nvec[0])
    return rng.integers(0, n_act, size=env.num_uavs, dtype=np.int64)


def _pick_axis_step(
    pos: Tuple[int, int, int],
    goal: Tuple[int, int, int],
) -> int:
    """One greedy axis-aligned step toward goal (layer, row, col)."""
    dl = goal[0] - pos[0]
    dr = goal[1] - pos[1]
    dc = goal[2] - pos[2]
    if dl != 0:
        return int(Action3D.LAYER_UP if dl > 0 else Action3D.LAYER_DOWN)
    if dr != 0:
        return int(Action3D.SOUTH if dr > 0 else Action3D.NORTH)
    if dc != 0:
        return int(Action3D.EAST if dc > 0 else Action3D.WEST)
    return int(Action3D.NORTH)


def greedy_nearest_unvisited_actions(env: GridWorld3DEnv) -> np.ndarray:
    """Each UAV moves one step toward its nearest unvisited free cell (Manhattan)."""
    L, H, W = env.map_shape
    actions = np.zeros(env.num_uavs, dtype=np.int64)
    for idx in range(env.num_uavs):
        pos = env.uav_positions[idx]
        best: Optional[Tuple[int, int, int]] = None
        best_d = 10**9
        for l in range(L):
            for r in range(H):
                for c in range(W):
                    if env.obstacle_map[l, r, c]:
                        continue
                    if env.visited_map[l, r, c]:
                        continue
                    d = abs(l - pos[0]) + abs(r - pos[1]) + abs(c - pos[2])
                    if d < best_d:
                        best_d = d
                        best = (l, r, c)
        if best is None:
            actions[idx] = int(Action3D.NORTH)
            continue
        actions[idx] = _pick_axis_step(pos, best)
    return actions
