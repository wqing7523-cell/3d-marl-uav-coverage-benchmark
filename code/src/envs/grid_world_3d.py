"""Multi-UAV 3D layered grid world (L x H x W) for cooperative coverage."""
from __future__ import annotations

from collections import deque
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Minimal subset so GridWorld3DEnv runs without installing gymnasium (offline / CI).
    class _Env:
        metadata = {"render_modes": []}

    class _MultiDiscrete:
        __slots__ = ("nvec", "shape")

        def __init__(self, nvec: List[int]) -> None:
            self.nvec = np.asarray(nvec, dtype=np.int64)
            self.shape = (len(self.nvec),)

    class _Box:
        __slots__ = ("low", "high", "shape", "dtype")

        def __init__(
            self,
            low: float,
            high: float,
            shape: Tuple[int, ...],
            dtype=np.float32,
        ) -> None:
            self.low = low
            self.high = high
            self.shape = tuple(shape)
            self.dtype = dtype

    class _Spaces:
        MultiDiscrete = _MultiDiscrete
        Box = _Box

    class _Gym:
        Env = _Env

    gym = _Gym()
    spaces = _Spaces()


class Action3D(IntEnum):
    NORTH = 0  # row - 1
    SOUTH = 1  # row + 1
    WEST = 2  # col - 1
    EAST = 3  # col + 1
    LAYER_UP = 4  # layer + 1
    LAYER_DOWN = 5  # layer - 1


class GridWorld3DEnv(gym.Env):
    """Voxel grid with discrete height layers; same reward/collision pattern as 2D GridWorldEnv."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        map_shape: Tuple[int, int, int] = (3, 5, 5),
        num_uavs: int = 2,
        obstacle_density: float = 0.08,
        max_steps: int = 2000,
        energy_budget: int = 3600,
        reward_new_cell_base: float = 358.74,
        reward_visited_cell: float = -31.14,
        reward_obstacle: float = -225.17,
        reward_collision: float = -100.0,
        reward_complete: float = 1000.0,
        reward_no_progress: float = -2.0,
        no_progress_patience: int = 30,
        shaping_weight: float = 10.0,
        obstacle_shaping_weight: float = 2.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_layers, self.height, self.width = map_shape
        if self.num_layers < 1 or self.height < 2 or self.width < 2:
            raise ValueError("map_shape must be (L>=1, H>=2, W>=2)")

        self.map_shape = map_shape
        self.num_uavs = num_uavs
        self.max_steps = max_steps
        self.energy_budget = energy_budget
        self.obstacle_density = max(0.0, float(obstacle_density))

        self.reward_new_cell_base = reward_new_cell_base
        self.reward_visited_cell = reward_visited_cell
        self.reward_obstacle = reward_obstacle
        self.reward_collision = reward_collision
        self.reward_complete = reward_complete
        self.reward_no_progress = reward_no_progress
        self.no_progress_patience = max(1, no_progress_patience)
        self.shaping_weight = shaping_weight
        self.obstacle_shaping_weight = obstacle_shaping_weight

        self.rng = np.random.default_rng(seed)
        self.obstacle_map: np.ndarray = np.zeros(map_shape, dtype=bool)
        self.visited_map: np.ndarray = np.zeros(map_shape, dtype=bool)
        self.uav_positions: List[Tuple[int, int, int]] = []
        self.uav_energy = np.zeros(self.num_uavs, dtype=int)
        self.step_count = 0
        self.visited_count = 0
        self.free_cell_count = 0
        self.total_cells = int(np.prod(map_shape))
        self.episode_reward = 0.0
        self.no_progress_steps = np.zeros(self.num_uavs, dtype=int)
        self.prev_potential = np.zeros(self.num_uavs, dtype=np.float32)

        self._build_spaces()

    def _build_spaces(self) -> None:
        n_act = len(Action3D)
        self.action_space = spaces.MultiDiscrete([n_act] * self.num_uavs)
        vox = self.num_layers * self.height * self.width
        map_obs_size = vox * 3
        scalar_obs_size = 1 + self.num_uavs * 4
        obs_size = map_obs_size + scalar_obs_size
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

    def _generate_obstacles(self) -> None:
        self.obstacle_map.fill(False)
        total = self.num_layers * self.height * self.width
        target = int(total * self.obstacle_density)
        if target <= 0:
            return
        coords = np.stack(
            np.meshgrid(
                np.arange(self.num_layers),
                np.arange(self.height),
                np.arange(self.width),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        perm = self.rng.permutation(len(coords))[:target]
        for i in perm:
            l, r, c = int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])
            self.obstacle_map[l, r, c] = True

    def _count_free_cells(self) -> int:
        return int(np.logical_not(self.obstacle_map).sum())

    def _clear_starts_from_obstacles(
        self, positions: List[Tuple[int, int, int]]
    ) -> None:
        for l, r, c in positions:
            if self.obstacle_map[l, r, c]:
                self.obstacle_map[l, r, c] = False

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._generate_obstacles()
        self.free_cell_count = self._count_free_cells()
        if self.free_cell_count < self.num_uavs:
            self.obstacle_map.fill(False)
            self.free_cell_count = self._count_free_cells()

        self.step_count = 0
        self.episode_reward = 0.0
        self.visited_map = np.zeros(self.map_shape, dtype=bool)
        self.uav_energy = np.full(self.num_uavs, self.energy_budget, dtype=int)
        self.uav_positions = self._initial_positions()
        self._clear_starts_from_obstacles(self.uav_positions)
        self.free_cell_count = self._count_free_cells()
        self.total_cells = self.free_cell_count

        self.visited_count = 0
        self.no_progress_steps = np.zeros(self.num_uavs, dtype=int)
        for l, r, c in self.uav_positions:
            if not self.visited_map[l, r, c]:
                self.visited_map[l, r, c] = True
                self.visited_count += 1

        self.prev_potential = np.array(
            [self._state_potential(pos) for pos in self.uav_positions], dtype=np.float32
        )

        return self._get_observation(), self._get_info()

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        if len(actions) != self.num_uavs:
            raise ValueError(
                f"Expected {self.num_uavs} actions, received {len(actions)}"
            )

        rewards = np.zeros(self.num_uavs, dtype=np.float32)
        attempted: List[Tuple[int, int, int]] = []
        final_positions = list(self.uav_positions)
        progress_made = [False] * self.num_uavs
        penalty_registered = [False] * self.num_uavs
        collisions = 0
        obstacle_hits = 0

        for idx, action in enumerate(actions):
            l, r, c = self.uav_positions[idx]
            if self.uav_energy[idx] <= 0:
                attempted.append((l, r, c))
                continue

            dl, dr, dc = self._action_to_delta(Action3D(int(action)))
            nl, nr, nc = l + dl, r + dr, c + dc

            if self._is_invalid_cell(nl, nr, nc):
                rewards[idx] += self.reward_obstacle
                obstacle_hits += 1
                self._register_no_progress(idx, rewards)
                penalty_registered[idx] = True
                attempted.append((l, r, c))
                continue

            attempted.append((nl, nr, nc))

        for idx, pos in enumerate(attempted):
            if self.uav_energy[idx] <= 0:
                continue
            if attempted.count(pos) > 1:
                rewards[idx] += self.reward_collision
                collisions += 1
                self._register_no_progress(idx, rewards)
                penalty_registered[idx] = True
                continue

            final_positions[idx] = pos
            self.uav_energy[idx] = max(0, self.uav_energy[idx] - 1)
            l, r, c = pos
            if not self.visited_map[l, r, c]:
                remaining = self._remaining_free_cells()
                scale = 1.0
                if remaining > 0:
                    scale += max(self.height, self.width, self.num_layers) / remaining
                rewards[idx] += self.reward_new_cell_base * scale
                self.visited_map[l, r, c] = True
                self.visited_count += 1
                self.no_progress_steps[idx] = 0
                progress_made[idx] = True
            else:
                rewards[idx] += self.reward_visited_cell
                self._register_no_progress(idx, rewards)
                penalty_registered[idx] = True

        for idx in range(self.num_uavs):
            if self.uav_energy[idx] <= 0:
                continue
            if progress_made[idx] or penalty_registered[idx]:
                continue
            self._register_no_progress(idx, rewards, gentle=True)

        self.uav_positions = final_positions

        for idx, pos in enumerate(self.uav_positions):
            if self.uav_energy[idx] <= 0:
                continue
            potential_new = self._state_potential(pos)
            delta = potential_new - self.prev_potential[idx]
            rewards[idx] += self.shaping_weight * delta
            self.prev_potential[idx] = potential_new

        self.step_count += 1
        self.episode_reward += float(rewards.sum())

        terminated = self._is_covered()
        truncated = self.step_count >= self.max_steps or np.all(self.uav_energy <= 0)

        if terminated:
            rewards += self.reward_complete / self.num_uavs

        info = self._get_info()
        info.update(
            {
                "collisions": collisions,
                "obstacle_hits": obstacle_hits,
                "reward_vector": rewards.copy(),
            }
        )
        return self._get_observation(), rewards, terminated, truncated, info

    def _action_to_delta(self, action: Action3D) -> Tuple[int, int, int]:
        if action == Action3D.NORTH:
            return 0, -1, 0
        if action == Action3D.SOUTH:
            return 0, 1, 0
        if action == Action3D.WEST:
            return 0, 0, -1
        if action == Action3D.EAST:
            return 0, 0, 1
        if action == Action3D.LAYER_UP:
            return 1, 0, 0
        if action == Action3D.LAYER_DOWN:
            return -1, 0, 0
        raise ValueError(f"Unsupported action: {action}")

    def _is_invalid_cell(self, l: int, r: int, c: int) -> bool:
        if (
            l < 0
            or l >= self.num_layers
            or r < 0
            or r >= self.height
            or c < 0
            or c >= self.width
        ):
            return True
        return bool(self.obstacle_map[l, r, c])

    def _remaining_free_cells(self) -> int:
        free = np.logical_not(self.obstacle_map)
        rem = np.logical_and(free, np.logical_not(self.visited_map))
        return int(rem.sum())

    def _is_covered(self) -> bool:
        free = np.logical_not(self.obstacle_map)
        return bool(np.all(np.logical_or(self.visited_map, self.obstacle_map)))

    def _initial_positions(self) -> List[Tuple[int, int, int]]:
        layer0 = self.obstacle_map[0]
        free = np.argwhere(~layer0)
        if len(free) < self.num_uavs:
            all_free = np.argwhere(~self.obstacle_map)
            if len(all_free) < self.num_uavs:
                raise RuntimeError("Not enough free cells for UAV initialisation")
            idx = self.rng.choice(len(all_free), size=self.num_uavs, replace=False)
            return [tuple(int(x) for x in all_free[i]) for i in idx]
        idx = self.rng.choice(len(free), size=self.num_uavs, replace=False)
        return [(0, int(free[i][0]), int(free[i][1])) for i in idx]

    def _state_potential(self, position: Tuple[int, int, int]) -> float:
        dist = self._nearest_unvisited_distance(position)
        clearance = self._nearest_obstacle_distance(position)
        potential = 0.0
        if dist is not None:
            potential -= float(dist)
        if clearance is not None:
            potential += self.obstacle_shaping_weight * float(min(clearance, 5))
        return potential

    def _nearest_obstacle_distance(self, position: Tuple[int, int, int]) -> Optional[int]:
        l, r, c = position
        if self.obstacle_map[l, r, c]:
            return 0

        def pred(pl: int, pr: int, pc: int) -> bool:
            return bool(self.obstacle_map[pl, pr, pc])

        visited = np.zeros(self.map_shape, dtype=bool)
        q: deque = deque()
        q.append((l, r, c, 0))
        visited[l, r, c] = True
        dirs = [
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
            (1, 0, 0),
            (-1, 0, 0),
        ]
        while q:
            cl, cr, cc, d = q.popleft()
            for dl, dr, dc in dirs:
                nl, nr, nc = cl + dl, cr + dr, cc + dc
                if nl < 0 or nl >= self.num_layers or nr < 0 or nr >= self.height or nc < 0 or nc >= self.width:
                    continue
                if self.obstacle_map[nl, nr, nc]:
                    return d + 1
                if visited[nl, nr, nc]:
                    continue
                visited[nl, nr, nc] = True
                q.append((nl, nr, nc, d + 1))
        return None

    def _nearest_unvisited_distance(
        self, position: Tuple[int, int, int]
    ) -> Optional[int]:
        l, r, c = position
        if not self.visited_map[l, r, c]:
            return 0

        def pred(pl: int, pr: int, pc: int) -> bool:
            if self.obstacle_map[pl, pr, pc]:
                return False
            return not self.visited_map[pl, pr, pc]

        visited = np.zeros(self.map_shape, dtype=bool)
        q: deque = deque()
        q.append((l, r, c, 0))
        visited[l, r, c] = True
        dirs = [
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
            (1, 0, 0),
            (-1, 0, 0),
        ]
        while q:
            cl, cr, cc, d = q.popleft()
            for dl, dr, dc in dirs:
                nl, nr, nc = cl + dl, cr + dr, cc + dc
                if self._is_invalid_cell(nl, nr, nc):
                    continue
                if pred(nl, nr, nc):
                    return d + 1
                if visited[nl, nr, nc]:
                    continue
                visited[nl, nr, nc] = True
                q.append((nl, nr, nc, d + 1))
        return None

    def _register_no_progress(
        self, idx: int, rewards: np.ndarray, gentle: bool = False
    ) -> None:
        self.no_progress_steps[idx] += 1
        if self.no_progress_steps[idx] >= self.no_progress_patience:
            penalty = (
                self.reward_no_progress if not gentle else self.reward_no_progress * 0.5
            )
            rewards[idx] += penalty
            self.no_progress_steps[idx] = 0

    def _get_observation(self) -> np.ndarray:
        visited_layer = self.visited_map.astype(np.float32)
        obstacle_layer = self.obstacle_map.astype(np.float32)
        uav_layer = np.zeros(self.map_shape, dtype=np.float32)
        for l, r, c in self.uav_positions:
            uav_layer[l, r, c] = 1.0
        maps = np.stack([visited_layer, obstacle_layer, uav_layer], axis=0).reshape(-1)

        denom = max(1, self.free_cell_count)
        coverage = self.visited_count / denom
        scalars: List[float] = [coverage]
        max_l = max(1, self.num_layers - 1)
        max_r = max(1, self.height - 1)
        max_c = max(1, self.width - 1)
        for idx, (l, r, c) in enumerate(self.uav_positions):
            scalars.extend(
                [
                    l / max_l,
                    r / max_r,
                    c / max_c,
                    self.uav_energy[idx] / max(1, self.energy_budget),
                ]
            )
        return np.concatenate([maps, np.array(scalars, dtype=np.float32)], axis=0)

    def _get_info(self) -> Dict[str, Any]:
        denom = max(1, self.free_cell_count)
        coverage = self.visited_count / denom
        return {
            "coverage": coverage,
            "steps": self.step_count,
            "energy": self.uav_energy.copy(),
            "visited_cells": self.visited_count,
            "visited_count": self.visited_count,
            "valid_cells": self.free_cell_count,
            "remaining_free_cells": self._remaining_free_cells(),
            "map_shape": self.map_shape,
        }

    def render(self) -> None:
        for l in range(self.num_layers):
            print(f"--- layer {l} ---")
            g = np.full((self.height, self.width), ".", dtype=object)
            g[self.obstacle_map[l]] = "#"
            g[self.visited_map[l] & ~self.obstacle_map[l]] = "v"
            for ul, ur, uc in self.uav_positions:
                if ul == l and not self.obstacle_map[l, ur, uc]:
                    g[ur, uc] = "U"
            print(g)

    def close(self) -> None:
        pass
