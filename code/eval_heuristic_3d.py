"""Evaluate random or greedy heuristic policies on GridWorld3DEnv (no learning)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import numpy as np

from src.envs.grid_world_3d import GridWorld3DEnv
from src.metrics.metrics import EpisodeStats, aggregate_episode_stats
from src.policies.heuristic_actions import greedy_nearest_unvisited_actions, random_actions
from src.utils.config import load_config
from src.utils.seeding import set_seed


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate heuristic policy on GridWorld3DEnv")
    p.add_argument("--policy", choices=("random", "greedy"), required=True)
    p.add_argument("--base-config", default="configs/base.yaml")
    p.add_argument("--env-config", type=str, required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=str, default=None)
    args = p.parse_args()

    base_cfg = load_config(args.base_config)
    env_cfg = load_config(args.env_config)

    env = GridWorld3DEnv(
        map_shape=tuple(int(x) for x in env_cfg["map_shape"]),
        num_uavs=int(env_cfg["num_uavs"][0]) if isinstance(env_cfg["num_uavs"], list) else int(env_cfg["num_uavs"]),
        obstacle_density=float(env_cfg["obstacle_density"]),
        max_steps=env_cfg.get("max_steps", 2500),
        energy_budget=env_cfg.get("energy_budget", 4500),
        shaping_weight=env_cfg.get("shaping_weight", 10.0),
        obstacle_shaping_weight=env_cfg.get("obstacle_shaping_weight", 2.0),
        seed=env_cfg.get("seed", args.seed),
    )

    map_shape = tuple(env.map_shape)
    num_uavs = env.num_uavs
    obstacle_density = float(env_cfg["obstacle_density"])

    set_seed(args.seed)
    stats_list: List[EpisodeStats] = []
    # Canonical coverage from last env info each episode (avoids any mismatch with EpisodeStats ratio).
    episode_final_coverage: List[float] = []

    print(
        f"Heuristic policy={args.policy} | map_shape={map_shape} | episodes={args.episodes} | seed_base={args.seed}",
        flush=True,
    )

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)
        # Independent RNG stream per episode for random policy so action noise is not coupled across maps.
        rng_ep = np.random.default_rng(int(np.random.SeedSequence([args.seed, ep]).generate_state(1)[0]))
        t0 = time.time()
        valid_cells = int(info.get("valid_cells", env.total_cells))
        if valid_cells <= 0:
            raise RuntimeError(f"reset(): invalid valid_cells={valid_cells}")
        st = EpisodeStats(
            start_time=t0,
            total_cells=valid_cells,
            per_uav_new_cells=[0 for _ in range(num_uavs)],
        )
        terminated = False
        truncated = False
        step_info: dict = {}
        while not (terminated or truncated):
            if args.policy == "random":
                actions = random_actions(env, rng_ep)
            else:
                actions = greedy_nearest_unvisited_actions(env)
            obs, rewards, terminated, truncated, step_info = env.step(actions)
            st.steps += 1
            st.total_actions += num_uavs
            st.collisions += step_info.get("collisions", 0)
            st.obstacle_hits += step_info.get("obstacle_hits", 0)
            st.visited_cells = step_info.get("visited_count", st.visited_cells)
            for idx, rv in enumerate(rewards):
                if rv >= env.reward_new_cell_base * 0.8:
                    st.new_cell_actions += 1
                    st.per_uav_new_cells[idx] += 1

        cov = float(step_info.get("coverage", 0.0))
        episode_final_coverage.append(cov)
        # Success only when environment signals full coverage termination (not mere truncation).
        st.success = bool(terminated and cov >= 0.99)
        st.end_time = time.time()
        stats_list.append(st)

    summary = aggregate_episode_stats(stats_list)
    cov_arr = np.array(episode_final_coverage, dtype=np.float64)
    cov_mu = float(np.mean(cov_arr))
    cov_sd = float(np.std(cov_arr, ddof=1)) if len(cov_arr) > 1 else 0.0
    out = {
        "policy": args.policy,
        "checkpoint": None,
        "mixer_type": "heuristic",
        "map_shape": list(map_shape),
        "num_uavs": num_uavs,
        "obstacle_density": obstacle_density,
        "episodes": args.episodes,
        "seed": args.seed,
        # Prefer env-reported terminal coverage per episode (same as info["coverage"] at last step).
        "coverage_mean": cov_mu,
        "coverage_std": cov_sd,
        "coverage_min": float(np.min(cov_arr)) if len(cov_arr) else 0.0,
        "coverage_max": float(np.max(cov_arr)) if len(cov_arr) else 0.0,
        "steps_mean": summary.get("steps_mean", 0.0),
        "success_mean": summary.get("success_mean", 0.0),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.output_json:
        outp = Path(args.output_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
