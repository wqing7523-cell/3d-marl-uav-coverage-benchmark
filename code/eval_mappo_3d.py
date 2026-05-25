"""Evaluate MAPPO checkpoint on GridWorld3DEnv (stochastic or greedy)."""
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
import torch
from torch.distributions import Categorical

from src.algos.mappo.networks import CentralCritic, MultiAgentActor
from src.envs.grid_world_3d import GridWorld3DEnv
from src.metrics.metrics import EpisodeStats, aggregate_episode_stats
from src.utils.config import load_config
from src.utils.seeding import set_seed


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate MAPPO checkpoint on GridWorld3DEnv")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--base-config", default="configs/base.yaml")
    p.add_argument("--env-config", type=str, required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from policy (recommended for reported MAPPO metrics)",
    )
    p.add_argument("--output-json", type=str, default=None)
    args = p.parse_args()

    base_cfg = load_config(args.base_config)
    env_cfg = load_config(args.env_config)
    device = torch.device(base_cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)

    map_shape = tuple(int(x) for x in ckpt["map_shape"])
    num_uavs = int(ckpt["num_uavs"])
    obstacle_density = float(ckpt["obstacle_density"])
    obs_dim = int(ckpt["obs_dim"])
    num_actions = int(ckpt["num_actions"])
    hidden_dim = int(ckpt.get("hidden_dim", 128))

    env = GridWorld3DEnv(
        map_shape=map_shape,
        num_uavs=num_uavs,
        obstacle_density=obstacle_density,
        max_steps=env_cfg.get("max_steps", 2500),
        energy_budget=env_cfg.get("energy_budget", 4500),
        shaping_weight=env_cfg.get("shaping_weight", 10.0),
        obstacle_shaping_weight=env_cfg.get("obstacle_shaping_weight", 2.0),
        seed=env_cfg.get("seed", args.seed),
    )

    if int(env.observation_space.shape[0]) != obs_dim:
        raise ValueError(
            f"Checkpoint obs_dim {obs_dim} != env {env.observation_space.shape[0]}"
        )

    actor = MultiAgentActor(obs_dim, num_uavs, num_actions, hidden_dim).to(device)
    critic = CentralCritic(obs_dim, hidden_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    actor.eval()
    critic.eval()

    set_seed(args.seed)
    stats_list: List[EpisodeStats] = []

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)
        t0 = time.time()
        st = EpisodeStats(
            start_time=t0,
            total_cells=int(info.get("valid_cells", env.total_cells)),
            per_uav_new_cells=[0 for _ in range(num_uavs)],
        )
        terminated = truncated = False
        step_info: dict = {}
        while not (terminated or truncated):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = actor(obs_t)
            actions = torch.zeros(num_uavs, dtype=torch.long, device=device)
            if args.stochastic:
                for i in range(num_uavs):
                    dist = Categorical(logits=logits[0, i])
                    actions[i] = dist.sample()
            else:
                for i in range(num_uavs):
                    actions[i] = logits[0, i].argmax()

            a = actions.cpu().numpy()
            obs, rewards, terminated, truncated, step_info = env.step(a)
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
        st.success = bool(terminated and cov >= 0.99)
        st.end_time = time.time()
        stats_list.append(st)

    summary = aggregate_episode_stats(stats_list)
    out = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "algorithm": "mappo",
        "map_shape": list(map_shape),
        "num_uavs": num_uavs,
        "obstacle_density": obstacle_density,
        "episodes": args.episodes,
        "seed": args.seed,
        "stochastic": bool(args.stochastic),
        "coverage_mean": summary.get("coverage_mean", 0.0),
        "coverage_std": summary.get("coverage_std", 0.0),
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
