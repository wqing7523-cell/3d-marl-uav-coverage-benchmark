"""Evaluate a saved checkpoint on GridWorld3DEnv (greedy policy, epsilon=0)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import numpy as np
import torch

from src.algos.qmix.agent_net import AgentNetwork
from src.envs.grid_world_3d import GridWorld3DEnv
from src.metrics.metrics import EpisodeStats, aggregate_episode_stats
from src.utils.config import load_config
from src.utils.seeding import set_seed


def _select_greedy(
    agents: List[AgentNetwork],
    obs: np.ndarray,
    hidden_states: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, torch.Tensor]:
    batch_size = obs.shape[0]
    num_agents = len(agents)
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    actions = np.zeros((batch_size, num_agents), dtype=np.int64)
    new_hidden = torch.zeros_like(hidden_states)
    for i, agent in enumerate(agents):
        q, h_new = agent(obs_tensor[:, i, :], hidden_states[i])
        new_hidden[i] = h_new
        actions[:, i] = q.argmax(dim=1).cpu().numpy()
    return actions, new_hidden


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate 3D QMIX/VDN checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--base-config", default="configs/base.yaml")
    p.add_argument("--env-config", type=str, required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=str, default=None)
    args = p.parse_args()

    base_cfg = load_config(args.base_config)
    env_cfg = load_config(args.env_config)
    device = torch.device(base_cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    map_shape = tuple(int(x) for x in ckpt["map_shape"])
    num_uavs = int(ckpt["num_uavs"])
    obstacle_density = float(ckpt["obstacle_density"])
    mixer_type = str(ckpt.get("mixer_type", "qmix")).lower()

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

    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.nvec[0])
    if "obs_dim" in ckpt and int(ckpt["obs_dim"]) != obs_dim:
        raise ValueError(
            f"Checkpoint obs_dim {ckpt['obs_dim']} != env {obs_dim}; check env config vs checkpoint."
        )

    hidden_dim = int(ckpt.get("agent_hidden_dim", 64))

    agents = [
        AgentNetwork(obs_dim, action_dim, hidden_dim).to(device) for _ in range(num_uavs)
    ]
    for agent, state in zip(agents, ckpt["agents"]):
        agent.load_state_dict(state)
    agents = [a.eval() for a in agents]

    if mixer_type == "qmix" and ckpt.get("mixer") is None:
        raise ValueError("QMIX checkpoint missing mixer weights")

    set_seed(args.seed)
    stats_list: List[EpisodeStats] = []
    termination_rows: List[dict] = []

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)
        t0 = time.time()
        st = EpisodeStats(
            start_time=t0,
            total_cells=int(info.get("valid_cells", env.total_cells)),
            per_uav_new_cells=[0 for _ in range(num_uavs)],
        )
        hidden = torch.zeros(len(agents), 1, hidden_dim, device=device)
        done = truncated = False
        while not (done or truncated):
            agent_obs = np.tile(obs, (num_uavs, 1)).reshape(1, num_uavs, obs_dim)
            actions, hidden = _select_greedy(agents, agent_obs, hidden, device)
            a = actions[0]
            obs, rewards, done, truncated, step_info = env.step(a)
            st.steps += 1
            st.total_actions += num_uavs
            st.collisions += step_info.get("collisions", 0)
            st.obstacle_hits += step_info.get("obstacle_hits", 0)
            st.visited_cells = step_info.get("visited_count", st.visited_cells)
            for idx, rv in enumerate(rewards):
                if rv >= env.reward_new_cell_base * 0.8:
                    st.new_cell_actions += 1
                    st.per_uav_new_cells[idx] += 1
        st.end_time = time.time()
        cov_end = float(step_info.get("coverage", 0.0))
        st.success = bool(done and cov_end >= 0.99)
        valid_cells = int(step_info.get("valid_cells", st.total_cells) or st.total_cells or 1)
        remaining = int(step_info.get("remaining_free_cells", 0))
        termination_rows.append(
            {
                "terminated": bool(done),
                "truncated": bool(truncated),
                "success": bool(st.success),
                "remaining_free_ratio": remaining / max(1, valid_cells),
                "coverage_end": cov_end,
            }
        )
        stats_list.append(st)

    summary = aggregate_episode_stats(stats_list)
    n_eps = max(1, len(termination_rows))
    truncated_frac = sum(r["truncated"] for r in termination_rows) / n_eps
    remaining_ratio_mean = sum(r["remaining_free_ratio"] for r in termination_rows) / n_eps
    out = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "mixer_type": mixer_type,
        "map_shape": list(map_shape),
        "num_uavs": num_uavs,
        "obstacle_density": obstacle_density,
        "episodes": args.episodes,
        "seed": args.seed,
        "coverage_mean": summary.get("coverage_mean", 0.0),
        "coverage_std": summary.get("coverage_std", 0.0),
        "steps_mean": summary.get("steps_mean", 0.0),
        "success_mean": summary.get("success_mean", 0.0),
        "truncated_fraction": float(truncated_frac),
        "remaining_free_ratio_mean": float(remaining_ratio_mean),
        "metrics_source": "eval_3d.py",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.output_json:
        outp = Path(args.output_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
