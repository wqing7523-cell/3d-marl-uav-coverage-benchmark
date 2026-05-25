"""Roll out GridWorld3DEnv and save 3D trajectory + per-layer occupancy heatmaps."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

_script_dir = Path(__file__).resolve().parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

import torch

from src.algos.qmix.agent_net import AgentNetwork
from src.envs.grid_world_3d import GridWorld3DEnv
from src.policies.heuristic_actions import greedy_nearest_unvisited_actions, random_actions
from src.utils.config import load_config
from src.utils.seeding import set_seed


def _select_greedy_qmix(
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


def _load_qmix_agents(
    checkpoint: Path,
    device: torch.device,
    obs_dim: int,
    action_dim: int,
) -> Tuple[List[AgentNetwork], int]:
    ckpt = torch.load(str(checkpoint), map_location=device)
    hidden_dim = int(ckpt.get("agent_hidden_dim", 64))
    num_uavs = int(ckpt["num_uavs"])
    agents = [
        AgentNetwork(obs_dim, action_dim, hidden_dim).to(device) for _ in range(num_uavs)
    ]
    for agent, state in zip(agents, ckpt["agents"]):
        agent.load_state_dict(state)
    agents = [a.eval() for a in agents]
    return agents, hidden_dim


def _record_positions(
    env: GridWorld3DEnv, visit_count: np.ndarray, traj: List[np.ndarray]
) -> None:
    pos = np.asarray(env.uav_positions, dtype=np.int32)
    traj.append(pos.copy())
    for l, r, c in env.uav_positions:
        visit_count[l, r, c] += 1


def _subsample_traj(traj: List[np.ndarray], max_points: int) -> np.ndarray:
    """traj: length T+1 list of (N,3); return (T', N, 3)."""
    arr = np.stack(traj, axis=0)
    t = arr.shape[0]
    if t <= max_points:
        return arr
    idx = np.unique(
        np.linspace(0, t - 1, num=max_points, dtype=np.int64)
    )
    return arr[idx]


def plot_trajectory_3d(
    traj_sn: np.ndarray,
    out_path: Path,
    title: str,
    dpi: int,
) -> None:
    """traj_sn: (T, N, 3) with (layer, row, col). Plot x=col, y=row, z=layer."""
    t, n, _ = traj_sn.shape
    fig = plt.figure(figsize=(7.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    try:
        base_cmap = matplotlib.colormaps["tab10"]
    except AttributeError:
        base_cmap = plt.cm.get_cmap("tab10")
    for i in range(n):
        pts = traj_sn[:, i, :]
        l, r, c = pts[:, 0], pts[:, 1], pts[:, 2]
        color = base_cmap(i % 10)
        ax.plot(c.astype(float), r.astype(float), l.astype(float), color=color, lw=1.6, label=f"UAV {i}")
        ax.scatter(c[0], r[0], l[0], color=color, s=36, marker="o", edgecolors="k", linewidths=0.5, zorder=5)
        ax.scatter(c[-1], r[-1], l[-1], color=color, s=40, marker="s", edgecolors="k", linewidths=0.5, zorder=5)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_zlabel("Layer")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=22, azim=-55)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_layer_occupancy(
    visit_count: np.ndarray,
    obstacle_map: np.ndarray,
    visited_final: np.ndarray,
    out_path: Path,
    title_prefix: str,
    dpi: int,
) -> None:
    l_dim, h_dim, w_dim = visit_count.shape
    ncols = min(4, l_dim)
    nrows = int(np.ceil(l_dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False)
    vmax = float(np.nanmax(np.where(obstacle_map, np.nan, visit_count))) or 1.0
    for layer in range(l_dim):
        r, c = divmod(layer, ncols)
        ax = axes[r][c]
        z = visit_count[layer].astype(np.float64)
        z = np.ma.array(z, mask=obstacle_map[layer])
        im = ax.imshow(z, origin="upper", cmap="magma", vmin=0.0, vmax=max(vmax, 1.0), interpolation="nearest")
        miss = np.logical_and(~obstacle_map[layer], ~visited_final[layer])
        if miss.any():
            miss_r, miss_c = np.where(miss)
            ax.scatter(miss_c, miss_r, s=8, c="cyan", marker="x", linewidths=0.6, alpha=0.85, label="unvisited")
        ax.set_title(f"{title_prefix} layer {layer}")
        ax.set_xticks(range(w_dim))
        ax.set_yticks(range(h_dim))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for layer in range(l_dim, nrows * ncols):
        r, c = divmod(layer, ncols)
        axes[r][c].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="3D trajectory + layer occupancy from one rollout")
    p.add_argument("--base-config", default="configs/base.yaml")
    p.add_argument("--env-config", type=str, required=True)
    p.add_argument("--policy", choices=("random", "greedy", "qmix"), default="greedy")
    p.add_argument("--checkpoint", type=str, default=None, help="Required when --policy qmix")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episode", type=int, default=1, help="Episode index; reset seed = seed + episode")
    p.add_argument("--traj-max-points", type=int, default=1200, help="Subsample length for 3D line clarity")
    p.add_argument("--out-dir", type=str, default="experiments/figures")
    p.add_argument("--tag", type=str, default=None, help="Filename tag (default from env + policy)")
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args()

    if args.policy == "qmix" and not args.checkpoint:
        raise SystemExit("--checkpoint is required for --policy qmix")

    base_cfg = load_config(args.base_config)
    env_cfg = load_config(args.env_config)
    device = torch.device(base_cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    set_seed(args.seed)
    ckpt: dict | None = None

    if args.policy == "qmix":
        ckpt_path = Path(args.checkpoint).resolve()
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        map_shape = tuple(int(x) for x in ckpt["map_shape"])
        num_uavs = int(ckpt["num_uavs"])
        obstacle_density = float(ckpt["obstacle_density"])
    else:
        map_shape = tuple(int(x) for x in env_cfg["map_shape"])
        num_uavs = int(env_cfg["num_uavs"][0]) if isinstance(env_cfg["num_uavs"], list) else int(env_cfg["num_uavs"])
        obstacle_density = float(env_cfg["obstacle_density"])

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
    agents: List[AgentNetwork] = []
    hidden_dim = 64
    if args.policy == "qmix":
        assert ckpt is not None
        if "obs_dim" in ckpt and int(ckpt["obs_dim"]) != obs_dim:
            raise ValueError(f"Checkpoint obs_dim {ckpt['obs_dim']} != env {obs_dim}")
        agents, hidden_dim = _load_qmix_agents(Path(args.checkpoint), device, obs_dim, action_dim)

    rng_ep = np.random.default_rng(
        int(np.random.SeedSequence([args.seed, args.episode]).generate_state(1)[0])
    )

    reset_seed = args.seed + args.episode
    obs, info = env.reset(seed=reset_seed)
    visit_count = np.zeros(env.obstacle_map.shape, dtype=np.float32)
    traj: List[np.ndarray] = []
    _record_positions(env, visit_count, traj)

    hidden = torch.zeros(len(agents), 1, hidden_dim, device=device) if agents else None
    terminated = truncated = False
    step_info: dict = {}
    while not (terminated or truncated):
        if args.policy == "random":
            actions = random_actions(env, rng_ep)
        elif args.policy == "greedy":
            actions = greedy_nearest_unvisited_actions(env)
        else:
            agent_obs = np.tile(obs, (num_uavs, 1)).reshape(1, num_uavs, obs_dim)
            assert hidden is not None
            act_b, hidden = _select_greedy_qmix(agents, agent_obs, hidden, device)
            actions = act_b[0]
        obs, _r, terminated, truncated, step_info = env.step(actions)
        _record_positions(env, visit_count, traj)

    visited_final = env.visited_map.copy()
    cov = float(step_info.get("coverage", 0.0))

    env_tag = Path(args.env_config).stem
    pol = args.policy
    tag = args.tag or f"{env_tag}_{pol}_seed{args.seed}_ep{args.episode}"
    out_dir = Path(args.out_dir)
    traj_path = out_dir / f"rollout_3d_trajectory_{tag}.png"
    heat_path = out_dir / f"rollout_layer_occupancy_{tag}.png"

    traj_sn = _subsample_traj(traj, args.traj_max_points)
    title_3d = f"3D trajectories ({pol}) | {map_shape} | cov={cov:.3f} | steps={len(traj)-1}"
    plot_trajectory_3d(traj_sn, traj_path, title_3d, args.dpi)

    title_h = f"Visit count / step"
    plot_layer_occupancy(visit_count, env.obstacle_map, visited_final, heat_path, title_h, args.dpi)

    print(f"Saved: {traj_path.resolve()}")
    print(f"Saved: {heat_path.resolve()}")
    print(f"coverage={cov:.4f} steps={len(traj)-1} success={terminated and cov >= 0.99}")


if __name__ == "__main__":
    main()
