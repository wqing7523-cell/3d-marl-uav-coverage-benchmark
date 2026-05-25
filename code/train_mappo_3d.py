"""Train MAPPO-style policy (shared encoder + per-agent heads) on GridWorld3DEnv."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam

from src.algos.mappo.networks import CentralCritic, MultiAgentActor
from src.envs.grid_world_3d import GridWorld3DEnv
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.seeding import set_seed


def ensure_map_shape(map_shape_entry: Any) -> Tuple[int, int, int]:
    if isinstance(map_shape_entry, (list, tuple)):
        if len(map_shape_entry) == 3:
            return int(map_shape_entry[0]), int(map_shape_entry[1]), int(map_shape_entry[2])
        if len(map_shape_entry) == 2:
            return 2, int(map_shape_entry[0]), int(map_shape_entry[1])
        if len(map_shape_entry) == 1:
            s = int(map_shape_entry[0])
            return 2, s, s
    s = int(map_shape_entry)
    return 2, s, s


def normalize_map_shapes(cfg_value: Any) -> List[Any]:
    if cfg_value is None:
        return [[2, 5, 5]]
    if isinstance(cfg_value, int):
        return [[2, cfg_value, cfg_value]]
    if isinstance(cfg_value, (list, tuple)) and len(cfg_value) > 0:
        if isinstance(cfg_value[0], (list, tuple)):
            return list(cfg_value)
        return [list(cfg_value) if isinstance(cfg_value, tuple) else cfg_value]
    return [[2, 5, 5]]


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """GAE-Lambda; next_values[t]=V(s_{t+1}) with 0 after terminal."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        nonterminal = 1.0 - float(dones[t])
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns


def collect_rollout(
    env: GridWorld3DEnv,
    actor: MultiAgentActor,
    critic: CentralCritic,
    rollout_steps: int,
    device: torch.device,
    reset_seed_counter: List[int],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    """Rollout buffers + rollout-level coverage/steps stats."""
    obs_dim = env.observation_space.shape[0]
    n_agents = env.num_uavs

    obs_buf = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
    act_buf = np.zeros((rollout_steps, n_agents), dtype=np.int64)
    logp_buf = np.zeros((rollout_steps,), dtype=np.float32)
    rew_buf = np.zeros((rollout_steps,), dtype=np.float32)
    val_buf = np.zeros((rollout_steps,), dtype=np.float32)
    next_val_buf = np.zeros((rollout_steps,), dtype=np.float32)
    done_buf = np.zeros((rollout_steps,), dtype=np.float32)

    episode_coverages: List[float] = []
    episode_steps_sum = 0
    episode_count = 0

    obs, info = env.reset(seed=reset_seed_counter[0])
    reset_seed_counter[0] += 1

    with torch.no_grad():
        for t in range(rollout_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = actor(obs_t)
            val = critic(obs_t)

            actions = torch.zeros(n_agents, dtype=torch.long, device=device)
            log_probs = []
            for i in range(n_agents):
                dist = Categorical(logits=logits[0, i])
                actions[i] = dist.sample()
                log_probs.append(dist.log_prob(actions[i]))
            log_prob_sum = torch.stack(log_probs).sum()

            act_np = actions.cpu().numpy()
            next_obs, rewards, terminated, truncated, info = env.step(act_np)
            r_team = float(np.sum(rewards))
            done = bool(terminated or truncated)

            obs_buf[t] = obs
            act_buf[t] = act_np
            logp_buf[t] = log_prob_sum.cpu().numpy()
            rew_buf[t] = r_team
            val_buf[t] = val.cpu().numpy().item()
            done_buf[t] = float(done)

            if done:
                next_val_buf[t] = 0.0
                episode_coverages.append(float(info.get("coverage", 0.0)))
                episode_steps_sum += int(info.get("steps", env.step_count))
                episode_count += 1
                obs, info = env.reset(seed=reset_seed_counter[0])
                reset_seed_counter[0] += 1
            else:
                nv_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                next_val_buf[t] = critic(nv_t).cpu().numpy().item()
                obs = next_obs

    cov_mean = float(np.mean(episode_coverages)) if episode_coverages else 0.0
    steps_mean = float(episode_steps_sum / max(1, episode_count))
    return (
        obs_buf,
        act_buf,
        logp_buf,
        rew_buf,
        val_buf,
        next_val_buf,
        done_buf,
        cov_mean,
        steps_mean,
    )


def ppo_update(
    actor: MultiAgentActor,
    critic: CentralCritic,
    optimizer: Adam,
    obs: np.ndarray,
    actions: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
    num_agents: int,
    device: torch.device,
    clip_range: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    ppo_epochs: int,
    num_minibatches: int,
) -> Tuple[float, float, float]:
    T = obs.shape[0]
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.tensor(actions, dtype=torch.long, device=device)
    old_log_t = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
    adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    ret_t = torch.tensor(returns, dtype=torch.float32, device=device)

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

    batch_size = T
    minibatch_size = max(1, batch_size // num_minibatches)

    total_pi = 0.0
    total_v = 0.0
    total_ent = 0.0
    n_updates = 0

    for _ in range(ppo_epochs):
        idx = np.arange(batch_size)
        np.random.shuffle(idx)
        for start in range(0, batch_size, minibatch_size):
            mb = idx[start : start + minibatch_size]
            if len(mb) == 0:
                continue

            ob = obs_t[mb]
            ac = act_t[mb]
            old_lp = old_log_t[mb]
            adv = adv_t[mb]
            ret = ret_t[mb]

            logits = actor(ob)
            values_new = critic(ob)

            new_lp = []
            ent_terms = []
            for i in range(num_agents):
                dist = Categorical(logits=logits[:, i])
                new_lp.append(dist.log_prob(ac[:, i]))
                ent_terms.append(dist.entropy())
            new_log_probs = torch.stack(new_lp, dim=1).sum(dim=1)
            entropy = torch.stack(ent_terms, dim=1).sum(dim=1).mean()

            ratio = torch.exp(new_log_probs - old_lp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values_new, ret)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()),
                max_grad_norm,
            )
            optimizer.step()

            total_pi += float(policy_loss.item())
            total_v += float(value_loss.item())
            total_ent += float(entropy.item())
            n_updates += 1

    if n_updates == 0:
        return 0.0, 0.0, 0.0
    return total_pi / n_updates, total_v / n_updates, total_ent / n_updates


def train_one(
    env_cfg: Dict[str, Any],
    algo_cfg: Dict[str, Any],
    base_cfg: Dict[str, Any],
    map_shape: Tuple[int, int, int],
    num_uavs: int,
    obstacle_density: float,
    logger: Any,
    seed: int,
    num_updates_override: Optional[int] = None,
    rollout_steps_override: Optional[int] = None,
) -> Path:
    device = torch.device(base_cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    env = GridWorld3DEnv(
        map_shape=map_shape,
        num_uavs=num_uavs,
        obstacle_density=obstacle_density,
        max_steps=env_cfg.get("max_steps", 2500),
        energy_budget=env_cfg.get("energy_budget", 4500),
        shaping_weight=env_cfg.get("shaping_weight", 10.0),
        obstacle_shaping_weight=env_cfg.get("obstacle_shaping_weight", 2.0),
        seed=env_cfg.get("seed", seed),
    )

    obs_dim = int(env.observation_space.shape[0])
    n_act = int(env.action_space.nvec[0])
    hidden_dim = int(algo_cfg.get("hidden_dim", 128))

    actor = MultiAgentActor(obs_dim, num_uavs, n_act, hidden_dim).to(device)
    critic = CentralCritic(obs_dim, hidden_dim).to(device)
    lr = float(algo_cfg.get("learning_rate", 3e-4))
    optimizer = Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    rollout_steps = int(
        rollout_steps_override
        if rollout_steps_override is not None
        else algo_cfg.get("rollout_steps", 2048)
    )
    num_updates = int(
        num_updates_override
        if num_updates_override is not None
        else algo_cfg.get("num_updates", 2000)
    )
    gamma = float(algo_cfg.get("gamma", 0.99))
    lam = float(algo_cfg.get("gae_lambda", 0.95))
    clip_range = float(algo_cfg.get("clip_range", 0.2))
    value_coef = float(algo_cfg.get("value_coef", 0.5))
    entropy_coef = float(algo_cfg.get("entropy_coef", 0.01))
    max_grad_norm = float(algo_cfg.get("max_grad_norm", 0.5))
    ppo_epochs = int(algo_cfg.get("ppo_epochs", 4))
    num_minibatches = int(algo_cfg.get("num_minibatches", 4))
    log_interval = int(algo_cfg.get("log_interval", 20))

    reset_seed_counter = [seed * 100_000 + 17]

    logger.info(
        "Training MAPPO map_shape=%s num_uavs=%s obstacle_density=%s rollout_steps=%s",
        map_shape,
        num_uavs,
        obstacle_density,
        rollout_steps,
    )

    for upd in range(1, num_updates + 1):
        (
            obs_buf,
            act_buf,
            logp_buf,
            rew_buf,
            val_buf,
            next_val_buf,
            done_buf,
            cov_roll,
            steps_roll,
        ) = collect_rollout(
            env,
            actor,
            critic,
            rollout_steps,
            device,
            reset_seed_counter,
        )

        advantages, returns = compute_gae(
            rew_buf, val_buf, next_val_buf, done_buf, gamma, lam
        )

        pi_l, v_l, ent = ppo_update(
            actor,
            critic,
            optimizer,
            obs_buf,
            act_buf,
            logp_buf,
            advantages,
            returns,
            num_uavs,
            device,
            clip_range,
            value_coef,
            entropy_coef,
            max_grad_norm,
            ppo_epochs,
            num_minibatches,
        )

        if upd == 1 or upd % log_interval == 0 or upd == num_updates:
            logger.info(
                "Upd %d | coverage_rollout_mean=%.3f | steps_rollout_mean=%.1f | pi_loss=%.4f | v_loss=%.4f | ent=%.4f",
                upd,
                cov_roll,
                steps_roll,
                pi_l,
                v_l,
                ent,
            )

    checkpoint_dir = Path(base_cfg.get("checkpoint_dir", "experiments/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    obs_density_str = f"obs{obstacle_density:.2f}".replace(".", "")
    L, H, W = map_shape
    ckpt_path = (
        checkpoint_dir
        / f"mappo3d_L{L}_H{H}_W{W}_uavs{num_uavs}_{obs_density_str}_{int(time.time())}.pt"
    )

    payload = {
        "algorithm": "mappo",
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "obs_dim": obs_dim,
        "num_agents": num_uavs,
        "num_actions": n_act,
        "hidden_dim": hidden_dim,
        "map_shape": map_shape,
        "num_uavs": num_uavs,
        "obstacle_density": obstacle_density,
    }
    torch.save(payload, ckpt_path)
    logger.info("MAPPO finished | checkpoint saved to %s", ckpt_path)
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAPPO on GridWorld3DEnv")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--env-config", default="configs/envs/grid_3d_medium.yaml")
    parser.add_argument("--algo-config", default="configs/algos/mappo.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-index", type=int, default=0)
    parser.add_argument("--uav-index", type=int, default=0)
    parser.add_argument("--obstacle-index", type=int, default=0)
    parser.add_argument(
        "--num-updates",
        type=int,
        default=None,
        help="Override mappo.yaml num_updates (for smoke tests)",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="Override mappo.yaml rollout_steps",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.base_config)
    env_cfg = load_config(args.env_config)
    algo_cfg = load_config(args.algo_config)
    env_cfg["seed"] = args.seed

    set_seed(args.seed)

    log_dir = Path(base_cfg.get("log_dir", "experiments/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("mappo3d", str(log_dir))

    map_shapes = normalize_map_shapes(env_cfg.get("map_shape"))
    num_uavs_list = env_cfg.get("num_uavs", [2])
    if not isinstance(num_uavs_list, list):
        num_uavs_list = [num_uavs_list]

    obstacle_density = env_cfg.get("obstacle_density", 0.08)
    if isinstance(obstacle_density, list):
        obstacle_density_list = obstacle_density
    else:
        obstacle_density_list = [obstacle_density]

    map_idx = max(0, min(args.map_index, len(map_shapes) - 1))
    uav_idx = max(0, min(args.uav_index, len(num_uavs_list) - 1))
    od_idx = (
        max(0, min(args.obstacle_index, len(obstacle_density_list) - 1))
        if len(obstacle_density_list) > 1
        else 0
    )

    ms = ensure_map_shape(map_shapes[map_idx])
    nu = int(num_uavs_list[uav_idx])
    od = float(obstacle_density_list[od_idx])

    train_one(
        env_cfg,
        algo_cfg,
        base_cfg,
        ms,
        nu,
        od,
        logger,
        args.seed,
        num_updates_override=args.num_updates,
        rollout_steps_override=args.rollout_steps,
    )


if __name__ == "__main__":
    main()
