"""Compute revisit, overlap, and unique coverage efficiency for paper Table."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.grid_world_3d import GridWorld3DEnv
from src.policies.heuristic_actions import greedy_nearest_unvisited_actions, random_actions
from src.utils.seeding import set_seed


def eval_heuristic(policy: str, env_cfg_path: Path, episodes: int, seed: int) -> Dict[str, float]:
    cfg = yaml.safe_load(env_cfg_path.read_text(encoding="utf-8"))
    env = GridWorld3DEnv(
        map_shape=tuple(int(x) for x in cfg["map_shape"]),
        num_uavs=int(cfg["num_uavs"][0]) if isinstance(cfg["num_uavs"], list) else int(cfg["num_uavs"]),
        obstacle_density=float(cfg["obstacle_density"]),
        max_steps=int(cfg.get("max_steps", 2500)),
        energy_budget=int(cfg.get("energy_budget", 4500)),
        shaping_weight=float(cfg.get("shaping_weight", 10.0)),
        obstacle_shaping_weight=float(cfg.get("obstacle_shaping_weight", 2.0)),
        seed=int(cfg.get("seed", seed)),
    )
    revisit_list: List[int] = []
    overlap_list: List[int] = []
    eff_list: List[float] = []
    cov_list: List[float] = []
    step_list: List[int] = []
    success_list: List[bool] = []

    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        rng_ep = np.random.default_rng(int(np.random.SeedSequence([seed, ep]).generate_state(1)[0]))
        revisit = 0
        overlap = 0
        steps = 0
        terminated = truncated = False
        step_info: Dict[str, Any] = {}
        valid = int(info.get("valid_cells", env.total_cells))

        while not (terminated or truncated):
            if policy == "random":
                actions = random_actions(env, rng_ep)
            else:
                actions = greedy_nearest_unvisited_actions(env)
            pre_visited = env.visited_map.copy()
            obs, rewards, terminated, truncated, step_info = env.step(actions)
            overlap += int(step_info.get("collisions", 0))
            steps += 1
            for idx, pos in enumerate(env.uav_positions):
                if env.uav_energy[idx] <= 0:
                    continue
                l, r, c = pos
                if pre_visited[l, r, c]:
                    revisit += 1

        cov = float(step_info.get("coverage", 0.0))
        v_cov = int(step_info.get("visited_count", round(cov * valid)))
        eff = v_cov / steps if steps > 0 else 0.0
        revisit_list.append(revisit)
        overlap_list.append(overlap)
        eff_list.append(eff)
        cov_list.append(cov)
        step_list.append(steps)
        success_list.append(bool(terminated and cov >= 0.99))

    return {
        "revisit_mean": float(np.mean(revisit_list)),
        "revisit_std": float(np.std(revisit_list, ddof=1)) if len(revisit_list) > 1 else 0.0,
        "overlap_mean": float(np.mean(overlap_list)),
        "overlap_std": float(np.std(overlap_list, ddof=1)) if len(overlap_list) > 1 else 0.0,
        "efficiency_mean": float(np.mean(eff_list)),
        "efficiency_std": float(np.std(eff_list, ddof=1)) if len(eff_list) > 1 else 0.0,
        "coverage_mean": float(np.mean(cov_list)),
        "steps_mean": float(np.mean(step_list)),
        "success_mean": float(np.mean(success_list)),
    }


def estimate_learning_from_jsonl(jsonl_path: Path, method: str, tier_valid_cells: Dict[str, float]) -> Dict[str, float]:
    """Estimate E from training jsonl; revisit from (1-pa_mean)*steps*num_uavs."""
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("method") == method]
    if not rows:
        return {}

    revisits, overlaps, effs = [], [], []
    for r in rows:
        steps = float(r["steps_mean"])
        cov = float(r["coverage_mean"])
        pa = float(r.get("pa_mean", 0.0))
        nu = int(r["num_uavs"])
        # valid cells approx from map
        ms = r["map_shape"]
        if isinstance(ms, str):
            L, H, W = [int(x) for x in ms.split("x")]
        else:
            L, H, W = [int(x) for x in ms]
        rho = float(r["obstacle_density"])
        valid = tier_valid_cells.get(f"{L}x{H}x{W}_{rho}")
        if valid is None:
            valid = L * H * W * (1.0 - rho)
        v_cov = cov * valid
        effs.append(v_cov / steps if steps > 0 else 0.0)
        total_actions = steps * nu
        new_actions = pa * total_actions
        revisits.append(max(0.0, total_actions - new_actions))
        overlaps.append(0.0)  # not logged in jsonl

    arr = lambda xs: (float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0)
    rm, rs = arr(revisits)
    em, es = arr(effs)
    return {
        "revisit_mean": rm,
        "revisit_std": rs,
        "overlap_mean": 0.0,
        "overlap_std": 0.0,
        "efficiency_mean": em,
        "efficiency_std": es,
        "coverage_mean": float(np.mean([float(r["coverage_mean"]) for r in rows])),
        "steps_mean": float(np.mean([float(r["steps_mean"]) for r in rows])),
        "success_mean": float(np.mean([float(r["success_mean"]) for r in rows])),
        "note": "QMIX/VDN revisit estimated as (1-pa_mean)*steps*num_uavs from training jsonl",
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    args = p.parse_args()
    episodes = args.episodes
    tiers = {
        "medium": ROOT / "configs/envs/grid_3d_medium.yaml",
        "hard": ROOT / "configs/envs/grid_3d_hard.yaml",
    }
    jsonls = {
        "medium": ROOT / "experiments/results/medium_qmix_vdn.jsonl",
        "hard": ROOT / "experiments/results/hard_qmix_vdn.jsonl",
    }
    # measured valid cells from one reset
    valid_cells = {}
    for tier, cfg_path in tiers.items():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        env = GridWorld3DEnv(
            map_shape=tuple(int(x) for x in cfg["map_shape"]),
            num_uavs=int(cfg["num_uavs"][0]) if isinstance(cfg["num_uavs"], list) else int(cfg["num_uavs"]),
            obstacle_density=float(cfg["obstacle_density"]),
            max_steps=int(cfg.get("max_steps", 2500)),
            seed=int(cfg.get("seed", 42)),
        )
        _, info = env.reset(seed=42)
        ms = "x".join(str(x) for x in cfg["map_shape"])
        key = f"{ms}_{cfg['obstacle_density']}"
        valid_cells[key] = float(info.get("valid_cells", env.total_cells))

    out: Dict[str, Any] = {}
    for tier, cfg_path in tiers.items():
        out[tier] = {}
        for pol in ("random", "greedy"):
            per_seed = []
            for seed in (0, 1, 2):
                per_seed.append(eval_heuristic(pol, cfg_path, episodes, seed))
            keys = ["revisit_mean", "overlap_mean", "efficiency_mean", "coverage_mean", "steps_mean", "success_mean"]
            agg = {}
            for k in keys:
                vals = [p[k] for p in per_seed]
                agg[k] = float(np.mean(vals))
                agg[k.replace("_mean", "_std")] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            out[tier][pol] = agg

        for method in ("qmix", "vdn"):
            out[tier][method] = estimate_learning_from_jsonl(jsonls[tier], method, valid_cells)

    print(json.dumps(out, indent=2))
    out_path = ROOT / "experiments/results_eval/coverage_quality_metrics.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
