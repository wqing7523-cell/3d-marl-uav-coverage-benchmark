"""Smoke test: random policy rollout for GridWorld3DEnv."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from src.envs.grid_world_3d import GridWorld3DEnv


def main() -> None:
    env = GridWorld3DEnv(
        map_shape=(3, 6, 6),
        num_uavs=2,
        obstacle_density=0.06,
        seed=0,
    )
    obs, info = env.reset(seed=0)
    print("obs shape:", obs.shape, "info:", {k: info[k] for k in ("coverage", "map_shape")})
    total_r = 0.0
    for t in range(50):
        actions = env.action_space.sample()
        obs, rewards, term, trunc, info = env.step(actions)
        total_r += float(rewards.sum())
        if term or trunc:
            print(f"done at t={t} term={term} trunc={trunc} coverage={info['coverage']:.3f}")
            break
    else:
        print(f"50 steps, coverage={info['coverage']:.3f} sum_reward={total_r:.1f}")
    env.close()


if __name__ == "__main__":
    main()
