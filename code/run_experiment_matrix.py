"""
Batch runs for paper-quality tables: multiple (method, seed) jobs with JSONL aggregate.

Example:
  python run_experiment_matrix.py \\
    --env-config configs/envs/grid_3d_medium.yaml \\
    --methods qmix,vdn \\
    --seeds 0,1,2 \\
    --results experiments/results/medium_runs.jsonl
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description="Run train_qmix_3d matrix (methods × seeds)")
    p.add_argument("--base-config", default="configs/base.yaml")
    p.add_argument("--env-config", required=True)
    p.add_argument(
        "--methods",
        default="qmix,vdn",
        help="Comma-separated: qmix uses configs/algos/qmix_paper.yaml, vdn uses vdn.yaml",
    )
    p.add_argument("--seeds", default="0,1,2", help="Comma-separated ints")
    p.add_argument(
        "--results",
        required=True,
        help="Append-only JSONL path (same as train --results-file)",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    algo_map = {
        "qmix": "configs/algos/qmix_paper.yaml",
        "vdn": "configs/algos/vdn.yaml",
    }
    for m in methods:
        if m not in algo_map:
            raise SystemExit(f"Unknown method {m}; choose from {list(algo_map)}")

    res_path = Path(args.results)
    res_path.parent.mkdir(parents=True, exist_ok=True)

    for m in methods:
        algo = algo_map[m]
        for seed in seeds:
            cmd = [
                sys.executable,
                str(ROOT / "train_qmix_3d.py"),
                "--base-config",
                str(ROOT / args.base_config),
                "--env-config",
                str(ROOT / args.env_config),
                "--algo-config",
                str(ROOT / algo),
                "--seed",
                str(seed),
                "--results-file",
                str(res_path.resolve()),
            ]
            print("RUN", " ".join(cmd))
            if args.dry_run:
                continue
            r = subprocess.run(cmd, cwd=str(ROOT))
            if r.returncode != 0:
                raise SystemExit(f"Failed seed={seed} method={m} code={r.returncode}")

    print("Done. Aggregate with experiments/summarize_runs.py or any JSONL tool.")


if __name__ == "__main__":
    main()
