# 3D MARL Multi-UAV Cooperative Coverage Benchmark (Reproducible)

Reproducible **GridWorld3DEnv** voxel benchmark and implementations for **QMIX + PBRS** multi-UAV 3D cooperative coverage (Dec-POMDP / CTDE), with Medium/Hard tiers, baselines (VDN, MAPPO, Greedy, Random), and paper-aligned evaluation artifacts.

**中文说明**：见 [README_复现说明.md](README_复现说明.md)

## Quick start

```bash
cd code
pip install -r requirements.txt
python demo_rollout.py
python train_qmix_3d.py --env-config configs/envs/grid_3d_medium.yaml --algo-config configs/algos/qmix_paper.yaml --seed 0
```

## Layout

| Path | Description |
|------|-------------|
| `code/` | Environment, algorithms, train/eval entry scripts |
| `experiments/` | Eval JSON, checkpoints, aggregation specs, figure/stats scripts |
| `paper/` | Chinese manuscript PDF/Markdown, appendix stats tables, figure assets |

## Reproduce paper tables

| Paper element | Path / command |
|---------------|----------------|
| Tables 4–5 (QMIX/VDN) | `experiments/results/medium_qmix_vdn.jsonl`, `hard_qmix_vdn.jsonl` |
| Table 7 (coverage quality) | `python experiments/compute_coverage_quality_metrics.py` (from repo root) |
| Tables 12–14 (bootstrap CI) | `python experiments/stats_bootstrap_ci.py` → `paper/appendix_stats_table_d.md` |
| Table 15 ($n=5$ Hard) | `paper/appendix_stats_table_e_n5_hard.md` |
| Dual-tier aggregation | See `experiments/DUAL_TABLE_README.md` |

Run aggregation and batch scripts from the repository root; train/eval entry points live under `code/`.

## Citation

If you use this benchmark, please cite the associated paper (author / title — to be filled when published).

## License

Research and reproducibility use; confirm with authors before redistribution.
