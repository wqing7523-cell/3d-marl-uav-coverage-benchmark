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
| `experiments/` | Eval JSON, checkpoints, aggregation specs, figure scripts |
| `paper/` | Chinese manuscript PDF/Markdown and figure assets |

## Citation

If you use this benchmark, please cite the associated paper (author / title — to be filled when published).

## License

Research and reproducibility use; confirm with authors before redistribution.
