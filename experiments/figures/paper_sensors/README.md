# Paper figures (*Sensors* layout)

Generate or refresh all figures:

```bash
cd /path/to/uav-qmix-3d
python experiments/generate_sensors_paper_figures.py --dpi 200 --tier medium --episodes-per-point 12
python experiments/generate_sensors_paper_figures.py --dpi 200 --tier hard --episodes-per-point 6
```

| File | Content |
|------|---------|
| `figure1_voxel_environment_{medium,hard}.png` | 3D scatter: obstacle voxels + subsampled traversable + UAV starts |
| `figure2_framework_ctde_pbrs_der.png` | CTDE + QMIX mixer + PBRS (+ optional DER) schematic (English) |
| `figure3_learning_curves_qmix_vdn.png` | Medium + hard side-by-side; runs `plot_learning_curves.py --lang en` then stitches `learning_curve_*_qmix_vdn_en.png` |
| `figure4_coverage_vs_obstacle_density_{medium,hard}.png` | Greedy vs Random final coverage vs ρ (heuristic sweep, not QMIX) |
| `figure5_rollout_3d_trajectory_hard_random.png` | From `../rollout_3d_trajectory_grid_3d_hard_random_seed0_ep1.png` with a stamped **Figure 5.** caption |
| `figure6_rollout_layer_occupancy_hard_random.png` | From `../rollout_layer_occupancy_grid_3d_hard_random_seed0_ep1.png` with a stamped **Figure 6.** caption |
| `figure7_ablation_potential_en.png` | PBRS ablation bar chart (English), from `plot_potential_ablation.py` if JSONL present |

**Note on Figure 4:** curves use **episode-end coverage** from **Greedy** and **Random** only (no checkpoint per ρ). For **QMIX vs ρ**, train/eval per density and replace this plot.
