# uav-qmix-3d

三维分层栅格多机协同覆盖环境（**L×H×W 体素**），与桌面 `uav-qmix` 中二维 `GridWorldEnv` 的奖励、能量、碰撞与势函数 shaping 思路对齐，便于后续接 QMIX / 其他 MARL。

## 环境说明

- **状态**：每层为 H×W 平面，共 L 层；障碍与访问标记均为三维布尔场。
- **动作**（每机 6 向）：北/南/西/东、层上/层下（`Action3D`）。
- **观测**：展平的 visited / obstacle / UAV 占用三通道体素 + 覆盖率与各机归一化位置、能量。
- **初始位置**：默认可行时优先在 **第 0 层** 随机选取，便于对应“地面起降”叙事。

## 依赖与运行

```bash
cd C:\Users\44358\Desktop\uav-qmix-3d
pip install -r requirements.txt
python demo_rollout.py
```

若暂时无法安装 **gymnasium**，环境类内置 **极简 shim**（仅用于本地启发式/调试）；完整 Gymnasium API 与上游兼容仍以安装官方包为准。

在代码中：

```python
from src.envs.grid_world_3d import GridWorld3DEnv

env = GridWorld3DEnv(map_shape=(4, 8, 8), num_uavs=3, obstacle_density=0.05, seed=42)
obs, info = env.reset(seed=42)
```

## QMIX 训练（已接入）

```bash
cd C:\Users\44358\Desktop\uav-qmix-3d
pip install -r requirements.txt
python train_qmix_3d.py
```

### MAPPO（已接入）

```bash
python train_mappo_3d.py --env-config configs/envs/grid_3d_medium.yaml --algo-config configs/algos/mappo.yaml --seed 0
python eval_mappo_3d.py --checkpoint experiments/checkpoints/<your>.pt --env-config configs/envs/grid_3d_medium.yaml --episodes 50 --seed 0 --stochastic --output-json experiments/results_eval/mappo_medium_s0.json
```

超参见 `configs/algos/mappo.yaml`；可用 `--num-updates`、`--rollout-steps` 覆盖以便快速试跑。

默认环境配置为 **`configs/envs/grid_3d_tiny.yaml`**（`[2,5,5]`、2 机、低密度障碍）。放大实验时可改 `map_shape`（如 `[3,6,6]`）并增大 `configs/algos/qmix.yaml` 中的 `episodes`。

检查点保存在 `experiments/checkpoints/`，日志在 `experiments/logs/qmix3d.log`。

### 论文质量：基线、曲线与汇总

- **QMIX / VDN 基线**：在算法 YAML 中设 `mixer_type: qmix`（默认）或 `mixer_type: vdn`（`configs/algos/vdn.yaml`）。VDN 使用 \(Q_{\mathrm{tot}}=\sum_i Q_i\)，与 QMIX 共用同一套网络与训练循环，便于公平对比。
- **较长训练**：`configs/algos/qmix_paper.yaml` 与 `vdn.yaml`（更大 buffer、`episodes: 2000` 等）。
- **多地图**：`configs/envs/grid_3d_medium.yaml`（3×6×6）、`grid_3d_hard.yaml`（更难设置示例）。
- **学习曲线**：训练时加 `--episode-log experiments/curves/run1.csv` 得到每 episode 覆盖率与步数。
- **批量实验**：`python run_experiment_matrix.py --env-config configs/envs/grid_3d_medium.yaml --methods qmix,vdn --seeds 0,1,2 --results experiments/results/runs.jsonl`，再用 `python experiments/summarize_runs.py experiments/results/runs.jsonl` 按方法汇总均值±标准差。
- **测试集式评估**：`python eval_3d.py --checkpoint experiments/checkpoints/xxx.pt --env-config configs/envs/grid_3d_medium.yaml --episodes 50 --seed 0 --output-json experiments/eval.json`
- **提速 / 省内存**：`qmix_paper.yaml` 与 `vdn.yaml` 默认使用较小 `buffer_size`、`batch_size`，并设 `bptt_detach_interval: 64`（周期性对 GRU hidden `detach`，缩短反传链）。若内存仍紧张可再减 `buffer_size`；若要更贴近完整 BPTT 可改为 `null` 或更大间隔（会更慢）。
- **学习曲线出图**：`pip install matplotlib` 后执行 `python experiments/plot_learning_curves.py`，默认读取 `experiments/curves/qmix_medium_s0.csv` 与 `vdn_medium_episode.csv`，输出 `experiments/figures/learning_curve_medium_qmix_vdn.png`。
- **五方法 × 两档主表**：`experiments/aggregate_dual_spec.json` + `python experiments/aggregate_eval_table.py --spec experiments/aggregate_dual_spec.json --skip-missing`；QMIX/VDN 可先 `python experiments/materialize_eval_from_training_jsonl.py --tier both` 与正文表 1/3 对齐，流程见 `experiments/DUAL_TABLE_README.md`。

## 论文（中文初稿与实验流程）

本仓库 **`paper/`** 目录内：

- `paper/IoTML_实验与写作流程.md`：先跑实验、再写中文、投稿前英译的步骤与命令（均在仓库根目录执行）。
- `paper/IoTML_论文初稿_中文.md`：会议论文中文模板（`【待填】` 占位）。

## 与二维工程的关系

本仓库已自带与 `uav-qmix` 对齐的 QMIX 实现；环境为 **6 向动作** 与 **三维体素观测**，无需再手动从二维工程改 import。
