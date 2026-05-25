# 可复现三维 MARL 协同覆盖基准与实现 — 备份说明

对应论文表述（摘要/结论）：

> **本文提供可复现的三维 MARL 协同覆盖基准与实现，供后续对比与 Sim-to-Real 研究参考。**

本备份包汇总支撑该表述所需的**环境、算法、配置、训练/评估管线、论文主表数据与图表**，便于归档、投稿附件或后续迁移研究。

---

## 1. 备份内容一览

| 目录 | 内容 | 对应论文要素 |
|------|------|----------------|
| `code/` | `GridWorld3DEnv`、QMIX/VDN/MAPPO、启发式策略及入口脚本 | 第3章环境、CTDE+QMIX+PBRS、训练/评估协议 |
| `code/configs/envs/` | `grid_3d_medium.yaml`、`grid_3d_hard.yaml` | 表1：Medium/Hard 两档 |
| `code/configs/algos/` | `qmix_paper.yaml`、`vdn.yaml`、`mappo.yaml`、`qmix_no_potential.yaml` | 主实验、消融 |
| `experiments/results_eval/` | 各方法 × 两档 × 3 种子评估 JSON | 表2、表3 原始数据 |
| `experiments/results/*.jsonl` | 训练日志 materialize 源 | 与正文表对齐的 QMIX/VDN 评估 |
| `experiments/checkpoints/` | 已训练权重（若存在） | 直接 `eval_3d.py` 复现 |
| `experiments/curves/` | 学习曲线、PBRS 消融 CSV | 图3、图7、表4 |
| `paper/` | `沁言学术.md` / `.pdf`、论文图 `figures/` | 文稿与插图 |
| `experiments/EXPERIMENT_DATA_BUNDLE.*` | 配置快照 + 指标汇总索引 | 一键查阅全部数值 |

---

## 2. 基准环境（GridWorld3DEnv）

- **实现**：`code/src/envs/grid_world_3d.py`
- **任务**：Dec-POMDP 多机三维体素协同覆盖；六邻域离散动作；全局体素观测向量（Visited / Obstacle / UAV 层 + 标量）。
- **奖励**：事件惩罚 + PBRS 势差（式(1)–(3)），见第3.3节。
- **两档配置**：
  - **Medium**：$3\times6\times6$，$N=2$，$\rho=0.08$，$T_{\max}=2500$，能量 4500
  - **Hard**：$4\times7\times7$，$N=3$，$\rho=0.12$，$T_{\max}=3500$，能量 6000

---

## 3. 算法与训练/评估协议

| 类别 | 方法 | 训练 | 独立评估 |
|------|------|------|----------|
| 学习类 | QMIX、VDN | 两档各 **2000 episodes** | `eval_3d.py`，$\epsilon=0$ 贪心 |
| 学习类 | MAPPO | 各档 **num_updates=2000** | `eval_mappo_3d.py --stochastic` |
| 说明性 | Greedy、Random | **不训练** | `eval_heuristic_3d.py` rollout |

**统计**：$n=3$ 随机种子，每种子 **50 episodes**，汇报 **mean ± std**。

**指标**（第4.1.2节）：coverage mean、success mean（终止式全覆盖比例）、steps mean。

---

## 4. 快速复现（在 `code/` 目录下）

```bash
pip install -r requirements.txt

# 环境冒烟
python demo_rollout.py

# QMIX 训练（论文主配置）
python train_qmix_3d.py --env-config configs/envs/grid_3d_medium.yaml --algo-config configs/algos/qmix_paper.yaml --seed 0

# 独立评估（50 episodes）
python eval_3d.py --checkpoint experiments/checkpoints/<your>.pt \
  --env-config configs/envs/grid_3d_medium.yaml --episodes 50 --seed 0 \
  --output-json experiments/results_eval/qmix_medium_s0.json

# 汇总表2/表3（需 JSON 齐全）
python experiments/aggregate_eval_table.py --spec experiments/aggregate_dual_spec.json
```

详细批处理流程见 `experiments/DUAL_TABLE_README.md`（本备份在 `experiments/batch_scripts/` 含 `.bat` 副本）。

---

## 5. 复现论文主表/主图

| 论文元素 | 备份内路径 / 命令 |
|----------|-------------------|
| 表2 / 表3 | `experiments/results_eval/*.json` + `aggregate_dual_spec.json` |
| 表4 PBRS 消融 | `configs/algos/qmix_no_potential.yaml` + `results/ablation_medium.jsonl` |
| 图1 环境示意 | `paper/figures/figure1_*` |
| 图2 架构 | `paper/figures/figure2_*` |
| 图3 训练曲线 | `experiments/plot_learning_curves.py` + `curves/*.csv` |
| 图4–5 Random 轨迹 | `experiments/plot_voxel_rollout_figures.py` |
| 图6 $\rho$ 扫参（启发式） | `experiments/generate_sensors_paper_figures.py` |
| 图7 消融 | `experiments/plot_potential_ablation.py` |

---

## 6. Sim-to-Real 参考（论文第5.2节方向）

本备份提供**离散体素仿真基准与 MARL 管线**，不含真实飞控接口。迁移研究可：

1. 以 `GridWorld3DEnv` 为教师环境预训练 QMIX+PBRS；
2. 将观测从体素向量替换为噪声点云/深度图，动作为连续控制时用蒸馏或微调；
3. 在真实机上做域随机化（障碍布局、感知噪声、动力学约束）。

相关讨论见 `paper/沁言学术.md` 第5.2节。

---

## 7. 重新生成本备份

在仓库根目录执行：

```bash
python tools/create_reproducibility_backup.py --out "C:\Users\44358\Desktop\lunwen\可复现基准与实现备份_3DMARL协同覆盖" --zip
```

---

## 8. 引用与许可

若公开发布代码，请与论文作者确认仓库许可；备份仅用于个人归档与复现，不等同于正式开源发布。
