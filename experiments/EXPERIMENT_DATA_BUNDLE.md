# 实验数据包（uav-qmix-3d）

本文件由 `experiments/export_experiment_data_bundle.py` 自动生成，便于归档与审阅；**非**论文正文。

- **生成时间（UTC）**：2026-05-15T03:03:26.990785+00:00
- **仓库根路径**：`C:\Users\44358\Desktop\uav-qmix-3d`

---

## 1. 双档五方法聚合（n=3 种子，均值 ± 样本标准差）

### Medium | grid_3d_medium | 3x6x6 | 2 UAV | obstacle_density 0.08

| 方法 | coverage_mean | success_mean | steps_mean |
|------|---------------|--------------|------------|
| QMIX | 0.8629 ± 0.0376 | 0.0993 ± 0.0385 | 2375.4 ± 50.7 |
| VDN | 0.7065 ± 0.0290 | 0.0918 ± 0.0387 | 2380.3 ± 49.8 |
| Greedy | 0.5427 ± 0.0081 | 0.1800 ± 0.0000 | 2060.9 ± 0.0 |
| Random | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 707.9 ± 6.6 |
| MAPPO | 0.6887 ± 0.0923 | 0.0533 ± 0.0503 | 2455.3 ± 38.8 |

<details><summary>逐方法 × 逐种子数值（展开）</summary>

- **QMIX**
  - **coverage_mean**：`qmix_medium_s0.json`:0.81987, `qmix_medium_s1.json`:0.87937, `qmix_medium_s2.json`:0.8894500000000001
  - **success_mean**：`qmix_medium_s0.json`:0.056, `qmix_medium_s1.json`:0.1295, `qmix_medium_s2.json`:0.1125
  - **steps_mean**：`qmix_medium_s0.json`:2432.462, `qmix_medium_s1.json`:2335.367, `qmix_medium_s2.json`:2358.5185
- **VDN**
  - **coverage_mean**：`vdn_medium_s0.json`:0.67595, `vdn_medium_s1.json`:0.7097650000000001, `vdn_medium_s2.json`:0.7336800000000001
  - **success_mean**：`vdn_medium_s0.json`:0.0495, `vdn_medium_s1.json`:0.1255, `vdn_medium_s2.json`:0.1005
  - **steps_mean**：`vdn_medium_s0.json`:2435.254, `vdn_medium_s1.json`:2338.1365, `vdn_medium_s2.json`:2367.553
- **Greedy**
  - **coverage_mean**：`greedy_medium_s0.json`:0.5489999999999999, `greedy_medium_s1.json`:0.5454, `greedy_medium_s2.json`:0.5336
  - **success_mean**：`greedy_medium_s0.json`:0.18, `greedy_medium_s1.json`:0.18, `greedy_medium_s2.json`:0.18
  - **steps_mean**：`greedy_medium_s0.json`:2060.88, `greedy_medium_s1.json`:2060.88, `greedy_medium_s2.json`:2060.88
- **Random**
  - **coverage_mean**：`random_medium_s0.json`:1.0, `random_medium_s1.json`:1.0, `random_medium_s2.json`:1.0
  - **success_mean**：`random_medium_s0.json`:1.0, `random_medium_s1.json`:1.0, `random_medium_s2.json`:1.0
  - **steps_mean**：`random_medium_s0.json`:712.56, `random_medium_s1.json`:700.3, `random_medium_s2.json`:710.72
- **MAPPO**
  - **coverage_mean**：`mappo_medium_s0.json`:0.7143999999999999, `mappo_medium_s1.json`:0.7654000000000001, `mappo_medium_s2.json`:0.5861999999999999
  - **success_mean**：`mappo_medium_s0.json`:0.06, `mappo_medium_s1.json`:0.1, `mappo_medium_s2.json`:0.0
  - **steps_mean**：`mappo_medium_s0.json`:2435.54, `mappo_medium_s1.json`:2430.22, `mappo_medium_s2.json`:2500.0

</details>

### Hard | grid_3d_hard | 4x7x7 | 3 UAV | obstacle_density 0.12

| 方法 | coverage_mean | success_mean | steps_mean |
|------|---------------|--------------|------------|
| QMIX | 0.9934 ± 0.0005 | 0.6013 ± 0.0095 | 2333.0 ± 6.3 |
| VDN | 0.9764 ± 0.0044 | 0.4652 ± 0.0593 | 2609.7 ± 119.6 |
| Greedy | 0.4416 ± 0.0008 | 0.0400 ± 0.0000 | 3365.0 ± 0.0 |
| Random | 0.9999 ± 0.0001 | 0.9867 ± 0.0115 | 1050.0 ± 31.4 |
| MAPPO | 0.9918 ± 0.0038 | 0.3800 ± 0.1908 | 3024.6 ± 315.0 |

<details><summary>逐方法 × 逐种子数值（展开）</summary>

- **QMIX**
  - **coverage_mean**：`qmix_hard_s0.json`:0.9934826589595375, `qmix_hard_s1.json`:0.9939017341040463, `qmix_hard_s2.json`:0.9928786127167629
  - **success_mean**：`qmix_hard_s0.json`:0.601, `qmix_hard_s1.json`:0.611, `qmix_hard_s2.json`:0.592
  - **steps_mean**：`qmix_hard_s0.json`:2330.455, `qmix_hard_s1.json`:2340.1545, `qmix_hard_s2.json`:2328.3815
- **VDN**
  - **coverage_mean**：`vdn_hard_s0.json`:0.9726473988439307, `vdn_hard_s1.json`:0.981271676300578, `vdn_hard_s2.json`:0.9753468208092486
  - **success_mean**：`vdn_hard_s0.json`:0.411, `vdn_hard_s1.json`:0.5285, `vdn_hard_s2.json`:0.456
  - **steps_mean**：`vdn_hard_s0.json`:2714.2845, `vdn_hard_s1.json`:2479.302, `vdn_hard_s2.json`:2635.4095
- **Greedy**
  - **coverage_mean**：`greedy_hard_s0.json`:0.44231213872832376, `greedy_hard_s1.json`:0.44080924855491327, `greedy_hard_s2.json`:0.44161849710982665
  - **success_mean**：`greedy_hard_s0.json`:0.04, `greedy_hard_s1.json`:0.04, `greedy_hard_s2.json`:0.04
  - **steps_mean**：`greedy_hard_s0.json`:3365.02, `greedy_hard_s1.json`:3365.02, `greedy_hard_s2.json`:3365.02
- **Random**
  - **coverage_mean**：`random_hard_s0.json`:0.9998843930635837, `random_hard_s1.json`:0.9998843930635837, `random_hard_s2.json`:1.0
  - **success_mean**：`random_hard_s0.json`:0.98, `random_hard_s1.json`:0.98, `random_hard_s2.json`:1.0
  - **steps_mean**：`random_hard_s0.json`:1053.36, `random_hard_s1.json`:1079.6, `random_hard_s2.json`:1017.1
- **MAPPO**
  - **coverage_mean**：`mappo_hard_s0.json`:0.9889017341040462, `mappo_hard_s1.json`:0.9902890173410406, `mappo_hard_s2.json`:0.9960693641618497
  - **success_mean**：`mappo_hard_s0.json`:0.26, `mappo_hard_s1.json`:0.28, `mappo_hard_s2.json`:0.6
  - **steps_mean**：`mappo_hard_s0.json`:3226.82, `mappo_hard_s1.json`:3185.36, `mappo_hard_s2.json`:2661.72

</details>

---

## 2. 每种子评估 JSON 摘要

| 文件 | coverage_mean | success_mean | steps_mean | 备注字段 |
|------|-----------------|--------------|------------|----------|
| `_smoke_mappo.json` | 1 | 1 | 361 | — |
| `greedy_hard_s0.json` | 0.442312 | 0.04 | 3365.02 | policy=greedy; mixer=heuristic |
| `greedy_hard_s1.json` | 0.440809 | 0.04 | 3365.02 | policy=greedy; mixer=heuristic |
| `greedy_hard_s2.json` | 0.441618 | 0.04 | 3365.02 | policy=greedy; mixer=heuristic |
| `greedy_medium_s0.json` | 0.549 | 0.18 | 2060.88 | policy=greedy; mixer=heuristic |
| `greedy_medium_s1.json` | 0.5454 | 0.18 | 2060.88 | policy=greedy; mixer=heuristic |
| `greedy_medium_s2.json` | 0.5336 | 0.18 | 2060.88 | policy=greedy; mixer=heuristic |
| `mappo_hard_s0.json` | 0.988902 | 0.26 | 3226.82 | — |
| `mappo_hard_s1.json` | 0.990289 | 0.28 | 3185.36 | — |
| `mappo_hard_s2.json` | 0.996069 | 0.6 | 2661.72 | — |
| `mappo_medium_s0.json` | 0.7144 | 0.06 | 2435.54 | — |
| `mappo_medium_s1.json` | 0.7654 | 0.1 | 2430.22 | — |
| `mappo_medium_s2.json` | 0.5862 | 0 | 2500 | — |
| `qmix_hard_s0.json` | 0.993483 | 0.601 | 2330.45 | mixer=qmix; 含 metrics_note; training_summary_jsonl |
| `qmix_hard_s1.json` | 0.993902 | 0.611 | 2340.15 | mixer=qmix; 含 metrics_note; training_summary_jsonl |
| `qmix_hard_s2.json` | 0.992879 | 0.592 | 2328.38 | mixer=qmix; 含 metrics_note; training_summary_jsonl |
| `qmix_medium_s0.json` | 0.81987 | 0.056 | 2432.46 | mixer=qmix; 含 metrics_note; training_summary_jsonl |
| `qmix_medium_s1.json` | 0.87937 | 0.1295 | 2335.37 | mixer=qmix; 含 metrics_note; training_summary_jsonl |
| `qmix_medium_s2.json` | 0.88945 | 0.1125 | 2358.52 | mixer=qmix; 含 metrics_note; training_summary_jsonl |
| `random_hard_s0.json` | 0.999884 | 0.98 | 1053.36 | policy=random; mixer=heuristic |
| `random_hard_s1.json` | 0.999884 | 0.98 | 1079.6 | policy=random; mixer=heuristic |
| `random_hard_s2.json` | 1 | 1 | 1017.1 | policy=random; mixer=heuristic |
| `random_medium_s0.json` | 1 | 1 | 712.56 | policy=random; mixer=heuristic |
| `random_medium_s1.json` | 1 | 1 | 700.3 | policy=random; mixer=heuristic |
| `random_medium_s2.json` | 1 | 1 | 710.72 | policy=random; mixer=heuristic |
| `vdn_hard_s0.json` | 0.972647 | 0.411 | 2714.28 | mixer=vdn; 含 metrics_note; training_summary_jsonl |
| `vdn_hard_s1.json` | 0.981272 | 0.5285 | 2479.3 | mixer=vdn; 含 metrics_note; training_summary_jsonl |
| `vdn_hard_s2.json` | 0.975347 | 0.456 | 2635.41 | mixer=vdn; 含 metrics_note; training_summary_jsonl |
| `vdn_medium_s0.json` | 0.67595 | 0.0495 | 2435.25 | mixer=vdn; 含 metrics_note; training_summary_jsonl |
| `vdn_medium_s1.json` | 0.709765 | 0.1255 | 2338.14 | mixer=vdn; 含 metrics_note; training_summary_jsonl |
| `vdn_medium_s2.json` | 0.73368 | 0.1005 | 2367.55 | mixer=vdn; 含 metrics_note; training_summary_jsonl |

<details><summary>各 JSON 全文（折叠）</summary>

#### `_smoke_mappo.json`

```json
{
  "checkpoint": "mappo3d_L2_H5_W5_uavs2_obs005_1778347853.pt",
  "algorithm": "mappo",
  "map_shape": [
    2,
    5,
    5
  ],
  "num_uavs": 2,
  "obstacle_density": 0.05,
  "episodes": 3,
  "seed": 0,
  "stochastic": true,
  "coverage_mean": 1.0,
  "coverage_std": 0.0,
  "steps_mean": 361.0,
  "success_mean": 1.0
}
```

#### `greedy_hard_s0.json`

```json
{
  "policy": "greedy",
  "checkpoint": null,
  "mixer_type": "heuristic",
  "map_shape": [
    4,
    7,
    7
  ],
  "num_uavs": 3,
  "obstacle_density": 0.12,
  "episodes": 50,
  "seed": 0,
  "coverage_mean": 0.44231213872832376,
  "coverage_std": 0.29378685756530476,
  "coverage_min": 0.03468208092485549,
  "coverage_max": 1.0,
  "steps_mean": 3365.02,
  "success_mean": 0.04
}
```

#### `greedy_hard_s1.json`

```json
{
  "policy": "greedy",
  "checkpoint": null,
  "mixer_type": "heuristic",
  "map_shape": [
    4,
    7,
    7
  ],
  "num_uavs": 3,
  "obstacle_density": 0.12,
  "episodes": 50,
  "seed": 1,
  "coverage_mean": 0.44080924855491327,
  "coverage_std": 0.2954384273063555,
  "coverage_min": 0.03468208092485549,
  "coverage_max": 1.0,
  "steps_mean": 3365.02,
  "success_mean": 0.04
}
```

#### `greedy_hard_s2.json`

```json
{
  "policy": "greedy",
  "checkpoint": null,
  "mixer_type": "heuristic",
  "map_shape": [
    4,
    7,
    7
  ],
  "num_uavs": 3,
  "obstacle_density": 0.12,
  "episodes": 50,
  "seed": 2,
  "coverage_mean": 0.44161849710982665,
  "coverage_std": 0.29447020741850793,
  "coverage_min": 0.03468208092485549,
  "coverage_max": 1.0,
  "steps_mean": 3365.02,
  "success_mean": 0.04
}
```

#### `greedy_medium_s0.json`

```json
{
  "policy": "greedy",
  "checkpoint": null,
  "mixer_type": "heuristic",
  "map_shape": [
    3,
    6,
    6
  ],
  "num_uavs": 2,
  "obstacle_density": 0.08,
  "episodes": 50,
  "seed": 0,
  "coverage_mean": 0.5489999999999999,
  "coverage_std": 0.35605196249985765,
  "steps_mean": 2060.88,
  "success_mean": 0.18
}
```

#### `greedy_medium_s1.json`

```json
{
  "policy": "greedy",
  "checkpoint": null,
  "mixer_type": "heuristic",
  "map_shape": [
    3,
    6,
    6
  ],
  "num_uavs": 2,
  "obstacle_density": 0.08,
  "episodes": 50,
  "seed": 1,
  "coverage_mean": 0.5454,
  "coverage_std": 0.36004560822206955,
  "steps_mean": 2060.88,
  "success_mean": 0.18
}
```

#### `greedy_medium_s2.json`

```json
{
  "policy": "greedy",
  "checkpoint": null,
  "mixer_type": "heuristic",
  "map_shape": [
    3,
    6,
    6
  ],
  "num_uavs": 2,
  "obstacle_density": 0.08,
  "episodes": 50,
  "seed": 2,
  "coverage_mean": 0.5336,
  "coverage_std": 0.3647506545573291,
  "steps_mean": 2060.88,
  "success_mean": 0.18
}
```

#### `mappo_hard_s0.json`

```json
{
  "checkpoint": "mappo3d_L4_H7_W7_uavs3_obs012_1778360796.pt",
  "algorithm": "mappo",
  "map_shape": [
    4,
    7,
    7
  ],
  "num_uavs": 3,
  "obstacle_density": 0.12,
  "episodes": 50,
  "seed": 0,
  "stochastic": true,
  "coverage_mean": 0.9889017341040462,
  "coverage_std": 0.01194953022836767,
  "steps_mean": 3226.82,
  "success_mean": 0.26
}
```

#### `mappo_hard_s1.json`

```json
{
  "checkpoint": "mappo3d_L4_H7_W7_uavs3_obs012_1778373399.pt",
  "algorithm": "mappo",
  "map_shape": [
    4,
    7,
    7
  ],
  "num_uavs": 3,
  "obstacle_density": 0.12,
  "episodes": 50,
  "seed": 1,
  "stochastic": true,
  "coverage_mean": 0.9902890173410406,
  "coverage_std": 0.009352022388974188,
  "steps_mean": 3185.36,
  "success_mean": 0.28
}
```

#### `mappo_hard_s2.json`

```json
{
  "checkpoint": "mappo3d_L4_H7_W7_uavs3_obs012_1778386065.pt",
  "algorithm": "mappo",
  "map_shape": [
    4,
    7,
    7
  ],
  "num_uavs": 3,
  "obstacle_density": 0.12,
  "episodes": 50,
  "seed": 2,
  "stochastic": true,
  "coverage_mean": 0.9960693641618497,
  "coverage_std": 0.006272675128901986,
  "steps_mean": 2661.72,
  "success_mean": 0.6
}
```

#### `mappo_medium_s0.json`

```json
{
  "checkpoint": "mappo3d_L3_H6_W6_uavs2_obs008_1778320633.pt",
  "algorithm": "mappo",
  "map_shape": [
    3,
    6,
    6
  ],
  "num_uavs": 2,
  "obstacle_density": 0.08,
  "episodes": 50,
  "seed": 0,
  "stochastic": true,
  "coverage_mean": 0.7143999999999999,
  "coverage_std": 0.2534494821458509,
  "steps_mean": 2435.54,
  "success_mean": 0.06
}
```

#### `mappo_medium_s1.json`

```json
{
  "checkpoint": "mappo3d_L3_H6_W6_uavs2_obs008_1778327797.pt",
  "algorithm": "mappo",
  "map_shape": [
    3,
    6,
    6
  ],
  "num_uavs": 2,
  "obstacle_density": 0.08,
  "episodes": 50,
  "seed": 1,
  "stochastic": true,
  "coverage_mean": 0.7654000000000001,
  "coverage_std": 0.243829530615141,
  "steps_mean": 2430.22,
  "success_mean": 0.1
}
```

#### `mappo_medium_s2.json`

```json
{
  "checkpoint": "mappo3d_L3_H6_W6_uavs2_obs008_1778336904.pt",
  "algorithm": "mappo",
  "map_shape": [
    3,
    6,
    6
  ],
  "num_uavs": 2,
  "obstacle_density": 0.08,
  "episodes": 50,
  "seed": 2,
  "stochastic": true,
  "coverage_mean": 0.5861999999999999,
  "coverage_std": 0.2283496441862785,
  "steps_mean": 2500.0,
  "success_mean": 0.0
}
```

#### `qmix_hard_s0.json`

```json
{
  "checkpoint": "qmix3d_L4_H7_W7_uavs3_obs012_1775082129.pt",
  "mixer_type": "qmix",
  "map_shape": [
    4,
    7,
    7
  ],
  "num_uavs": 3,
  "obstacle_density": 0.12,
  "episodes": null,
  "seed": 0,
  "coverage_mean": 0.9934826589595375,
  "coverage_std": 0.01218028086180411,
  "steps_mean": 2330.455,
  "success_mean": 0.601,
  "metrics_source": "training_summary_jsonl",
  "metrics_note": "Aligned with Table 3 / hard_qmix_vdn.jsonl. Replace with eval_3d.py --episodes 50 when checkpoints exist."
}
```

#### `qmix_hard_s1.json`

```json
{
  "checkpoint": "qmix3d_L4_H7_W7_uavs3_obs012_1775126118.pt",
  "mixer_type": "qmix",
  "map_shape": [
    4,
    7,
    7
  ],
  "num_uavs": 3,
  "obstacle_density": 0.12,
  "episodes": null,
  "seed": 1,
  "coverage_mean": 0.9939017341040463,
  "coverage_std": 0.011034218795629226,
  "steps_mean": 2340.1545,
  "success_mean": 0.611,
  "metrics_source": "training_summary_jsonl",
  "metrics_note": "Aligned with Table 3 / hard_qmix_vdn.jsonl. Replace with eval_3d.py --episodes 50 when checkpoints exist."
}
```

#### `qmix_hard_s2.json`

```json
{
  "checkpoint": "qmix3d_L4_H7_W7_uavs3_obs012_1775171227.pt",
  "mixer_type": "qmix",
  "map_shape": [
    4,
    7,
    7
  ],
  "num_uavs": 3,
  "obstacle_density": 0.12,
  "episodes": null,
  "seed": 2,
  "coverage_mean": 0.9928786127167629,
  "coverage_std": 0.012467574301679165,
  "steps_mean": 2328.3815,
  "success_mean": 0.592,
  "metrics_source": "training_summary_jsonl",
  "metrics_note": "Aligned with Table 3 / hard_qmix_vdn.jsonl. Replace with eval_3d.py --episodes 50 when checkpoints exist."
}
```

#### `qmix_medium_s0.json`

```json
{
  "checkpoint": "qmix3d_L3_H6_W6_uavs2_obs008_1778068048.pt",
  "mixer_type": "qmix",
  "map_shape": [
    3,
    6,
    6
  ],
  "num_uavs": 2,
  "obstacle_density": 0.08,
  "episodes": null,
  "seed": 0,
  "coverage_mean": 0.81987,
  "coverage_std": 0.1123707395187911,
  "steps_mean": 2432.462,
  "success_mean": 0.056,
  "metrics_source": "training_summary_jsonl",
  "metrics_note": "Aligned with Table 1 / medium_qmix_vdn.jsonl. Replace with eval_3d.py --episodes 50 when checkpoints exist."
}
```

#### `qmix_medium_s1.json`

```json
{
  "checkpoint": "qmix3d_L3_H6_W6_uavs2_obs008_1778183251.pt",
  "mixer_type": "qmix",
  "map_shape": [
    3,
    6,
    6
  ],
  "num_uavs": 2,
  "obstacle_density": 0.08,
  "episodes": null,
  "seed": 1,
  "coverage_mean": 0.87937,
  "coverage_std": 0.09073589752683332,
  "steps_mean": 2335.367,
  "success_mean": 0.1295,
  "metrics_source": "training_summary_jsonl",
  "metrics_note": "Aligned with Table 1 / medium_qmix_vdn.jsonl. Replace with eval_3d.py --episodes 50 when checkpoints exist."
}
```

#### `qmix_medium_s2.json`

```json
{
  "checkpoint": "qmix3d_L3_H6_W6_uavs2_obs008_1778267952.pt",
  "mixer_type": "qmix",
  "map_shape": [
    3,
    6,
    6
  ],
  "num_uavs": 2,
  "obstacle_density": 0.08,
  "episodes": null,
  "seed": 2,
  "coverage_mean": 0.8894500000000001,
  "coverage_std": 0.08359005622680248,
  "steps_mean": 2358.5185,
  "success_mean": 0.1125,
  "metrics_source": "training_summary_jsonl",
  "metrics_note": "Aligned with Table 1 / medium_qmix_vdn.jsonl. Replace with eval_3d.py --episodes 50 when checkpoints exist."
}
```

#### `random_hard_s0.json`

```json
{
  "policy": "random",
  "checkpoint": null,
  "mixer_type": "heuristic",
  "map_shape": [
    4,
    7,
    7
  ],
  "num_uavs": 3,
  "obstacle_density": 0.12,
  "episodes": 50,
  "seed": 0,
  "coverage_mean": 0.9998843930635837,
  "coverage_std": 0.0008174644869208714,
  "coverage_min": 0.9942196531791907,
  "coverage_max": 1.0,
  "steps_mean": 1053.36,
  "success_mean": 0.98
}
```

#### `random_hard_s1.json`

```json
{
  "policy": "random",
  "checkpoint": null,
  "mixer_type": "heuristic",
  "map_shape": [
    4,
    7,
    7
  ],
  "num_uavs": 3,
  "obstacle_density": 0.12,
  "episodes": 50,
  "seed": 1,
  "coverage_mean": 0.9998843930635837,
  "coverage_std": 0.0008174644869208714,
  "coverage_min": 0.9942196531791907,
  "coverage_max": 1.0,
  "steps_mean": 1079.6,
  "success_mean": 0.98
}
```

#### `random_hard_s2.json`

```json
{
  "policy": "random",
  "checkpoint": null,
  "mixer_type": "heuristic",
  "map_shape": [
    4,
    7,
    7
  ],
  "num_uavs": 3,
  "obstacle_density": 0.12,
  "episodes": 50,
  "seed": 2,
  "coverage_mean": 1.0,
  "coverage_std": 0.0,
  "coverage_min": 1.0,
  "coverage_max": 1.0,
  "steps_mean": 1017.1,
  "success_mean": 1.0
}
```

#### `random_medium_s0.json`

```json
{
  "policy": "random",
  "checkpoint": null,
  "mixer_type": "heuristic",
  "map_shape": [
    3,
    6,
    6
  ],
  "num_uavs": 2,
  "obstacle_density": 0.08,
  "episodes": 50,
  "seed": 0,
  "coverage_mean": 1.0,
  "coverage_std": 0.0,
  "coverage_min": 1.0,
  "coverage_max": 1.0,
  "steps_mean": 712.56,
  "success_mean": 1.0
}
```

#### `random_medium_s1.json`

```json
{
  "policy": "random",
  "checkpoint": null,
  "mixer_type": "heuristic",
  "map_shape": [
    3,
    6,
    6
  ],
  "num_uavs": 2,
  "obstacle_density": 0.08,
  "episodes": 50,
  "seed": 1,
  "coverage_mean": 1.0,
  "coverage_std": 0.0,
  "coverage_min": 1.0,
  "coverage_max": 1.0,
  "steps_mean": 700.3,
  "success_mean": 1.0
}
```

#### `random_medium_s2.json`

```json
{
  "policy": "random",
  "checkpoint": null,
  "mixer_type": "heuristic",
  "map_shape": [
    3,
    6,
    6
  ],
  "num_uavs": 2,
  "obstacle_density": 0.08,
  "episodes": 50,
  "seed": 2,
  "coverage_mean": 1.0,
  "coverage_std": 0.0,
  "coverage_min": 1.0,
  "coverage_max": 1.0,
  "steps_mean": 710.72,
  "success_mean": 1.0
}
```

#### `vdn_hard_s0.json`

```json
{
  "checkpoint": "vdn3d_L4_H7_W7_uavs3_obs012_1775256641.pt",
  "mixer_type": "vdn",
  "map_shape": [
    4,
    7,
    7
  ],
  "num_uavs": 3,
  "obstacle_density": 0.12,
  "episodes": null,
  "seed": 0,
  "coverage_mean": 0.9726473988439307,
  "coverage_std": 0.04540835908887821,
  "steps_mean": 2714.2845,
  "success_mean": 0.411,
  "metrics_source": "training_summary_jsonl",
  "metrics_note": "Aligned with Table 3 / hard_qmix_vdn.jsonl. Replace with eval_3d.py --episodes 50 when checkpoints exist."
}
```

#### `vdn_hard_s1.json`

```json
{
  "checkpoint": "vdn3d_L4_H7_W7_uavs3_obs012_1775284777.pt",
  "mixer_type": "vdn",
  "map_shape": [
    4,
    7,
    7
  ],
  "num_uavs": 3,
  "obstacle_density": 0.12,
  "episodes": null,
  "seed": 1,
  "coverage_mean": 0.981271676300578,
  "coverage_std": 0.03497630496757871,
  "steps_mean": 2479.302,
  "success_mean": 0.5285,
  "metrics_source": "training_summary_jsonl",
  "metrics_note": "Aligned with Table 3 / hard_qmix_vdn.jsonl. Replace with eval_3d.py --episodes 50 when checkpoints exist."
}
```

#### `vdn_hard_s2.json`

```json
{
  "checkpoint": "vdn3d_L4_H7_W7_uavs3_obs012_1775316135.pt",
  "mixer_type": "vdn",
  "map_shape": [
    4,
    7,
    7
  ],
  "num_uavs": 3,
  "obstacle_density": 0.12,
  "episodes": null,
  "seed": 2,
  "coverage_mean": 0.9753468208092486,
  "coverage_std": 0.04354975402010134,
  "steps_mean": 2635.4095,
  "success_mean": 0.456,
  "metrics_source": "training_summary_jsonl",
  "metrics_note": "Aligned with Table 3 / hard_qmix_vdn.jsonl. Replace with eval_3d.py --episodes 50 when checkpoints exist."
}
```

#### `vdn_medium_s0.json`

```json
{
  "checkpoint": "vdn3d_L3_H6_W6_uavs2_obs008_1774861056.pt",
  "mixer_type": "vdn",
  "map_shape": [
    3,
    6,
    6
  ],
  "num_uavs": 2,
  "obstacle_density": 0.08,
  "episodes": null,
  "seed": 0,
  "coverage_mean": 0.67595,
  "coverage_std": 0.1823814615030815,
  "steps_mean": 2435.254,
  "success_mean": 0.0495,
  "metrics_source": "training_summary_jsonl",
  "metrics_note": "Aligned with Table 1 / medium_qmix_vdn.jsonl. Replace with eval_3d.py --episodes 50 when checkpoints exist."
}
```

#### `vdn_medium_s1.json`

```json
{
  "checkpoint": "vdn3d_L3_H6_W6_uavs2_obs008_1774876572.pt",
  "mixer_type": "vdn",
  "map_shape": [
    3,
    6,
    6
  ],
  "num_uavs": 2,
  "obstacle_density": 0.08,
  "episodes": null,
  "seed": 1,
  "coverage_mean": 0.7097650000000001,
  "coverage_std": 0.1968395152783099,
  "steps_mean": 2338.1365,
  "success_mean": 0.1255,
  "metrics_source": "training_summary_jsonl",
  "metrics_note": "Aligned with Table 1 / medium_qmix_vdn.jsonl. Replace with eval_3d.py --episodes 50 when checkpoints exist."
}
```

#### `vdn_medium_s2.json`

```json
{
  "checkpoint": "vdn3d_L3_H6_W6_uavs2_obs008_1774892113.pt",
  "mixer_type": "vdn",
  "map_shape": [
    3,
    6,
    6
  ],
  "num_uavs": 2,
  "obstacle_density": 0.08,
  "episodes": null,
  "seed": 2,
  "coverage_mean": 0.7336800000000001,
  "coverage_std": 0.17855519482781787,
  "steps_mean": 2367.553,
  "success_mean": 0.1005,
  "metrics_source": "training_summary_jsonl",
  "metrics_note": "Aligned with Table 1 / medium_qmix_vdn.jsonl. Replace with eval_3d.py --episodes 50 when checkpoints exist."
}
```

</details>

---

## 3. 配置文件原文（YAML）

### `configs/algos/mappo.yaml`

```yaml
# Multi-agent PPO (shared encoder + per-agent heads + centralized critic)
rollout_steps: 2048
num_updates: 2000
hidden_dim: 128
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
learning_rate: 0.0003
entropy_coef: 0.01
value_coef: 0.5
max_grad_norm: 0.5
ppo_epochs: 4
num_minibatches: 4
log_interval: 20
```

### `configs/algos/qmix_no_potential.yaml`

```yaml
# 消融：关闭环境势函数 shaping（与 qmix_paper 其余一致）
mixer_type: qmix
enable_potential_reward: false
agent_hidden_dim: 64
mixing_hidden_dim: 32
hyper_hidden_dim: 64
learning_rate: 0.0005
buffer_size: 6000
batch_size: 16
bptt_detach_interval: 64
min_buffer: 400
episodes: 2000
target_update_interval: 200
gamma: 0.99
grad_clip: 10.0
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay: 0.998
epsilon_min: 0.05
epsilon_plateaus: []
log_interval: 50
recovery:
  enabled: false
```

### `configs/algos/qmix_paper.yaml`

```yaml
# 论文实验建议配置（更长训练、更大缓冲）
mixer_type: qmix
enable_potential_reward: true
agent_hidden_dim: 64
mixing_hidden_dim: 32
hyper_hidden_dim: 64
learning_rate: 0.0005
# 16GB 内存机器：过大 buffer + 2500 步轨迹易触发大量换页；可按机器再调
buffer_size: 6000
batch_size: 16
# 每 K 步对 RNN hidden detach，截断 BPTT，显著加快长轨迹反传（略改梯度偏差，常用折中）
bptt_detach_interval: 64
min_buffer: 400
episodes: 2000
target_update_interval: 200
gamma: 0.99
grad_clip: 10.0
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay: 0.998
epsilon_min: 0.05
epsilon_plateaus: []
log_interval: 50
recovery:
  enabled: false
```

### `configs/algos/vdn.yaml`

```yaml
# VDN 基线：Q_tot = sum_i Q_i，无 hypernetwork 混合器
mixer_type: vdn
enable_potential_reward: true
agent_hidden_dim: 64
mixing_hidden_dim: 32
hyper_hidden_dim: 64
learning_rate: 0.0005
buffer_size: 6000
batch_size: 16
bptt_detach_interval: 64
min_buffer: 400
episodes: 2000
target_update_interval: 200
gamma: 0.99
grad_clip: 10.0
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay: 0.998
epsilon_min: 0.05
epsilon_plateaus: []
log_interval: 50
recovery:
  enabled: false
```

### `configs/envs/grid_3d_hard.yaml`

```yaml
map_shape: [4, 7, 7]
num_uavs: [3]
obstacle_density: 0.12
max_steps: 3500
energy_budget: 6000
shaping_weight: 10.0
obstacle_shaping_weight: 2.0
seed: 42
```

### `configs/envs/grid_3d_medium.yaml`

```yaml
map_shape: [3, 6, 6]
num_uavs: [2]
obstacle_density: 0.08
max_steps: 2500
energy_budget: 4500
shaping_weight: 10.0
obstacle_shaping_weight: 2.0
seed: 42
```

---

## 4. 训练过程 JSONL（原文逐行）

### `experiments/results/ablation_medium.jsonl`

- 行数：3，字节：724

```jsonl
{"method": "qmix", "map_shape": "3x6x6", "num_uavs": 2, "obstacle_density": 0.08, "seed": 0, "coverage_mean": 0.8163700000000002, "coverage_std": 0.11288012712608009, "pa_mean": 0.01766469019489836, "steps_mean": 2432.713, "success_mean": 0.054}
{"method": "qmix", "map_shape": "3x6x6", "num_uavs": 2, "obstacle_density": 0.08, "seed": 1, "coverage_mean": 0.88248, "coverage_std": 0.08819721991083392, "pa_mean": 0.021443481127651952, "steps_mean": 2336.235, "success_mean": 0.1285}
{"method": "qmix", "map_shape": "3x6x6", "num_uavs": 2, "obstacle_density": 0.08, "seed": 2, "coverage_mean": 0.883165, "coverage_std": 0.08831836035049563, "pa_mean": 0.020527132484855083, "steps_mean": 2362.742, "success_mean": 0.1075}
```

### `experiments/results/curve_aux.jsonl`

- 行数：1，字节：236

```jsonl
{"method": "vdn", "map_shape": "3x6x6", "num_uavs": 2, "obstacle_density": 0.08, "seed": 0, "coverage_mean": 0.67595, "coverage_std": 0.1823814615030815, "pa_mean": 0.014827599527980023, "steps_mean": 2435.254, "success_mean": 0.0495}
```

### `experiments/results/hard_curve_aux.jsonl`

- 行数：2，字节：496

```jsonl
{"method": "qmix", "map_shape": "4x7x7", "num_uavs": 3, "obstacle_density": 0.12, "seed": 0, "coverage_mean": 0.9934826589595375, "coverage_std": 0.01218028086180411, "pa_mean": 0.033783683376572934, "steps_mean": 2330.455, "success_mean": 0.601}
{"method": "vdn", "map_shape": "4x7x7", "num_uavs": 3, "obstacle_density": 0.12, "seed": 0, "coverage_mean": 0.9726473988439307, "coverage_std": 0.04540835908887821, "pa_mean": 0.027493978677411178, "steps_mean": 2714.2845, "success_mean": 0.411}
```

### `experiments/results/hard_qmix_only.jsonl`

- 行数：3，字节：747

```jsonl
{"method": "qmix", "map_shape": "4x7x7", "num_uavs": 3, "obstacle_density": 0.12, "seed": 0, "coverage_mean": 0.9934826589595375, "coverage_std": 0.01218028086180411, "pa_mean": 0.033783683376572934, "steps_mean": 2330.455, "success_mean": 0.601}
{"method": "qmix", "map_shape": "4x7x7", "num_uavs": 3, "obstacle_density": 0.12, "seed": 1, "coverage_mean": 0.9939017341040463, "coverage_std": 0.011034218795629226, "pa_mean": 0.033810112542829035, "steps_mean": 2340.1545, "success_mean": 0.611}
{"method": "qmix", "map_shape": "4x7x7", "num_uavs": 3, "obstacle_density": 0.12, "seed": 2, "coverage_mean": 0.9928786127167629, "coverage_std": 0.012467574301679165, "pa_mean": 0.03435701655449204, "steps_mean": 2328.3815, "success_mean": 0.592}
```

### `experiments/results/hard_qmix_vdn.jsonl`

- 行数：6，字节：1490

```jsonl
{"method": "qmix", "map_shape": "4x7x7", "num_uavs": 3, "obstacle_density": 0.12, "seed": 0, "coverage_mean": 0.9934826589595375, "coverage_std": 0.01218028086180411, "pa_mean": 0.033783683376572934, "steps_mean": 2330.455, "success_mean": 0.601}
{"method": "qmix", "map_shape": "4x7x7", "num_uavs": 3, "obstacle_density": 0.12, "seed": 1, "coverage_mean": 0.9939017341040463, "coverage_std": 0.011034218795629226, "pa_mean": 0.033810112542829035, "steps_mean": 2340.1545, "success_mean": 0.611}
{"method": "qmix", "map_shape": "4x7x7", "num_uavs": 3, "obstacle_density": 0.12, "seed": 2, "coverage_mean": 0.9928786127167629, "coverage_std": 0.012467574301679165, "pa_mean": 0.03435701655449204, "steps_mean": 2328.3815, "success_mean": 0.592}
{"method": "vdn", "map_shape": "4x7x7", "num_uavs": 3, "obstacle_density": 0.12, "seed": 0, "coverage_mean": 0.9726473988439307, "coverage_std": 0.04540835908887821, "pa_mean": 0.027493978677411178, "steps_mean": 2714.2845, "success_mean": 0.411}
{"method": "vdn", "map_shape": "4x7x7", "num_uavs": 3, "obstacle_density": 0.12, "seed": 1, "coverage_mean": 0.981271676300578, "coverage_std": 0.03497630496757871, "pa_mean": 0.031059457257546882, "steps_mean": 2479.302, "success_mean": 0.5285}
{"method": "vdn", "map_shape": "4x7x7", "num_uavs": 3, "obstacle_density": 0.12, "seed": 2, "coverage_mean": 0.9753468208092486, "coverage_std": 0.04354975402010134, "pa_mean": 0.028824274749103273, "steps_mean": 2635.4095, "success_mean": 0.456}
```

### `experiments/results/medium_qmix_vdn.jsonl`

- 行数：6，字节：1453

```jsonl
{"method": "qmix", "map_shape": "3x6x6", "num_uavs": 2, "obstacle_density": 0.08, "seed": 0, "coverage_mean": 0.81987, "coverage_std": 0.1123707395187911, "pa_mean": 0.017739251878670096, "steps_mean": 2432.462, "success_mean": 0.056}
{"method": "qmix", "map_shape": "3x6x6", "num_uavs": 2, "obstacle_density": 0.08, "seed": 1, "coverage_mean": 0.87937, "coverage_std": 0.09073589752683332, "pa_mean": 0.02141255605238252, "steps_mean": 2335.367, "success_mean": 0.1295}
{"method": "qmix", "map_shape": "3x6x6", "num_uavs": 2, "obstacle_density": 0.08, "seed": 2, "coverage_mean": 0.8894500000000001, "coverage_std": 0.08359005622680248, "pa_mean": 0.020759097720999653, "steps_mean": 2358.5185, "success_mean": 0.1125}
{"method": "vdn", "map_shape": "3x6x6", "num_uavs": 2, "obstacle_density": 0.08, "seed": 0, "coverage_mean": 0.67595, "coverage_std": 0.1823814615030815, "pa_mean": 0.014827599527980023, "steps_mean": 2435.254, "success_mean": 0.0495}
{"method": "vdn", "map_shape": "3x6x6", "num_uavs": 2, "obstacle_density": 0.08, "seed": 1, "coverage_mean": 0.7097650000000001, "coverage_std": 0.1968395152783099, "pa_mean": 0.01798545286681636, "steps_mean": 2338.1365, "success_mean": 0.1255}
{"method": "vdn", "map_shape": "3x6x6", "num_uavs": 2, "obstacle_density": 0.08, "seed": 2, "coverage_mean": 0.7336800000000001, "coverage_std": 0.17855519482781787, "pa_mean": 0.01749182692775907, "steps_mean": 2367.553, "success_mean": 0.1005}
```

### `experiments/results/single_lines.jsonl`

- 行数：1，字节：236

```jsonl
{"method": "qmix", "map_shape": "3x6x6", "num_uavs": 2, "obstacle_density": 0.08, "seed": 0, "coverage_mean": 0.81987, "coverage_std": 0.1123707395187911, "pa_mean": 0.017739251878670096, "steps_mean": 2432.462, "success_mean": 0.056}
```

---

## 5. 学习曲线 CSV 索引

| 文件名 | 相对路径 | 字节数 |
|--------|----------|--------|
| `ablation_no_potential_s0.csv` | `experiments/curves/ablation_no_potential_s0.csv` | 60892 |
| `ablation_no_potential_s1.csv` | `experiments/curves/ablation_no_potential_s1.csv` | 60816 |
| `ablation_no_potential_s2.csv` | `experiments/curves/ablation_no_potential_s2.csv` | 60838 |
| `qmix_hard_s0.csv` | `experiments/curves/qmix_hard_s0.csv` | 60606 |
| `qmix_medium_s0.csv` | `experiments/curves/qmix_medium_s0.csv` | 60893 |
| `vdn_hard_s0.csv` | `experiments/curves/vdn_hard_s0.csv` | 60723 |
| `vdn_medium_episode.csv` | `experiments/curves/vdn_medium_episode.csv` | 60893 |
