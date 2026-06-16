**附表 E** Hard 档 QMIX vs VDN（$n=5$ 种子；与表 3 **同协议**：各 seed 训练 2000 episodes 后的 **训练汇总 success mean**，非 `eval_3d.py` greedy 复评）

| 算法 | 五种子 success | mean | bootstrap 95% CI |
|------|----------------|------|------------------|
| QMIX | 0.601, 0.611, 0.592, 0.579, 0.556 | 0.588 | [0.570, 0.603] |
| VDN | 0.411, 0.528, 0.456, 0.564, 0.532 | 0.498 | [0.444, 0.544] |

- Mann–Whitney U（QMIX vs VDN，$n=5$）$p \approx 0.016

**说明**：`eval_3d.py` 纯 greedy 在本仓库 checkpoint 上 success≈0（与训练期 ε-greedy 汇总不一致）；附表 E 与主表口径一致，采用 `hard_qmix_vdn.jsonl`。

主文表 3 仍为 $n=3$ 描述性主表；本表为 $n=5$ 探索性扩展。
