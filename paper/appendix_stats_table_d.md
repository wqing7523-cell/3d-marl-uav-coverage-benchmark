**附表 D** 探索性统计（$n=3$ 种子；**exploratory CI**，bootstrap 10000 次重采样；**仅 QMIX/VDN，同基准评测协议**）



数据：`experiments/results_eval/{qmix,vdn}_{medium,hard}_s{0,1,2}.json`（与表 2、表 3 同源的训练汇总物化结果）。**$n=3$ 时 exploratory CI 仅作探索性参考**，不能替代更大样本的显著性结论。MAPPO 因评测协议不同，**不纳入本附表**（见附录 C）。



**表 D.1** Hard 档 **success mean** 的 exploratory CI（95%）



| 算法 | mean | exploratory 95% CI |

|------|------|---------------------|

| QMIX | 0.601 | [0.592, 0.611] |

| VDN | 0.465 | [0.411, 0.528] |



**表 D.2** Medium 档 **success mean** 的 exploratory CI（95%）



| 算法 | mean | exploratory 95% CI |

|------|------|---------------------|

| QMIX | 0.099 | [0.056, 0.130] |

| VDN | 0.092 | [0.050, 0.126] |



**表 D.3** Hard 档 QMIX vs VDN（success mean，探索性）



- QMIX 三种子：0.601, 0.611, 0.592

- VDN 三种子：0.411, 0.528, 0.456

- Mann–Whitney U 双侧检验 $p \approx 0.100$（$n=3$，**exploratory，仅供参考**）

