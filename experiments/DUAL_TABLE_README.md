# 五方法 × 两档场景（Medium / Hard）主表

## 一键汇总（Markdown）

```text
experiments\run_fill_hard_and_aggregate.bat
```

或分步：

```text
python experiments\materialize_eval_from_training_jsonl.py --tier both
python experiments\run_heuristic_hard_eval.bat
python experiments\run_aggregate_dual.bat
```

最后一行等价于：

```text
python experiments\aggregate_eval_table.py --spec experiments\aggregate_dual_spec.json --skip-missing
```

## 输入文件约定

| 方法 | Medium | Hard |
|------|--------|------|
| QMIX / VDN | 默认由 `medium_qmix_vdn.jsonl` materialize 到 `qmix_medium_s*.json`（与表 1 一致）；也可用 `eval_3d.py` 覆盖 | 默认由 `hard_qmix_vdn.jsonl` → `qmix_hard_s*.json`（与表 3 一致）；也可用 `run_eval_qmix_vdn_hard.bat` 覆盖为 50 局测试 |
| Greedy / Random | `eval_heuristic_3d.py` + `grid_3d_medium.yaml` | `run_heuristic_hard_eval.bat` |
| MAPPO | 已有 `mappo_medium_s*.json` | 需训练并写出 `mappo_hard_s*.json`（见 `mappo_hard_note.md`） |

## 规格文件

- `aggregate_medium_spec.json`：仅 medium  
- `aggregate_hard_spec.json`：仅 hard  
- `aggregate_dual_spec.json`：两档各一张表  

`--skip-missing`：某方法 JSON 不全时跳过该行（例如尚未跑 hard 启发式 / MAPPO）。
