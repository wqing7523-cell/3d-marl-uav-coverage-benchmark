"""Bootstrap 95% CI and exploratory tests from results_eval per-seed JSON (n=3)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "experiments" / "results_eval"
OUT_MD = ROOT / "paper" / "appendix_stats_table_d.md"


def load_seed_values(tier: str, algo: str, metric: str = "success_mean") -> list[float]:
    vals: list[float] = []
    for s in range(3):
        p = EVAL_DIR / f"{algo}_{tier}_s{s}.json"
        if not p.is_file():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        vals.append(float(d[metric]))
    return vals


def bootstrap_ci(values: list[float], n_boot: int = 10000, alpha: float = 0.05) -> tuple[float, float, float]:
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean, mean
    rng = np.random.default_rng(42)
    boots = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return mean, float(lo), float(hi)


def mann_whitney_p(a: list[float], b: list[float]) -> float | None:
  try:
      from scipy.stats import mannwhitneyu
  except ImportError:
      return None
  if len(a) < 2 or len(b) < 2:
      return None
  _, p = mannwhitneyu(a, b, alternative="two-sided")
  return float(p)


def fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def write_markdown() -> None:
    lines = [
        "**附表 D** 探索性统计（$n=3$ 种子；bootstrap 95% CI，10000 次重采样）",
        "",
        "数据：`experiments/results_eval/{algo}_{tier}_s{0,1,2}.json`（与表 2、表 3 同源的训练汇总物化结果）。**$n=3$ 时 CI 与 p 值仅作探索性参考**，不能替代更大样本的显著性结论。",
        "",
        "**表 D.1** Hard 档 **success mean** 的 bootstrap 95% CI",
        "",
        "| 算法 | mean | bootstrap 95% CI |",
        "|------|------|------------------|",
    ]
    for algo in ("qmix", "vdn", "mappo"):
        v = load_seed_values("hard", algo)
        m, lo, hi = bootstrap_ci(v)
        lines.append(f"| {algo.upper()} | {m:.3f} | [{lo:.3f}, {hi:.3f}] |")

    lines += [
        "",
        "**表 D.2** Medium 档 **success mean** 的 bootstrap 95% CI",
        "",
        "| 算法 | mean | bootstrap 95% CI |",
        "|------|------|------------------|",
    ]
    for algo in ("qmix", "vdn", "mappo"):
        v = load_seed_values("medium", algo)
        m, lo, hi = bootstrap_ci(v)
        lines.append(f"| {algo.upper()} | {m:.3f} | [{lo:.3f}, {hi:.3f}] |")

    qmix_h = load_seed_values("hard", "qmix")
    vdn_h = load_seed_values("hard", "vdn")
    p_h = mann_whitney_p(qmix_h, vdn_h)
    lines += [
        "",
        "**表 D.3** Hard 档 QMIX vs VDN（success mean，探索性）",
        "",
        f"- QMIX 三种子：{', '.join(f'{x:.3f}' for x in qmix_h)}",
        f"- VDN 三种子：{', '.join(f'{x:.3f}' for x in vdn_h)}",
    ]
    if p_h is not None:
        lines.append(f"- Mann–Whitney U 双侧检验 $p \\approx {p_h:.3f}$（$n=3$，**仅供参考**）")
    else:
        lines.append("- 未安装 `scipy`，未报告 Mann–Whitney p 值（`pip install scipy` 后可重跑本脚本）")

    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", OUT_MD)


if __name__ == "__main__":
    write_markdown()
