"""Plot potential-based shaping ablation (with vs without potential) as bars.

Inputs are JSONL files produced by train_qmix_3d.py --results-file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_metric(rows: List[Dict[str, Any]], method: str, key: str) -> np.ndarray:
    vals: List[float] = []
    for r in rows:
        if r.get("method") != method:
            continue
        if key not in r:
            continue
        vals.append(float(r[key]))
    if not vals:
        raise ValueError(f"No values for method={method!r}, key={key!r}")
    return np.asarray(vals, dtype=np.float64)


def mean_std(a: np.ndarray) -> Tuple[float, float]:
    if a.size == 1:
        return float(a.mean()), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--with-potential",
        type=str,
        default=str(ROOT / "experiments/results/medium_qmix_vdn.jsonl"),
        help="JSONL containing QMIX runs with potential shaping (method=qmix rows used).",
    )
    p.add_argument(
        "--without-potential",
        type=str,
        default=str(ROOT / "experiments/results/ablation_medium.jsonl"),
        help="JSONL containing QMIX runs without potential shaping (method=qmix rows used).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "experiments/figures/ablation_potential_comparison.png"),
    )
    p.add_argument(
        "--lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="Label language: zh or en",
    )
    args = p.parse_args()

    with_path = Path(args.with_potential)
    without_path = Path(args.without_potential)
    for path in (with_path, without_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing JSONL: {path}")

    rows_with = _load_rows(with_path)
    rows_without = _load_rows(without_path)

    # Metrics (per-seed aggregates).
    cov_with = _extract_metric(rows_with, method="qmix", key="coverage_mean")
    cov_wo = _extract_metric(rows_without, method="qmix", key="coverage_mean")
    steps_with = _extract_metric(rows_with, method="qmix", key="steps_mean")
    steps_wo = _extract_metric(rows_without, method="qmix", key="steps_mean")
    succ_with = _extract_metric(rows_with, method="qmix", key="success_mean")
    succ_wo = _extract_metric(rows_without, method="qmix", key="success_mean")

    cov_m = [mean_std(cov_with), mean_std(cov_wo)]
    steps_m = [mean_std(steps_with), mean_std(steps_wo)]
    succ_m = [mean_std(succ_with), mean_std(succ_wo)]

    # Use two-line tick labels to avoid overlap.
    if args.lang == "en":
        labels = ["QMIX\n(with potential)", "QMIX\n(without potential)"]
    else:
        labels = ["QMIX\n（有势函数）", "QMIX\n（无势函数）"]
    x = np.arange(len(labels), dtype=np.float64)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), dpi=160)

    def bar(ax, stats, title, ylim=None, fmt="{:.3f}"):
        means = [s[0] for s in stats]
        stds = [s[1] for s in stats]
        bars = ax.bar(x, means, yerr=stds, capsize=4, alpha=0.9, color=["#4C78A8", "#F58518"])
        ax.set_xticks(x, labels, rotation=0)
        ax.tick_params(axis="x", labelsize=9, pad=6)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        if ylim is not None:
            ax.set_ylim(*ylim)
        # Value labels
        for b, m in zip(bars, means):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                fmt.format(m),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    if args.lang == "en":
        bar(axes[0], cov_m, "Coverage (mean ± std)", ylim=(0.0, 1.05), fmt="{:.3f}")
        bar(axes[1], steps_m, "Steps (mean ± std)", ylim=None, fmt="{:.1f}")
        bar(axes[2], succ_m, "Success rate (mean ± std)", ylim=(0.0, 0.25), fmt="{:.3f}")
        fig.suptitle("Potential shaping ablation (grid_3d_medium, n=3 seeds)", y=1.02)
    else:
        bar(axes[0], cov_m, "覆盖率（均值±std）", ylim=(0.0, 1.05), fmt="{:.3f}")
        bar(axes[1], steps_m, "平均步数（均值±std）", ylim=None, fmt="{:.1f}")
        bar(axes[2], succ_m, "成功率（均值±std）", ylim=(0.0, 0.25), fmt="{:.3f}")
        fig.suptitle("势函数 shaping 消融对比（grid_3d_medium，n=3 seeds）", y=1.02)
    # Add extra room for multi-line x tick labels and suptitle.
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.resolve()}")


if __name__ == "__main__":
    main()

