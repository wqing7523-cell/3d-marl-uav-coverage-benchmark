"""Plot per-episode coverage curves from train_qmix_3d.py --episode-log CSV files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    episodes: list[int] = []
    coverages: list[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            coverages.append(float(row["coverage"]))
    return np.array(episodes, dtype=np.int32), np.array(coverages, dtype=np.float64)


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return y
    k = np.ones(w, dtype=np.float64) / w
    return np.convolve(y, k, mode="same")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--qmix",
        type=str,
        default=str(ROOT / "experiments/curves/qmix_medium_s0.csv"),
    )
    p.add_argument(
        "--vdn",
        type=str,
        default=str(ROOT / "experiments/curves/vdn_medium_episode.csv"),
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "experiments/figures/learning_curve_medium_qmix_vdn.png"),
    )
    p.add_argument(
        "--lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="Label language: zh or en",
    )
    p.add_argument("--smooth", type=int, default=0, help="Moving average window; 0=off")
    p.add_argument(
        "--map",
        type=str,
        default="grid_3d_medium",
        help="Map tag shown in figure title (e.g. grid_3d_hard).",
    )
    args = p.parse_args()

    q_path, v_path = Path(args.qmix), Path(args.vdn)
    for path in (q_path, v_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing CSV: {path}")

    eq, yq = read_curve(q_path)
    ev, yv = read_curve(v_path)
    if args.smooth > 1:
        yq = smooth(yq, args.smooth)
        yv = smooth(yv, args.smooth)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    ax.plot(eq, yq, label="QMIX", linewidth=1.2, alpha=0.9)
    ax.plot(ev, yv, label="VDN", linewidth=1.2, alpha=0.9)
    map_tag = args.map
    if args.lang == "en":
        ax.set_xlabel("Episode")
        ax.set_ylabel("Coverage (end-of-episode)")
        ax.set_title(f"Learning Curve ({map_tag}, seed=0)")
    else:
        ax.set_xlabel("Episode")
        ax.set_ylabel("Coverage（单局末覆盖率）")
        ax.set_title(f"训练过程覆盖率曲线（{map_tag}，seed=0）")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.resolve()}")


if __name__ == "__main__":
    main()
