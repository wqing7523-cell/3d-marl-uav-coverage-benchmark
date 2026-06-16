"""Generate Phase-B paper figures: coverage–success joint, training stability, sync to paper/figures."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

QMIX_MEDIUM_CKPTS = [
    "qmix3d_L3_H6_W6_uavs2_obs008_1778068048.pt",
    "qmix3d_L3_H6_W6_uavs2_obs008_1778183251.pt",
    "qmix3d_L3_H6_W6_uavs2_obs008_1778267952.pt",
]
MAPPO_MEDIUM_CKPTS = [
    "mappo3d_L3_H6_W6_uavs2_obs008_1778320633.pt",
    "mappo3d_L3_H6_W6_uavs2_obs008_1778327797.pt",
    "mappo3d_L3_H6_W6_uavs2_obs008_1778336904.pt",
]

# Main-table aggregates (Table 4 Medium) for bar comparison panel
TABLE4_MEDIUM = {
    "QMIX": {"coverage": 0.863, "success": 0.099},
    "VDN": {"coverage": 0.707, "success": 0.092},
    "MAPPO": {"coverage": 0.689, "success": 0.053},
}


def _setup_plt() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def read_training_csv(path: Path) -> dict[str, np.ndarray]:
    episodes, coverage, success, steps = [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            coverage.append(float(row["coverage"]))
            success.append(float(row["success"]))
            steps.append(float(row["steps"]))
    return {
        "episode": np.array(episodes, dtype=np.int32),
        "coverage": np.array(coverage, dtype=np.float64),
        "success": np.array(success, dtype=np.float64),
        "steps": np.array(steps, dtype=np.float64),
    }


def rolling_mean(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return y.copy()
    out = np.convolve(y, np.ones(w) / w, mode="same")
    return out


def plot_figure8_coverage_success(out: Path, dpi: int) -> None:
    """Joint view: training scatter + table-derived termination bars (Medium)."""
    _setup_plt()
    qmix_csv = ROOT / "experiments/curves/qmix_medium_s0.csv"
    if not qmix_csv.is_file():
        raise FileNotFoundError(qmix_csv)
    data = read_training_csv(qmix_csv)
    # Last 300 training episodes (late training, near eval regime)
    tail = slice(-300, None)
    cov = data["coverage"][tail]
    succ = data["success"][tail]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), dpi=dpi)

    ax0 = axes[0]
    colors = np.where(succ >= 0.5, "#2ca02c", "#d62728")
    ax0.scatter(cov, succ, c=colors, alpha=0.35, s=12, edgecolors="none")
    ax0.set_xlabel("Episode-end coverage")
    ax0.set_ylabel("Completion success (0/1)")
    ax0.set_title("(a) QMIX training tail (seed 0, last 300 ep.)")
    ax0.set_xlim(0, 1.02)
    ax0.set_ylim(-0.05, 1.05)
    ax0.axhline(0.99, color="gray", ls="--", lw=0.8, alpha=0.6)

    ax1 = axes[1]
    methods = list(TABLE4_MEDIUM.keys())
    trunc_est = [1.0 - TABLE4_MEDIUM[m]["success"] for m in methods]
    x = np.arange(len(methods))
    ax1.bar(x, trunc_est, color=["#1f77b4", "#ff7f0e", "#9467bd"], alpha=0.85)
    ax1.set_xticks(x, methods)
    ax1.set_ylabel("Estimated truncated / non-success fraction")
    ax1.set_title("(b) From Table 4 success mean ($1 - \\mathrm{success}$)")
    ax1.set_ylim(0, 1.05)

    ax2 = axes[2]
    remain_est = [1.0 - TABLE4_MEDIUM[m]["coverage"] for m in methods]
    ax2.bar(x, remain_est, color=["#1f77b4", "#ff7f0e", "#9467bd"], alpha=0.85)
    ax2.set_xticks(x, methods)
    ax2.set_ylabel("Estimated unvisited ratio ($1 - \\mathrm{coverage}$)")
    ax2.set_title("(c) From Table 2 coverage mean (coarse)")
    ax2.set_ylim(0, max(0.5, max(remain_est) * 1.2))

    fig.suptitle("Medium tier: coverage vs completion success (QMIX training + learning-class aggregates)", y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


def plot_figure3b_training_stability(out: Path, dpi: int) -> None:
    """Multi-curve training success rate (rolling) for Medium/Hard."""
    _setup_plt()
    series = [
        ("QMIX Medium s0", ROOT / "experiments/curves/qmix_medium_s0.csv", "#1f77b4"),
        ("VDN Medium", ROOT / "experiments/curves/vdn_medium_episode.csv", "#ff7f0e"),
        ("QMIX Hard s0", ROOT / "experiments/curves/qmix_hard_s0.csv", "#2ca02c"),
        ("VDN Hard s0", ROOT / "experiments/curves/vdn_hard_s0.csv", "#d62728"),
    ]
    ablation_paths = [
        ROOT / f"experiments/curves/ablation_no_potential_s{i}.csv" for i in range(3)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=dpi)
    w = 80

    ax_m = axes[0]
    for label, path, color in series[:2]:
        if not path.is_file():
            continue
        d = read_training_csv(path)
        ax_m.plot(d["episode"], rolling_mean(d["success"], w), label=label, color=color, lw=1.2)
    for i, p in enumerate(ablation_paths):
        if p.is_file():
            d = read_training_csv(p)
            ax_m.plot(
                d["episode"],
                rolling_mean(d["success"], w),
                label=f"QMIX w/o PBRS s{i}",
                color="#9467bd",
                lw=0.9,
                alpha=0.55,
            )
    ax_m.set_xlabel("Training episode")
    ax_m.set_ylabel(f"Rolling mean success (window={w})")
    ax_m.set_title("Medium tier")
    ax_m.legend(fontsize=7, loc="upper left")
    ax_m.set_ylim(-0.02, 1.02)

    ax_h = axes[1]
    for label, path, color in series[2:]:
        if not path.is_file():
            continue
        d = read_training_csv(path)
        ax_h.plot(d["episode"], rolling_mean(d["success"], w), label=label, color=color, lw=1.2)
    ax_h.set_xlabel("Training episode")
    ax_h.set_ylabel(f"Rolling mean success (window={w})")
    ax_h.set_title("Hard tier")
    ax_h.legend(fontsize=8)
    ax_h.set_ylim(-0.02, 1.02)

    fig.suptitle("Training-phase completion success (not identical to Table 4–5 eval aggregates)", y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


def parse_mappo_log_medium() -> tuple[np.ndarray, np.ndarray] | None:
    log_path = ROOT / "experiments/logs/mappo3d.log"
    if not log_path.is_file():
        return None
    pat = re.compile(
        r"map_shape=\(3, 6, 6\).*?Upd (\d+) \| episodes=(\d+) \| coverage_mean=([\d.]+)"
    )
    # Simpler line-by-line for medium 3x6x6 after first medium block
    upds, covs = [], []
    in_medium = False
    for line in open(log_path, encoding="utf-8", errors="ignore"):
        if "map_shape=(3, 6, 6)" in line and "obstacle_density=0.08" in line:
            in_medium = True
            continue
        if in_medium and "map_shape=(4, 7, 7)" in line:
            break
        if in_medium and "Upd " in line and "coverage_mean=" in line:
            m = re.search(r"Upd (\d+) \| episodes=(\d+) \| coverage_mean=([\d.]+)", line)
            if m:
                upds.append(int(m.group(1)))
                covs.append(float(m.group(3)))
    if not upds:
        return None
    return np.array(upds), np.array(covs)


def plot_mappo_training_coverage(out: Path, dpi: int) -> None:
    parsed = parse_mappo_log_medium()
    if parsed is None:
        print("Skip MAPPO training curve: no medium entries in mappo3d.log")
        return
    upds, covs = parsed
    _setup_plt()
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=dpi)
    ax.plot(upds, covs, color="#9467bd", lw=1.2)
    ax.set_xlabel("MAPPO update")
    ax.set_ylabel("coverage_mean (training log)")
    ax.set_title("MAPPO Medium training coverage (from log)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


def run_mappo_greedy_eval(seed: int, ckpt_name: str, tier: str) -> dict:
    env = f"configs/envs/grid_3d_{tier}.yaml"
    ckpt = ROOT / "experiments/checkpoints" / ckpt_name
    out = ROOT / f"experiments/results_eval/mappo_{tier}_s{seed}_greedy.json"
    cmd = [
        sys.executable,
        str(ROOT / "eval_mappo_3d.py"),
        "--checkpoint",
        str(ckpt),
        "--env-config",
        env,
        "--episodes",
        "50",
        "--seed",
        str(seed),
        "--output-json",
        str(out),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return json.loads(out.read_text(encoding="utf-8"))


def aggregate_mappo_greedy_table() -> dict:
    rows = []
    for tier, ckpts in [("medium", MAPPO_MEDIUM_CKPTS), ("hard", [
        "mappo3d_L4_H7_W7_uavs3_obs012_1778360796.pt",
        "mappo3d_L4_H7_W7_uavs3_obs012_1778373399.pt",
        "mappo3d_L4_H7_W7_uavs3_obs012_1778386065.pt",
    ])]:
        for s, ck in enumerate(ckpts):
            p = ROOT / f"experiments/results_eval/mappo_{tier}_s{s}_greedy.json"
            if not p.is_file():
                try:
                    run_mappo_greedy_eval(s, ck, tier)
                except Exception as e:
                    print("WARN greedy eval failed", tier, s, e)
                    continue
            if p.is_file():
                rows.append(json.loads(p.read_text(encoding="utf-8")))
    return {"rows": rows}


def write_appendix_c_mappo_greedy_md(path: Path) -> None:
    """Markdown snippet for paper appendix / 附表 B."""
    lines = [
        "**附表 B** MAPPO 评估解码对照（50 episodes/种子，$n=3$）",
        "",
        "| 档位 | 解码 | coverage mean | success mean | steps mean |",
        "|------|------|---------------|--------------|------------|",
    ]
    for tier in ("medium", "hard"):
        for stochastic in (True, False):
            tag = "stochastic" if stochastic else "greedy"
            suffix = "" if stochastic else "_greedy"
            vals = {"coverage": [], "success": [], "steps": []}
            for s in range(3):
                p = ROOT / f"experiments/results_eval/mappo_{tier}_s{s}{suffix}.json"
                if not p.is_file():
                    continue
                d = json.loads(p.read_text(encoding="utf-8"))
                vals["coverage"].append(d["coverage_mean"])
                vals["success"].append(d["success_mean"])
                vals["steps"].append(d["steps_mean"])
            if not vals["coverage"]:
                continue
            def fmt(key: str) -> str:
                a = np.array(vals[key])
                return f"{a.mean():.3f} $\\pm$ {a.std(ddof=1) if len(a)>1 else 0:.3f}"

            lines.append(
                f"| {tier.capitalize()} | {tag} | {fmt('coverage')} | {fmt('success')} | {fmt('steps')} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", path)


def sync_to_paper(*png_names: str) -> None:
    src = ROOT / "experiments/figures/paper_sensors"
    dst = ROOT / "paper/figures"
    dst.mkdir(parents=True, exist_ok=True)
    for name in png_names:
        for base in (src, ROOT / "experiments/figures"):
            p = base / name
            if p.is_file():
                import shutil
                shutil.copy2(p, dst / name)
                print("Synced", dst / name)
                break


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--skip-mappo-greedy", action="store_true")
    p.add_argument("--out-dir", type=str, default="experiments/figures/paper_sensors")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    f8 = out_dir / "figure8_coverage_success_medium.png"
    f3b = out_dir / "figure3b_training_success_stability.png"
    fmappo = out_dir / "figure_mappo_medium_training_coverage.png"

    plot_figure8_coverage_success(f8, args.dpi)
    plot_figure3b_training_stability(f3b, args.dpi)
    plot_mappo_training_coverage(fmappo, args.dpi)

    if not args.skip_mappo_greedy:
        aggregate_mappo_greedy_table()
    snippet = ROOT / "paper" / "appendix_mappo_greedy_table.md"
    write_appendix_c_mappo_greedy_md(snippet)

    sync_to_paper(
        "figure8_coverage_success_medium.png",
        "figure3b_training_success_stability.png",
        "figure_mappo_medium_training_coverage.png",
    )


if __name__ == "__main__":
    main()
