"""Generate / assemble *Sensors*-style paper figures (Fig1–7 layout).

Outputs under experiments/figures/paper_sensors/:
  figure1_voxel_environment_*.png  — 3D obstacle + UAV start positions
  figure2_framework_ctde_pbrs_der.png — schematic (English labels)
  figure3_learning_curves_qmix_vdn.png — medium+hard side-by-side (English labels via plot_learning_curves --lang en)
  figure4_coverage_vs_obstacle_density.png — heuristic sweep (see --tier)
  figure5_rollout_3d_trajectory_*.png — rollout PNG + optional descriptive top line (no “Figure N”)
  figure6_rollout_layer_occupancy_*.png — rollout heatmaps + optional descriptive top line (no “Figure N”)
  figure7_ablation_potential.png — regenerate bar chart if JSONL exists

Figure 4 default uses **medium** tier and capped steps for speed; use --tier hard
with smaller --episodes-per-point for heavy runs.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.grid_world_3d import GridWorld3DEnv
from src.policies.heuristic_actions import greedy_nearest_unvisited_actions, random_actions


def plot_figure1_voxel_env(
    out_path: Path,
    map_shape: tuple[int, int, int],
    num_uavs: int,
    obstacle_density: float,
    seed: int,
    dpi: int,
) -> None:
    env = GridWorld3DEnv(
        map_shape=map_shape,
        num_uavs=num_uavs,
        obstacle_density=obstacle_density,
        max_steps=3500,
        energy_budget=6000,
        seed=seed,
    )
    obs, _ = env.reset(seed=seed)
    _ = obs

    obs_idx = np.argwhere(env.obstacle_map)
    free_idx = np.argwhere(~env.obstacle_map)

    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    if len(obs_idx) > 0:
        ax.scatter(
            obs_idx[:, 2],
            obs_idx[:, 1],
            obs_idx[:, 0],
            c="#6c757d",
            s=36,
            alpha=0.75,
            marker="s",
            edgecolors="#343a40",
            linewidths=0.2,
            label="Obstacle voxels",
        )
    # subsample free voxels for clarity
    rng = np.random.default_rng(seed + 1)
    if len(free_idx) > 400:
        pick = rng.choice(len(free_idx), size=400, replace=False)
        free_idx = free_idx[pick]
    ax.scatter(
        free_idx[:, 2],
        free_idx[:, 1],
        free_idx[:, 0],
        c="#cfe2ff",
        s=8,
        alpha=0.35,
        marker=".",
        label="Traversable voxels (subsample)",
    )
    try:
        cmap = matplotlib.colormaps["tab10"]
    except AttributeError:
        cmap = plt.cm.get_cmap("tab10")
    for i, (l, r, c) in enumerate(env.uav_positions):
        ax.scatter(
            [c],
            [r],
            [l],
            color=cmap(i),
            s=220,
            marker="^",
            edgecolors="k",
            linewidths=0.8,
            label=f"UAV {i} start",
            zorder=10,
        )
    ax.set_xlabel("Column (x)")
    ax.set_ylabel("Row (y)")
    ax.set_zlabel("Layer (z)")
    L, H, W = map_shape
    ax.set_title(
        f"Voxelized environment (L×H×W = {L}×{H}×{W}, "
        f"ρ={obstacle_density:.2f}, N={num_uavs})"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=18, azim=-60)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_figure2_framework(out_path: Path, dpi: int) -> None:
    """Wider figure + pad_inches so FancyBboxPatch corners (esp. PBRS) are not clipped."""
    fig, ax = plt.subplots(figsize=(14.0, 6.2))
    ax.set_xlim(-0.15, 13.2)
    ax.set_ylim(-0.25, 5.95)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.04)

    def box(x, y, w, h, text, fc="#e9ecef", ec="#495057"):
        r = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.2, edgecolor=ec, facecolor=fc,
        )
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2, style="->", color="#212529"):
        a = FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
            linewidth=1.1, color=color, zorder=5,
        )
        ax.add_patch(a)

    box(0.75, 2.65, 2.35, 1.15, "Voxel env\n(GridWorld3D)", fc="#d0ebff")
    box(3.35, 2.65, 2.05, 1.15, "Local obs\n+ scalars", fc="#fff3bf")
    box(5.65, 2.65, 1.95, 1.15, "Agent nets\n(GRU–Q)", fc="#e7f5ff")
    box(7.85, 2.65, 2.45, 1.15, "QMIX mixer\n(monotonic)", fc="#ffe3e3")

    arrow(3.1, 3.225, 3.35, 3.225)
    arrow(5.4, 3.225, 5.65, 3.225)
    arrow(7.6, 3.225, 7.85, 3.225)

    box(3.15, 0.45, 3.15, 1.05, "CTDE:\ncentralized train /\ndistributed exec", fc="#e6fcf5")
    box(7.05, 0.45, 3.35, 1.05, "VDN / MAPPO\n(baselines)", fc="#f3f0ff")

    arrow(6.0, 2.65, 5.8, 1.55, color="#495057")
    arrow(9.0, 2.65, 8.7, 1.55, color="#495057")

    box(0.55, 4.25, 2.55, 0.72, "PBRS\n(potential ΔΦ)", fc="#fff9db", ec="#c92a2a")
    arrow(1.825, 4.25, 1.825, 3.85, color="#c92a2a")
    ax.text(2.38, 4.02, "shaping", ha="left", fontsize=8, color="#c92a2a")

    box(10.55, 4.25, 2.45, 0.72, "DER / recovery\n(optional, 2D foundation)", fc="#f8f9fa", ec="#868e96")
    # DER → QMIX mixer top (training stability auxiliary path)
    arrow(11.775, 4.25, 9.08, 3.82, color="#868e96")
    ax.text(10.35, 3.92, "train stab.", ha="center", fontsize=8, color="#495057")

    ax.text(
        6.55,
        5.55,
        "Cooperative coverage decision stack\n"
        "(QMIX + CTDE; PBRS as enabling reward shaping).",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.45)
    plt.close(fig)


def assemble_figure3_learning_curves(out_path: Path, dpi: int) -> None:
    """Stitch medium + hard QMIX/VDN curves with English titles (not the default zh PNGs)."""
    med_csv_q = ROOT / "experiments/curves/qmix_medium_s0.csv"
    med_csv_v = ROOT / "experiments/curves/vdn_medium_episode.csv"
    hard_csv_q = ROOT / "experiments/curves/qmix_hard_s0.csv"
    hard_csv_v = ROOT / "experiments/curves/vdn_hard_s0.csv"
    for p in (med_csv_q, med_csv_v, hard_csv_q, hard_csv_v):
        if not p.is_file():
            raise FileNotFoundError(f"Missing curve CSV: {p}")

    med_en = ROOT / "experiments/figures/learning_curve_medium_qmix_vdn_en.png"
    hard_en = ROOT / "experiments/figures/learning_curve_hard_qmix_vdn_en.png"
    plot_script = ROOT / "experiments/plot_learning_curves.py"
    subprocess.run(
        [
            sys.executable,
            str(plot_script),
            "--lang",
            "en",
            "--qmix",
            str(med_csv_q),
            "--vdn",
            str(med_csv_v),
            "--map",
            "grid_3d_medium",
            "--out",
            str(med_en),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(plot_script),
            "--lang",
            "en",
            "--qmix",
            str(hard_csv_q),
            "--vdn",
            str(hard_csv_v),
            "--map",
            "grid_3d_hard",
            "--out",
            str(hard_en),
        ],
        check=True,
    )

    im1 = Image.open(med_en).convert("RGB")
    im2 = Image.open(hard_en).convert("RGB")
    w = im1.width + im2.width + 40
    h = max(im1.height, im2.height) + 50
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(im1, (20, 40))
    canvas.paste(im2, (im1.width + 20, 40))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(dpi, dpi))


def sweep_coverage_vs_density(
    out_path: Path,
    tier: str,
    episodes_per_point: int,
    dpi: int,
) -> None:
    if tier == "medium":
        map_shape = (3, 6, 6)
        num_uavs = 2
        max_steps = 2000
        energy = 4500
        base_seed = 100
    else:
        map_shape = (4, 7, 7)
        num_uavs = 3
        max_steps = 2500
        energy = 6000
        base_seed = 200

    densities = np.array([0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18], dtype=np.float64)
    g_mean, g_std = [], []
    r_mean, r_std = [], []

    for j, rho in enumerate(densities):
        g_cov, r_cov = [], []
        for ep in range(episodes_per_point):
            seed_ep = int(base_seed + j * 97 + ep * 11)
            env = GridWorld3DEnv(
                map_shape=map_shape,
                num_uavs=num_uavs,
                obstacle_density=float(rho),
                max_steps=max_steps,
                energy_budget=energy,
                seed=seed_ep,
            )
            rng_ep = np.random.default_rng(seed_ep + 999)

            # Greedy
            obs, _ = env.reset(seed=seed_ep)
            term = trunc = False
            info = {}
            while not (term or trunc):
                a = greedy_nearest_unvisited_actions(env)
                obs, _, term, trunc, info = env.step(a)
            g_cov.append(float(info.get("coverage", 0.0)))

            # Random
            env = GridWorld3DEnv(
                map_shape=map_shape,
                num_uavs=num_uavs,
                obstacle_density=float(rho),
                max_steps=max_steps,
                energy_budget=energy,
                seed=seed_ep + 1,
            )
            obs, _ = env.reset(seed=seed_ep + 1)
            term = trunc = False
            info = {}
            while not (term or trunc):
                a = random_actions(env, rng_ep)
                obs, _, term, trunc, info = env.step(a)
            r_cov.append(float(info.get("coverage", 0.0)))

        g_mean.append(float(np.mean(g_cov)))
        g_std.append(float(np.std(g_cov, ddof=1)) if len(g_cov) > 1 else 0.0)
        r_mean.append(float(np.mean(r_cov)))
        r_std.append(float(np.std(r_cov, ddof=1)) if len(r_cov) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.errorbar(
        densities, r_mean, yerr=r_std, fmt="-o", capsize=4, color="#2f9e44",
        label="Random policy",
    )
    ax.errorbar(
        densities, g_mean, yerr=g_std, fmt="-s", capsize=4, color="#1971c2",
        label="Greedy heuristic",
    )
    ax.set_xlabel("Obstacle density ρ")
    ax.set_ylabel("Episode final coverage")
    ax.set_title(
        f"Coverage vs. obstacle density ({tier} tier, "
        f"{episodes_per_point} episodes/point, T_max={max_steps})"
    )
    ax.grid(True, alpha=0.35)
    ax.legend()
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def stamp_png_with_figure_line(src: Path, dst: Path, figure_line: str, dpi: int) -> None:
    """Add a top-of-canvas descriptive line (no document figure number) above rollout PNGs."""
    if not src.is_file():
        raise FileNotFoundError(src)
    arr = plt.imread(str(src))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    ih, iw = int(arr.shape[0]), int(arr.shape[1])
    fig_w = iw / dpi + 0.35
    fig_h = ih / dpi + 0.55
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[0.16, 1.0], hspace=0.1)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.axis("off")
    ax_top.text(
        0.5,
        0.5,
        figure_line,
        transform=ax_top.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    ax_img = fig.add_subplot(gs[1, 0])
    ax_img.imshow(arr, interpolation="nearest")
    ax_img.axis("off")
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def regenerate_figure7(out_path: Path, dpi: int) -> None:
    import subprocess

    with_json = ROOT / "experiments/results/medium_qmix_vdn.jsonl"
    without_json = ROOT / "experiments/results/ablation_medium.jsonl"
    if not with_json.is_file() or not without_json.is_file():
        shutil.copy2(ROOT / "experiments/figures/ablation_potential_comparison_en.png", out_path)
        return
    tmp = out_path.with_suffix(".tmp.png")
    cmd = [
        sys.executable,
        str(ROOT / "experiments/plot_potential_ablation.py"),
        "--with-potential",
        str(with_json),
        "--without-potential",
        str(without_json),
        "--out",
        str(tmp),
        "--lang",
        "en",
    ]
    subprocess.run(cmd, check=True)
    shutil.move(tmp, out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="experiments/figures/paper_sensors")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--tier", choices=("medium", "hard"), default="medium",
                   help="Tier for Fig.1 snapshot and Fig.4 density sweep")
    p.add_argument("--episodes-per-point", type=int, default=12)
    args = p.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.tier == "medium":
        ms, nu, od = (3, 6, 6), 2, 0.08
    else:
        ms, nu, od = (4, 7, 7), 3, 0.12

    f1 = out_dir / f"figure1_voxel_environment_{args.tier}.png"
    plot_figure1_voxel_env(f1, ms, nu, od, seed=42, dpi=args.dpi)
    print("Wrote", f1, flush=True)

    f2 = out_dir / "figure2_framework_ctde_pbrs_der.png"
    plot_figure2_framework(f2, dpi=args.dpi)
    print("Wrote", f2, flush=True)

    f3 = out_dir / "figure3_learning_curves_qmix_vdn.png"
    assemble_figure3_learning_curves(f3, dpi=args.dpi)
    print("Wrote", f3, flush=True)

    f4 = out_dir / f"figure4_coverage_vs_obstacle_density_{args.tier}.png"
    sweep_coverage_vs_density(f4, args.tier, args.episodes_per_point, dpi=args.dpi)
    print("Wrote", f4, flush=True)

    src5 = ROOT / "experiments/figures/rollout_3d_trajectory_grid_3d_hard_random_seed0_ep1.png"
    src6 = ROOT / "experiments/figures/rollout_layer_occupancy_grid_3d_hard_random_seed0_ep1.png"
    stamp_png_with_figure_line(
        src5,
        out_dir / "figure5_rollout_3d_trajectory_hard_random.png",
        "Random-policy rollout — 3D UAV trajectories (grid_3d_hard, seed 0, episode 1).",
        dpi=args.dpi,
    )
    stamp_png_with_figure_line(
        src6,
        out_dir / "figure6_rollout_layer_occupancy_hard_random.png",
        "Random-policy rollout — per-layer visit counts / step (grid_3d_hard, seed 0, episode 1).",
        dpi=args.dpi,
    )
    print("Wrote Fig 5–6 (stamped captions) from experiments/figures/ sources.", flush=True)

    f7 = out_dir / "figure7_ablation_potential_en.png"
    regenerate_figure7(f7, dpi=args.dpi)
    print("Wrote", f7, flush=True)

    paper_fig = ROOT / "paper" / "figures"
    paper_fig.mkdir(parents=True, exist_ok=True)
    for png in out_dir.glob("figure*.png"):
        shutil.copy2(png, paper_fig / png.name)
    print(f"Synced figure*.png -> {paper_fig.resolve()}")

    print(f"Done in {time.time()-t0:.1f}s. Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
