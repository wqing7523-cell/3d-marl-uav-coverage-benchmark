"""
C6 方案 B：Hard 档 QMIX/VDN，补训 seed 3–4 后做 50-episode greedy eval。

一次性跑完（训练 + 评估 + 附表 E 统计）：

    python experiments/run_c6_scheme_b_hard.py

可选：已有 checkpoint 时跳过训练，仅 eval + 统计：

    python experiments/run_c6_scheme_b_hard.py --skip-train

断点续跑（跳过 manifest 里已完成的训练，不重复跑 seed 3）：

    python experiments/run_c6_scheme_b_hard.py --skip-completed
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "experiments" / "checkpoints"
EVAL_DIR = ROOT / "experiments" / "results_eval"
RESULTS_JSONL = ROOT / "experiments" / "results" / "hard_qmix_vdn.jsonl"
OUT_TABLE_E = ROOT / "paper" / "appendix_stats_table_e_n5_hard.md"
MANIFEST = EVAL_DIR / "c6_scheme_b_hard_manifest.json"

ENV_CONFIG = "configs/envs/grid_3d_hard.yaml"
BASE_CONFIG = "configs/base.yaml"
ALGO = {"qmix": "configs/algos/qmix_paper.yaml", "vdn": "configs/algos/vdn.yaml"}
CKPT_PREFIX = {"qmix": "qmix3d_L4_H7_W7_uavs3_obs012", "vdn": "vdn3d_L4_H7_W7_uavs3_obs012"}
NEW_SEEDS = (3, 4)
EPISODES = 50


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"Command failed (code {r.returncode}): {' '.join(cmd)}")


def load_manifest() -> dict[str, str]:
    if MANIFEST.is_file():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def save_manifest_entry(method: str, seed: int, ckpt: Path) -> None:
    data = load_manifest()
    data[f"{method}_s{seed}"] = str(ckpt.resolve())
    MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_checkpoint(method: str, seed: int, not_before: float) -> Path:
    key = f"{method}_s{seed}"
    data = load_manifest()
    if key in data and Path(data[key]).is_file():
        return Path(data[key])
    return newest_checkpoint(method, not_before)


def newest_checkpoint(method: str, not_before: float) -> Path:
    pattern = f"{CKPT_PREFIX[method]}_*.pt"
    cands = [
        p
        for p in CKPT_DIR.glob(pattern)
        if p.is_file() and p.stat().st_mtime >= not_before - 2.0
    ]
    if not cands:
        raise FileNotFoundError(
            f"No new checkpoint matching {pattern} under {CKPT_DIR}. "
            "Train may have failed or checkpoint_dir differs from configs/base.yaml."
        )
    return max(cands, key=lambda p: p.stat().st_mtime)


def train(method: str, seed: int) -> Path:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run(
        [
            sys.executable,
            str(ROOT / "train_qmix_3d.py"),
            "--base-config",
            BASE_CONFIG,
            "--env-config",
            ENV_CONFIG,
            "--algo-config",
            ALGO[method],
            "--seed",
            str(seed),
            "--results-file",
            str(RESULTS_JSONL),
        ]
    )
    ckpt = newest_checkpoint(method, t0)
    save_manifest_entry(method, seed, ckpt)
    print(f"    checkpoint: {ckpt}", flush=True)
    return ckpt


def eval_ckpt(method: str, seed: int, ckpt: Path) -> Path:
    out = EVAL_DIR / f"{method}_hard_s{seed}.json"
    run(
        [
            sys.executable,
            str(ROOT / "eval_3d.py"),
            "--checkpoint",
            str(ckpt),
            "--env-config",
            ENV_CONFIG,
            "--episodes",
            str(EPISODES),
            "--seed",
            str(seed),
            "--output-json",
            str(out),
        ]
    )
    return out


def load_success(method: str, seeds: range) -> list[float]:
    vals: list[float] = []
    for s in seeds:
        p = EVAL_DIR / f"{method}_hard_s{s}.json"
        if not p.is_file():
            raise FileNotFoundError(f"Missing eval JSON: {p}")
        d = json.loads(p.read_text(encoding="utf-8"))
        vals.append(float(d["success_mean"]))
    return vals


def bootstrap_ci(values: list[float], n_boot: int = 10000) -> tuple[float, float, float]:
    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean, mean
    rng = np.random.default_rng(42)
    boots = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
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


def write_table_e() -> None:
    seeds5 = range(5)
    qmix = load_success("qmix", seeds5)
    vdn = load_success("vdn", seeds5)
    mq, lq, hq = bootstrap_ci(qmix)
    mv, lv, hv = bootstrap_ci(vdn)
    p = mann_whitney_p(qmix, vdn)
    lines = [
        "**附表 E** Hard 档 QMIX vs VDN（$n=5$ 种子；seed 0–2 与主表同源，seed 3–4 为 C6 补训后 `eval_3d.py` 50 episodes）",
        "",
        "| 算法 | 五种子 success | mean | bootstrap 95% CI |",
        "|------|----------------|------|------------------|",
        f"| QMIX | {', '.join(f'{x:.3f}' for x in qmix)} | {mq:.3f} | [{lq:.3f}, {hq:.3f}] |",
        f"| VDN | {', '.join(f'{x:.3f}' for x in vdn)} | {mv:.3f} | [{lv:.3f}, {hv:.3f}] |",
        "",
        f"- Mann–Whitney U（QMIX vs VDN，$n=5$）$p \\approx {p:.3f}" if p is not None else "- 未安装 scipy，未报告 p 值",
        "",
        "主文表 3 仍为 $n=3$ 描述性主表；本表为探索性扩展，勿与 MAPPO 横比。",
        "",
    ]
    OUT_TABLE_E.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_TABLE_E}", flush=True)
    print(f"  QMIX n=5: mean={mq:.3f} CI=[{lq:.3f},{hq:.3f}]", flush=True)
    print(f"  VDN  n=5: mean={mv:.3f} CI=[{lv:.3f},{hv:.3f}]", flush=True)
    if p is not None:
        print(f"  Mann-Whitney p={p:.3f}", flush=True)


def manifest_has(method: str, seed: int) -> bool:
    key = f"{method}_s{seed}"
    data = load_manifest()
    return key in data and Path(data[key]).is_file()


def main() -> None:
    ap = argparse.ArgumentParser(description="C6 scheme B: Hard QMIX/VDN seeds 3-4 train+eval")
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip all training; eval using manifest/newest checkpoint",
    )
    ap.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip training when manifest already has checkpoint for that method/seed",
    )
    ap.add_argument("--methods", default="qmix,vdn", help="Comma-separated: qmix,vdn")
    ap.add_argument("--seeds", default="3,4", help="Comma-separated seed ids, e.g. 4 or 3,4")
    args = ap.parse_args()

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    ckpt_map: dict[tuple[str, int], Path] = {}

    for method in methods:
        for seed in seeds:
            key = (method, seed)
            if args.skip_train:
                ckpt = resolve_checkpoint(method, seed, 0.0)
                print(f"[skip-train] {method} seed={seed} -> {ckpt}", flush=True)
            elif args.skip_completed and manifest_has(method, seed):
                ckpt = Path(load_manifest()[f"{method}_s{seed}"])
                print(f"[skip-completed] {method} seed={seed} -> {ckpt}", flush=True)
            else:
                print(f"\n=== Train {method.upper()} seed={seed} (Hard, 2000 episodes) ===", flush=True)
                ckpt = train(method, seed)
            ckpt_map[key] = ckpt

    for method in methods:
        for seed in seeds:
            out_json = EVAL_DIR / f"{method}_hard_s{seed}.json"
            if out_json.is_file() and args.skip_completed:
                print(f"[skip-completed] eval exists: {out_json}", flush=True)
                continue
            print(f"\n=== Eval {method.upper()} seed={seed} ({EPISODES} episodes) ===", flush=True)
            eval_ckpt(method, seed, ckpt_map[(method, seed)])

    print("\n=== Table E (n=5 Hard QMIX vs VDN) ===", flush=True)
    write_table_e()
    print("\nC6 scheme B finished.", flush=True)


if __name__ == "__main__":
    main()
