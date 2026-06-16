"""Rebuild附表 E from hard_qmix_vdn.jsonl (same protocol as Table 3, n=5 seeds)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "experiments" / "results" / "hard_qmix_vdn.jsonl"
OUT = ROOT / "paper" / "appendix_stats_table_e_n5_hard.md"
MANIFEST = ROOT / "experiments" / "results_eval" / "c6_scheme_b_hard_manifest.json"


def load_jsonl() -> list[dict]:
    rows = []
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def success_by_seed(method: str, seeds: range) -> list[float]:
    rows = load_jsonl()
    by_seed = {int(r["seed"]): float(r["success_mean"]) for r in rows if r["method"] == method}
    return [by_seed[s] for s in seeds]


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
    _, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(p)


def sync_eval_json(method: str, seed: int, row: dict, ckpt: str | None) -> None:
    out_dir = ROOT / "experiments" / "results_eval"
    ms = row["map_shape"].split("x")
    out = {
        "checkpoint": ckpt,
        "mixer_type": method,
        "map_shape": [int(ms[0]), int(ms[1]), int(ms[2])],
        "num_uavs": int(row["num_uavs"]),
        "obstacle_density": float(row["obstacle_density"]),
        "episodes": None,
        "seed": seed,
        "coverage_mean": float(row["coverage_mean"]),
        "coverage_std": float(row["coverage_std"]),
        "steps_mean": float(row["steps_mean"]),
        "success_mean": float(row["success_mean"]),
        "metrics_source": "training_summary_jsonl",
        "metrics_note": "C6 n=5 extension; same training-end aggregate as Table 3 (not greedy eval_3d.py).",
    }
    path = out_dir / f"{method}_hard_s{seed}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}
    rows = load_jsonl()
    for r in rows:
        m = str(r["method"])
        s = int(r["seed"])
        if s > 4:
            continue
        ckpt = manifest.get(f"{m}_s{s}")
        sync_eval_json(m, s, r, ckpt)

    seeds5 = range(5)
    qmix = success_by_seed("qmix", seeds5)
    vdn = success_by_seed("vdn", seeds5)
    mq, lq, hq = bootstrap_ci(qmix)
    mv, lv, hv = bootstrap_ci(vdn)
    p = mann_whitney_p(qmix, vdn)

    lines = [
        "**附表 E** Hard 档 QMIX vs VDN（$n=5$ 种子；与表 3 **同协议**：各 seed 训练 2000 episodes 后的 **训练汇总 success mean**，非 `eval_3d.py` greedy 复评）",
        "",
        "| 算法 | 五种子 success | mean | bootstrap 95% CI |",
        "|------|----------------|------|------------------|",
        f"| QMIX | {', '.join(f'{x:.3f}' for x in qmix)} | {mq:.3f} | [{lq:.3f}, {hq:.3f}] |",
        f"| VDN | {', '.join(f'{x:.3f}' for x in vdn)} | {mv:.3f} | [{lv:.3f}, {hv:.3f}] |",
        "",
        f"- Mann–Whitney U（QMIX vs VDN，$n=5$）$p \\approx {p:.3f}" if p is not None else "- 未安装 scipy",
        "",
        "**说明**：`eval_3d.py` 纯 greedy 在本仓库 checkpoint 上 success≈0（与训练期 ε-greedy 汇总不一致）；附表 E 与主表口径一致，采用 `hard_qmix_vdn.jsonl`。",
        "",
        "主文表 3 仍为 $n=3$ 描述性主表；本表为 $n=5$ 探索性扩展。",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT)
    print(f"QMIX n=5 mean={mq:.3f} CI=[{lq:.3f},{hq:.3f}]")
    print(f"VDN  n=5 mean={mv:.3f} CI=[{lv:.3f},{hv:.3f}]")
    if p is not None:
        print(f"Mann-Whitney p={p:.3f}")


if __name__ == "__main__":
    main()
