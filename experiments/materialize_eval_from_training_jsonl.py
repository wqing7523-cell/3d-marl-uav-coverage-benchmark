"""Write eval-shaped JSON files from training-result JSONL (QMIX/VDN).

Training summaries align manuscript Tables 1 (medium) / 3 (hard). For strict
post-hoc 50-episode ``eval_3d.py`` metrics, overwrite outputs by running the
corresponding ``run_eval_*`` batch once checkpoints exist on disk.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent

TIER_CONFIG: Dict[str, Dict[str, Any]] = {
    "hard": {
        "jsonl": ROOT / "experiments/results/hard_qmix_vdn.jsonl",
        "table_ref": "Table 3 / hard_qmix_vdn.jsonl",
        "ckpt": {
            "qmix": [
                "qmix3d_L4_H7_W7_uavs3_obs012_1775082129.pt",
                "qmix3d_L4_H7_W7_uavs3_obs012_1775126118.pt",
                "qmix3d_L4_H7_W7_uavs3_obs012_1775171227.pt",
            ],
            "vdn": [
                "vdn3d_L4_H7_W7_uavs3_obs012_1775256641.pt",
                "vdn3d_L4_H7_W7_uavs3_obs012_1775284777.pt",
                "vdn3d_L4_H7_W7_uavs3_obs012_1775316135.pt",
            ],
        },
        "prefix": {"qmix": "qmix_hard_s", "vdn": "vdn_hard_s"},
    },
    "medium": {
        "jsonl": ROOT / "experiments/results/medium_qmix_vdn.jsonl",
        "table_ref": "Table 1 / medium_qmix_vdn.jsonl",
        "ckpt": {
            "qmix": [
                "qmix3d_L3_H6_W6_uavs2_obs008_1778068048.pt",
                "qmix3d_L3_H6_W6_uavs2_obs008_1778183251.pt",
                "qmix3d_L3_H6_W6_uavs2_obs008_1778267952.pt",
            ],
            "vdn": [
                "vdn3d_L3_H6_W6_uavs2_obs008_1774861056.pt",
                "vdn3d_L3_H6_W6_uavs2_obs008_1774876572.pt",
                "vdn3d_L3_H6_W6_uavs2_obs008_1774892113.pt",
            ],
        },
        "prefix": {"qmix": "qmix_medium_s", "vdn": "vdn_medium_s"},
    },
}


def _parse_map_shape(row: Dict[str, Any]) -> List[int]:
    ms = row.get("map_shape")
    if isinstance(ms, str) and "x" in ms:
        return [int(x) for x in ms.split("x")]
    if isinstance(ms, (list, tuple)) and len(ms) == 3:
        return [int(ms[0]), int(ms[1]), int(ms[2])]
    raise ValueError(f"Cannot parse map_shape from row: {row}")


def run_one_tier(
    tier: str,
    jsonl_path: Path,
    out_dir: Path,
    ckpt_dir: Path,
) -> None:
    cfg = TIER_CONFIG[tier]
    ckpt_hints: Dict[str, List[str]] = cfg["ckpt"]
    prefixes: Dict[str, str] = cfg["prefix"]

    rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        m = str(r.get("method", "")).lower()
        by_method.setdefault(m, []).append(r)

    out_dir.mkdir(parents=True, exist_ok=True)

    for method, items in by_method.items():
        prefix = prefixes.get(method)
        if prefix is None:
            continue
        hints = ckpt_hints.get(method, [None] * len(items))
        for i, row in enumerate(items):
            seed = int(row.get("seed", i))
            ckpt_name = hints[i] if i < len(hints) else None
            ckpt_path = str((ckpt_dir / ckpt_name).resolve()) if ckpt_name else None
            map_shape = _parse_map_shape(row)

            out: Dict[str, Any] = {
                "checkpoint": ckpt_path,
                "mixer_type": method,
                "map_shape": map_shape,
                "num_uavs": int(row.get("num_uavs", 2)),
                "obstacle_density": float(row.get("obstacle_density", 0.0)),
                "episodes": None,
                "seed": seed,
                "coverage_mean": float(row.get("coverage_mean", 0.0)),
                "coverage_std": float(row.get("coverage_std", 0.0)),
                "steps_mean": float(row.get("steps_mean", 0.0)),
                "success_mean": float(row.get("success_mean", 0.0)),
                "metrics_source": "training_summary_jsonl",
                "metrics_note": (
                    f"Aligned with {cfg['table_ref']}. "
                    "Replace with eval_3d.py --episodes 50 when checkpoints exist."
                ),
            }
            dest = out_dir / f"{prefix}{seed}.json"
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"Wrote {dest}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tier",
        choices=("medium", "hard", "both"),
        default="both",
        help="Which tier JSON to emit from training summaries.",
    )
    p.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Override JSONL path (defaults per tier).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "experiments/results_eval",
        help="Directory for output *_s*.json files.",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "experiments/checkpoints",
        help="Used only to fill checkpoint paths in JSON metadata.",
    )
    args = p.parse_args()

    tiers: Tuple[str, ...]
    if args.tier == "both":
        tiers = ("medium", "hard")
    else:
        tiers = (args.tier,)

    for tier in tiers:
        jsonl = args.jsonl if args.jsonl is not None else TIER_CONFIG[tier]["jsonl"]
        if not jsonl.is_file():
            print(f"[skip] tier={tier}: JSONL not found: {jsonl}", flush=True)
            continue
        run_one_tier(tier, jsonl, args.out_dir, args.checkpoint_dir)


if __name__ == "__main__":
    main()
