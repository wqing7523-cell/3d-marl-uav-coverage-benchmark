"""Summarize JSONL lines from train_qmix_3d --results-file (mean/std per method)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=str)
    args = p.parse_args()
    path = Path(args.jsonl)
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    by_method: DefaultDict[str, List[float]] = defaultdict(list)
    by_method_steps: DefaultDict[str, List[float]] = defaultdict(list)
    for r in rows:
        m = r.get("method", "unknown")
        by_method[m].append(float(r.get("coverage_mean", 0.0)))
        by_method_steps[m].append(float(r.get("steps_mean", 0.0)))

    print(f"Loaded {len(rows)} runs from {path}")
    for m in sorted(by_method.keys()):
        arr = np.array(by_method[m], dtype=float)
        st = np.array(by_method_steps[m], dtype=float)
        print(
            f"{m:6s}  coverage_mean {arr.mean():.4f} ± {arr.std(ddof=1) if len(arr) > 1 else 0.0:.4f}  "
            f"(n={len(arr)})  steps_mean {st.mean():.1f} ± {st.std(ddof=1) if len(st) > 1 else 0.0:.1f}"
        )


if __name__ == "__main__":
    main()
