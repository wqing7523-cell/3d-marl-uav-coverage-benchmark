"""Aggregate eval_*.py JSON outputs into mean ± std table (multi-seed)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _aggregate_row(name: str, paths: Sequence[Path], metrics: List[str]) -> List[str]:
    arrays: Dict[str, List[float]] = {m: [] for m in metrics}
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Missing file: {path}")
        data = _load(path)
        for m in metrics:
            v = data.get(m)
            if v is None:
                arrays[m].append(float("nan"))
            else:
                arrays[m].append(float(v))

    cells = [name]
    for m in metrics:
        arr = np.array(arrays[m], dtype=float)
        mu = float(np.nanmean(arr))
        sd = float(np.nanstd(arr, ddof=1)) if np.sum(~np.isnan(arr)) > 1 else 0.0
        if m == "steps_mean":
            cells.append(f"{mu:.1f} ± {sd:.1f}")
        else:
            cells.append(f"{mu:.4f} ± {sd:.4f}")
    return cells


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Build multi-seed summary table from eval JSON files. "
            "Use --spec FILE on Windows to avoid cmd.exe quoting issues with '|'."
        )
    )
    p.add_argument(
        "--spec",
        type=Path,
        help='JSON file with {"rows":[{"name":"QMIX","paths":["..."]}, ...]}',
    )
    p.add_argument(
        "--row",
        action="append",
        dest="rows_cli",
        help='One row: MethodName|path1.json|path2.json',
    )
    p.add_argument(
        "--metrics",
        default="coverage_mean,success_mean,steps_mean",
        help="Comma-separated keys to aggregate (mean ± std over seeds)",
    )
    p.add_argument("--csv", action="store_true", help="Print CSV instead of Markdown")
    p.add_argument(
        "--skip-missing",
        action="store_true",
        help="Omit a method row if any of its JSON paths are missing (stderr warning).",
    )
    args = p.parse_args()

    if args.spec and args.rows_cli:
        raise SystemExit("Use either --spec or --row, not both.")
    if not args.spec and not args.rows_cli:
        raise SystemExit("Provide --spec PATH or one/more --row ...")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    tier_sections: List[Tuple[str, List[Tuple[str, List[Path]]]]] = []

    if args.spec:
        spec_path = args.spec
        if not spec_path.is_file():
            raise SystemExit(f"Spec not found: {spec_path}")
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)

        def rows_from_block(block: Dict[str, Any]) -> List[Tuple[str, List[Path]]]:
            jobs: List[Tuple[str, List[Path]]] = []
            for row in block.get("rows", []):
                name = str(row["name"])
                paths = [_resolve(x) for x in row["paths"]]
                jobs.append((name, paths))
            return jobs

        if "tiers" in spec:
            for tier in spec["tiers"]:
                title = str(tier.get("title", "Scenario"))
                tier_sections.append((title, rows_from_block(tier)))
        else:
            tier_sections.append(("", rows_from_block(spec)))
    else:
        assert args.rows_cli is not None
        jobs: List[Tuple[str, List[Path]]] = []
        for line in args.rows_cli:
            parts = line.split("|")
            if len(parts) < 2:
                raise SystemExit(f"Bad --row (need Name|path1|path2...): {line}")
            name = parts[0].strip()
            paths = [_resolve(x.strip()) for x in parts[1:] if x.strip()]
            jobs.append((name, paths))
        tier_sections.append(("", jobs))

    print("Aggregating...", flush=True)
    header = ["Method"] + metrics

    def emit_table(rows_out: List[List[str]]) -> None:
        if args.csv:
            w = csv.writer(sys.stdout)
            w.writerow(header)
            for r in rows_out:
                w.writerow(r)
        else:
            print("| " + " | ".join(header) + " |")
            print("| " + " | ".join(["---"] * len(header)) + " |")
            for r in rows_out:
                print("| " + " | ".join(r) + " |")

    first = True
    for title, row_jobs in tier_sections:
        rows_out: List[List[str]] = []
        for name, paths in row_jobs:
            if args.skip_missing and any(not p.is_file() for p in paths):
                print(f"[skip-missing] row={name} missing one of: {paths}", file=sys.stderr)
                continue
            rows_out.append(_aggregate_row(name, paths, metrics))
        if not rows_out:
            print(f"[skip-missing] tier empty: {title!r}", file=sys.stderr)
            continue
        if not first and not args.csv:
            print()
        first = False
        if title and not args.csv:
            print(f"### {title}")
            print()
        emit_table(rows_out)


if __name__ == "__main__":
    main()
