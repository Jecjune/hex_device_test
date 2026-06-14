#!/usr/bin/env python3
"""Check sample_interval_s consistency across trajectory JSON files."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from hex_device_test.tools.trajectory_loader import read_trajectory_metadata


def check_files(json_paths: List[Path]) -> int:
    if not json_paths:
        print("No JSON files given.")
        return 1

    rows: List[Dict[str, Optional[float]]] = []
    errors: List[str] = []

    for json_path in json_paths:
        resolved = json_path.resolve()
        if not resolved.is_file():
            errors.append(f"file not found: {resolved}")
            continue
        try:
            meta = read_trajectory_metadata(resolved)
        except Exception as exc:
            errors.append(f"{resolved}: {exc}")
            continue

        interval = meta.get("sample_interval_s")
        if interval is None:
            errors.append(f"{resolved}: missing sample_interval_s")
            continue

        rows.append({
            "path": resolved,
            "sample_interval_s": float(interval),
            "sample_count": meta.get("sample_count"),
            "joint_count": meta.get("joint_count"),
        })

    if errors:
        print("FAIL")
        for err in errors:
            print(f"  - {err}")
        return 1

    if not rows:
        print("FAIL: no valid files")
        return 1

    name_width = max(len(row["path"].name) for row in rows)
    print(f"{'file':<{name_width}}  sample_interval_s  sample_count  joint_count")
    print("-" * (name_width + 40))

    intervals = set()
    for row in rows:
        intervals.add(row["sample_interval_s"])
        sample_count = row["sample_count"] if row["sample_count"] is not None else "n/a"
        joint_count = row["joint_count"] if row["joint_count"] is not None else "n/a"
        print(
            f"{row['path'].name:<{name_width}}  "
            f"{row['sample_interval_s']:<17}  {sample_count!s:<12}  {joint_count}"
        )

    print()
    if len(intervals) == 1:
        print(f"PASS: all {len(rows)} file(s) have sample_interval_s = {intervals.pop()}")
        return 0

    print("FAIL: sample_interval_s mismatch")
    by_interval: Dict[float, List[str]] = {}
    for row in rows:
        by_interval.setdefault(row["sample_interval_s"], []).append(row["path"].name)
    for interval, names in sorted(by_interval.items()):
        print(f"  {interval}: {', '.join(names)}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check sample_interval_s consistency across trajectory JSON files"
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        type=Path,
        metavar="JSON",
        help="Trajectory JSON file(s) to check",
    )
    args = parser.parse_args()
    return check_files(args.json_files)


if __name__ == "__main__":
    sys.exit(main())
