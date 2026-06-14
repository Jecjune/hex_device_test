#!/usr/bin/env python3
"""Verify load_waypoints_from_json against raw JSON samples."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from hex_device_test.tools.trajectory_loader import load_waypoints_from_json


def _expected_waypoint_count(sample_count: int, stride: int) -> int:
    return (sample_count + stride - 1) // stride


def _load_raw_samples(json_path: Path) -> Tuple[dict, List[dict]]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data, data["samples"]


def _pick_spot_indices(total: int, count: int) -> List[int]:
    if total <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count <= 1:
        return [0]
    step = (total - 1) / (count - 1)
    indices = [int(round(i * step)) for i in range(count)]
    indices[0] = 0
    indices[-1] = total - 1
    return sorted(set(indices))


def _positions_from_joint_feedback(sample: dict) -> Optional[List[float]]:
    feedback = sample.get("joint_feedback")
    if not feedback:
        return None
    ordered = sorted(feedback, key=lambda item: item["joint_index"])
    return [item["position_rad"] for item in ordered]


def _compare_waypoint(
    waypoint_idx: int,
    got: List[float],
    expected: List[float],
) -> List[str]:
    errors: List[str] = []
    if got != expected:
        errors.append(f"waypoint[{waypoint_idx}] positions_rad mismatch")
        errors.append(f"  got:      {got}")
        errors.append(f"  expected: {expected}")
    if not all(isinstance(v, (int, float)) for v in got):
        errors.append(
            f"waypoint[{waypoint_idx}] contains non-numeric values: {got}"
        )
    return errors


def verify(
    json_path: Path,
    stride: int,
    max_errors: int = 10,
) -> Tuple[List[str], int, int]:
    errors: List[str] = []
    checked = 0
    mismatch_count = 0

    if not json_path.is_file():
        return [f"file not found: {json_path}"], 0, 0

    meta, samples = _load_raw_samples(json_path)
    expected_positions = [s["positions_rad"] for s in samples[::stride]]
    expected_count = _expected_waypoint_count(len(samples), stride)

    try:
        waypoints = load_waypoints_from_json(json_path, stride=stride)
    except Exception as exc:
        return [f"load_waypoints_from_json raised: {exc}"], 0, 0

    if len(waypoints) != len(expected_positions):
        errors.append(
            f"waypoint count mismatch: got {len(waypoints)}, "
            f"expected {len(expected_positions)} (samples={len(samples)}, stride={stride})"
        )

    if len(waypoints) != expected_count:
        errors.append(
            f"waypoint count vs formula mismatch: got {len(waypoints)}, "
            f"expected {expected_count}"
        )

    sample_count = meta.get("sample_count")
    if sample_count is not None and sample_count != len(samples):
        errors.append(
            f"metadata sample_count={sample_count} != len(samples)={len(samples)}"
        )

    joint_count = meta.get("joint_count")
    if joint_count is not None:
        for idx, wp in enumerate(waypoints):
            if len(wp) != joint_count:
                errors.append(
                    f"waypoint[{idx}] length {len(wp)} != joint_count {joint_count}"
                )
                break

    for idx, (got, expected) in enumerate(zip(waypoints, expected_positions)):
        checked += 1
        frame_errors = _compare_waypoint(idx, got, expected)
        if frame_errors:
            mismatch_count += 1
            if len(errors) < max_errors:
                errors.extend(frame_errors)
            elif len(errors) == max_errors:
                errors.append("... additional mismatches omitted ...")

    return errors, checked, mismatch_count


def verify_spot_frames(
    json_path: Path,
    stride: int,
    spot_count: int,
    frame_indices: Optional[List[int]] = None,
) -> List[str]:
    errors: List[str] = []
    meta, samples = _load_raw_samples(json_path)
    waypoints = load_waypoints_from_json(json_path, stride=stride)

    if frame_indices:
        indices = [i for i in frame_indices if 0 <= i < len(waypoints)]
        invalid = [i for i in frame_indices if i < 0 or i >= len(waypoints)]
        for idx in invalid:
            errors.append(
                f"frame index {idx} out of range [0, {len(waypoints) - 1}]"
            )
    else:
        indices = _pick_spot_indices(len(waypoints), spot_count)

    print("Spot frame checks:")
    for waypoint_idx in indices:
        sample_idx = waypoint_idx * stride
        sample = samples[sample_idx]
        got = waypoints[waypoint_idx]
        expected = sample["positions_rad"]
        frame_index = sample.get("frame_index", "n/a")
        feedback_positions = _positions_from_joint_feedback(sample)

        ok_loader = got == expected
        ok_feedback = (
            feedback_positions is None or got == feedback_positions
        )

        status = "OK" if ok_loader and ok_feedback else "FAIL"
        print(
            f"  [{status}] waypoint={waypoint_idx}, "
            f"sample={sample_idx}, frame_index={frame_index}"
        )
        print(f"         loader:   {got}")

        if not ok_loader:
            errors.append(
                f"spot waypoint[{waypoint_idx}] loader vs sample mismatch"
            )
            errors.append(f"  loader:   {got}")
            errors.append(f"  sample:   {expected}")

        if feedback_positions is not None:
            print(f"         feedback: {feedback_positions}")
            if not ok_feedback:
                errors.append(
                    f"spot waypoint[{waypoint_idx}] loader vs joint_feedback mismatch"
                )
                errors.append(f"  loader:   {got}")
                errors.append(f"  feedback: {feedback_positions}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify trajectory JSON position extraction"
    )
    parser.add_argument(
        "--points-json",
        type=Path,
        required=True,
        help="Recorded trajectory JSON file to verify",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sample every N frames (default: 1)",
    )
    parser.add_argument(
        "--spots",
        type=int,
        default=10,
        help="Number of evenly distributed frames to print in detail (default: 10)",
    )
    parser.add_argument(
        "--frame-indices",
        type=str,
        default=None,
        help="Comma-separated waypoint indices to check, e.g. 0,1,100,500",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="Maximum mismatch details to print for full scan (default: 10)",
    )
    args = parser.parse_args()

    if args.stride < 1:
        print(f"FAIL: stride must be >= 1, got {args.stride}")
        return 1
    if args.spots < 1:
        print(f"FAIL: spots must be >= 1, got {args.spots}")
        return 1

    frame_indices = None
    if args.frame_indices:
        try:
            frame_indices = [
                int(item.strip()) for item in args.frame_indices.split(",") if item.strip()
            ]
        except ValueError:
            print(f"FAIL: invalid --frame-indices: {args.frame_indices}")
            return 1

    json_path = args.points_json.resolve()
    print(f"JSON:   {json_path}")
    print(f"stride: {args.stride}")
    print()

    errors, checked, mismatch_count = verify(
        json_path,
        stride=args.stride,
        max_errors=args.max_errors,
    )
    spot_errors = verify_spot_frames(
        json_path,
        stride=args.stride,
        spot_count=args.spots,
        frame_indices=frame_indices,
    )
    errors.extend(spot_errors)

    waypoints = load_waypoints_from_json(json_path, stride=args.stride)
    meta, samples = _load_raw_samples(json_path)

    print()
    print(f"Full scan: checked {checked} waypoints, mismatches {mismatch_count}")

    if errors:
        print("FAIL")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("PASS")
    print(f"  samples in file:     {len(samples)}")
    if meta.get("sample_count") is not None:
        print(f"  sample_count meta:   {meta['sample_count']}")
    print(f"  waypoints extracted: {len(waypoints)}")
    print(f"  joint_count meta:    {meta.get('joint_count', 'n/a')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
