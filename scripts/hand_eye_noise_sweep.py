#!/usr/bin/env python3
"""Sweep hand-eye AX=XB sensitivity to noise and pose-sample count (simulated data)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robot_pose_pipeline.hand_eye import (
    HAND_EYE_METHODS,
    evaluate_hand_eye,
    generate_eye_in_hand_dataset,
    solve_eye_in_hand,
)


def _run_one(
    samples: int,
    rotation_noise_deg: float,
    translation_noise_m: float,
    seed: int,
    methods: list[str],
) -> dict:
    dataset = generate_eye_in_hand_dataset(
        sample_count=samples,
        seed=seed,
        noise_rotation_deg=rotation_noise_deg,
        noise_translation=translation_noise_m,
    )
    method_metrics: dict[str, dict[str, float]] = {}
    for name in methods:
        estimated = solve_eye_in_hand(
            dataset.base_to_gripper,
            dataset.camera_to_target,
            HAND_EYE_METHODS[name],
        )
        method_metrics[name] = evaluate_hand_eye(
            estimated, dataset.gripper_to_camera_ground_truth
        )
    return {
        "samples": samples,
        "rotation_noise_deg": rotation_noise_deg,
        "translation_noise_m": translation_noise_m,
        "seed": seed,
        "methods": method_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-counts",
        type=int,
        nargs="+",
        default=[8, 15, 30, 50],
        help="Pose-pair counts (motion diversity proxy)",
    )
    parser.add_argument(
        "--rotation-noise-degs",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.5, 1.0, 2.0],
        help="Rotation noise sigma in degrees",
    )
    parser.add_argument(
        "--translation-noise-ms",
        type=float,
        nargs="+",
        default=[0.0, 0.001, 0.002, 0.005],
        help="Translation noise sigma in meters",
    )
    parser.add_argument("--methods", nargs="+", default=["tsai", "park", "horaud"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trials", type=int, default=3, help="Seeds per setting (seed..seed+trials-1)")
    parser.add_argument("--output", default="outputs/hand_eye_noise_sweep.json")
    args = parser.parse_args()

    for name in args.methods:
        if name not in HAND_EYE_METHODS:
            raise SystemExit(f"Unknown method {name}; choose from {list(HAND_EYE_METHODS)}")

    rows: list[dict] = []
    # 1) Fix samples=30, sweep noise (paired rot/trans grids on diagonal + extremes)
    noise_grid = [
        (r, t)
        for r in args.rotation_noise_degs
        for t in args.translation_noise_ms
        if (r == 0.0 and t == 0.0)
        or (r > 0 and t > 0 and abs(args.rotation_noise_degs.index(r) - args.translation_noise_ms.index(t)) <= 1)
        or (r in (0.5, 1.0, 2.0) and t in (0.001, 0.002, 0.005))
    ]
    # de-duplicate while preserving order
    seen: set[tuple[float, float]] = set()
    unique_noise: list[tuple[float, float]] = []
    for pair in noise_grid:
        if pair not in seen:
            seen.add(pair)
            unique_noise.append(pair)

    for rot, trans in unique_noise:
        for trial in range(args.trials):
            rows.append(
                _run_one(30, rot, trans, args.seed + trial, args.methods)
            )

    # 2) Fix moderate noise, sweep sample count
    for n in args.sample_counts:
        for trial in range(args.trials):
            rows.append(
                _run_one(n, 0.5, 0.002, args.seed + 100 + trial, args.methods)
            )

    # Aggregate mean metrics for reporting
    def key_of(row: dict) -> tuple:
        return (row["samples"], row["rotation_noise_deg"], row["translation_noise_m"])

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(key_of(row), []).append(row)

    summary = []
    for key, items in sorted(groups.items()):
        samples, rot, trans = key
        entry = {
            "samples": samples,
            "rotation_noise_deg": rot,
            "translation_noise_m": trans,
            "trials": len(items),
            "methods": {},
        }
        for method in args.methods:
            rots = [it["methods"][method]["rotation_error_deg"] for it in items]
            trans_errs = [it["methods"][method]["translation_error"] for it in items]
            entry["methods"][method] = {
                "rotation_error_deg_mean": float(np.mean(rots)),
                "rotation_error_deg_std": float(np.std(rots)),
                "translation_error_mean": float(np.mean(trans_errs)),
                "translation_error_std": float(np.std(trans_errs)),
            }
        summary.append(entry)

    payload = {
        "note": (
            "Simulated eye-in-hand AX=XB. T_gripper_camera is simulated extrinsic, "
            "NOT a real-robot calibration result."
        ),
        "methods": args.methods,
        "summary": summary,
        "raw_trials": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("samples  rot_noise  trans_noise  method    rot_err_deg  trans_err_m")
    for entry in summary:
        for method, m in entry["methods"].items():
            print(
                f"{entry['samples']:7d}  {entry['rotation_noise_deg']:9.2f}  "
                f"{entry['translation_noise_m']:11.4f}  {method:8s}  "
                f"{m['rotation_error_deg_mean']:11.4f}  {m['translation_error_mean']:.6f}"
            )
    print(output.resolve())


if __name__ == "__main__":
    main()
