from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robot_pose_pipeline.pose_metrics import evaluate_pose


def load_matrix(path: str) -> np.ndarray:
    value = np.loadtxt(path, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"pose must be 4x4: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a predicted T_camera_object pose.")
    parser.add_argument("--predicted", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--model-points", required=True, help="Nx3 .npy or whitespace text")
    parser.add_argument("--camera-matrix", help="Optional 3x3 whitespace text")
    parser.add_argument("--output", default="outputs/pose_metrics.json")
    args = parser.parse_args()

    points_path = Path(args.model_points)
    points = np.load(points_path) if points_path.suffix == ".npy" else np.loadtxt(points_path)
    camera_matrix = np.loadtxt(args.camera_matrix) if args.camera_matrix else None
    metrics = evaluate_pose(
        load_matrix(args.predicted), load_matrix(args.ground_truth), points, camera_matrix
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
