#!/usr/bin/env python3
"""Batch evaluate predicted 4x4 poses against GT poses (ADD / ADD-S / rot / trans / proj)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robot_pose_pipeline.pose_metrics import evaluate_pose


def _load_T(path: Path) -> np.ndarray:
    T = np.loadtxt(path, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"expected 4x4: {path}")
    return T


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True, help="dir of <id>.txt predicted poses")
    parser.add_argument("--gt-dir", type=Path, required=True, help="dir of <id>.txt GT poses")
    parser.add_argument("--model-points", type=Path, required=True, help=".npy Nx3 or whitespace Nx3")
    parser.add_argument("--camera-matrix", type=Path, help="optional 3x3")
    parser.add_argument("--output", type=Path, default=Path("outputs/ycbv_pose_metrics.json"))
    args = parser.parse_args()

    points_path = args.model_points
    points = np.load(points_path) if points_path.suffix == ".npy" else np.loadtxt(points_path)
    K = np.loadtxt(args.camera_matrix) if args.camera_matrix else None

    rows = []
    for gt_path in sorted(args.gt_dir.glob("*.txt")):
        pred_path = args.pred_dir / gt_path.name
        if not pred_path.is_file():
            continue
        metrics = evaluate_pose(_load_T(pred_path), _load_T(gt_path), points, K)
        metrics["frame_id"] = gt_path.stem
        rows.append(metrics)

    if not rows:
        raise SystemExit("no overlapping pred/gt frames")

    keys = [k for k in rows[0] if k != "frame_id"]
    summary = {
        "num_frames": len(rows),
        "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
        "median": {k: float(np.median([r[k] for r in rows])) for k in keys},
        "per_frame": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"num_frames": summary["num_frames"], "mean": summary["mean"], "median": summary["median"]}, indent=2))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
