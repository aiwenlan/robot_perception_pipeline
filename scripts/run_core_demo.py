from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cv2

from robot_pose_pipeline.dataset_manifest import load_manifest
from robot_pose_pipeline.hand_eye import (
    HAND_EYE_METHODS,
    evaluate_hand_eye,
    generate_eye_in_hand_dataset,
    solve_eye_in_hand,
)
from robot_pose_pipeline.io import imread
from robot_pose_pipeline.mask_metrics import mask_coverage
from robot_pose_pipeline.pose_metrics import evaluate_pose
from robot_pose_pipeline.rgbd import depth_to_points, save_ply
from robot_pose_pipeline.synthetic_scene import write_synthetic_dataset
from robot_pose_pipeline.transforms import compose


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Windows-ready core demo without FoundationPose/ROS2.")
    parser.add_argument("--output-dir", default="outputs/core_demo")
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_dir = Path("data/synthetic")
    manifest_path = write_synthetic_dataset(synthetic_dir, frame_count=args.frames)
    records = load_manifest(manifest_path)

    # 1) Hand-eye AX=XB
    dataset = generate_eye_in_hand_dataset(sample_count=20, noise_rotation_deg=0.2, noise_translation=0.001)
    estimated = solve_eye_in_hand(dataset.base_to_gripper, dataset.camera_to_target, HAND_EYE_METHODS["tsai"])
    hand_eye = {
        "metrics": evaluate_hand_eye(estimated, dataset.gripper_to_camera_ground_truth),
        "T_gripper_camera": estimated.tolist(),
    }
    (output_dir / "hand_eye.json").write_text(json.dumps(hand_eye, indent=2), encoding="utf-8")
    np.savetxt(output_dir / "T_gripper_camera.txt", estimated, fmt="%.9f")

    # 2) Pose metrics on first frame
    first = records[0]
    model_points = np.loadtxt(first.mesh)
    pose_metrics = evaluate_pose(
        np.loadtxt(first.predicted_pose),
        np.loadtxt(first.gt_pose),
        model_points,
        first.camera_matrix,
    )
    (output_dir / "pose_metrics.json").write_text(json.dumps(pose_metrics, indent=2), encoding="utf-8")

    # 3) Mask metrics
    gt_mask = imread(first.gt_mask, cv2.IMREAD_GRAYSCALE)
    pred_mask = imread(first.predicted_mask, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None or pred_mask is None:
        raise FileNotFoundError("failed to read synthetic masks")
    mask_metrics = mask_coverage(pred_mask, gt_mask)
    (output_dir / "mask_metrics.json").write_text(json.dumps(mask_metrics, indent=2), encoding="utf-8")

    # 4) Compose base object using synthetic T_base_camera
    base_camera = np.loadtxt(synthetic_dir / "T_base_camera.txt")
    base_object = compose(base_camera, np.loadtxt(first.predicted_pose))
    np.savetxt(output_dir / "T_base_object.txt", base_object, fmt="%.9f")

    # 5) Target point cloud from mask + depth
    depth = imread(first.depth, cv2.IMREAD_UNCHANGED)
    points = depth_to_points(depth, first.camera_matrix, first.depth_scale_to_m, mask=gt_mask)
    save_ply(output_dir / "target_cloud.ply", points)

    summary = {
        "manifest": str(manifest_path.resolve()),
        "hand_eye_metrics": hand_eye["metrics"],
        "pose_metrics": pose_metrics,
        "mask_metrics": mask_metrics,
        "point_count": len(points),
        "notes": [
            "This demo uses synthetic GT/predicted poses to validate geometry and evaluation.",
            "Replace predicted_pose with FoundationPose outputs after Docker setup.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"demo outputs -> {output_dir.resolve()}")


if __name__ == "__main__":
    main()
