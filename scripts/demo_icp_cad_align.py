#!/usr/bin/env python3
"""M3 demo: CAD/model cloud @ T_camera_object + ICP vs observed object cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robot_pose_pipeline.io import imread
from robot_pose_pipeline.pointcloud import numpy_to_open3d, open3d_to_numpy, rgbd_to_pointcloud, voxel_downsample
from robot_pose_pipeline.registration import (
    icp_point_to_plane,
    icp_point_to_point,
    transform_model_to_camera,
)
from robot_pose_pipeline.transforms import as_transform


def load_frame(manifest: Path, index: int = 0) -> dict:
    lines = [x for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    record = json.loads(lines[index])
    base = manifest.parent
    for key in ("rgb", "depth", "gt_mask", "predicted_pose", "gt_pose", "mesh"):
        if record.get(key) and not Path(record[key]).is_absolute():
            record[key] = str((base / record[key]).resolve())
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/synthetic/manifest.jsonl"))
    parser.add_argument("--model-points", type=Path, default=Path("data/synthetic/model/points.txt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/m3_icp"))
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()

    import open3d as o3d

    record = load_frame(args.manifest, 0)
    rgb = imread(record["rgb"])
    depth = imread(record["depth"], flags=-1)
    mask = imread(record.get("gt_mask"), flags=0) if record.get("gt_mask") else None
    K = np.asarray(record["camera_matrix"], dtype=float)
    scale = float(record.get("depth_scale_to_m", 0.001))
    pose_path = record.get("predicted_pose") or record.get("gt_pose")
    if not pose_path:
        raise SystemExit("manifest frame needs predicted_pose or gt_pose")
    T_init = as_transform(np.loadtxt(pose_path), name="T_camera_object")

    # Model cloud in object frame (synthetic ships sampled cube points).
    model_xyz = np.loadtxt(args.model_points)
    model = numpy_to_open3d(model_xyz)
    source = transform_model_to_camera(model, T_init)
    target = rgbd_to_pointcloud(depth, K, rgb_bgr=rgb, mask=mask, depth_scale_to_m=scale, median_ksize=5)
    target = voxel_downsample(target, 0.003)

    # Perturb init slightly so ICP has work to do when pose is already perfect.
    noise = np.eye(4)
    noise[:3, 3] = [0.004, -0.003, 0.002]
    T_noisy = noise @ T_init
    source_noisy = transform_model_to_camera(model, T_noisy)

    result_pt = icp_point_to_point(source_noisy, target, T_noisy, threshold=args.threshold)
    result_pl = icp_point_to_plane(source_noisy, target, T_noisy, threshold=args.threshold)

    args.out.mkdir(parents=True, exist_ok=True)
    before = transform_model_to_camera(model, T_noisy)
    after = transform_model_to_camera(model, result_pt.T_camera_object_refined)
    before.paint_uniform_color([1, 0.2, 0.2])
    after.paint_uniform_color([0.2, 1, 0.2])
    target.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.io.write_point_cloud(str(args.out / "target_object.ply"), target)
    o3d.io.write_point_cloud(str(args.out / "model_before_icp.ply"), before)
    o3d.io.write_point_cloud(str(args.out / "model_after_icp.ply"), after)

    print("point-to-point:", f"fitness={result_pt.fitness:.4f}", f"rmse={result_pt.rmse:.6f}")
    print("point-to-plane:", f"fitness={result_pl.fitness:.4f}", f"rmse={result_pl.rmse:.6f}")
    print("T_correction (p2p) translation:", result_pt.T_correction[:3, 3])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
