#!/usr/bin/env python3
"""Optional offline helper for M6: write processed/object/CAD clouds without ROS.

For live ROS topics see robot_pose_pipeline_ros/cloud_processor_node.py (WSL).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robot_pose_pipeline.io import imread
from robot_pose_pipeline.pointcloud import (
    open3d_to_numpy,
    remove_plane_if_dominant,
    rgbd_to_pointcloud,
    statistical_outlier_removal,
    voxel_downsample,
)
from robot_pose_pipeline.registration import transform_model_to_camera
from robot_pose_pipeline.pointcloud.convert import numpy_to_open3d
from robot_pose_pipeline.transforms import as_transform


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/synthetic/manifest.jsonl"))
    parser.add_argument("--model-points", type=Path, default=Path("data/synthetic/model/points.txt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/m6_clouds"))
    args = parser.parse_args()

    import open3d as o3d

    line = next(x for x in args.manifest.read_text(encoding="utf-8").splitlines() if x.strip())
    record = json.loads(line)
    base = args.manifest.parent
    for key in ("rgb", "depth", "gt_mask", "gt_pose", "predicted_pose"):
        if record.get(key) and not Path(record[key]).is_absolute():
            record[key] = str((base / record[key]).resolve())

    rgb = imread(record["rgb"])
    depth = imread(record["depth"], flags=-1)
    mask = imread(record["gt_mask"], flags=0) if record.get("gt_mask") else None
    K = np.asarray(record["camera_matrix"], dtype=float)
    scale = float(record.get("depth_scale_to_m", 0.001))
    pose = record.get("predicted_pose") or record.get("gt_pose")
    T = as_transform(np.loadtxt(pose))

    raw = rgbd_to_pointcloud(depth, K, rgb_bgr=rgb, depth_scale_to_m=scale, median_ksize=5)
    processed = statistical_outlier_removal(voxel_downsample(raw, 0.004))
    processed, _ = remove_plane_if_dominant(processed, distance_threshold=0.012, min_inlier_ratio=0.4)
    obj = rgbd_to_pointcloud(depth, K, rgb_bgr=rgb, mask=mask, depth_scale_to_m=scale, median_ksize=5)
    model = numpy_to_open3d(np.loadtxt(args.model_points))
    cad = transform_model_to_camera(model, T)
    cad.paint_uniform_color([1.0, 0.75, 0.1])

    args.out.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.out / "raw_cloud.ply"), raw)
    o3d.io.write_point_cloud(str(args.out / "processed_cloud.ply"), processed)
    o3d.io.write_point_cloud(str(args.out / "object_cloud.ply"), obj)
    o3d.io.write_point_cloud(str(args.out / "cad_aligned_cloud.ply"), cad)
    meta = {
        "frame": "camera_color_optical_frame",
        "counts": {
            "raw": len(raw.points),
            "processed": len(processed.points),
            "object": len(obj.points),
            "cad_aligned": len(cad.points),
        },
        "topics_ros": [
            "/perception/target_cloud",
            "/perception/processed_cloud",
            "/perception/object_cloud",
            "/perception/cad_aligned_cloud",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
