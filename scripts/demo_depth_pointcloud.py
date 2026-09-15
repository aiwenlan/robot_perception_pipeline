#!/usr/bin/env python3
"""M1 demo: depth preprocess + Open3D RGB-D point cloud (camera frame)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robot_pose_pipeline.depth import depth_stats, masked_depth_stats, preprocess_depth
from robot_pose_pipeline.io import imread
from robot_pose_pipeline.pointcloud import open3d_to_numpy, rgbd_to_pointcloud
from robot_pose_pipeline.rgbd import depth_to_points, save_ply


def load_first_frame(manifest: Path) -> dict:
    line = next(x for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip())
    record = json.loads(line)
    base = manifest.parent
    for key in ("rgb", "depth", "gt_mask", "predicted_mask"):
        if record.get(key) and not Path(record[key]).is_absolute():
            record[key] = str((base / record[key]).resolve())
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/synthetic/manifest.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/m1_depth_pointcloud"))
    parser.add_argument("--median-ksize", type=int, default=5)
    args = parser.parse_args()

    record = load_first_frame(args.manifest)
    rgb = imread(record["rgb"])
    depth = imread(record["depth"], flags=-1)
    mask_path = record.get("gt_mask") or record.get("predicted_mask")
    mask = imread(mask_path, flags=0) if mask_path else None
    K = np.asarray(record["camera_matrix"], dtype=float)
    scale = float(record.get("depth_scale_to_m", 0.001))

    raw_m = preprocess_depth(depth, depth_scale_to_m=scale, median_ksize=0)
    filt_m = preprocess_depth(depth, depth_scale_to_m=scale, median_ksize=args.median_ksize)
    print("raw depth:", depth_stats(raw_m))
    print("filtered:", depth_stats(filt_m))
    if mask is not None:
        print("masked target depth:", masked_depth_stats(filt_m, mask))

    cloud_scene = rgbd_to_pointcloud(depth, K, rgb_bgr=rgb, mask=None, depth_scale_to_m=scale, median_ksize=args.median_ksize)
    cloud_obj = rgbd_to_pointcloud(depth, K, rgb_bgr=rgb, mask=mask, depth_scale_to_m=scale, median_ksize=args.median_ksize)
    pts_np = depth_to_points(depth, K, depth_scale_to_m=scale, mask=mask)
    pts_o3d, _ = open3d_to_numpy(cloud_obj)

    args.out.mkdir(parents=True, exist_ok=True)
    import open3d as o3d

    o3d.io.write_point_cloud(str(args.out / "scene.ply"), cloud_scene)
    o3d.io.write_point_cloud(str(args.out / "object.ply"), cloud_obj)
    save_ply(args.out / "object_numpy.ply", pts_np)

    print(f"scene points: {len(cloud_scene.points)}")
    print(f"object Open3D points: {len(cloud_obj.points)}")
    print(f"object numpy points: {len(pts_np)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
