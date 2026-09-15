#!/usr/bin/env python3
"""Minimal grasp demo: object point cloud → grasp candidates → visualization.

Uses GraspNet-compatible T_camera_grasp representation. Default proposer is a
geometric heuristic (no cloud GPU / no GraspNet weights). Optional:
  --grasps path.npy|.npz  to load official GraspGroup / repo npz instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robot_pose_pipeline.grasping import (
    grasps_to_npy,
    grasps_from_graspnet_npy,
    propose_grasps_on_cloud,
    save_grasp_overlay_image,
)
from robot_pose_pipeline.io import imread
from robot_pose_pipeline.pointcloud import open3d_to_numpy, rgbd_to_pointcloud


def _load_object_cloud_from_manifest(manifest: Path):
    line = next(x for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip())
    record = json.loads(line)
    base = manifest.parent
    for key in ("rgb", "depth", "gt_mask"):
        if record.get(key) and not Path(record[key]).is_absolute():
            record[key] = str((base / record[key]).resolve())
    rgb = imread(record["rgb"])
    depth = imread(record["depth"], flags=-1)
    mask = imread(record["gt_mask"], flags=0) if record.get("gt_mask") else None
    K = np.asarray(record["camera_matrix"], dtype=float)
    scale = float(record.get("depth_scale_to_m", 0.001))
    cloud = rgbd_to_pointcloud(
        depth, K, rgb_bgr=rgb, mask=mask, depth_scale_to_m=scale, median_ksize=5, max_depth_m=2.0
    )
    return cloud


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/synthetic/manifest.jsonl"))
    parser.add_argument("--cloud-ply", type=Path, default=None, help="Optional object cloud PLY override")
    parser.add_argument("--grasps", type=Path, default=None, help="Optional GraspGroup npy / npz")
    parser.add_argument("--out", type=Path, default=Path("outputs/grasp_demo"))
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--num-samples", type=int, default=80)
    args = parser.parse_args()

    import open3d as o3d

    args.out.mkdir(parents=True, exist_ok=True)
    if args.cloud_ply is not None:
        cloud = o3d.io.read_point_cloud(str(args.cloud_ply))
    else:
        cloud = _load_object_cloud_from_manifest(args.manifest)

    points, _ = open3d_to_numpy(cloud)
    print(f"object cloud points={len(points)}")
    if len(points) < 10:
        raise SystemExit("object cloud too small; check mask/manifest")

    if args.grasps is not None:
        grasps = grasps_from_graspnet_npy(args.grasps).topk(args.topk)
        print(f"loaded grasps from {args.grasps}: n={len(grasps)}")
    else:
        grasps = propose_grasps_on_cloud(points, num_samples=args.num_samples, topk=args.topk, seed=3)
        print(f"proposed geometric grasps: n={len(grasps)}")

    best = grasps.best()
    if best is None:
        raise SystemExit("no grasps")
    print(
        f"best score={best.score:.3f} width={best.width*1000:.1f}mm "
        f"center={np.round(best.center(), 4).tolist()} approach={np.round(best.approach(), 4).tolist()}"
    )
    np.savetxt(args.out / "T_camera_grasp_best.txt", best.T_camera_grasp)
    grasps_to_npy(grasps, args.out / "grasps.npz")
    o3d.io.write_point_cloud(str(args.out / "object_cloud.ply"), cloud)

    # Also dump grippers merged with cloud for Open3D viewers.
    geoms = [cloud]
    from robot_pose_pipeline.grasping import grasps_to_open3d_geometry

    geoms.extend(grasps_to_open3d_geometry(grasps, topk=args.topk))
    # Save a combined triangle mesh of grippers only for reference.
    preview = save_grasp_overlay_image(cloud, grasps, args.out / "grasps_overlay.png", topk=min(8, args.topk))
    print(f"wrote {preview}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
