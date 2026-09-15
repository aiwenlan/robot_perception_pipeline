#!/usr/bin/env python3
"""M2 demo: voxel/SOR filter, RANSAC plane, DBSCAN clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robot_pose_pipeline.io import imread
from robot_pose_pipeline.pointcloud import (
    cluster_dbscan,
    remove_plane_if_dominant,
    rgbd_to_pointcloud,
    statistical_outlier_removal,
    voxel_downsample,
)


def load_first_frame(manifest: Path) -> dict:
    line = next(x for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip())
    record = json.loads(line)
    base = manifest.parent
    for key in ("rgb", "depth", "gt_mask"):
        if record.get(key) and not Path(record[key]).is_absolute():
            record[key] = str((base / record[key]).resolve())
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/synthetic/manifest.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/m2_filter_segment"))
    args = parser.parse_args()

    import open3d as o3d

    record = load_first_frame(args.manifest)
    rgb = imread(record["rgb"])
    depth = imread(record["depth"], flags=-1)
    K = np.asarray(record["camera_matrix"], dtype=float)
    scale = float(record.get("depth_scale_to_m", 0.001))

    cloud = rgbd_to_pointcloud(depth, K, rgb_bgr=rgb, depth_scale_to_m=scale, median_ksize=5)
    n0 = len(cloud.points)
    cloud = voxel_downsample(cloud, voxel_size=0.004)
    n1 = len(cloud.points)
    cloud = statistical_outlier_removal(cloud, nb_neighbors=20, std_ratio=2.0)
    n2 = len(cloud.points)
    rest, removed = remove_plane_if_dominant(cloud, distance_threshold=0.01, min_inlier_ratio=0.4)
    n3 = len(rest.points)
    clusters, infos = cluster_dbscan(rest, eps=0.03, min_points=15)

    args.out.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.out / "filtered.ply"), cloud)
    o3d.io.write_point_cloud(str(args.out / "after_plane.ply"), rest)
    palette = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ]
    for i, (c, info) in enumerate(zip(clusters, infos)):
        c.paint_uniform_color(palette[i % len(palette)])
        o3d.io.write_point_cloud(str(args.out / f"cluster_{info.label:02d}.ply"), c)
        print(
            f"cluster {info.label}: n={info.num_points} "
            f"center={tuple(round(x, 3) for x in info.center)} "
            f"aabb={info.min_bound} -> {info.max_bound}"
        )

    print(
        f"points: raw={n0} voxel={n1} sor={n2} after_plane={n3} "
        f"plane_removed={removed} clusters={len(clusters)}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
