from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from robot_pose_pipeline.io import imread
from robot_pose_pipeline.rgbd import depth_to_points, save_ply, save_xyz


def main() -> None:
    parser = argparse.ArgumentParser(description="Back-project mask+depth into camera-frame points.")
    parser.add_argument("--depth", required=True)
    parser.add_argument("--camera-matrix", required=True, help="3x3 whitespace text")
    parser.add_argument("--mask")
    parser.add_argument("--rgb", help="Optional RGB for colored PLY")
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--output", default="outputs/target_cloud.ply")
    args = parser.parse_args()

    depth = imread(args.depth, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(args.depth)
    mask = None
    if args.mask:
        mask = imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(args.mask)
    camera_matrix = np.loadtxt(args.camera_matrix)
    points = depth_to_points(depth, camera_matrix, depth_scale_to_m=args.depth_scale, mask=mask)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".ply":
        colors = None
        if args.rgb and mask is not None:
            rgb = imread(args.rgb, cv2.IMREAD_COLOR)
            if rgb is None:
                raise FileNotFoundError(args.rgb)
            depth_m = depth.astype(np.float64)
            if depth.dtype != np.float32 and depth.dtype != np.float64:
                depth_m = depth_m * args.depth_scale
            valid = np.isfinite(depth_m) & (depth_m > 0) & (depth_m <= 5.0) & (mask > 0)
            ys, xs = np.where(valid)
            colors = rgb[ys, xs, ::-1]  # BGR -> RGB
        save_ply(output, points, colors)
    else:
        save_xyz(output, points)
    print(f"points={len(points)} -> {output.resolve()}")


if __name__ == "__main__":
    main()
