#!/usr/bin/env python3
"""Render pcl_demo PCD outputs into a README strip image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_pcd_xyz(path: Path) -> np.ndarray:
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(cloud.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError(f"empty cloud: {path}")
    return pts


def render_cloud(points: np.ndarray, color, width=420, height=320) -> np.ndarray:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.paint_uniform_color(color)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(cloud)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.12, 0.12, 0.14])
    opt.point_size = 3.0
    vis.poll_events()
    vis.update_renderer()
    vis.get_view_control().set_zoom(0.7)
    vis.poll_events()
    vis.update_renderer()
    tmp = Path("outputs/pcl_viz_tmp.png")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    vis.capture_screen_image(str(tmp), do_render=True)
    vis.destroy_window()
    data = np.fromfile(str(tmp), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcl-out", type=Path, default=Path("pcl_demo/outputs"))
    parser.add_argument("--out", type=Path, default=Path("docs/assets/viz_pcl_demo.png"))
    args = parser.parse_args()

    stages = [
        ("input", args.pcl_out / "00_input.pcd", [0.7, 0.7, 0.7]),
        ("no plane", args.pcl_out / "04_no_plane.pcd", [0.3, 0.7, 1.0]),
        ("object", args.pcl_out / "06_object_cluster.pcd", [0.2, 0.95, 0.35]),
        ("ICP aligned", args.pcl_out / "08_model_after_icp.pcd", [1.0, 0.75, 0.15]),
    ]
    panels = []
    for label, path, color in stages:
        pts = load_pcd_xyz(path)
        img = render_cloud(pts, color)
        cv2.putText(img, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        panels.append(img)
    canvas = np.concatenate(panels, axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", canvas)
    buf.tofile(str(args.out))
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
