#!/usr/bin/env python3
"""Render Open3D PLY / mesh previews into docs/assets for README."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def render_geometries(geoms, path: Path, width: int = 640, height: int = 480) -> None:
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.12, 0.12, 0.14])
    opt.point_size = 3.0
    vis.poll_events()
    vis.update_renderer()
    # Fit view
    vis.get_view_control().set_zoom(0.7)
    vis.poll_events()
    vis.update_renderer()
    path.parent.mkdir(parents=True, exist_ok=True)
    vis.capture_screen_image(str(path), do_render=True)
    vis.destroy_window()


def load_cloud(path: Path, color=None):
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    if color is not None and len(cloud.points):
        cloud.paint_uniform_color(color)
    return cloud


def hstack_images(paths: list[Path], out: Path, labels: list[str] | None = None) -> None:
    imgs = []
    for i, p in enumerate(paths):
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        if labels and i < len(labels):
            cv2.putText(img, labels[i], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        imgs.append(img)
    h = min(im.shape[0] for im in imgs)
    resized = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in imgs]
    canvas = np.concatenate(resized, axis=1)
    ok, buf = cv2.imencode(".png", canvas)
    out.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(out))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    import open3d as o3d

    out = args.out_dir
    tmp = Path("outputs/readme_viz_tmp")
    tmp.mkdir(parents=True, exist_ok=True)

    # M1: scene vs object
    scene = load_cloud(Path("outputs/m1_depth_pointcloud/scene.ply"))
    obj = load_cloud(Path("outputs/m1_depth_pointcloud/object.ply"), [0.95, 0.75, 0.15])
    render_geometries([scene], tmp / "m1_scene.png")
    render_geometries([obj], tmp / "m1_object.png")
    hstack_images([tmp / "m1_scene.png", tmp / "m1_object.png"], out / "viz_m1_pointcloud.png", ["scene cloud", "object cloud"])

    # M2: cluster
    c0 = Path("outputs/m2_filter_segment/cluster_00.ply")
    if c0.is_file() and len(o3d.io.read_point_cloud(str(c0)).points) > 0:
        cloud = load_cloud(c0, [0.2, 0.85, 0.35])
        render_geometries([cloud], out / "viz_m2_cluster.png")
    else:
        filt = load_cloud(Path("outputs/m2_filter_segment/filtered.ply"), [0.3, 0.7, 1.0])
        render_geometries([filt], out / "viz_m2_cluster.png")

    # M3: before / after ICP
    before = load_cloud(Path("outputs/m3_icp/model_before_icp.ply"), [1.0, 0.25, 0.25])
    after = load_cloud(Path("outputs/m3_icp/model_after_icp.ply"), [0.25, 1.0, 0.35])
    target = load_cloud(Path("outputs/m3_icp/target_object.ply"), [0.55, 0.55, 0.55])
    render_geometries([target, before], tmp / "m3_before.png")
    render_geometries([target, after], tmp / "m3_after.png")
    hstack_images([tmp / "m3_before.png", tmp / "m3_after.png"], out / "viz_m3_icp.png", ["before ICP", "after ICP"])

    # M4: matches already png
    matches = Path("outputs/m4_orb_pnp/matches.png")
    if matches.is_file():
        data = np.fromfile(str(matches), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        cv2.putText(img, "ORB matches (query | train)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        ok, buf = cv2.imencode(".png", img)
        buf.tofile(str(out / "viz_m4_orb_matches.png"))

    # M5: TSDF mesh
    mesh = o3d.io.read_triangle_mesh(str(Path("outputs/m5_tsdf/tsdf_mesh.ply")))
    if not mesh.is_empty():
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.75, 0.78, 0.85])
        render_geometries([mesh], out / "viz_m5_tsdf.png")

    # M6: four clouds strip
    raw = load_cloud(Path("outputs/m6_clouds/raw_cloud.ply"))
    proc = load_cloud(Path("outputs/m6_clouds/processed_cloud.ply"), [0.2, 0.75, 1.0])
    obj6 = load_cloud(Path("outputs/m6_clouds/object_cloud.ply"), [0.2, 0.95, 0.3])
    cad = load_cloud(Path("outputs/m6_clouds/cad_aligned_cloud.ply"), [1.0, 0.75, 0.15])
    render_geometries([raw], tmp / "m6_raw.png")
    render_geometries([proc], tmp / "m6_proc.png")
    render_geometries([obj6], tmp / "m6_obj.png")
    render_geometries([cad], tmp / "m6_cad.png")
    hstack_images(
        [tmp / "m6_raw.png", tmp / "m6_proc.png", tmp / "m6_obj.png", tmp / "m6_cad.png"],
        out / "viz_m6_clouds.png",
        ["raw", "processed", "object", "CAD aligned"],
    )

    print("wrote README viz under", out.resolve())
    for p in sorted(out.glob("viz_*.png")):
        print(" ", p.name, p.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
