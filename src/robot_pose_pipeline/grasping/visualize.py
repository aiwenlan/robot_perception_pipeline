"""Open3D visualization helpers for grasps."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import Grasp6D, GraspSet


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover
        raise ImportError('Open3D required: pip install -e ".[open3d]"') from exc
    return o3d


def _gripper_mesh(width: float, depth: float = 0.02):
    """Simple parallel-jaw gripper mesh in grasp frame (meters)."""
    o3d = _require_open3d()
    w = max(float(width), 0.01)
    finger_h = 0.045
    finger_t = 0.006
    palm_t = 0.01
    left = o3d.geometry.TriangleMesh.create_box(finger_t, finger_h, depth)
    left.translate([-w / 2 - finger_t, -finger_h / 2, -depth])
    right = o3d.geometry.TriangleMesh.create_box(finger_t, finger_h, depth)
    right.translate([w / 2, -finger_h / 2, -depth])
    palm = o3d.geometry.TriangleMesh.create_box(w + 2 * finger_t, palm_t, palm_t)
    palm.translate([-w / 2 - finger_t, -palm_t / 2, -depth - palm_t])
    mesh = left + right + palm
    mesh.compute_vertex_normals()
    return mesh


def grasp_to_open3d_mesh(grasp: Grasp6D, color=(0.1, 0.85, 0.25)):
    mesh = _gripper_mesh(grasp.width, grasp.depth)
    mesh.paint_uniform_color(color)
    mesh.transform(grasp.T_camera_grasp)
    return mesh


def grasps_to_open3d_geometry(grasps: GraspSet, topk: int = 20):
    """Return list of colored gripper meshes for top-k grasps."""
    geoms = []
    subset = grasps.topk(topk)
    if not subset.grasps:
        return geoms
    scores = np.asarray([g.score for g in subset.grasps], dtype=np.float64)
    smin, smax = float(scores.min()), float(scores.max())
    for g in subset.grasps:
        t = 0.5 if smax - smin < 1e-9 else (g.score - smin) / (smax - smin)
        color = (1.0 - 0.7 * t, 0.2 + 0.7 * t, 0.15)  # low=red → high=green
        geoms.append(grasp_to_open3d_mesh(g, color=color))
    return geoms


def save_grasp_overlay_image(
    cloud,
    grasps: GraspSet,
    out_path: str | Path,
    topk: int = 15,
    width: int = 900,
    height: int = 650,
) -> Path:
    """Offscreen render cloud + gripper meshes to PNG."""
    o3d = _require_open3d()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    geoms = [cloud] + grasps_to_open3d_geometry(grasps, topk=topk)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.12, 0.12, 0.14])
    opt.point_size = 3.0
    vis.poll_events()
    vis.update_renderer()
    vis.get_view_control().set_zoom(0.62)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(out_path), do_render=True)
    vis.destroy_window()
    return out_path
