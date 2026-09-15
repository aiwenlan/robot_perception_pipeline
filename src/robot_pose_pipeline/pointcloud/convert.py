"""RGB-D → Open3D PointCloud in the camera optical frame (meters)."""

from __future__ import annotations

import numpy as np

from robot_pose_pipeline.depth import preprocess_depth
from robot_pose_pipeline.rgbd import depth_to_points


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Open3D is required. Install with: pip install -e \".[open3d]\"") from exc
    return o3d


def numpy_to_open3d(points: np.ndarray, colors: np.ndarray | None = None):
    """Build Open3D PointCloud from (N,3) XYZ meters and optional (N,3) RGB uint8/float[0,1]."""
    o3d = _require_open3d()
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {points.shape}")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.asarray(colors)
        if colors.dtype == np.uint8:
            colors = colors.astype(np.float64) / 255.0
        if colors.shape != points.shape:
            raise ValueError("colors must match points shape (N,3)")
        cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def open3d_to_numpy(cloud) -> tuple[np.ndarray, np.ndarray | None]:
    points = np.asarray(cloud.points, dtype=np.float64)
    colors = np.asarray(cloud.colors, dtype=np.float64) if cloud.has_colors() else None
    return points, colors


def colors_from_rgb(rgb_bgr: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample BGR image at integer pixel coords uv (N,2)=(u,v) → RGB uint8 (N,3)."""
    rgb_bgr = np.asarray(rgb_bgr)
    uv = np.asarray(uv, dtype=np.int64)
    h, w = rgb_bgr.shape[:2]
    u = np.clip(uv[:, 0], 0, w - 1)
    v = np.clip(uv[:, 1], 0, h - 1)
    bgr = rgb_bgr[v, u]
    return bgr[:, ::-1].copy()


def rgbd_to_pointcloud(
    depth: np.ndarray,
    camera_matrix: np.ndarray,
    rgb_bgr: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    depth_scale_to_m: float = 0.001,
    min_depth_m: float = 0.05,
    max_depth_m: float = 2.0,
    median_ksize: int = 0,
    preprocess: bool = True,
):
    """
    Back-project depth to an Open3D point cloud in the **camera optical frame**.

    Uses the same pinhole model as `robot_pose_pipeline.rgbd.depth_to_points`.
    If preprocess=True, depth is scaled/clipped (meters, NaN invalid) first, then
    converted to a dense float map for back-projection (NaN → 0 so existing
    depth_to_points invalid logic drops them).
    """
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    if preprocess:
        depth_m = preprocess_depth(
            depth,
            depth_scale_to_m=depth_scale_to_m,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            median_ksize=median_ksize,
        )
        depth_for_proj = np.where(np.isfinite(depth_m), depth_m, 0.0).astype(np.float64)
        points = depth_to_points(
            depth_for_proj,
            camera_matrix,
            depth_scale_to_m=1.0,
            mask=mask,
            max_depth_m=max_depth_m,
        )
    else:
        points = depth_to_points(
            depth,
            camera_matrix,
            depth_scale_to_m=depth_scale_to_m,
            mask=mask,
            max_depth_m=max_depth_m,
        )

    colors = None
    if rgb_bgr is not None and len(points):
        # Recompute uv from points for color sampling.
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        z = points[:, 2]
        u = (points[:, 0] * fx / z + cx).round().astype(np.int64)
        v = (points[:, 1] * fy / z + cy).round().astype(np.int64)
        colors = colors_from_rgb(rgb_bgr, np.stack([u, v], axis=1))

    return numpy_to_open3d(points, colors)
