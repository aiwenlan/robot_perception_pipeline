"""Open3D filtering and normals (camera frame point sets; frame unchanged)."""

from __future__ import annotations

import numpy as np


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Open3D is required. Install with: pip install -e \".[open3d]\"") from exc
    return o3d


def voxel_downsample(cloud, voxel_size: float = 0.005):
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    return cloud.voxel_down_sample(voxel_size)


def statistical_outlier_removal(cloud, nb_neighbors: int = 20, std_ratio: float = 2.0):
    cleaned, _ = cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cleaned


def radius_outlier_removal(cloud, nb_points: int = 16, radius: float = 0.02):
    cleaned, _ = cloud.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return cleaned


def crop_box(cloud, min_bound: np.ndarray, max_bound: np.ndarray):
    """Axis-aligned crop in the cloud's current frame (usually camera)."""
    o3d = _require_open3d()
    min_bound = np.asarray(min_bound, dtype=np.float64).reshape(3)
    max_bound = np.asarray(max_bound, dtype=np.float64).reshape(3)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return cloud.crop(bbox)


def estimate_normals(cloud, radius: float = 0.02, max_nn: int = 30):
    o3d = _require_open3d()
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return cloud
