"""Plane RANSAC and DBSCAN clustering on Open3D clouds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Open3D is required. Install with: pip install -e \".[open3d]\"") from exc
    return o3d


@dataclass(frozen=True)
class ClusterInfo:
    label: int
    num_points: int
    min_bound: tuple[float, float, float]
    max_bound: tuple[float, float, float]
    center: tuple[float, float, float]


def segment_plane_ransac(cloud, distance_threshold: float = 0.01, ransac_n: int = 3, num_iterations: int = 1000):
    """
    Returns (plane_model, inlier_cloud, outlier_cloud).
    plane_model: [a,b,c,d] with a x + b y + c z + d = 0 in the cloud frame.
    """
    model, inliers = cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    plane = cloud.select_by_index(inliers)
    rest = cloud.select_by_index(inliers, invert=True)
    return np.asarray(model, dtype=np.float64), plane, rest


def remove_plane_ransac(cloud, distance_threshold: float = 0.01, ransac_n: int = 3, num_iterations: int = 1000):
    _, _, rest = segment_plane_ransac(cloud, distance_threshold, ransac_n, num_iterations)
    return rest


def remove_plane_if_dominant(cloud, distance_threshold: float = 0.01, min_inlier_ratio: float = 0.35):
    """Remove RANSAC plane only when it explains enough points; else return cloud unchanged."""
    n = len(cloud.points)
    if n < 50:
        return cloud, False
    _, plane, rest = segment_plane_ransac(cloud, distance_threshold=distance_threshold)
    if len(plane.points) / max(n, 1) >= min_inlier_ratio and len(rest.points) > 20:
        return rest, True
    return cloud, False


def cluster_dbscan(cloud, eps: float = 0.02, min_points: int = 30) -> tuple[list, list[ClusterInfo]]:
    """
    DBSCAN on the cloud. Returns (list of cluster clouds, stats).
    Noise label (-1) is excluded from cluster clouds.
    """
    o3d = _require_open3d()
    labels = np.asarray(cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors) if cloud.has_colors() else None
    clusters = []
    infos: list[ClusterInfo] = []
    for label in sorted(set(labels.tolist()) - {-1}):
        idx = np.where(labels == label)[0]
        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(points[idx])
        if colors is not None:
            sub.colors = o3d.utility.Vector3dVector(colors[idx])
        clusters.append(sub)
        pts = points[idx]
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        center = pts.mean(axis=0)
        infos.append(
            ClusterInfo(
                label=int(label),
                num_points=int(len(idx)),
                min_bound=(float(mn[0]), float(mn[1]), float(mn[2])),
                max_bound=(float(mx[0]), float(mx[1]), float(mx[2])),
                center=(float(center[0]), float(center[1]), float(center[2])),
            )
        )
    return clusters, infos
