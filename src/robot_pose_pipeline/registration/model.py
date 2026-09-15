"""Sample CAD/mesh into a model point cloud and place it in the camera frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_pose_pipeline.transforms import as_transform, transform_points


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Open3D is required. Install with: pip install -e \".[open3d]\"") from exc
    return o3d


def load_mesh_as_pointcloud(mesh_path: str | Path, num_points: int = 2000):
    """Load mesh (.ply/.obj/.stl) and sample to Open3D PointCloud in **object/model frame**."""
    return sample_mesh_to_pointcloud(mesh_path, num_points=num_points)


def sample_mesh_to_pointcloud(mesh_path: str | Path, num_points: int = 2000):
    o3d = _require_open3d()
    mesh_path = Path(mesh_path)
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"failed to load mesh: {mesh_path}")
    mesh.compute_vertex_normals()
    return mesh.sample_points_uniformly(number_of_points=int(num_points))


def transform_model_to_camera(model_cloud, T_camera_object: np.ndarray):
    """
    Apply T_camera_object so that model points expressed in object frame become camera-frame points:

        p_camera = T_camera_object · p_object

    Returns a new Open3D cloud (copy).
    """
    o3d = _require_open3d()
    T = as_transform(T_camera_object, name="T_camera_object")
    points = np.asarray(model_cloud.points, dtype=np.float64)
    transformed = transform_points(T, points)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(transformed)
    if model_cloud.has_colors():
        out.colors = model_cloud.colors
    if model_cloud.has_normals():
        R = T[:3, :3]
        normals = np.asarray(model_cloud.normals, dtype=np.float64) @ R.T
        out.normals = o3d.utility.Vector3dVector(normals)
    return out
