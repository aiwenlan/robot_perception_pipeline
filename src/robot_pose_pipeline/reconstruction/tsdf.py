"""Open3D TSDF integration with known camera poses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_pose_pipeline.depth import apply_depth_scale
from robot_pose_pipeline.transforms import as_transform, invert_transform


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Open3D is required. Install with: pip install -e \".[open3d]\"") from exc
    return o3d


@dataclass(frozen=True)
class TSDFResult:
    mesh: object
    cloud: object
    num_frames: int
    voxel_length: float
    sdf_trunc: float


def integrate_rgbd_frames(
    rgbs_bgr: list[np.ndarray],
    depths: list[np.ndarray],
    T_camera_world_or_T_world_camera: list[np.ndarray],
    camera_matrix: np.ndarray,
    *,
    pose_is_T_world_camera: bool = True,
    depth_scale_to_m: float = 0.001,
    depth_max_m: float = 2.0,
    voxel_length: float = 0.005,
    sdf_trunc: float = 0.02,
) -> TSDFResult:
    """
    Fuse RGB-D frames into a TSDF volume.

    Pose convention:
    - If pose_is_T_world_camera=True (default), each pose is T_world_camera
      (camera extrinsic: maps camera points → world), which Open3D expects
      as the camera-to-world transform for integration.
    - If False, inputs are T_camera_world and will be inverted.

    Depth is converted to meters via depth_scale_to_m.
    """
    o3d = _require_open3d()
    if not (len(rgbs_bgr) == len(depths) == len(T_camera_world_or_T_world_camera)):
        raise ValueError("rgbs, depths, and poses must have the same length")
    if len(rgbs_bgr) == 0:
        raise ValueError("need at least one frame")

    K = np.asarray(camera_matrix, dtype=np.float64)
    h, w = rgbs_bgr[0].shape[:2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_length),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for rgb_bgr, depth, pose in zip(rgbs_bgr, depths, T_camera_world_or_T_world_camera):
        pose = as_transform(pose)
        T_world_camera = pose if pose_is_T_world_camera else invert_transform(pose)
        depth_m = apply_depth_scale(depth, depth_scale_to_m).astype(np.float32)
        depth_m = np.where(np.isfinite(depth_m), depth_m, 0.0)
        color = o3d.geometry.Image(np.ascontiguousarray(rgb_bgr[:, :, ::-1]))  # RGB
        depth_img = o3d.geometry.Image(np.ascontiguousarray(depth_m))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth_img,
            depth_scale=1.0,  # already meters
            depth_trunc=float(depth_max_m),
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(T_world_camera))

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    cloud = volume.extract_point_cloud()
    return TSDFResult(mesh=mesh, cloud=cloud, num_frames=len(rgbs_bgr), voxel_length=voxel_length, sdf_trunc=sdf_trunc)


def save_tsdf_result(result: TSDFResult, out_dir: str | Path) -> dict[str, Path]:
    o3d = _require_open3d()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = out_dir / "tsdf_mesh.ply"
    cloud_path = out_dir / "tsdf_cloud.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), result.mesh)
    o3d.io.write_point_cloud(str(cloud_path), result.cloud)
    return {"mesh": mesh_path, "cloud": cloud_path}
