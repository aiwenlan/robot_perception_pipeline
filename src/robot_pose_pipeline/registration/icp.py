"""ICP refinement. Initial pose comes from FoundationPose T_camera_object."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_pose_pipeline.transforms import as_transform, compose


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Open3D is required. Install with: pip install -e \".[open3d]\"") from exc
    return o3d


@dataclass(frozen=True)
class ICPResult:
    """
    T_correction: transforms the *source* cloud further into the target frame.
    If source was already placed by T_camera_object_init, then:

        T_camera_object_refined = T_correction @ T_camera_object_init

    (left-multiply when both map object→camera via the same composition convention).
    """

    fitness: float
    rmse: float
    T_correction: np.ndarray
    T_camera_object_refined: np.ndarray


def _run_icp(source, target, T_camera_object_init: np.ndarray, threshold: float, estimation):
    o3d = _require_open3d()
    T_init = as_transform(T_camera_object_init, name="T_camera_object_init")
    # Source is assumed already in camera frame (model transformed by init pose).
    # We refine with identity seed relative to current placement, OR pass init as
    # identity on already-transformed source. Convention used here:
    #   source = model already transformed by T_camera_object_init
    #   ICP finds T_correction close to I such that T_correction @ source ≈ target
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    reg = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        np.eye(4),
        estimation,
        criteria,
    )
    T_corr = np.asarray(reg.transformation, dtype=np.float64)
    T_refined = compose(T_corr, T_init)
    return ICPResult(
        fitness=float(reg.fitness),
        rmse=float(reg.inlier_rmse),
        T_correction=T_corr,
        T_camera_object_refined=T_refined,
    )


def icp_point_to_point(source_camera, target_camera, T_camera_object_init: np.ndarray, threshold: float = 0.01):
    o3d = _require_open3d()
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    return _run_icp(source_camera, target_camera, T_camera_object_init, threshold, estimation)


def icp_point_to_plane(source_camera, target_camera, T_camera_object_init: np.ndarray, threshold: float = 0.01):
    o3d = _require_open3d()
    if not target_camera.has_normals():
        target_camera.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(threshold * 2, 0.02), max_nn=30)
        )
    if not source_camera.has_normals():
        source_camera.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(threshold * 2, 0.02), max_nn=30)
        )
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    return _run_icp(source_camera, target_camera, T_camera_object_init, threshold, estimation)
