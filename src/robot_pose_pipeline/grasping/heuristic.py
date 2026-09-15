"""Geometric grasp proposals on an object point cloud (camera frame).

This is NOT the official GraspNet neural network. It produces GraspNet-compatible
`T_camera_grasp` candidates so the rest of the pipeline (viz / ROS / ICP pose)
can be connected to a downstream "operation" step without cloud GPUs.
"""

from __future__ import annotations

import numpy as np

from .types import GraspSet, make_grasp_from_center_approach


def propose_grasps_on_cloud(
    points: np.ndarray,
    *,
    num_samples: int = 64,
    topk: int = 20,
    width_margin: float = 0.01,
    min_width: float = 0.02,
    max_width: float = 0.10,
    approach_prefer_camera: bool = True,
    seed: int = 0,
) -> GraspSet:
    """
    Sample points on the object cloud and build top-down / side grasps.

    Score favors:
      - approach aligned with camera +Z (looking at object) when approach_prefer_camera
      - grasp center near cloud centroid
      - moderate gripper width
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 10:
        raise ValueError(f"points must be (N,3) with N>=10, got {points.shape}")

    rng = np.random.default_rng(seed)
    centroid = points.mean(axis=0)
    # PCA for local axes
    centered = points - centroid
    cov = centered.T @ centered / max(len(points) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    # Object extent along principal axes
    extents = []
    for i in range(3):
        proj = centered @ eigvecs[:, i]
        extents.append(float(proj.max() - proj.min()))
    extents = np.asarray(extents, dtype=np.float64)

    idx = rng.choice(len(points), size=min(num_samples, len(points)), replace=False)
    grasps = []
    cam_z = np.array([0.0, 0.0, 1.0])

    for i, pid in enumerate(idx):
        center = points[int(pid)].copy()
        # Alternate approach: mostly from camera (+Z toward scene is -?);
        # optical frame: camera looks along +Z, so approach toward object from camera is +Z.
        if approach_prefer_camera and (i % 3 != 2):
            approach = cam_z.copy()
        else:
            # Approach along smallest PCA axis (thin direction) — pinch the thin side.
            approach = eigvecs[:, 2].copy()
            if approach[2] < 0:
                approach = -approach

        # Closing along the medium extent axis.
        closing = eigvecs[:, 1].copy()
        width = float(np.clip(extents[1] + width_margin, min_width, max_width))
        # Score
        align = float(np.dot(approach / (np.linalg.norm(approach) + 1e-12), cam_z))
        dist = float(np.linalg.norm(center - centroid))
        score = 0.55 * max(align, 0.0) + 0.30 * float(np.exp(-dist / 0.05)) + 0.15 * (1.0 - abs(width - 0.05) / 0.05)
        score = float(np.clip(score + 0.02 * rng.normal(), 0.0, 1.0))
        g = make_grasp_from_center_approach(
            center=center,
            approach=approach,
            width=width,
            score=score,
            depth=0.02,
            closing_axis=closing,
        )
        grasps.append(g)

    return GraspSet(grasps).topk(topk)
