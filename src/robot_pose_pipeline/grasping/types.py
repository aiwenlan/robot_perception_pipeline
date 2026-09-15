"""6D grasp representation in the camera optical frame.

Convention (aligned with GraspNet / graspnetAPI GraspGroup):
  - T_camera_grasp maps grasp/gripper frame points → camera frame:
        p_camera = T_camera_grasp · p_grasp
  - translation = grasp center in camera coordinates (meters)
  - approach direction = -Z axis of the grasp frame
  - width = gripper opening (meters)
  - score = higher is better
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_pose_pipeline.transforms import as_transform, make_transform


@dataclass(frozen=True)
class Grasp6D:
    T_camera_grasp: np.ndarray
    width: float
    score: float = 1.0
    depth: float = 0.02

    def center(self) -> np.ndarray:
        return as_transform(self.T_camera_grasp, "T_camera_grasp")[:3, 3].copy()

    def approach(self) -> np.ndarray:
        """Unit approach direction in camera frame (gripper -Z)."""
        R = as_transform(self.T_camera_grasp, "T_camera_grasp")[:3, :3]
        v = -R[:, 2]
        n = float(np.linalg.norm(v))
        return v / max(n, 1e-12)


@dataclass
class GraspSet:
    grasps: list[Grasp6D]

    def __len__(self) -> int:
        return len(self.grasps)

    def topk(self, k: int = 20) -> "GraspSet":
        ordered = sorted(self.grasps, key=lambda g: g.score, reverse=True)
        return GraspSet(ordered[: max(0, int(k))])

    def best(self) -> Grasp6D | None:
        if not self.grasps:
            return None
        return self.topk(1).grasps[0]

    def as_arrays(self) -> dict[str, np.ndarray]:
        if not self.grasps:
            return {
                "T_camera_grasp": np.zeros((0, 4, 4), dtype=np.float64),
                "width": np.zeros((0,), dtype=np.float64),
                "score": np.zeros((0,), dtype=np.float64),
                "depth": np.zeros((0,), dtype=np.float64),
            }
        return {
            "T_camera_grasp": np.stack([g.T_camera_grasp for g in self.grasps], axis=0),
            "width": np.asarray([g.width for g in self.grasps], dtype=np.float64),
            "score": np.asarray([g.score for g in self.grasps], dtype=np.float64),
            "depth": np.asarray([g.depth for g in self.grasps], dtype=np.float64),
        }

    @staticmethod
    def from_arrays(
        T_camera_grasp: np.ndarray,
        width: np.ndarray,
        score: np.ndarray | None = None,
        depth: np.ndarray | None = None,
    ) -> "GraspSet":
        T = np.asarray(T_camera_grasp, dtype=np.float64)
        w = np.asarray(width, dtype=np.float64).reshape(-1)
        if T.ndim != 3 or T.shape[1:] != (4, 4):
            raise ValueError(f"T_camera_grasp must be (N,4,4), got {T.shape}")
        n = T.shape[0]
        if score is None:
            score_arr = np.ones(n, dtype=np.float64)
        else:
            score_arr = np.asarray(score, dtype=np.float64).reshape(-1)
        if depth is None:
            depth_arr = np.full(n, 0.02, dtype=np.float64)
        else:
            depth_arr = np.asarray(depth, dtype=np.float64).reshape(-1)
        grasps = [
            Grasp6D(
                T_camera_grasp=as_transform(T[i]),
                width=float(w[i]),
                score=float(score_arr[i]),
                depth=float(depth_arr[i]),
            )
            for i in range(n)
        ]
        return GraspSet(grasps)


def make_grasp_from_center_approach(
    center: np.ndarray,
    approach: np.ndarray,
    width: float,
    score: float = 1.0,
    depth: float = 0.02,
    closing_axis: np.ndarray | None = None,
) -> Grasp6D:
    """Build T_camera_grasp from center + approach (-Z) and optional closing axis (Y)."""
    approach = np.asarray(approach, dtype=np.float64).reshape(3)
    approach = approach / max(float(np.linalg.norm(approach)), 1e-12)
    if closing_axis is None:
        aux = np.array([0.0, 0.0, 1.0]) if abs(approach[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        closing = np.cross(approach, aux)
        closing = closing / max(float(np.linalg.norm(closing)), 1e-12)
    else:
        closing = np.asarray(closing_axis, dtype=np.float64).reshape(3)
        closing = closing - approach * float(np.dot(closing, approach))
        closing = closing / max(float(np.linalg.norm(closing)), 1e-12)
    binormal = np.cross(closing, approach)
    binormal = binormal / max(float(np.linalg.norm(binormal)), 1e-12)
    # Columns: X=binormal, Y=closing, Z=-approach  ⇒  -Z = approach
    R = np.stack([binormal, closing, -approach], axis=1)
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1.0
    T = make_transform(R, np.asarray(center, dtype=np.float64).reshape(3), validate=True)
    return Grasp6D(T_camera_grasp=T, width=float(width), score=float(score), depth=float(depth))
