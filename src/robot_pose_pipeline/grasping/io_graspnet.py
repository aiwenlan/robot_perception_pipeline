"""Import/export GraspNet-style grasp arrays.

Official graspnetAPI GraspGroup uses an (N,17) array. This repo also uses:

  T_camera_grasp: (N,4,4)
  width, score, depth: (N,)
saved via np.savez_compressed (.npz).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_pose_pipeline.transforms import as_transform, make_transform

from .types import Grasp6D, GraspSet


def grasps_to_npy(grasps: GraspSet, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = grasps.as_arrays()
    # Always write .npz for structured fields.
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    np.savez_compressed(path, **arrays)
    return path


def grasps_from_graspnet_npy(path: str | Path) -> GraspSet:
    """
    Load either:
      - this repo's .npz with T_camera_grasp/width/score/depth
      - GraspNet GraspGroup (N,17) float array (.npy)
    """
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        return GraspSet.from_arrays(
            data["T_camera_grasp"],
            data["width"],
            score=data["score"] if "score" in data.files else None,
            depth=data["depth"] if "depth" in data.files else None,
        )

    arr = np.load(path, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):
        return GraspSet.from_arrays(
            arr["T_camera_grasp"],
            arr["width"],
            score=arr["score"] if "score" in arr.files else None,
            depth=arr["depth"] if "depth" in arr.files else None,
        )

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 17:
        return _from_graspgroup_array(arr)
    raise ValueError(f"unsupported grasp file shape {getattr(arr, 'shape', None)}: {path}")


def _from_graspgroup_array(gg: np.ndarray) -> GraspSet:
    """
    GraspNet GraspGroup layout (graspnetAPI):
      score, width, height, depth, rotation_matrix(9), translation(3), object_id
    """
    grasps: list[Grasp6D] = []
    for row in gg:
        score = float(row[0])
        width = float(row[1])
        depth = float(row[3])
        R = row[4:13].reshape(3, 3)
        t = row[13:16]
        try:
            T = make_transform(R, t, validate=True)
        except ValueError:
            u, _, vt = np.linalg.svd(R)
            R_ortho = u @ vt
            if np.linalg.det(R_ortho) < 0:
                u[:, -1] *= -1
                R_ortho = u @ vt
            T = make_transform(R_ortho, t, validate=True)
        T = as_transform(T)
        grasps.append(Grasp6D(T_camera_grasp=T, width=width, score=score, depth=depth))
    return GraspSet(grasps)
