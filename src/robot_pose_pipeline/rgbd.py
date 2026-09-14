from __future__ import annotations

from pathlib import Path

import numpy as np


def depth_to_points(
    depth: np.ndarray,
    camera_matrix: np.ndarray,
    depth_scale_to_m: float = 0.001,
    mask: np.ndarray | None = None,
    max_depth_m: float = 5.0,
) -> np.ndarray:
    """Back-project depth into camera-frame XYZ points (meters)."""
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D, got {depth.shape}")
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    if camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix must be 3x3")

    if depth.dtype == np.float32 or depth.dtype == np.float64:
        depth_m = depth.astype(np.float64)
        # Heuristic: if values look like millimeters already stored as float.
        if np.nanmax(depth_m) > 20.0 and depth_scale_to_m == 0.001:
            depth_m = depth_m * depth_scale_to_m
    else:
        depth_m = depth.astype(np.float64) * float(depth_scale_to_m)

    valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= max_depth_m)
    if mask is not None:
        valid &= np.asarray(mask) > 0
    v_idx, u_idx = np.where(valid)
    if len(u_idx) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    z = depth_m[v_idx, u_idx]
    x = (u_idx.astype(np.float64) - cx) * z / fx
    y = (v_idx.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def save_xyz(path: str | Path, points: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(points, dtype=np.float64), fmt="%.6f")


def save_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
        if colors.shape != (len(points), 3):
            raise ValueError("colors must match points shape (N,3)")
        header.extend(["property uchar red", "property uchar green", "property uchar blue"])
    header.append("end_header")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(header) + "\n")
        if colors is None:
            for point in points:
                handle.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        else:
            for point, color in zip(points, colors):
                handle.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n"
                )
