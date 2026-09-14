from __future__ import annotations

from typing import Iterable

import numpy as np


def as_transform(value: np.ndarray | Iterable[Iterable[float]], name: str = "transform") -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    return transform


def validate_rotation(rotation: np.ndarray, atol: float = 1e-6) -> None:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation contains non-finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError("rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=atol):
        raise ValueError(f"rotation determinant must be +1, got {determinant:.9f}")


def make_transform(rotation: np.ndarray, translation: np.ndarray, validate: bool = True) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    if translation.shape != (3,):
        raise ValueError(f"translation must contain 3 values, got {translation.shape}")
    if validate:
        validate_rotation(rotation)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = as_transform(transform)
    rotation = transform[:3, :3]
    validate_rotation(rotation)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -(rotation.T @ transform[:3, 3])
    return result


def compose(*transforms: np.ndarray) -> np.ndarray:
    if not transforms:
        return np.eye(4, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    for index, transform in enumerate(transforms):
        result = result @ as_transform(transform, name=f"transform[{index}]")
    return result


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = as_transform(transform)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("points contain non-finite values")
    return points @ transform[:3, :3].T + transform[:3, 3]


def rotation_diagnostics(transform: np.ndarray) -> dict[str, float]:
    rotation = as_transform(transform)[:3, :3]
    orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
    return {
        "determinant": float(np.linalg.det(rotation)),
        "orthogonality_error_fro": orthogonality_error,
    }

