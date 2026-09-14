from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .transforms import as_transform, transform_points


def rotation_error_deg(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    pred_rotation = as_transform(predicted, "predicted")[:3, :3]
    gt_rotation = as_transform(ground_truth, "ground_truth")[:3, :3]
    relative = pred_rotation @ gt_rotation.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def translation_error(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    pred_translation = as_transform(predicted, "predicted")[:3, 3]
    gt_translation = as_transform(ground_truth, "ground_truth")[:3, 3]
    return float(np.linalg.norm(pred_translation - gt_translation))


def _validate_model_points(model_points: np.ndarray) -> np.ndarray:
    points = np.asarray(model_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"model_points must have non-empty shape (N, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("model_points contain non-finite values")
    return points


def add_error(predicted: np.ndarray, ground_truth: np.ndarray, model_points: np.ndarray) -> float:
    points = _validate_model_points(model_points)
    predicted_points = transform_points(predicted, points)
    ground_truth_points = transform_points(ground_truth, points)
    return float(np.linalg.norm(predicted_points - ground_truth_points, axis=1).mean())


def adds_error(predicted: np.ndarray, ground_truth: np.ndarray, model_points: np.ndarray) -> float:
    points = _validate_model_points(model_points)
    predicted_points = transform_points(predicted, points)
    ground_truth_points = transform_points(ground_truth, points)
    tree = cKDTree(ground_truth_points)
    distances, _ = tree.query(predicted_points, k=1, workers=-1)
    return float(distances.mean())


def project_points(transform: np.ndarray, model_points: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    points = transform_points(transform, _validate_model_points(model_points))
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"camera_matrix must have shape (3, 3), got {camera_matrix.shape}")
    if np.any(points[:, 2] <= 0):
        raise ValueError("all transformed model points must be in front of the camera")
    pixels_homogeneous = points @ camera_matrix.T
    return pixels_homogeneous[:, :2] / pixels_homogeneous[:, 2:3]


def projection_error_px(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    model_points: np.ndarray,
    camera_matrix: np.ndarray,
) -> float:
    predicted_pixels = project_points(predicted, model_points, camera_matrix)
    ground_truth_pixels = project_points(ground_truth, model_points, camera_matrix)
    return float(np.linalg.norm(predicted_pixels - ground_truth_pixels, axis=1).mean())


def evaluate_pose(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    model_points: np.ndarray,
    camera_matrix: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {
        "rotation_error_deg": rotation_error_deg(predicted, ground_truth),
        "translation_error": translation_error(predicted, ground_truth),
        "add": add_error(predicted, ground_truth, model_points),
        "adds": adds_error(predicted, ground_truth, model_points),
    }
    if camera_matrix is not None:
        metrics["projection_error_px"] = projection_error_px(
            predicted, ground_truth, model_points, camera_matrix
        )
    return metrics

