from __future__ import annotations

import numpy as np


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask.shape}")
    return mask > 0


def mask_iou(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    pred = _as_binary_mask(predicted)
    gt = _as_binary_mask(ground_truth)
    if pred.shape != gt.shape:
        raise ValueError(f"mask shapes differ: {pred.shape} vs {gt.shape}")
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def mask_coverage(predicted: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    pred = _as_binary_mask(predicted)
    gt = _as_binary_mask(ground_truth)
    if pred.shape != gt.shape:
        raise ValueError(f"mask shapes differ: {pred.shape} vs {gt.shape}")
    gt_area = float(gt.sum())
    pred_area = float(pred.sum())
    intersection = float(np.logical_and(pred, gt).sum())
    return {
        "iou": mask_iou(pred, gt),
        "precision": float(intersection / pred_area) if pred_area else 0.0,
        "recall": float(intersection / gt_area) if gt_area else 0.0,
        "gt_area_px": gt_area,
        "predicted_area_px": pred_area,
    }
