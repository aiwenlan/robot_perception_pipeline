"""Depth preprocessing in meters (camera optical frame depth values)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthStats:
    valid_count: int
    invalid_count: int
    min_m: float
    max_m: float
    mean_m: float


def apply_depth_scale(depth: np.ndarray, depth_scale_to_m: float = 0.001) -> np.ndarray:
    """Convert raw depth to meters. Float depths already in meters are left as-is when max<=20."""
    depth = np.asarray(depth)
    if depth.dtype == np.float32 or depth.dtype == np.float64:
        depth_m = depth.astype(np.float64)
        if np.nanmax(depth_m) > 20.0 and depth_scale_to_m == 0.001:
            depth_m = depth_m * float(depth_scale_to_m)
        return depth_m
    return depth.astype(np.float64) * float(depth_scale_to_m)


def invalidate_depth(depth_m: np.ndarray, max_depth_m: float = 5.0, min_depth_m: float = 1e-4) -> np.ndarray:
    """Return a copy with invalid depths set to NaN."""
    depth_m = np.asarray(depth_m, dtype=np.float64).copy()
    invalid = (~np.isfinite(depth_m)) | (depth_m < min_depth_m) | (depth_m > max_depth_m)
    depth_m[invalid] = np.nan
    return depth_m


def clip_depth(depth_m: np.ndarray, min_depth_m: float = 0.05, max_depth_m: float = 2.0) -> np.ndarray:
    """Keep only depths inside [min, max]; others become NaN."""
    depth_m = np.asarray(depth_m, dtype=np.float64).copy()
    keep = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    out = np.full_like(depth_m, np.nan)
    out[keep] = depth_m[keep]
    return out


def median_filter_depth(depth_m: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Median filter on finite depths; invalid stay NaN (no zero bleed)."""
    if ksize % 2 == 0 or ksize < 3:
        raise ValueError("ksize must be odd and >= 3")
    depth_m = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(depth_m)
    if not np.any(valid):
        return np.full_like(depth_m, np.nan)
    # Fill holes with local valid mean via inpainting on 8-bit scaled map is heavy;
    # use median only where the original pixel is valid.
    fill_value = float(np.nanmedian(depth_m[valid]))
    filled = np.where(valid, depth_m, fill_value).astype(np.float32)
    filtered = cv2.medianBlur(filled, ksize).astype(np.float64)
    out = np.full_like(depth_m, np.nan)
    out[valid] = filtered[valid]
    return out


def depth_stats(depth_m: np.ndarray) -> DepthStats:
    depth_m = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(depth_m)
    vals = depth_m[valid]
    if vals.size == 0:
        return DepthStats(0, int(depth_m.size), float("nan"), float("nan"), float("nan"))
    return DepthStats(
        valid_count=int(vals.size),
        invalid_count=int((~valid).sum()),
        min_m=float(vals.min()),
        max_m=float(vals.max()),
        mean_m=float(vals.mean()),
    )


def masked_depth_stats(depth_m: np.ndarray, mask: np.ndarray) -> DepthStats:
    depth_m = np.asarray(depth_m, dtype=np.float64)
    mask = np.asarray(mask) > 0
    if mask.shape != depth_m.shape:
        raise ValueError(f"mask shape {mask.shape} != depth shape {depth_m.shape}")
    selected = np.where(mask, depth_m, np.nan)
    return depth_stats(selected)


def preprocess_depth(
    depth: np.ndarray,
    depth_scale_to_m: float = 0.001,
    min_depth_m: float = 0.05,
    max_depth_m: float = 2.0,
    median_ksize: int = 0,
) -> np.ndarray:
    """Scale → invalidate → clip → optional median. Output float64 meters with NaN invalid."""
    depth_m = apply_depth_scale(depth, depth_scale_to_m)
    depth_m = invalidate_depth(depth_m, max_depth_m=max_depth_m, min_depth_m=min_depth_m)
    depth_m = clip_depth(depth_m, min_depth_m=min_depth_m, max_depth_m=max_depth_m)
    if median_ksize and median_ksize >= 3:
        depth_m = median_filter_depth(depth_m, ksize=median_ksize)
    return depth_m
