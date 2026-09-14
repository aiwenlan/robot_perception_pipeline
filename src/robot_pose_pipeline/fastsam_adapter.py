from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=np.float64).reshape(4)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0.0, None), axis=1)
    box_area = np.prod(np.clip(box[2:] - box[:2], 0.0, None))
    boxes_area = np.prod(np.clip(boxes[:, 2:] - boxes[:, :2], 0.0, None), axis=1)
    return intersection / np.maximum(box_area + boxes_area - intersection, 1e-12)


def select_mask_by_box(
    detector_box_xyxy: np.ndarray,
    candidate_boxes_xyxy: np.ndarray,
    candidate_masks: np.ndarray,
    min_iou: float = 0.05,
) -> tuple[np.ndarray, int, float]:
    candidate_masks = np.asarray(candidate_masks)
    if candidate_masks.ndim != 3:
        raise ValueError(f"candidate_masks must have shape (N,H,W), got {candidate_masks.shape}")
    if len(candidate_masks) != len(candidate_boxes_xyxy):
        raise ValueError("candidate box and mask counts differ")
    scores = box_iou_xyxy(detector_box_xyxy, candidate_boxes_xyxy)
    index = int(np.argmax(scores))
    score = float(scores[index])
    if score < min_iou:
        raise RuntimeError(f"no FastSAM candidate overlaps detector box enough; best IoU={score:.4f}")
    return (candidate_masks[index] > 0.5).astype(np.uint8) * 255, index, score


@dataclass
class FastSAMResult:
    mask: np.ndarray
    candidate_index: int
    candidate_box_iou: float


class FastSAMSegmenter:
    """Thin adapter over the local FastSAM implementation used by the course repositories."""

    def __init__(self, weights: str | Path, device: str = "cuda:0", image_size: int = 1024):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("FastSAM requires the optional 'fastsam' dependencies") from exc
        self.model = YOLO(str(weights))
        self.device = device
        self.image_size = image_size

    def segment_box(self, image_bgr: np.ndarray, detector_box_xyxy: np.ndarray) -> FastSAMResult:
        results = self.model(
            image_bgr,
            imgsz=self.image_size,
            device=self.device,
            retina_masks=True,
            verbose=False,
        )
        result = results[0]
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            raise RuntimeError("FastSAM returned no candidate masks")
        masks = result.masks.data.detach().cpu().numpy()
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        if masks.shape[1:] != image_bgr.shape[:2]:
            masks = np.stack(
                [cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST) for mask in masks]
            )
        mask, index, score = select_mask_by_box(detector_box_xyxy, boxes, masks)
        return FastSAMResult(mask=mask, candidate_index=index, candidate_box_iou=score)

