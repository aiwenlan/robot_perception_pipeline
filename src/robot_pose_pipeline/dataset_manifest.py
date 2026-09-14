from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    timestamp_sec: float
    rgb: Path
    depth: Path
    camera_matrix: np.ndarray
    depth_scale_to_m: float
    gt_mask: Path | None = None
    predicted_mask: Path | None = None
    mesh: Path | None = None
    gt_pose: Path | None = None
    predicted_pose: Path | None = None
    object_id: str | None = None


def _resolve(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_manifest(path: str | Path, check_files: bool = True) -> list[FrameRecord]:
    manifest_path = Path(path).resolve()
    base = manifest_path.parent
    records: list[FrameRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            raw = json.loads(line)
            camera_matrix = np.asarray(raw["camera_matrix"], dtype=np.float64)
            if camera_matrix.shape != (3, 3):
                raise ValueError(f"line {line_number}: camera_matrix must be 3x3")
            record = FrameRecord(
                frame_id=str(raw["frame_id"]),
                timestamp_sec=float(raw.get("timestamp_sec", len(records) / 30.0)),
                rgb=_resolve(base, raw["rgb"]),
                depth=_resolve(base, raw["depth"]),
                camera_matrix=camera_matrix,
                depth_scale_to_m=float(raw.get("depth_scale_to_m", 0.001)),
                gt_mask=_resolve(base, raw.get("gt_mask")),
                predicted_mask=_resolve(base, raw.get("predicted_mask")),
                mesh=_resolve(base, raw.get("mesh")),
                gt_pose=_resolve(base, raw.get("gt_pose")),
                predicted_pose=_resolve(base, raw.get("predicted_pose")),
                object_id=raw.get("object_id"),
            )
            if check_files:
                for field_name in ("rgb", "depth", "gt_mask", "predicted_mask", "mesh", "gt_pose", "predicted_pose"):
                    value = getattr(record, field_name)
                    if value is not None and not value.exists():
                        raise FileNotFoundError(f"line {line_number}: {field_name} does not exist: {value}")
            records.append(record)
    if not records:
        raise ValueError(f"manifest contains no records: {manifest_path}")
    return records


def iter_manifest(path: str | Path, check_files: bool = True) -> Iterator[FrameRecord]:
    yield from load_manifest(path, check_files=check_files)

