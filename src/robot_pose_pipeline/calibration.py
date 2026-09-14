from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .io import imread, imwrite


@dataclass
class CalibrationResult:
    rms: float
    camera_matrix: list[list[float]]
    distortion_coefficients: list[float]
    mean_reprojection_error_px: float
    per_image_reprojection_error_px: list[float]
    used_images: list[str]
    rejected_images: list[str]
    image_size: tuple[int, int]
    board_inner_corners: tuple[int, int]
    square_size: float
    square_size_unit: str

    def to_dict(self) -> dict:
        return asdict(self)


def chessboard_object_points(board_cols: int, board_rows: int, square_size: float) -> np.ndarray:
    if board_cols < 2 or board_rows < 2 or square_size <= 0:
        raise ValueError("invalid chessboard definition")
    points = np.zeros((board_rows * board_cols, 3), np.float32)
    points[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    points[:, :2] *= float(square_size)
    return points


def reprojection_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rotation_vectors: tuple[np.ndarray, ...],
    translation_vectors: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> list[float]:
    errors: list[float] = []
    for object_set, observed, rvec, tvec in zip(
        object_points, image_points, rotation_vectors, translation_vectors
    ):
        projected, _ = cv2.projectPoints(object_set, rvec, tvec, camera_matrix, distortion)
        residuals = observed.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.linalg.norm(residuals, axis=1).mean()))
    return errors


def calibrate_chessboard_images(
    image_paths: Iterable[str | Path],
    board_cols: int,
    board_rows: int,
    square_size: float,
    square_size_unit: str = "arbitrary",
    preview_dir: str | Path | None = None,
) -> CalibrationResult:
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("no calibration images were provided")

    object_template = chessboard_object_points(board_cols, board_rows, square_size)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_images: list[str] = []
    rejected_images: list[str] = []
    image_size: tuple[int, int] | None = None
    preview_path = Path(preview_dir) if preview_dir else None
    if preview_path:
        preview_path.mkdir(parents=True, exist_ok=True)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-3,
    )
    pattern_size = (board_cols, board_rows)

    for path in paths:
        image = imread(path, cv2.IMREAD_COLOR)
        if image is None:
            rejected_images.append(str(path))
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError(f"all calibration images must share one size: {path} is {current_size}, expected {image_size}")

        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            rejected_images.append(str(path))
            continue
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_template.copy())
        image_points.append(refined)
        used_images.append(str(path))
        if preview_path:
            annotated = image.copy()
            cv2.drawChessboardCorners(annotated, pattern_size, refined, found)
            imwrite(preview_path / f"{path.stem}_corners.png", annotated)

    if image_size is None or len(used_images) < 3:
        raise RuntimeError(f"at least 3 valid chessboard images are required; found {len(used_images)}")

    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    errors = reprojection_errors(
        object_points, image_points, rvecs, tvecs, camera_matrix, distortion
    )
    return CalibrationResult(
        rms=float(rms),
        camera_matrix=camera_matrix.tolist(),
        distortion_coefficients=distortion.reshape(-1).tolist(),
        mean_reprojection_error_px=float(np.mean(errors)),
        per_image_reprojection_error_px=errors,
        used_images=used_images,
        rejected_images=rejected_images,
        image_size=image_size,
        board_inner_corners=(board_cols, board_rows),
        square_size=float(square_size),
        square_size_unit=square_size_unit,
    )
