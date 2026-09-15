"""ORB matching and PnP → T_camera_object (experimental)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from robot_pose_pipeline.transforms import make_transform


@dataclass(frozen=True)
class MatchResult:
    keypoints_query: tuple
    keypoints_train: tuple
    matches: tuple
    num_raw: int
    num_filtered: int


@dataclass(frozen=True)
class PnPResult:
    """Pose of the object/model in the camera frame: p_camera = T_camera_object · p_object."""

    T_camera_object: np.ndarray
    inlier_count: int
    reprojection_error_px: float
    success: bool


def match_orb(
    image_query_bgr: np.ndarray,
    image_train_bgr: np.ndarray,
    max_features: int = 1500,
    ratio: float = 0.75,
) -> MatchResult:
    """
    Match ORB features: query ↔ train.
    Lowe ratio test on knn matches (k=2).
    """
    orb = cv2.ORB_create(nfeatures=max_features)
    gray_q = cv2.cvtColor(image_query_bgr, cv2.COLOR_BGR2GRAY)
    gray_t = cv2.cvtColor(image_train_bgr, cv2.COLOR_BGR2GRAY)
    kq, dq = orb.detectAndCompute(gray_q, None)
    kt, dt = orb.detectAndCompute(gray_t, None)
    if dq is None or dt is None or len(kq) < 4 or len(kt) < 4:
        return MatchResult(tuple(), tuple(), tuple(), 0, 0)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(dq, dt, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return MatchResult(tuple(kq), tuple(kt), tuple(good), num_raw=len(knn), num_filtered=len(good))


def solve_pnp_from_2d3d(
    image_points: np.ndarray,
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    ransac: bool = True,
    reproj_threshold: float = 3.0,
) -> PnPResult:
    """
    Solve PnP. object_points are in the **object/model frame**;
    image_points are pixels in the query image.

    OpenCV returns rvec/tvec that map object → camera:
        p_camera = R @ p_object + t  ⇒  T_camera_object
    """
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    dist = np.zeros(5) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)

    if len(object_points) < 4:
        return PnPResult(np.eye(4), 0, float("inf"), False)

    def _solve(use_ransac: bool, flag: int):
        if use_ransac:
            return cv2.solvePnPRansac(
                object_points,
                image_points,
                camera_matrix,
                dist,
                flags=flag,
                reprojectionError=reproj_threshold,
            )
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist, flags=flag)
        return ok, rvec, tvec, None

    # OpenCV 4.x: ITERATIVE DLT needs >=6 points; try EPNP then ITERATIVE.
    attempts = []
    if ransac and len(object_points) >= 6:
        attempts.append((True, cv2.SOLVEPNP_EPNP))
        attempts.append((True, cv2.SOLVEPNP_ITERATIVE))
    attempts.append((False, cv2.SOLVEPNP_EPNP))
    if len(object_points) >= 6:
        attempts.append((False, cv2.SOLVEPNP_ITERATIVE))

    ok = False
    rvec = tvec = inliers = None
    for use_ransac, flag in attempts:
        try:
            out = _solve(use_ransac, flag)
            ok, rvec, tvec, inliers = out
            if ok and rvec is not None and tvec is not None and np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec)):
                break
            ok = False
        except cv2.error:
            ok = False
            continue

    inlier_count = 0 if inliers is None else int(len(inliers))
    if ok and inliers is None:
        inlier_count = int(len(object_points))

    if not ok:
        return PnPResult(np.eye(4), 0, float("inf"), False)

    R, _ = cv2.Rodrigues(rvec)
    try:
        T = make_transform(R, tvec.reshape(3), validate=True)
    except ValueError:
        return PnPResult(np.eye(4), 0, float("inf"), False)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist)
    projected = projected.reshape(-1, 2)
    observed = image_points.reshape(-1, 2)
    if inliers is not None and len(inliers):
        idx = inliers.reshape(-1)
        err = float(np.linalg.norm(projected[idx] - observed[idx], axis=1).mean())
    else:
        err = float(np.linalg.norm(projected - observed, axis=1).mean())
    if not np.isfinite(err):
        return PnPResult(np.eye(4), 0, float("inf"), False)
    return PnPResult(T_camera_object=T, inlier_count=inlier_count, reprojection_error_px=err, success=True)
