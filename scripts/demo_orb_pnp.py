#!/usr/bin/env python3
"""M4 demo: ORB match + 2D-3D correspondences + PnP (experimental branch)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from robot_pose_pipeline.io import imread, imwrite
from robot_pose_pipeline.matching import match_orb, solve_pnp_from_2d3d
from robot_pose_pipeline.pose_metrics import rotation_error_deg, translation_error
from robot_pose_pipeline.transforms import as_transform, make_transform, transform_points
from scipy.spatial.transform import Rotation


def load_frame(manifest: Path, index: int) -> dict:
    lines = [x for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    record = json.loads(lines[index])
    base = manifest.parent
    for key in ("rgb", "depth", "gt_mask", "gt_pose", "predicted_pose"):
        if record.get(key) and not Path(record[key]).is_absolute():
            record[key] = str((base / record[key]).resolve())
    return record


def run_synthetic_pnp_check(out: Path) -> None:
    """Fallback teaching path when real ORB correspondences are too few (weak texture)."""
    K = np.array([[400.0, 0, 320.0], [0, 400.0, 240.0], [0, 0, 1.0]])
    obj = np.array(
        [[0, 0, 0], [0.05, 0, 0], [0.05, 0.04, 0], [0, 0.04, 0], [0.02, 0.02, 0.03], [-0.02, 0.01, 0.02]],
        dtype=float,
    )
    T = make_transform(Rotation.from_euler("xyz", [12, -8, 4], degrees=True).as_matrix(), [0.01, -0.02, 0.55])
    pts_cam = transform_points(T, obj)
    uv = np.stack(
        [K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2], K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]],
        axis=1,
    )
    result = solve_pnp_from_2d3d(uv, obj, K, ransac=False)
    np.savetxt(out / "T_camera_object_pnp_synthetic.txt", result.T_camera_object)
    print(
        f"[fallback synthetic PnP] success={result.success} "
        f"reproj={result.reprojection_error_px:.4f}px "
        f"(use this to learn PnP when ORB matches are insufficient)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/synthetic/manifest.jsonl"))
    parser.add_argument("--train-index", type=int, default=0)
    parser.add_argument("--query-index", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("outputs/m4_orb_pnp"))
    args = parser.parse_args()

    train = load_frame(args.manifest, args.train_index)
    query = load_frame(args.manifest, args.query_index)
    img_t = imread(train["rgb"])
    img_q = imread(query["rgb"])
    depth_t = imread(train["depth"], flags=-1)
    mask_t = imread(train["gt_mask"], flags=0) if train.get("gt_mask") else None
    K = np.asarray(train["camera_matrix"], dtype=float)
    scale = float(train.get("depth_scale_to_m", 0.001))

    matches = match_orb(img_q, img_t, max_features=4000, ratio=0.8)
    print(f"ORB matches: raw={matches.num_raw} filtered={matches.num_filtered}")

    pose_train = train.get("gt_pose") or train.get("predicted_pose")
    if not pose_train:
        raise SystemExit("train frame needs gt_pose/predicted_pose")
    T_cam_obj_train = as_transform(np.loadtxt(pose_train))
    T_obj_cam_train = np.linalg.inv(T_cam_obj_train)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    depth_m = depth_t.astype(np.float64) * scale

    pts2d, pts3d_obj = [], []
    for m in matches.matches:
        u_t, v_t = matches.keypoints_train[m.trainIdx].pt
        u_q, v_q = matches.keypoints_query[m.queryIdx].pt
        ui, vi = int(round(u_t)), int(round(v_t))
        if vi < 0 or ui < 0 or vi >= depth_m.shape[0] or ui >= depth_m.shape[1]:
            continue
        if mask_t is not None and mask_t[vi, ui] == 0:
            continue
        z = float(depth_m[vi, ui])
        if not np.isfinite(z) or z <= 0 or z > 2.0:
            continue
        x = (ui - cx) * z / fx
        y = (vi - cy) * z / fy
        p_obj = (T_obj_cam_train @ np.array([x, y, z, 1.0]))[:3]
        pts2d.append([u_q, v_q])
        pts3d_obj.append(p_obj)

    pts2d = np.asarray(pts2d, dtype=float)
    pts3d_obj = np.asarray(pts3d_obj, dtype=float)
    print(f"2D-3D correspondences: {len(pts2d)}")

    args.out.mkdir(parents=True, exist_ok=True)
    vis = np.concatenate([img_q, img_t], axis=1)
    for m in list(matches.matches)[:50]:
        uq, vq = matches.keypoints_query[m.queryIdx].pt
        ut, vt = matches.keypoints_train[m.trainIdx].pt
        cv2.line(vis, (int(uq), int(vq)), (int(ut) + img_q.shape[1], int(vt)), (0, 255, 0), 1)
    imwrite(args.out / "matches.png", vis)

    result = None
    if len(pts2d) >= 4:
        result = solve_pnp_from_2d3d(pts2d, pts3d_obj, K, ransac=len(pts2d) >= 8)
    if result is not None and result.success:
        print(
            f"PnP success={result.success} inliers={result.inlier_count} "
            f"reproj={result.reprojection_error_px:.3f}px"
        )
        np.savetxt(args.out / "T_camera_object_pnp.txt", result.T_camera_object)
        pose_query = query.get("gt_pose") or query.get("predicted_pose")
        if pose_query:
            T_gt = as_transform(np.loadtxt(pose_query))
            print(
                f"vs GT: rot={rotation_error_deg(result.T_camera_object, T_gt):.3f}deg "
                f"trans={translation_error(result.T_camera_object, T_gt)*1000:.2f}mm"
            )
    else:
        print("ORB+depth correspondences insufficient on this synthetic texture — expected teaching failure mode.")
        run_synthetic_pnp_check(args.out)

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
