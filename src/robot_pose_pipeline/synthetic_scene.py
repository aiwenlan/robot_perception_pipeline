from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .io import imwrite
from .pose_metrics import project_points
from .transforms import make_transform


def cube_corners(half_size: float = 0.04) -> np.ndarray:
    axes = np.array([-half_size, half_size], dtype=np.float64)
    return np.array([[x, y, z] for x in axes for y in axes for z in axes], dtype=np.float64)


def sample_cube_points(half_size: float = 0.04, samples_per_edge: int = 8) -> np.ndarray:
    values = np.linspace(-half_size, half_size, samples_per_edge)
    points = []
    for x in values:
        for y in values:
            for z in values:
                on_face = (
                    abs(abs(x) - half_size) < 1e-9
                    or abs(abs(y) - half_size) < 1e-9
                    or abs(abs(z) - half_size) < 1e-9
                )
                if on_face:
                    points.append([x, y, z])
    return np.asarray(points, dtype=np.float64)


def default_camera_matrix(width: int = 640, height: int = 480) -> np.ndarray:
    fx = fy = 600.0
    return np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def render_synthetic_frame(
    camera_matrix: np.ndarray,
    camera_object: np.ndarray,
    width: int = 640,
    height: int = 480,
    half_size: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return RGB, depth(mm uint16), mask, and model points for a cube."""
    model_points = sample_cube_points(half_size=half_size)
    corners = cube_corners(half_size=half_size)
    pixels = project_points(camera_object, corners, camera_matrix)
    hull = cv2.convexHull(pixels.astype(np.float32))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 255)

    depth = np.zeros((height, width), dtype=np.float32)
    camera_points = (camera_object[:3, :3] @ model_points.T).T + camera_object[:3, 3]
    projected = project_points(camera_object, model_points, camera_matrix)
    for pixel, point in zip(np.round(projected).astype(int), camera_points):
        u, v = int(pixel[0]), int(pixel[1])
        if 0 <= u < width and 0 <= v < height and point[2] > 0:
            depth_mm = point[2] * 1000.0
            previous = depth[v, u]
            if previous == 0 or depth_mm < previous:
                depth[v, u] = depth_mm

    # Dense fill inside mask using planar approximation at object center depth.
    center_depth_m = float(camera_object[2, 3])
    ys, xs = np.where(mask > 0)
    depth[ys, xs] = np.where(depth[ys, xs] > 0, depth[ys, xs], center_depth_m * 1000.0)

    rgb = np.full((height, width, 3), 30, dtype=np.uint8)
    rgb[mask > 0] = (40, 170, 220)
    for pixel in np.round(project_points(camera_object, corners, camera_matrix)).astype(int):
        u, v = int(pixel[0]), int(pixel[1])
        if 0 <= u < width and 0 <= v < height:
            cv2.circle(rgb, (u, v), 3, (0, 255, 0), -1)

    depth_u16 = np.clip(depth, 0, 65535).astype(np.uint16)
    return rgb, depth_u16, mask, model_points


def write_synthetic_dataset(
    output_dir: str | Path,
    frame_count: int = 5,
    seed: int = 11,
) -> Path:
    output_dir = Path(output_dir)
    for name in ("rgb", "depth", "mask", "pred_mask", "gt_pose", "pred_pose", "model"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    camera_matrix = default_camera_matrix()
    model_points = sample_cube_points()
    np.savetxt(output_dir / "model" / "points.txt", model_points, fmt="%.6f")
    np.savetxt(output_dir / "camera_matrix.txt", camera_matrix, fmt="%.6f")

    # Simulated fixed hand-eye / base-camera extrinsics for TF demos.
    base_camera = make_transform(
        Rotation.from_euler("xyz", [180.0, 0.0, 0.0], degrees=True).as_matrix(),
        np.array([0.4, 0.0, 0.6]),
    )
    np.savetxt(output_dir / "T_base_camera.txt", base_camera, fmt="%.9f")

    manifest_path = output_dir / "manifest.jsonl"
    lines: list[str] = []
    for index in range(frame_count):
        frame_id = f"{index + 1:06d}"
        yaw = float(rng.uniform(-25, 25))
        pitch = float(rng.uniform(-15, 15))
        translation = np.array(
            [
                float(rng.uniform(-0.08, 0.08)),
                float(rng.uniform(-0.06, 0.06)),
                float(rng.uniform(0.45, 0.65)),
            ]
        )
        gt_pose = make_transform(
            Rotation.from_euler("xyz", [pitch, yaw, 0.0], degrees=True).as_matrix(),
            translation,
        )
        # Small synthetic prediction noise for evaluation demos.
        noise_r = Rotation.from_euler(
            "xyz",
            rng.normal(0.0, 1.5, size=3),
            degrees=True,
        ).as_matrix()
        pred_pose = make_transform(
            noise_r @ gt_pose[:3, :3],
            gt_pose[:3, 3] + rng.normal(0.0, 0.003, size=3),
        )

        rgb, depth, mask, _ = render_synthetic_frame(camera_matrix, gt_pose)
        # Predicted mask = slightly eroded GT to emulate FastSAM boundary error.
        kernel = np.ones((5, 5), np.uint8)
        pred_mask = cv2.erode(mask, kernel, iterations=1)

        rgb_rel = f"rgb/{frame_id}.png"
        depth_rel = f"depth/{frame_id}.png"
        mask_rel = f"mask/{frame_id}.png"
        pred_mask_rel = f"pred_mask/{frame_id}.png"
        gt_pose_rel = f"gt_pose/{frame_id}.txt"
        pred_pose_rel = f"pred_pose/{frame_id}.txt"

        imwrite(output_dir / rgb_rel, rgb)
        imwrite(output_dir / depth_rel, depth)
        imwrite(output_dir / mask_rel, mask)
        imwrite(output_dir / pred_mask_rel, pred_mask)
        np.savetxt(output_dir / gt_pose_rel, gt_pose, fmt="%.9f")
        np.savetxt(output_dir / pred_pose_rel, pred_pose, fmt="%.9f")

        record = {
            "frame_id": frame_id,
            "timestamp_sec": index / 5.0,
            "rgb": rgb_rel,
            "depth": depth_rel,
            "depth_scale_to_m": 0.001,
            "camera_matrix": camera_matrix.tolist(),
            "gt_mask": mask_rel,
            "predicted_mask": pred_mask_rel,
            "mesh": "model/points.txt",
            "gt_pose": gt_pose_rel,
            "predicted_pose": pred_pose_rel,
            "object_id": "synthetic_cube",
        }
        lines.append(json.dumps(record, ensure_ascii=False))

    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path
