#!/usr/bin/env python3
"""Generate a FoundationPose YcbineoatReader-compatible mini synthetic scene.

Layout matches official mustard0 expectations:
  rgb/, depth/, masks/, mesh/textured_simple.obj, cam_K.txt, annotated_poses/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robot_pose_pipeline.io import imwrite
from robot_pose_pipeline.synthetic_scene import (
    default_camera_matrix,
    render_synthetic_frame,
    sample_cube_points,
)
from robot_pose_pipeline.transforms import make_transform

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "foundationpose" / "demo_data" / "mustard0_minisynth"


def write_cube_obj(path: Path, half_size: float = 0.04) -> None:
    """Write a simple axis-aligned cube OBJ (meters) usable as textured_simple.obj."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h = half_size
    verts = [
        (-h, -h, -h),
        (h, -h, -h),
        (h, h, -h),
        (-h, h, -h),
        (-h, -h, h),
        (h, -h, h),
        (h, h, h),
        (-h, h, h),
    ]
    # 1-based faces (triangles)
    faces = [
        (1, 2, 3),
        (1, 3, 4),
        (5, 6, 7),
        (5, 7, 8),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 4, 8),
        (3, 8, 7),
        (4, 1, 5),
        (4, 5, 8),
    ]
    lines = ["# mustard0_minisynth cube mesh (meters)", "mtllib textured_simple.mtl", "o cube"]
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    lines.append("usemtl cube_mat")
    for f in faces:
        lines.append(f"f {f[0]} {f[1]} {f[2]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mtl = path.with_suffix(".mtl")
    mtl.write_text(
        "newmtl cube_mat\nKd 0.9 0.7 0.1\nKa 0.1 0.1 0.1\nNs 10\n",
        encoding="utf-8",
    )


def generate_scene(output_dir: Path, frame_count: int = 5, seed: int = 42) -> Path:
    output_dir = Path(output_dir)
    for name in ("rgb", "depth", "masks", "annotated_poses", "pred_pose", "mesh", "model"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    camera_matrix = default_camera_matrix()
    np.savetxt(output_dir / "cam_K.txt", camera_matrix, fmt="%.6f")
    write_cube_obj(output_dir / "mesh" / "textured_simple.obj")
    model_points = sample_cube_points()
    np.savetxt(output_dir / "model" / "points.txt", model_points, fmt="%.6f")

    for index in range(frame_count):
        frame_id = f"{index:06d}"
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
        imwrite(output_dir / "rgb" / f"{frame_id}.png", rgb)
        imwrite(output_dir / "depth" / f"{frame_id}.png", depth)
        imwrite(output_dir / "masks" / f"{frame_id}.png", mask)
        np.savetxt(output_dir / "annotated_poses" / f"{frame_id}.txt", gt_pose, fmt="%.9f")
        np.savetxt(output_dir / "pred_pose" / f"{frame_id}.txt", pred_pose, fmt="%.9f")

    readme = output_dir / "README.txt"
    readme.write_text(
        "Synthetic stand-in for FoundationPose demo_data/mustard0.\n"
        "Matches YcbineoatReader layout (rgb/depth/masks/cam_K/mesh/annotated_poses).\n"
        "NOT the official mustard bottle sequence — use for manifest/ROS/geometry smoke only.\n"
        "For real run_demo.py quality, place official mustard0 under sibling mustard0/.\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote {frame_count} frames -> {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_scene(args.out.resolve(), frame_count=args.frames, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
