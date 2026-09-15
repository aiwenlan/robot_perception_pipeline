#!/usr/bin/env python3
"""M5 demo: multi-frame RGB-D TSDF fusion with known poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robot_pose_pipeline.io import imread
from robot_pose_pipeline.reconstruction.tsdf import integrate_rgbd_frames, save_tsdf_result
from robot_pose_pipeline.transforms import as_transform, invert_transform


def load_frames(manifest: Path, max_frames: int) -> list[dict]:
    base = manifest.parent
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for key in ("rgb", "depth", "gt_pose", "predicted_pose"):
            if record.get(key) and not Path(record[key]).is_absolute():
                record[key] = str((base / record[key]).resolve())
        rows.append(record)
        if len(rows) >= max_frames:
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/synthetic/manifest.jsonl"))
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("outputs/m5_tsdf"))
    parser.add_argument("--voxel-length", type=float, default=0.004)
    parser.add_argument("--sdf-trunc", type=float, default=0.015)
    args = parser.parse_args()

    frames = load_frames(args.manifest, args.max_frames)
    if len(frames) < 2:
        raise SystemExit("need >=2 frames")

    # Use first camera as world: T_world_camera_i = T_cam0_object @ inv(T_cami_object)
    # Actually easier: world = first camera frame.
    # T_world_camera_0 = I
    # T_camera_i_object known; T_camera_0_object known
    # T_world_camera_i = T_camera_0_object @ inv(T_camera_i_object)
    # because p_cam0 = T_c0_obj @ p_obj; p_cami = T_ci_obj @ p_obj
    # ⇒ p_cam0 = T_c0_obj @ inv(T_ci_obj) @ p_cami = T_world_camera_i @ p_cami
    pose0 = frames[0].get("gt_pose") or frames[0].get("predicted_pose")
    T_c0_obj = as_transform(np.loadtxt(pose0))

    rgbs, depths, poses = [], [], []
    K = np.asarray(frames[0]["camera_matrix"], dtype=float)
    scale = float(frames[0].get("depth_scale_to_m", 0.001))
    for fr in frames:
        rgbs.append(imread(fr["rgb"]))
        depths.append(imread(fr["depth"], flags=-1))
        pose_i = fr.get("gt_pose") or fr.get("predicted_pose")
        T_ci_obj = as_transform(np.loadtxt(pose_i))
        T_world_camera = T_c0_obj @ invert_transform(T_ci_obj)
        poses.append(T_world_camera)

    result = integrate_rgbd_frames(
        rgbs,
        depths,
        poses,
        K,
        pose_is_T_world_camera=True,
        depth_scale_to_m=scale,
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
    )
    paths = save_tsdf_result(result, args.out)
    print(f"frames={result.num_frames} voxel={result.voxel_length} trunc={result.sdf_trunc}")
    print(f"mesh vertices={len(result.mesh.vertices)} cloud={len(result.cloud.points)}")
    print(f"wrote {paths}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
