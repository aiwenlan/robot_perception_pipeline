#!/usr/bin/env python3
"""Convert one BOP YCB-V test scene + one object into a mustard0-like folder with GT poses.

BOP mask files are named ``{im_id:06d}_{gt_index:06d}.png`` where ``gt_index`` is the
index inside ``scene_gt.json[im_id]`` (NOT the object id).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _as_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ycbv-root", type=Path, required=True)
    parser.add_argument("--scene", type=str, default="000050")
    parser.add_argument("--ob-id", type=int, default=5, help="BOP object id (mustard bottle = 5)")
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.ycbv_root.resolve()
    scene = root / "test" / args.scene
    if not scene.is_dir():
        raise SystemExit(f"scene not found: {scene}")

    models_dir = root / "models"
    ply = models_dir / f"obj_{args.ob_id:06d}.ply"
    if not ply.is_file():
        raise SystemExit(f"mesh not found: {ply}")

    scene_gt = _load_json(scene / "scene_gt.json")
    scene_cam = _load_json(scene / "scene_camera.json")

    frame_items: list[tuple[int, int, dict]] = []
    for fid, objs in scene_gt.items():
        for gt_idx, obj in enumerate(objs):
            if int(obj["obj_id"]) == args.ob_id:
                frame_items.append((int(fid), gt_idx, obj))
                break
    frame_items = sorted(frame_items, key=lambda x: x[0])[: args.max_frames]
    if not frame_items:
        raise SystemExit(f"object {args.ob_id} not in scene {args.scene}")

    out = args.out.resolve()
    for sub in ("rgb", "depth", "masks", "gt_pose", "mesh"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    cam0 = scene_cam[str(frame_items[0][0])]
    K = np.asarray(cam0["cam_K"], dtype=np.float64).reshape(3, 3)
    np.savetxt(out / "cam_K.txt", K)

    mesh = trimesh.load(ply, force="mesh", process=False)
    mesh.apply_scale(0.001)  # BOP mm -> meters
    mesh_path = out / "mesh" / "textured_simple.obj"
    mesh.export(mesh_path)

    kept = 0
    for fid, gt_idx, gt in frame_items:
        sid = f"{fid:06d}"
        rgb_src = scene / "rgb" / f"{sid}.png"
        depth_src = scene / "depth" / f"{sid}.png"
        mask_src = scene / "mask_visib" / f"{sid}_{gt_idx:06d}.png"
        if not mask_src.is_file():
            mask_src = scene / "mask" / f"{sid}_{gt_idx:06d}.png"
        if not (rgb_src.is_file() and depth_src.is_file() and mask_src.is_file()):
            print(f"[skip] {sid}: missing rgb/depth/mask (gt_idx={gt_idx})")
            continue

        shutil.copy2(rgb_src, out / "rgb" / f"{sid}.png")
        # BOP depth: meters = raw_uint16 * depth_scale / 1000
        # YcbineoatReader does raw/1000 only, so bake depth_scale into the saved depth.
        import cv2

        depth_scale = float(scene_cam[str(fid)].get("depth_scale", 1.0))
        depth_raw = cv2.imread(str(depth_src), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            print(f"[skip] {sid}: cannot read depth")
            continue
        depth_baked = np.clip(np.round(depth_raw.astype(np.float64) * depth_scale), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(out / "depth" / f"{sid}.png"), depth_baked)
        shutil.copy2(mask_src, out / "masks" / f"{sid}.png")

        R = np.asarray(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(gt["cam_t_m2c"], dtype=np.float64).reshape(3) * 0.001
        np.savetxt(out / "gt_pose" / f"{sid}.txt", _as_T(R, t))
        kept += 1

    meta = {
        "source": "BOP ycbv",
        "scene": args.scene,
        "ob_id": args.ob_id,
        "frames": kept,
        "depth_scale_to_m": 0.001,
        "mesh": str(mesh_path),
        "note": "GT poses are T_camera_object in meters; mesh scaled mm->m. Mask index = scene_gt list index.",
    }
    (out / "object_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
