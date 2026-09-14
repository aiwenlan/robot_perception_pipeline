#!/usr/bin/env python3
"""Convert FoundationPose demo_data/mustard0 into project JSONL manifest format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENES = [
    ROOT / "data" / "foundationpose" / "demo_data" / "mustard0",
    ROOT / "data" / "foundationpose" / "demo_data" / "mustard0_minisynth",
]
DEFAULT_OUT = ROOT / "data" / "foundationpose" / "demo_data" / "mustard0_manifest.jsonl"


def _rel(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def _load_K(scene: Path) -> list[list[float]]:
    k_path = scene / "cam_K.txt"
    if not k_path.is_file():
        raise FileNotFoundError(f"Missing cam_K.txt in {scene}")
    K = np.loadtxt(k_path).reshape(3, 3)
    return K.tolist()


def build_manifest(scene: Path, out: Path, object_id: str = "mustard0") -> int:
    rgb_dir = scene / "rgb"
    depth_dir = scene / "depth"
    mask_dir = scene / "masks"
    if not mask_dir.is_dir():
        mask_dir = scene / "mask"
    mesh_candidates = [
        scene / "mesh" / "textured_simple.obj",
        scene / "mesh" / "textured.obj",
    ]
    mesh = next((p for p in mesh_candidates if p.is_file()), None)
    if mesh is None:
        raise FileNotFoundError(f"No mesh obj under {scene / 'mesh'}")

    rgb_files = sorted(rgb_dir.glob("*.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No rgb pngs in {rgb_dir}")

    K = _load_K(scene)
    gt_dir = scene / "annotated_poses"
    out.parent.mkdir(parents=True, exist_ok=True)
    base = out.parent

    n = 0
    with out.open("w", encoding="utf-8") as f:
        for i, rgb in enumerate(rgb_files):
            frame_id = rgb.stem
            depth = depth_dir / f"{frame_id}.png"
            if not depth.is_file():
                # some dumps use same stem with different extension already png
                alts = list(depth_dir.glob(f"{frame_id}.*"))
                if not alts:
                    print(f"[warn] skip {frame_id}: missing depth", file=sys.stderr)
                    continue
                depth = alts[0]

            mask = None
            for cand in (
                mask_dir / f"{frame_id}.png",
                mask_dir / f"{frame_id}.jpg",
            ):
                if cand.is_file():
                    mask = cand
                    break
            if mask is None and mask_dir.is_dir():
                # first-frame-only mask common in mustard0
                masks = sorted(mask_dir.glob("*.png")) + sorted(mask_dir.glob("*.jpg"))
                if masks:
                    mask = masks[0] if i == 0 else None

            gt_pose = None
            if gt_dir.is_dir():
                for cand in (
                    gt_dir / f"{frame_id}.txt",
                    gt_dir / f"{i:06d}.txt",
                    gt_dir / f"{i}.txt",
                ):
                    if cand.is_file():
                        gt_pose = cand
                        break
                if gt_pose is None:
                    gts = sorted(gt_dir.glob("*.txt"))
                    if i < len(gts):
                        gt_pose = gts[i]

            row = {
                "frame_id": frame_id,
                "timestamp_sec": float(i) * 0.1,
                "rgb": _rel(rgb, base),
                "depth": _rel(depth, base),
                "depth_scale_to_m": 0.001,
                "camera_matrix": K,
                "mesh": _rel(mesh, base),
                "object_id": object_id,
            }
            if mask is not None:
                row["gt_mask"] = _rel(mask, base)
            if gt_pose is not None:
                row["gt_pose"] = _rel(gt_pose, base)
            # Placeholder for FoundationPose outputs (attach_pose_results.py can fill)
            pred_dir = scene / "pred_pose"
            pred_dir.mkdir(exist_ok=True)
            pred_pose = pred_dir / f"{frame_id}.txt"
            row["predicted_pose"] = _rel(pred_pose, base)

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"[ok] wrote {n} frames -> {out}")
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--object-id", default=None)
    args = parser.parse_args()
    if args.scene is None:
        scene = next((p for p in DEFAULT_SCENES if p.is_dir() and (p / "rgb").is_dir()), None)
        if scene is None:
            raise SystemExit(
                "No demo scene found. Run scripts/download_assets.py "
                "or scripts/make_mustard0_minisynth.py first."
            )
    else:
        scene = args.scene.resolve()
        if not scene.is_dir():
            raise SystemExit(f"Scene not found: {scene}")
    object_id = args.object_id or scene.name
    build_manifest(scene, args.out.resolve(), object_id=object_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
