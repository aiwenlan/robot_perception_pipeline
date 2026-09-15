#!/usr/bin/env python3
"""Create a synthetic plane + box cloud (and a perturbed model) as PCD for pcl_demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def write_pcd_xyz(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA ascii\n"
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def make_scene(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Table plane z=0.55, camera looking roughly +Z (optical).
    uu, vv = np.meshgrid(np.linspace(-0.25, 0.25, 80), np.linspace(-0.20, 0.20, 70))
    plane = np.stack([uu.ravel(), vv.ravel(), np.full(uu.size, 0.55)], axis=1)
    plane += rng.normal(0.0, 0.0015, plane.shape)

    # Box object above the table.
    xs, ys, zs = np.meshgrid(
        np.linspace(-0.04, 0.04, 25),
        np.linspace(-0.03, 0.03, 20),
        np.linspace(0.42, 0.52, 18),
    )
    box = np.stack([xs.ravel() + 0.02, ys.ravel() - 0.05, zs.ravel()], axis=1)
    box += rng.normal(0.0, 0.001, box.shape)

    # Sparse clutter blob far from object.
    clutter = rng.normal(scale=[0.02, 0.02, 0.01], size=(300, 3)) + np.array([-0.18, 0.12, 0.48])

    scene = np.vstack([plane, box, clutter]).astype(np.float32)
    model = box.copy().astype(np.float32)
    return scene, model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    scene, model = make_scene(rng)
    # Perturb model pose slightly so ICP has work to do.
    model_src = model.copy()
    model_src[:, 0] += 0.012
    model_src[:, 1] -= 0.008
    model_src[:, 2] += 0.004

    write_pcd_xyz(args.out_dir / "scene.pcd", scene)
    write_pcd_xyz(args.out_dir / "model.pcd", model_src)
    print(f"scene points={len(scene)} -> {args.out_dir / 'scene.pcd'}")
    print(f"model points={len(model_src)} -> {args.out_dir / 'model.pcd'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
