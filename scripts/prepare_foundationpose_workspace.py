#!/usr/bin/env python3
"""Verify FoundationPose workspace junctions and print ready-to-run demo commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FP = ROOT.parent / "code" / "FoundationPose-main"
LOCAL_WEIGHTS = ROOT / "data" / "foundationpose" / "weights"
LOCAL_DEMO = ROOT / "data" / "foundationpose" / "demo_data"
REFINER = "2023-10-28-18-33-37"
SCORER = "2024-01-11-20-02-45"


def ok(label: str, path: Path, required: bool = True) -> bool:
    exists = path.exists()
    status = "OK" if exists else "MISSING"
    print(f"[{status:7}] {label}: {path}")
    return exists or not required


def resolve_scene(demo_root: Path) -> Path | None:
    for name in ("mustard0", "mustard0_minisynth"):
        scene = demo_root / name
        if (scene / "rgb").is_dir() and (scene / "mesh" / "textured_simple.obj").is_file():
            return scene
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundationpose", type=Path, default=DEFAULT_FP)
    args = parser.parse_args()
    fp = args.foundationpose.resolve()

    print("=== Local assets (canonical) ===")
    checks = [
        ok("FP repository", fp / "estimater.py"),
        ok("local refiner", LOCAL_WEIGHTS / REFINER / "model_best.pth"),
        ok("local scorer", LOCAL_WEIGHTS / SCORER / "model_best.pth"),
        ok("FP weights junction", fp / "weights"),
        ok("FP demo_data junction", fp / "demo_data"),
    ]

    scene = resolve_scene(LOCAL_DEMO)
    if scene is None:
        ok("demo scene (mustard0 or minisynth)", LOCAL_DEMO / "mustard0")
        checks.append(False)
    else:
        print(f"[OK     ] demo scene: {scene}")
        checks.append(True)
        if scene.name != "mustard0":
            print(
                "[NOTE   ] Using synthetic stand-in. Official mustard0 preferred when Drive quota clears."
            )

    print()
    print("=== Suggested commands ===")
    print("# Windows: refresh junctions if needed")
    print(f'cd "{ROOT}"')
    print(r"python scripts\link_foundationpose_assets.py")
    print()
    if scene is not None:
        mesh = scene / "mesh" / "textured_simple.obj"
        # Prefer paths as seen from FoundationPose repo via junctions
        fp_scene = fp / "demo_data" / scene.name
        fp_mesh = fp_scene / "mesh" / "textured_simple.obj"
        print("# Inside FoundationPose Docker / Linux container:")
        print(f"cd {fp.as_posix()}")
        print("python run_demo.py \\")
        print(f"  --mesh_file {fp_mesh.as_posix()} \\")
        print(f"  --test_scene_dir {fp_scene.as_posix()}")
        print()
        print("# Windows host equivalents (if CUDA stack installed locally):")
        print(f'cd "{fp}"')
        print(f'python run_demo.py --mesh_file "{mesh}" --test_scene_dir "{scene}"')
    else:
        print("# No scene yet. Run:")
        print(r"python scripts\download_assets.py")
        print(r"python scripts\make_mustard0_minisynth.py")

    print()
    if all(checks):
        print("Workspace ready.")
        return 0
    print("Workspace incomplete — see MISSING items above.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
