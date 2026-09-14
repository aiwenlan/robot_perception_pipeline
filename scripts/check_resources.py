#!/usr/bin/env python3
"""Check local project assets and optional FoundationPose repo links."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def state(path: Path) -> str:
    return "OK" if path.exists() else "MISSING"


parser = argparse.ArgumentParser(
    description="Check resources under this repo (and optional FoundationPose links)."
)
parser.add_argument(
    "--foundationpose",
    default=str(ROOT.parent / "code" / "FoundationPose-main"),
    help="Course FoundationPose source tree (for junction / estimater.py check)",
)
parser.add_argument("--data", default=str(ROOT / "data"))
args = parser.parse_args()

foundation = Path(args.foundationpose).resolve()
data = Path(args.data).resolve()
fp_assets = data / "foundationpose"
mustard0 = fp_assets / "demo_data" / "mustard0"
minisynth = fp_assets / "demo_data" / "mustard0_minisynth"
usable_demo = mustard0 if mustard0.exists() else minisynth if minisynth.exists() else None

required = {
    "FoundationPose repository": foundation / "estimater.py",
    "local refiner weights": fp_assets / "weights" / "2023-10-28-18-33-37" / "model_best.pth",
    "local scorer weights": fp_assets / "weights" / "2024-01-11-20-02-45" / "model_best.pth",
    "usable demo scene (mustard0 or minisynth)": usable_demo or (fp_assets / "demo_data" / "mustard0"),
    "FP repo weights link/dir": foundation / "weights",
    "FP repo demo_data link/dir": foundation / "demo_data",
    "FastSAM weights": data / "models" / "FastSAM-s.pt",
    "calibration samples": data / "calibration" / "opencv_left",
    "synthetic dataset": data / "synthetic" / "manifest.jsonl",
    "dataset manifest": data / "manifest.jsonl",
}
optional = {
    "official mustard0": mustard0,
    "mustard0_minisynth": minisynth,
    "demo JSONL manifest": fp_assets / "demo_data" / "mustard0_manifest.jsonl",
    "optional LINEMOD models": data / "datasets" / "linemod" / "models",
    "manual download notes": fp_assets / "demo_data" / "README_MANUAL.md",
}

missing = 0
for label, path in required.items():
    status = state(path)
    if status != "OK":
        missing += 1
    print(f"[{status:7}] {label}: {path}")
print("--- optional ---")
for label, path in optional.items():
    print(f"[{state(path):7}] {label}: {path}")

print()
print("Canonical assets live under data/foundationpose/ (gitignored).")
print("Download: python scripts/download_assets.py")
print("Link into course repo: python scripts/link_foundationpose_assets.py")
print("Workspace check: python scripts/prepare_foundationpose_workspace.py")
print()
if missing:
    print(f"{missing} required resource(s) missing. See docs/DOWNLOAD_CHECKLIST.md")
else:
    note = "official mustard0" if mustard0.exists() else "minisynth stand-in (official preferred)"
    print(f"All required resources found ({note}).")
