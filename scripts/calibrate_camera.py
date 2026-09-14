from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_pose_pipeline.calibration import calibrate_chessboard_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate one camera from chessboard images.")
    parser.add_argument("images", help="Image glob, for example data/calibration/*.jpg")
    parser.add_argument("--cols", type=int, default=7, help="Inner-corner columns")
    parser.add_argument("--rows", type=int, default=6, help="Inner-corner rows")
    parser.add_argument("--square-size", type=float, default=0.03, help="Physical square size")
    parser.add_argument("--unit", default="m")
    parser.add_argument("--output", default="outputs/camera_calibration.json")
    parser.add_argument("--preview-dir", default="outputs/calibration_previews")
    args = parser.parse_args()

    paths = sorted(Path().glob(args.images))
    result = calibrate_chessboard_images(
        paths, args.cols, args.rows, args.square_size, args.unit, args.preview_dir
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"used={len(result.used_images)} rejected={len(result.rejected_images)}")
    print(f"RMS={result.rms:.6f}, mean reprojection error={result.mean_reprojection_error_px:.4f}px")
    print(output.resolve())


if __name__ == "__main__":
    main()
