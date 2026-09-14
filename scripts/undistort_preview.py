from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from robot_pose_pipeline.io import imread, imwrite


def main() -> None:
    parser = argparse.ArgumentParser(description="Save undistorted preview from camera_calibration.json.")
    parser.add_argument("--calibration", default="outputs/camera_calibration.json")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/undistort_preview.png")
    args = parser.parse_args()

    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    image = imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    camera_matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(calibration["distortion_coefficients"], dtype=np.float64)
    undistorted = cv2.undistort(image, camera_matrix, distortion)
    side_by_side = np.hstack([image, undistorted])
    cv2.putText(side_by_side, "original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(
        side_by_side,
        "undistorted",
        (image.shape[1] + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    imwrite(output, side_by_side)
    print(output.resolve())


if __name__ == "__main__":
    main()
