from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from robot_pose_pipeline.io import imread
from robot_pose_pipeline.mask_metrics import mask_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare predicted mask against GT mask.")
    parser.add_argument("--predicted", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", default="outputs/mask_metrics.json")
    args = parser.parse_args()
    predicted = imread(args.predicted, cv2.IMREAD_GRAYSCALE)
    ground_truth = imread(args.ground_truth, cv2.IMREAD_GRAYSCALE)
    if predicted is None or ground_truth is None:
        raise FileNotFoundError("cannot read predicted/ground-truth mask")
    metrics = mask_coverage(predicted, ground_truth)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
