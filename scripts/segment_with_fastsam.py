from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from robot_pose_pipeline.fastsam_adapter import FastSAMSegmenter
from robot_pose_pipeline.io import imread, imwrite


def main() -> None:
    parser = argparse.ArgumentParser(description="Use detector bbox to choose a FastSAM mask.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--box", required=True, nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="outputs/predicted_mask.png")
    args = parser.parse_args()
    image = imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    result = FastSAMSegmenter(args.weights, args.device).segment_box(image, np.array(args.box))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    imwrite(output, result.mask)
    print(f"candidate={result.candidate_index}, bbox IoU={result.candidate_box_iou:.4f}")
    print(output.resolve())


if __name__ == "__main__":
    main()
