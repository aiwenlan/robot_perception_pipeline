from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from robot_pose_pipeline.transforms import compose


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose T_base_camera and T_camera_object.")
    parser.add_argument("--base-camera", required=True)
    parser.add_argument("--camera-object", required=True)
    parser.add_argument("--output", default="outputs/T_base_object.txt")
    args = parser.parse_args()
    result = compose(np.loadtxt(args.base_camera), np.loadtxt(args.camera_object))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, result, fmt="%.9f")
    print("T_base_object = T_base_camera @ T_camera_object")
    print(result)


if __name__ == "__main__":
    main()
