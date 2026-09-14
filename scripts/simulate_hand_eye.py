from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robot_pose_pipeline.hand_eye import (
    HAND_EYE_METHODS,
    evaluate_hand_eye,
    generate_eye_in_hand_dataset,
    solve_eye_in_hand,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Practice eye-in-hand AX=XB with synthetic poses.")
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--method", choices=HAND_EYE_METHODS, default="tsai")
    parser.add_argument("--rotation-noise-deg", type=float, default=0.1)
    parser.add_argument("--translation-noise-m", type=float, default=0.001)
    parser.add_argument("--output", default="outputs/hand_eye_result.json")
    args = parser.parse_args()

    dataset = generate_eye_in_hand_dataset(
        sample_count=args.samples,
        noise_rotation_deg=args.rotation_noise_deg,
        noise_translation=args.translation_noise_m,
    )
    estimated = solve_eye_in_hand(
        dataset.base_to_gripper,
        dataset.camera_to_target,
        HAND_EYE_METHODS[args.method],
    )
    payload = {
        "convention": "T_parent_child maps child-frame coordinates into parent frame",
        "method": args.method,
        "T_gripper_camera_estimated": estimated.tolist(),
        "T_gripper_camera_ground_truth": dataset.gripper_to_camera_ground_truth.tolist(),
        "metrics": evaluate_hand_eye(estimated, dataset.gripper_to_camera_ground_truth),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.set_printoptions(precision=5, suppress=True)
    print("T_gripper_camera estimated:\n", estimated)
    print(json.dumps(payload["metrics"], indent=2))
    print(output.resolve())


if __name__ == "__main__":
    main()
