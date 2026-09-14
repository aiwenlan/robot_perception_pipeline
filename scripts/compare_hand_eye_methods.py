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
    parser = argparse.ArgumentParser(description="Compare OpenCV hand-eye solvers under noise.")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--rotation-noise-deg", type=float, default=0.5)
    parser.add_argument("--translation-noise-m", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="outputs/hand_eye_method_compare.json")
    args = parser.parse_args()

    dataset = generate_eye_in_hand_dataset(
        sample_count=args.samples,
        seed=args.seed,
        noise_rotation_deg=args.rotation_noise_deg,
        noise_translation=args.translation_noise_m,
    )
    results = {}
    for name, method in HAND_EYE_METHODS.items():
        estimated = solve_eye_in_hand(dataset.base_to_gripper, dataset.camera_to_target, method)
        results[name] = {
            "metrics": evaluate_hand_eye(estimated, dataset.gripper_to_camera_ground_truth),
            "T_gripper_camera": estimated.tolist(),
        }
    payload = {
        "samples": args.samples,
        "rotation_noise_deg": args.rotation_noise_deg,
        "translation_noise_m": args.translation_noise_m,
        "methods": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, item in results.items():
        print(f"{name:10} rot={item['metrics']['rotation_error_deg']:.4f} deg  "
              f"trans={item['metrics']['translation_error']:.6f} m")
    print(output.resolve())


if __name__ == "__main__":
    main()
