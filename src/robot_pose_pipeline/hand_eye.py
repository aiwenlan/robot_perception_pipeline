from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .pose_metrics import rotation_error_deg, translation_error
from .transforms import compose, invert_transform, make_transform


@dataclass
class HandEyeDataset:
    base_to_gripper: list[np.ndarray]
    camera_to_target: list[np.ndarray]
    gripper_to_camera_ground_truth: np.ndarray
    base_to_target: np.ndarray


def _random_transform(rng: np.random.Generator, translation_scale: float) -> np.ndarray:
    rotation = Rotation.random(random_state=rng).as_matrix()
    translation = rng.uniform(-translation_scale, translation_scale, size=3)
    return make_transform(rotation, translation)


def generate_eye_in_hand_dataset(
    sample_count: int = 20,
    seed: int = 7,
    translation_scale: float = 0.35,
    noise_rotation_deg: float = 0.0,
    noise_translation: float = 0.0,
) -> HandEyeDataset:
    if sample_count < 4:
        raise ValueError("sample_count must be at least 4")
    rng = np.random.default_rng(seed)
    gripper_to_camera = _random_transform(rng, 0.08)
    base_to_target = _random_transform(rng, 0.25)
    base_to_target[:3, 3] += np.array([0.0, 0.0, 0.8])

    base_to_gripper: list[np.ndarray] = []
    camera_to_target: list[np.ndarray] = []
    for _ in range(sample_count):
        base_to_gripper_i = _random_transform(rng, translation_scale)
        base_to_gripper_i[:3, 3] += np.array([0.0, 0.0, 0.5])
        base_to_camera_i = compose(base_to_gripper_i, gripper_to_camera)
        camera_to_target_i = compose(invert_transform(base_to_camera_i), base_to_target)
        base_to_gripper.append(
            add_pose_noise(base_to_gripper_i, rng, noise_rotation_deg, noise_translation)
        )
        camera_to_target.append(
            add_pose_noise(camera_to_target_i, rng, noise_rotation_deg, noise_translation)
        )
    return HandEyeDataset(
        base_to_gripper=base_to_gripper,
        camera_to_target=camera_to_target,
        gripper_to_camera_ground_truth=gripper_to_camera,
        base_to_target=base_to_target,
    )


def add_pose_noise(
    transform: np.ndarray,
    rng: np.random.Generator,
    rotation_sigma_deg: float,
    translation_sigma: float,
) -> np.ndarray:
    if rotation_sigma_deg <= 0 and translation_sigma <= 0:
        return transform.copy()
    delta_rotation = Rotation.from_rotvec(
        np.radians(rng.normal(0.0, rotation_sigma_deg, size=3))
    ).as_matrix()
    noisy_rotation = delta_rotation @ transform[:3, :3]
    noisy_translation = transform[:3, 3] + rng.normal(0.0, translation_sigma, size=3)
    return make_transform(noisy_rotation, noisy_translation)


def solve_eye_in_hand(
    base_to_gripper: list[np.ndarray],
    camera_to_target: list[np.ndarray],
    method: int = cv2.CALIB_HAND_EYE_TSAI,
) -> np.ndarray:
    if len(base_to_gripper) != len(camera_to_target) or len(base_to_gripper) < 4:
        raise ValueError("hand-eye inputs must contain matching lists with at least 4 poses")
    rotations_gripper_to_base = [pose[:3, :3] for pose in base_to_gripper]
    translations_gripper_to_base = [pose[:3, 3].reshape(3, 1) for pose in base_to_gripper]
    rotations_target_to_camera = [pose[:3, :3] for pose in camera_to_target]
    translations_target_to_camera = [pose[:3, 3].reshape(3, 1) for pose in camera_to_target]
    rotation_camera_to_gripper, translation_camera_to_gripper = cv2.calibrateHandEye(
        rotations_gripper_to_base,
        translations_gripper_to_base,
        rotations_target_to_camera,
        translations_target_to_camera,
        method=method,
    )
    return make_transform(rotation_camera_to_gripper, translation_camera_to_gripper.reshape(3))


def evaluate_hand_eye(estimated: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    return {
        "rotation_error_deg": rotation_error_deg(estimated, ground_truth),
        "translation_error": translation_error(estimated, ground_truth),
    }


HAND_EYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

