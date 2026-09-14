import unittest

from robot_pose_pipeline.hand_eye import evaluate_hand_eye, generate_eye_in_hand_dataset, solve_eye_in_hand


class HandEyeTests(unittest.TestCase):
    def test_noise_free_solution(self):
        data = generate_eye_in_hand_dataset(sample_count=30)
        estimated = solve_eye_in_hand(data.base_to_gripper, data.camera_to_target)
        metrics = evaluate_hand_eye(estimated, data.gripper_to_camera_ground_truth)
        self.assertLess(metrics["rotation_error_deg"], 1e-4)
        self.assertLess(metrics["translation_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()
