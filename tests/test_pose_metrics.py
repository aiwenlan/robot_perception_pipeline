import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from robot_pose_pipeline.pose_metrics import evaluate_pose
from robot_pose_pipeline.transforms import make_transform


class PoseMetricTests(unittest.TestCase):
    def test_known_error(self):
        gt = make_transform(np.eye(3), [0, 0, 1])
        pred = make_transform(Rotation.from_euler("z", 10, degrees=True).as_matrix(), [0.01, 0, 1])
        points = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=float)
        metrics = evaluate_pose(pred, gt, points, np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]))
        self.assertAlmostEqual(metrics["rotation_error_deg"], 10.0, places=6)
        self.assertAlmostEqual(metrics["translation_error"], 0.01, places=8)
        self.assertGreater(metrics["projection_error_px"], 0)


if __name__ == "__main__":
    unittest.main()
