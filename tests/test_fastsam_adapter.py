import unittest

import numpy as np

from robot_pose_pipeline.fastsam_adapter import select_mask_by_box


class FastSAMAdapterTests(unittest.TestCase):
    def test_selects_best_overlapping_candidate(self):
        masks = np.zeros((2, 8, 8), dtype=float)
        masks[1, 2:6, 2:6] = 1
        selected, index, score = select_mask_by_box(
            np.array([2, 2, 6, 6]), np.array([[0, 0, 1, 1], [2, 2, 6, 6]]), masks
        )
        self.assertEqual(index, 1)
        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(int(selected.sum()), 16 * 255)


if __name__ == "__main__":
    unittest.main()
