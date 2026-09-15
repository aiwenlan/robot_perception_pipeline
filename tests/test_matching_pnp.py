import unittest

import numpy as np

from robot_pose_pipeline.matching import solve_pnp_from_2d3d
from robot_pose_pipeline.transforms import make_transform, transform_points
from scipy.spatial.transform import Rotation


class OrbPnpTests(unittest.TestCase):
    def test_pnp_synthetic_correspondences(self):
        K = np.array([[400.0, 0, 320.0], [0, 400.0, 240.0], [0, 0, 1.0]])
        obj = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.05, 0.04, 0.0],
                [0.0, 0.04, 0.0],
                [0.02, 0.02, 0.03],
                [-0.02, 0.01, 0.02],
            ],
            dtype=float,
        )
        T = make_transform(Rotation.from_euler("xyz", [15, -10, 5], degrees=True).as_matrix(), [0.02, -0.01, 0.6])
        pts_cam = transform_points(T, obj)
        uv = np.stack(
            [
                K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2],
                K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2],
            ],
            axis=1,
        )
        result = solve_pnp_from_2d3d(uv, obj, K, ransac=False)
        self.assertTrue(result.success)
        self.assertLess(result.reprojection_error_px, 1.0)
        np.testing.assert_allclose(result.T_camera_object[:3, 3], T[:3, 3], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
