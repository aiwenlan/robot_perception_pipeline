import unittest

import numpy as np


class RegistrationIcpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import open3d  # noqa: F401

            cls.have = True
        except ImportError:
            cls.have = False

    def test_icp_recovers_small_translation(self):
        if not self.have:
            self.skipTest("open3d not installed")
        from robot_pose_pipeline.pointcloud.convert import numpy_to_open3d
        from robot_pose_pipeline.registration import icp_point_to_point, transform_model_to_camera

        rng = np.random.default_rng(1)
        model = rng.normal(size=(500, 3)) * 0.05
        T_true = np.eye(4)
        T_true[:3, 3] = [0.1, -0.05, 0.8]
        target = transform_model_to_camera(numpy_to_open3d(model), T_true)
        T_init = T_true.copy()
        T_init[:3, 3] += [0.01, -0.008, 0.005]
        source = transform_model_to_camera(numpy_to_open3d(model), T_init)
        result = icp_point_to_point(source, target, T_init, threshold=0.05)
        self.assertGreater(result.fitness, 0.8)
        self.assertLess(result.rmse, 0.01)
        np.testing.assert_allclose(result.T_camera_object_refined[:3, 3], T_true[:3, 3], atol=5e-3)


if __name__ == "__main__":
    unittest.main()
