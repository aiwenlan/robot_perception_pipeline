import unittest

import numpy as np


class TsdfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import open3d  # noqa: F401

            cls.have = True
        except ImportError:
            cls.have = False

    def test_tsdf_two_frames(self):
        if not self.have:
            self.skipTest("open3d not installed")
        from robot_pose_pipeline.reconstruction import integrate_rgbd_frames

        h, w = 48, 64
        K = np.array([[80.0, 0, w / 2], [0, 80.0, h / 2], [0, 0, 1.0]])
        depth = np.full((h, w), 1000, dtype=np.uint16)  # 1m plane
        rgb = np.full((h, w, 3), 180, dtype=np.uint8)
        T0 = np.eye(4)
        T1 = np.eye(4)
        T1[:3, 3] = [0.02, 0.0, 0.0]
        result = integrate_rgbd_frames(
            [rgb, rgb],
            [depth, depth],
            [T0, T1],
            K,
            pose_is_T_world_camera=True,
            depth_scale_to_m=0.001,
            voxel_length=0.02,
            sdf_trunc=0.05,
        )
        self.assertEqual(result.num_frames, 2)
        self.assertGreater(len(result.cloud.points), 10)


if __name__ == "__main__":
    unittest.main()
