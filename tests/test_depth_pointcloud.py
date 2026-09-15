import unittest

import numpy as np

from robot_pose_pipeline.depth import apply_depth_scale, depth_stats, preprocess_depth
from robot_pose_pipeline.rgbd import depth_to_points


class DepthPreprocessTests(unittest.TestCase):
    def test_scale_uint16(self):
        depth = np.array([[1000, 0], [2000, 5000]], dtype=np.uint16)
        depth_m = apply_depth_scale(depth, 0.001)
        np.testing.assert_allclose(depth_m[0, 0], 1.0)

    def test_preprocess_invalid(self):
        depth = np.array([[1000, 0], [8000, 500]], dtype=np.uint16)
        out = preprocess_depth(depth, max_depth_m=2.0, min_depth_m=0.05, median_ksize=0)
        self.assertTrue(np.isnan(out[0, 1]))
        self.assertTrue(np.isnan(out[1, 0]))  # 8m clipped
        self.assertAlmostEqual(out[0, 0], 1.0)
        stats = depth_stats(out)
        self.assertEqual(stats.valid_count, 2)

    def test_aligns_with_depth_to_points(self):
        depth = np.zeros((8, 8), dtype=np.uint16)
        depth[4, 4] = 1000
        K = np.array([[100.0, 0, 4.0], [0, 100.0, 4.0], [0, 0, 1.0]])
        pts = depth_to_points(depth, K, depth_scale_to_m=0.001)
        depth_m = preprocess_depth(depth, median_ksize=0, max_depth_m=5.0)
        filled = np.where(np.isfinite(depth_m), depth_m, 0.0)
        pts2 = depth_to_points(filled, K, depth_scale_to_m=1.0)
        np.testing.assert_allclose(pts, pts2, atol=1e-9)


class Open3DPointCloudTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import open3d  # noqa: F401
            cls.have_o3d = True
        except ImportError:
            cls.have_o3d = False

    def test_rgbd_to_pointcloud_center(self):
        if not self.have_o3d:
            self.skipTest("open3d not installed")
        from robot_pose_pipeline.pointcloud import open3d_to_numpy, rgbd_to_pointcloud

        depth = np.zeros((6, 6), dtype=np.uint16)
        depth[3, 3] = 1000
        K = np.array([[100.0, 0, 3.0], [0, 100.0, 3.0], [0, 0, 1.0]])
        cloud = rgbd_to_pointcloud(depth, K, depth_scale_to_m=0.001, median_ksize=0, max_depth_m=5.0)
        pts, _ = open3d_to_numpy(cloud)
        self.assertEqual(len(pts), 1)
        np.testing.assert_allclose(pts[0], [0.0, 0.0, 1.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
