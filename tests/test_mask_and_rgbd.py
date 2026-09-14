import unittest

import numpy as np

from robot_pose_pipeline.mask_metrics import mask_coverage, mask_iou
from robot_pose_pipeline.rgbd import depth_to_points
from robot_pose_pipeline.synthetic_scene import default_camera_matrix, render_synthetic_frame, write_synthetic_dataset
from robot_pose_pipeline.transforms import make_transform
from scipy.spatial.transform import Rotation


class MaskMetricTests(unittest.TestCase):
    def test_perfect_iou(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 255
        self.assertAlmostEqual(mask_iou(mask, mask), 1.0)

    def test_coverage_fields(self):
        gt = np.zeros((8, 8), dtype=np.uint8)
        pred = np.zeros((8, 8), dtype=np.uint8)
        gt[2:6, 2:6] = 255
        pred[3:7, 3:7] = 255
        metrics = mask_coverage(pred, gt)
        self.assertIn("iou", metrics)
        self.assertGreater(metrics["iou"], 0.0)
        self.assertLess(metrics["iou"], 1.0)


class RgbdTests(unittest.TestCase):
    def test_backproject_center(self):
        depth = np.zeros((4, 4), dtype=np.uint16)
        depth[2, 2] = 1000  # 1 meter
        camera = np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]])
        points = depth_to_points(depth, camera, depth_scale_to_m=0.001)
        self.assertEqual(len(points), 1)
        np.testing.assert_allclose(points[0], [0.0, 0.0, 1.0], atol=1e-6)


class SyntheticSceneTests(unittest.TestCase):
    def test_render_and_write(self):
        camera = default_camera_matrix()
        pose = make_transform(Rotation.from_euler("xyz", [10, -8, 0], degrees=True).as_matrix(), [0.0, 0.0, 0.5])
        rgb, depth, mask, points = render_synthetic_frame(camera, pose)
        self.assertEqual(rgb.shape[:2], depth.shape)
        self.assertGreater(int((mask > 0).sum()), 100)
        self.assertGreater(len(points), 10)


if __name__ == "__main__":
    unittest.main()
