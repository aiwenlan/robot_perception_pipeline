import unittest

import numpy as np

from robot_pose_pipeline.grasping import (
    GraspSet,
    grasps_from_graspnet_npy,
    grasps_to_npy,
    make_grasp_from_center_approach,
    propose_grasps_on_cloud,
)
from robot_pose_pipeline.transforms import as_transform


class GraspingTests(unittest.TestCase):
    def test_make_grasp_approach_axis(self):
        g = make_grasp_from_center_approach([0, 0, 0.5], [0, 0, 1], width=0.04, score=0.9)
        T = as_transform(g.T_camera_grasp)
        approach = -T[:3, 2]
        np.testing.assert_allclose(approach, [0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(g.center(), [0, 0, 0.5], atol=1e-6)

    def test_propose_and_roundtrip_npz(self):
        rng = np.random.default_rng(0)
        pts = rng.normal(scale=0.02, size=(500, 3)) + np.array([0.0, 0.0, 0.5])
        grasps = propose_grasps_on_cloud(pts, num_samples=40, topk=10, seed=1)
        self.assertGreaterEqual(len(grasps), 1)
        best = grasps.best()
        self.assertIsNotNone(best)
        self.assertGreater(best.width, 0.0)

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            path = grasps_to_npy(grasps, Path(td) / "g.npz")
            loaded = grasps_from_graspnet_npy(path)
            self.assertEqual(len(loaded), len(grasps))
            np.testing.assert_allclose(loaded.best().T_camera_grasp, best.T_camera_grasp, atol=1e-8)

    def test_graspgroup_layout_17(self):
        g = make_grasp_from_center_approach([0.1, -0.2, 0.4], [0, 0, 1], width=0.05, score=0.8, depth=0.03)
        R = g.T_camera_grasp[:3, :3].reshape(-1)
        t = g.T_camera_grasp[:3, 3]
        row = np.concatenate([[g.score, g.width, 0.02, g.depth], R, t, [-1.0]])
        gg = row.reshape(1, 17)
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "gg.npy"
            np.save(p, gg)
            loaded = grasps_from_graspnet_npy(p)
            self.assertEqual(len(loaded), 1)
            np.testing.assert_allclose(loaded.grasps[0].center(), g.center(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
