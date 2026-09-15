import unittest

import numpy as np


class PointCloudFilterSegmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import open3d as o3d

            cls.o3d = o3d
            cls.have = True
        except ImportError:
            cls.have = False

    def _plane_plus_blob(self):
        o3d = self.o3d
        # z=1 plane
        uu, vv = np.meshgrid(np.linspace(-0.2, 0.2, 40), np.linspace(-0.2, 0.2, 40))
        plane = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], axis=1)
        blob = np.random.default_rng(0).normal(scale=0.01, size=(200, 3)) + np.array([0.0, 0.0, 0.7])
        pts = np.vstack([plane, blob])
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts)
        return cloud

    def test_voxel_and_plane_and_cluster(self):
        if not self.have:
            self.skipTest("open3d not installed")
        from robot_pose_pipeline.pointcloud import (
            cluster_dbscan,
            remove_plane_ransac,
            voxel_downsample,
        )

        cloud = self._plane_plus_blob()
        n0 = len(cloud.points)
        cloud = voxel_downsample(cloud, 0.01)
        self.assertLess(len(cloud.points), n0)
        rest = remove_plane_ransac(cloud, distance_threshold=0.02)
        self.assertGreater(len(rest.points), 50)
        clusters, infos = cluster_dbscan(rest, eps=0.05, min_points=20)
        self.assertGreaterEqual(len(clusters), 1)
        self.assertTrue(all(i.num_points > 0 for i in infos))


if __name__ == "__main__":
    unittest.main()
