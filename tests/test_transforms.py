import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from robot_pose_pipeline.transforms import compose, invert_transform, make_transform, transform_points


class TransformTests(unittest.TestCase):
    def test_inverse_and_composition(self):
        transform = make_transform(Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix(), [1, 2, 3])
        np.testing.assert_allclose(compose(transform, invert_transform(transform)), np.eye(4), atol=1e-10)

    def test_transform_points(self):
        transform = make_transform(np.eye(3), [1, 2, 3])
        np.testing.assert_allclose(transform_points(transform, [[0, 0, 0]]), [[1, 2, 3]])


if __name__ == "__main__":
    unittest.main()
