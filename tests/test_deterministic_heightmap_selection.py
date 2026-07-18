import unittest

import numpy as np

from shelf_gym.utils.mapping_utils import deterministic_top_point_indices


class DeterministicHeightmapSelectionTest(unittest.TestCase):
    def test_selects_highest_point_once_per_pixel(self):
        linear = np.asarray([4, 2, 4, 2, 8, 4], dtype=np.int64)
        z = np.asarray([0.1, 0.5, 0.7, 0.4, 0.2, 0.6], dtype=np.float64)
        selected = deterministic_top_point_indices(
            linear, z, array_module=np
        )

        self.assertEqual(selected.tolist(), [1, 2, 4])
        self.assertEqual(linear[selected].tolist(), [2, 4, 8])
        self.assertEqual(z[selected].tolist(), [0.5, 0.7, 0.2])

    def test_equal_height_tie_uses_later_input_point(self):
        linear = np.asarray([3, 3, 3], dtype=np.int64)
        z = np.asarray([0.2, 0.9, 0.9], dtype=np.float64)
        first = deterministic_top_point_indices(linear, z, array_module=np)
        second = deterministic_top_point_indices(linear, z, array_module=np)

        self.assertEqual(first.tolist(), [2])
        self.assertTrue(np.array_equal(first, second))

    def test_rejects_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "equal 1D arrays"):
            deterministic_top_point_indices(
                np.asarray([1, 2]), np.asarray([[0.1, 0.2]]), array_module=np
            )


if __name__ == "__main__":
    unittest.main()
