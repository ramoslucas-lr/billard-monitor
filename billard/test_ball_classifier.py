import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from billard.ball_classifier import find_nearest

class TestFindNearest(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(find_nearest([1, 2, 3, 4, 5], 3), 3)

    def test_nearest_larger(self):
        self.assertEqual(find_nearest([10, 20, 30], 18), 20)

    def test_nearest_smaller(self):
        self.assertEqual(find_nearest([10, 20, 30], 22), 20)

    def test_negative_numbers(self):
        self.assertEqual(find_nearest([-10, -5, 0, 5], -3), -5)
        self.assertEqual(find_nearest([-10, -5, 0, 5], -7), -5)

    def test_equidistant_values(self):
        # argmin will pick the first minimum index.
        # np.abs([1, 3] - 2) -> [1, 1], argmin -> 0 -> index 0 which is 1
        self.assertEqual(find_nearest([1, 3], 2), 1)
        self.assertEqual(find_nearest([3, 1], 2), 3)

    def test_single_element_array(self):
        self.assertEqual(find_nearest([42], 100), 42)

    def test_numpy_array_input(self):
        arr = np.array([10.5, 20.5, 30.5])
        self.assertEqual(find_nearest(arr, 21.0), 20.5)

    def test_float_values(self):
        self.assertAlmostEqual(find_nearest([1.1, 2.2, 3.3], 2.0), 2.2)

if __name__ == '__main__':
    unittest.main()
