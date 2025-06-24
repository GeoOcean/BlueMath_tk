import unittest

import numpy as np
import pandas as pd

from bluemath_tk.datamining.mda import (
    MDA,
    MDAError,
    calculate_normalized_squared_distance,
    find_nearest_indices,
)


class TestMDAHelperFunctions(unittest.TestCase):
    def test_calculate_normalized_squared_distance(self):
        # Test basic distance calculation
        data_array = np.array([[0.5, 0.5]])
        array_to_compare = np.array([[0.5, 0.5], [1.0, 1.0]])
        distances = calculate_normalized_squared_distance(data_array, array_to_compare)
        self.assertEqual(len(distances), 2)
        self.assertAlmostEqual(distances[0], 0.0)  # Same point
        self.assertAlmostEqual(distances[1], 0.5)  # Distance to [1.0, 1.0]

        # Test with directional indices
        data_array = np.array([[0.0, 0.9]])  # Angle near 1.0 (360 degrees normalized)
        array_to_compare = np.array([[0.1, 0.0]])  # Small angle
        distances = calculate_normalized_squared_distance(
            data_array, array_to_compare, directional_indices=[1]
        )
        # Should use minimum circular distance
        self.assertTrue(
            distances[0] < 0.5
        )  # Distance should be small due to circular nature

        # Test with weights
        data_array = np.array([[0.5, 0.5]])
        array_to_compare = np.array([[1.0, 1.0]])
        distances = calculate_normalized_squared_distance(
            data_array, array_to_compare, weights=[2.0, 1.0]
        )
        self.assertAlmostEqual(distances[0], 1.25)  # ((0.5 * 2)^2 + (0.5 * 1)^2)

    def test_find_nearest_indices(self):
        query_points = np.array(
            [
                [0.1, 0.1],
                [0.9, 0.9],
            ]
        )
        reference_points = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        indices = find_nearest_indices(query_points, reference_points)
        self.assertEqual(len(indices), 2)
        self.assertEqual(indices[0], 0)  # Closest to (0,0)
        self.assertEqual(indices[1], 2)  # Closest to (1,1)


class TestMDA(unittest.TestCase):
    def setUp(self):
        # Create fixed test data instead of random
        self.data = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "angle": [
                    0.0,
                    45.0,
                    90.0,
                    135.0,
                    180.0,
                    225.0,
                    270.0,
                    315.0,
                    360.0,
                    0.0,
                ],
            }
        )
        self.mda = MDA(num_centers=3)

    def test_initialization(self):
        # Test valid initialization
        mda = MDA(num_centers=5)
        self.assertEqual(mda.num_centers, 5)

        # Test invalid initialization
        with self.assertRaises(ValueError):
            MDA(num_centers=0)
        with self.assertRaises(ValueError):
            MDA(num_centers=-1)

    def test_fit_basic(self):
        # Test basic fitting without directional variables
        self.mda.fit(data=self.data)
        self.assertTrue(self.mda.is_fitted)
        self.assertEqual(len(self.mda.centroids), 3)
        self.assertEqual(list(self.mda.centroids.columns), ["x", "y", "angle"])

    def test_fit_with_directional(self):
        # Test fitting with directional variables
        self.mda.fit(
            data=self.data,
            directional_variables=["angle"],
            custom_scale_factor={"angle": [0, 360]},
        )
        self.assertTrue(self.mda.is_fitted)
        self.assertEqual(len(self.mda.centroids), 3)
        # Check that angles are within [0, 360]
        self.assertTrue(all(0 <= angle <= 360 for angle in self.mda.centroids["angle"]))

    def test_seed_reproducibility(self):
        # Test that same seed gives same results
        mda1 = MDA(num_centers=3)
        mda2 = MDA(num_centers=3)

        seed = 5
        mda1.fit(data=self.data, first_centroid_seed=seed)
        mda2.fit(data=self.data, first_centroid_seed=seed)

        pd.testing.assert_frame_equal(mda1.centroids, mda2.centroids)

    def test_predict(self):
        self.mda.fit(data=self.data)

        # Create test points
        test_data = pd.DataFrame(
            {
                "x": [1.1, 5.5, 9.9],
                "y": [0.11, 0.55, 0.99],
                "angle": [10.0, 180.0, 350.0],
            }
        )

        indices, nearest_df = self.mda.predict(test_data)
        self.assertEqual(len(indices), 3)
        self.assertEqual(len(nearest_df), 3)
        self.assertEqual(list(nearest_df.columns), ["x", "y", "angle"])

    def test_fit_predict(self):
        indices, nearest_df = self.mda.fit_predict(data=self.data)
        self.assertEqual(len(indices), len(self.data))
        self.assertEqual(len(nearest_df), len(self.data))
        self.assertTrue(self.mda.is_fitted)

    def test_error_cases(self):
        # Test predict before fit
        mda = MDA(num_centers=3)
        with self.assertRaises(MDAError):
            mda.predict(self.data)

        # Test invalid data type
        with self.assertRaises(TypeError):
            self.mda.fit(data=[[1, 2, 3], [4, 5, 6]])  # Not a DataFrame

        # Test invalid directional variables
        with self.assertRaises(TypeError):
            self.mda.fit(
                data=self.data, directional_variables="angle"
            )  # Should be list

        # Test invalid custom scale factor
        with self.assertRaises(TypeError):
            self.mda.fit(data=self.data, custom_scale_factor=[0, 360])  # Should be dict

        # Test invalid first centroid seed
        with self.assertRaises(ValueError):
            self.mda.fit(data=self.data, first_centroid_seed=len(self.data) + 1)


if __name__ == "__main__":
    unittest.main()
