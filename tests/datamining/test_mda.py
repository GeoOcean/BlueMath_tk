import unittest
import numpy as np
import pandas as pd
from bluemath_tk.datamining.mda import MDA


class TestMDA(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "Hs": np.random.rand(1000) * 7,
                "Tp": np.random.rand(1000) * 20,
                "Dir": np.random.rand(1000) * 360,
            }
        )
        self.mda = MDA(num_centers=10)

    def test_fit(self):
        self.mda.fit(
            data=self.df,
            directional_variables=["Dir"],
            custom_scale_factor={"Dir": [0, 360]},
            first_centroid_seed=10,
        )
        self.assertIsInstance(self.mda.centroids, pd.DataFrame)
        self.assertEqual(self.mda.centroids.shape[0], 10)

    def test_predict(self):
        data_sample = pd.DataFrame(
            {
                "Hs": np.random.rand(15) * 7,
                "Tp": np.random.rand(15) * 20,
                "Dir": np.random.rand(15) * 360,
            }
        )
        self.mda.fit(
            data=self.df,
            directional_variables=["Dir"],
            custom_scale_factor={},
        )
        nearest_centroids, nearest_centroid_df = self.mda.predict(
            data=data_sample,
        )
        self.assertIsInstance(nearest_centroids, np.ndarray)
        self.assertEqual(len(nearest_centroids), 15)
        self.assertIsInstance(nearest_centroid_df, pd.DataFrame)
        self.assertEqual(nearest_centroid_df.shape[0], 15)

    def test_fit_predict(self):
        nearest_centroids, nearest_centroid_df = self.mda.fit_predict(
            data=self.df,
            directional_variables=["Dir"],
            custom_scale_factor={"Dir": [0, 360]},
        )
        self.assertIsInstance(nearest_centroids, np.ndarray)
        self.assertEqual(len(nearest_centroids), 1000)
        self.assertIsInstance(nearest_centroid_df, pd.DataFrame)
        self.assertEqual(nearest_centroid_df.shape[0], 1000)


if __name__ == "__main__":
    unittest.main()
