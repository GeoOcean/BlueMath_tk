import unittest
import numpy as np
import pandas as pd
from bluemath_tk.datamining.kma import KMA


class TestKMA(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "Hs": np.random.rand(1000) * 7,
                "Tp": np.random.rand(1000) * 20,
                "Dir": np.random.rand(1000) * 360,
            }
        )
        self.kma = KMA(num_clusters=10)

    def test_fit(self):
        self.kma.fit(data=self.df)
        self.assertIsInstance(self.kma.centroids, pd.DataFrame)
        self.assertEqual(self.kma.centroids.shape[0], 10)

    def test_predict(self):
        data_sample = pd.DataFrame(
            {
                "Hs": np.random.rand(15) * 7,
                "Tp": np.random.rand(15) * 20,
                "Dir": np.random.rand(15) * 360,
            }
        )
        self.kma.fit(data=self.df)
        nearest_centroids, nearest_centroid_df = self.kma.predict(data=data_sample)
        self.assertIsInstance(nearest_centroids, np.ndarray)
        self.assertEqual(len(nearest_centroids), 15)
        self.assertIsInstance(nearest_centroid_df, pd.DataFrame)
        self.assertEqual(nearest_centroid_df.shape[0], 15)

    def test_fit_predict(self):
        nearest_centroids, nearest_centroid_df = self.kma.fit_predict(data=self.df)
        self.assertIsInstance(nearest_centroids, np.ndarray)
        self.assertEqual(len(nearest_centroids), 1000)
        self.assertIsInstance(nearest_centroid_df, pd.DataFrame)
        self.assertEqual(nearest_centroid_df.shape[0], 1000)


if __name__ == "__main__":
    unittest.main()
