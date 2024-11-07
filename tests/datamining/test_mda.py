import unittest
import numpy as np
import pandas as pd
from bluemath_tk.datamining.mda import MDA


class TestMDA(unittest.TestCase):
    def setUp(self):
        self.mda = MDA(num_centers=10)
        self.df = pd.DataFrame(
            {
                "Hs": np.random.rand(1000) * 7,
                "Tp": np.random.rand(1000) * 20,
                "Dir": np.random.rand(1000) * 360,
            }
        )

    def test_run(self):
        centroids = self.mda.fit(
            data=self.df,
            directional_variables=["Dir"],
            custom_scale_factor={"Dir": [0, 360]},
        )
        self.assertEqual(centroids.shape[0], 10)

    def test_nearest_centroid_indices(self):
        data_sample = pd.DataFrame(
            {
                "Hs": np.random.rand(15) * 7,
                "Tp": np.random.rand(15) * 20,
                "Dir": np.random.rand(15) * 360,
            }
        ).values
        _centroids = self.mda.fit(
            data=self.df,
            directional_variables=["Dir"],
            custom_scale_factor={},
        )
        nearest_centroids = self.mda.nearest_centroid_indices(
            data_q=data_sample,
        )
        self.assertEqual(len(nearest_centroids), 15)


if __name__ == "__main__":
    unittest.main()
