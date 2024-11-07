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
        self.kma = KMA(num_clusters=5)

    def test_fit(self):
        centroids = self.kma.fit(
            data=self.df,
            directional_variables=["Dir"],
            custom_scale_factor={"Dir": [0, 360]},
        )
        self.assertIsInstance(centroids, pd.DataFrame)
        self.assertEqual(centroids.shape[0], 5)


if __name__ == "__main__":
    unittest.main()
