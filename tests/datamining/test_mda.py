import unittest
import numpy as np
import pandas as pd
from bluemath_tk.datamining.mda import MDA


class TestMDA(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame(
            {
                "Hs": np.random.rand(1000) * 7,
                "Tp": np.random.rand(1000) * 20,
                "Dir": np.random.rand(1000) * 360,
            }
        )
        self.mda = MDA(data=df, ix_directional=["Dir"])

    def test_run(self):
        self.mda.run(10)  # Run mda to assign centroids and test
        self.assertEqual(self.mda.centroids.shape[0], 10)


if __name__ == "__main__":
    unittest.main()
