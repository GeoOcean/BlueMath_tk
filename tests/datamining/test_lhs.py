import unittest
import os
import pandas as pd
from bluemath_tk.datamining.lhs import LHS


class TestLHS(unittest.TestCase):
    def setUp(self):
        self.dimensions_names = ["CM", "SS", "Qb"]
        self.lower_bounds = [0.5, -0.2, 1]
        self.upper_bounds = [5.3, 1.5, 200]
        self.lhs = LHS(num_dimensions=3, seed=0)

    def test_generate(self):
        lhs_sampled_df = self.lhs.generate(
            dimensions_names=self.dimensions_names,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            num_samples=100,
        )
        self.assertIsInstance(lhs_sampled_df, pd.DataFrame)
        self.assertEqual(lhs_sampled_df.shape[0], 100)

    # def test_save_model(self):
    #     self.lhs.save_model("/tmp/lhs_model.pkl")
    #     self.assertTrue(os.path.exists("/tmp/lhs_model.pkl"))

    # def test_load_model(self):
    #     lhs_loaded = self.lhs.load_model("/tmp/lhs_model.pkl")
    #     self.assertIsInstance(lhs_loaded, LHS)


if __name__ == "__main__":
    unittest.main()
