import unittest
import numpy as np
import pandas as pd
from bluemath_tk.interpolation.rbf import RBF


class TestRBF(unittest.TestCase):
    def setUp(self):
        self.dataset = pd.DataFrame(
            {
                "Hs": np.random.rand(1000) * 7,
                "Tp": np.random.rand(1000) * 20,
                "Dir": np.random.rand(1000) * 360,
            }
        )
        self.subset = self.dataset.sample(frac=0.25)
        self.target = pd.DataFrame(
            {
                "HsPred": self.subset["Hs"] * 2 + self.subset["Tp"] * 3,
                "DirPred": -self.subset["Dir"],
            }
        )
        self.rbf = RBF()

    def test_fit(self):
        self.rbf.fit(
            subset_data=self.subset,
            subset_directional_variables=["Dir"],
            target_data=self.target,
            target_directional_variables=["DirPred"],
            normalize_target_data=True,
        )
        self.assertTrue(self.rbf.is_fitted)
        self.assertTrue(self.rbf.is_target_normalized)
        self.assertIn("Dir_u", self.rbf.normalized_subset_data.columns)
        self.assertIn("Dir_v", self.rbf.normalized_subset_data.columns)
        self.assertIn("DirPred_u", self.rbf.normalized_target_data.columns)
        self.assertIn("DirPred_v", self.rbf.normalized_target_data.columns)
        self.assertFalse(self.rbf.rbf_coeffs.empty)
        self.assertFalse(self.rbf.opt_sigmas == {})

    def test_predict(self):
        self.rbf.fit(
            subset_data=self.subset,
            subset_directional_variables=["Dir"],
            target_data=self.target,
            target_directional_variables=["DirPred"],
            normalize_target_data=True,
        )
        predictions = self.rbf.predict(dataset=self.dataset)
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertIn("HsPred", predictions.columns)
        self.assertIn("DirPred", predictions.columns)

    def test_fit_predict(self):
        predictions = self.rbf.fit_predict(
            subset_data=self.subset,
            subset_directional_variables=["Dir"],
            target_data=self.target,
            target_directional_variables=["DirPred"],
            normalize_target_data=True,
            dataset=self.dataset,
            num_threads=4,
        )
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertIn("HsPred", predictions.columns)
        self.assertIn("DirPred", predictions.columns)


if __name__ == "__main__":
    unittest.main()
