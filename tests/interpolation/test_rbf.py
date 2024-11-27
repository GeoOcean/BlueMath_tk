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
                # "TpPred": self.subset["Tp"] * 1.8 + 5,
                "DirPred": self.subset["Dir"] % 45,
            }
        )
        self.rbf = RBF(
            sigma_min=0.001,
            sigma_max=0.1,
            sigma_diff=0.0001,
            kernel="gaussian",
        )

    # def test_fit(self):
    #     self.rbf.fit(
    #         subset_data=self.subset,
    #         target_data=self.target,
    #     )
    #     self.assertEqual(self.rbf.is_fitted, True)

    def test_predict(self):
        self.rbf.fit(
            subset_data=self.subset,
            target_data=self.target,
            subset_directional_variables=["Dir"],
            target_directional_variables=["DirPred"],
        )
        prediction = self.rbf.predict(dataset=self.dataset)
        self.assertEqual(prediction.shape[0], self.dataset.shape[0])


if __name__ == "__main__":
    unittest.main()
