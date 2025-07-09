import unittest
import numpy as np
import pandas as pd

from bluemath_tk.distributions.nonstat_gev import NonStatGEV

def get_sample_gev_data(n=100):
    np.random.seed(42)
    t = np.linspace(0, 10, n)
    xt = np.random.gumbel(loc=5 + 0.1 * t, scale=1 + 0.05 * t, size=n)
    covariates = pd.DataFrame({
        "cov1": np.sin(t),
        "cov2": np.cos(t)
    })
    return xt, t, covariates

class TestNonStatGEV(unittest.TestCase):
    def setUp(self):
        self.xt, self.t, self.covariates = get_sample_gev_data()
        self.model = NonStatGEV(
            xt=self.xt,
            t=self.t,
            covariates=self.covariates,
            trends=True,
            var_name="test_var"
        )

    def test_fit_basic(self):
        result = self.model.fit(
            nmu=0, npsi=0, ngamma=0,
            ntrend_loc=1, list_loc=[0],
            ntrend_sc=1, list_sc=[1],
            ntrend_sh=0, list_sh=[]
        )
        self.assertIn("negloglikelihood", result)
        self.assertTrue(result["success"])
        self.assertIsInstance(result["beta0"], float)
        self.assertIsInstance(result["alpha0"], float)

    def test_quantile_shape(self):
        self.model.fit(
            nmu=0, npsi=0, ngamma=0,
            ntrend_loc=1, list_loc=[0],
            ntrend_sc=1, list_sc=[1],
            ntrend_sh=0, list_sh=[]
        )
        q = self.model._quantile()
        self.assertEqual(q.shape, self.xt.shape)

if __name__ == "__main__":
    unittest.main()