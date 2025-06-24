import unittest

import numpy as np

from bluemath_tk.distributions._base_distributions import FitResult
from bluemath_tk.distributions.gev import GEV


class TestGEV(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(1000) * 2
        self.p = np.random.rand(1000)
        self.loc = 0.0
        self.scale = 1.0
        self.shape_frechet = 0.1  # GEV (Frechet)
        self.shape_weibull = -0.1  # GEV (Weibull)
        self.shape_gumbel = 0.0  # Gumbel case

    def test_pdf(self):
        for shape in [self.shape_frechet, self.shape_weibull, self.shape_gumbel]:
            custom_pdf = GEV.pdf(self.x, self.loc, self.scale, shape)
            self.assertIsInstance(custom_pdf, np.ndarray)
            self.assertEqual(custom_pdf.shape[0], 1000)

    def test_cdf(self):
        for shape in [self.shape_frechet, self.shape_weibull, self.shape_gumbel]:
            custom_cdf = GEV.cdf(self.x, self.loc, self.scale, shape)
            self.assertIsInstance(custom_cdf, np.ndarray)
            self.assertEqual(custom_cdf.shape[0], 1000)

    def test_sf(self):
        for shape in [self.shape_frechet, self.shape_weibull, self.shape_gumbel]:
            custom_sf = GEV.sf(self.x, self.loc, self.scale, shape)
            self.assertIsInstance(custom_sf, np.ndarray)
            self.assertEqual(custom_sf.shape[0], 1000)

    def test_qf(self):
        for shape in [self.shape_frechet, self.shape_weibull, self.shape_gumbel]:
            custom_qf = GEV.qf(self.p, self.loc, self.scale, shape)
            self.assertIsInstance(custom_qf, np.ndarray)
            self.assertEqual(custom_qf.shape[0], 1000)

    def test_nll(self):
        for shape in [self.shape_frechet, self.shape_weibull, self.shape_gumbel]:
            nll = GEV.nll(self.x, self.loc, self.scale, shape)
            self.assertIsInstance(nll, float)

    def test_random(self):
        for shape in [self.shape_frechet, self.shape_weibull, self.shape_gumbel]:
            random_values = GEV.random(1000, self.loc, self.scale, shape)
            self.assertIsInstance(random_values, np.ndarray)
            self.assertEqual(random_values.shape[0], 1000)

    def test_mean(self):
        for shape in [self.shape_frechet, self.shape_weibull, self.shape_gumbel]:
            mean = GEV.mean(self.loc, self.scale, shape)
            self.assertIsInstance(mean, float)

    def test_median(self):
        for shape in [self.shape_frechet, self.shape_weibull, self.shape_gumbel]:
            median = GEV.median(self.loc, self.scale, shape)
            self.assertIsInstance(median, float)

    def test_variance(self):
        for shape in [self.shape_frechet, self.shape_weibull, self.shape_gumbel]:
            variance = GEV.variance(self.loc, self.scale, shape)
            self.assertIsInstance(variance, float)

    def test_std(self):
        for shape in [self.shape_frechet, self.shape_weibull, self.shape_gumbel]:
            std = GEV.std(self.loc, self.scale, shape)
            self.assertIsInstance(std, float)

    def test_stats(self):
        for shape in [self.shape_frechet, self.shape_weibull, self.shape_gumbel]:
            stats = GEV.stats(self.loc, self.scale, shape)
            self.assertIsInstance(stats, dict)
            self.assertIn("mean", stats)
            self.assertIn("median", stats)
            self.assertIn("variance", stats)
            self.assertIn("std", stats)

    def test_invalid_scale(self):
        with self.assertRaises(ValueError):
            GEV.pdf(self.x, self.loc, 0.0, self.shape_frechet)
        with self.assertRaises(ValueError):
            GEV.cdf(self.x, self.loc, 0.0, self.shape_frechet)
        with self.assertRaises(ValueError):
            GEV.sf(self.x, self.loc, 0.0, self.shape_frechet)
        with self.assertRaises(ValueError):
            GEV.qf(self.p, self.loc, 0.0, self.shape_frechet)
        with self.assertRaises(ValueError):
            GEV.random(1000, self.loc, 0.0, self.shape_frechet)
        with self.assertRaises(ValueError):
            GEV.mean(self.loc, 0.0, self.shape_frechet)
        with self.assertRaises(ValueError):
            GEV.median(self.loc, 0.0, self.shape_frechet)
        with self.assertRaises(ValueError):
            GEV.variance(self.loc, 0.0, self.shape_frechet)
        with self.assertRaises(ValueError):
            GEV.std(self.loc, 0.0, self.shape_frechet)
        with self.assertRaises(ValueError):
            GEV.stats(self.loc, 0.0, self.shape_frechet)

    def test_fit(self):
        # Generate data using specific parameters
        loc, scale, shape = 0.5, 1.5, 0.2
        data = GEV.random(1000, loc, scale, shape, random_state=42)

        # Fit the GEV distribution to the data
        fit_result = GEV.fit(data)

        # Check the fit result
        self.assertIsInstance(fit_result, FitResult)
        self.assertTrue(fit_result.success)
        self.assertEqual(len(fit_result.params), 3)  # loc, scale, shape
        self.assertGreater(fit_result.params[1], 0)  # Scale must be > 0
        self.assertIsInstance(fit_result.nll, float)

        # Verify that the fitted parameters are close to the original ones
        self.assertAlmostEqual(fit_result.params[0], loc, delta=0.2)
        self.assertAlmostEqual(fit_result.params[1], scale, delta=0.2)
        self.assertAlmostEqual(fit_result.params[2], shape, delta=0.1)


if __name__ == "__main__":
    unittest.main()
