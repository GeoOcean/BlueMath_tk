import unittest

import numpy as np

from bluemath_tk.core.metrics import (
    AVAILABLE_METRICS,
    DEFAULT_METRICS,
    bias,
    circular_bias,
    circular_difference,
    circular_mean,
    circular_r2,
    circular_rmse,
    compute_metrics,
    hh,
    linear_fit,
    mae,
    mse,
    paired_finite,
    pearson_r,
    percentile_bias,
    r2,
    rmse,
    rrmse,
    si,
    willmott_d,
)


class TestConvention(unittest.TestCase):
    """Argument order and sign convention: metric(reference, estimate)."""

    def setUp(self):
        """Build paired reference/estimate samples."""

        rng = np.random.default_rng(0)
        self.ref = rng.gamma(2.0, 1.0, 500)
        # estimate: smoothed (lower variance) and 10% too high
        self.est = 1.1 * (0.5 * self.ref + 0.5 * self.ref.mean())

    def test_positive_bias_is_overestimation(self):
        """Positive bias is overestimation."""

        self.assertAlmostEqual(bias([1.0, 2.0], [1.5, 2.5]), 0.5)
        self.assertAlmostEqual(bias([1.5, 2.5], [1.0, 2.0]), -0.5)

    def test_r2_normalised_by_reference_variance(self):
        """R2 normalised by reference variance."""

        ref, est = self.ref, self.est
        expected = 1 - np.sum((est - ref) ** 2) / np.sum((ref - ref.mean()) ** 2)
        self.assertAlmostEqual(r2(ref, est), expected)
        # asymmetric: swapping arguments changes the score
        self.assertNotAlmostEqual(r2(ref, est), r2(est, ref))

    def test_symmetric_metrics(self):
        """Symmetric metrics."""

        for func in (mae, mse, rmse, pearson_r):
            self.assertAlmostEqual(func(self.ref, self.est), func(self.est, self.ref))

    def test_keyword_names(self):
        """Keyword names."""

        self.assertAlmostEqual(bias(reference=[1.0, 2.0], estimate=[2.0, 3.0]), 1.0)


class TestLinearMetrics(unittest.TestCase):
    """Tests for linear metrics."""

    def test_perfect_estimate(self):
        """Perfect estimate."""

        ref = np.array([1.0, 2.0, 3.0, 4.0])
        m = compute_metrics(ref, ref, AVAILABLE_METRICS)
        self.assertEqual(m["n"], 4)
        for name in ("bias", "mae", "mse", "rmse", "rrmse", "si", "hh"):
            self.assertAlmostEqual(m[name], 0.0)
        for name in ("r", "r2", "willmott_d", "slope"):
            self.assertAlmostEqual(m[name], 1.0)
        self.assertAlmostEqual(m["intercept"], 0.0)

    def test_known_values(self):
        """Known values."""

        ref = np.array([1.0, 2.0, 3.0, 4.0])
        est = np.array([2.0, 2.0, 4.0, 4.0])
        self.assertAlmostEqual(mae(ref, est), 0.5)
        self.assertAlmostEqual(mse(ref, est), 0.5)
        self.assertAlmostEqual(rmse(ref, est), np.sqrt(0.5))
        self.assertAlmostEqual(rrmse(ref, est), np.sqrt(0.5) / 2.5)
        self.assertAlmostEqual(
            rrmse(ref, est, normalization="rms"), np.sqrt(0.5) / np.sqrt(7.5)
        )
        self.assertAlmostEqual(hh(ref, est), np.sqrt(2.0 / 34.0))

    def test_rrmse_bad_normalization(self):
        """Rrmse bad normalization."""

        with self.assertRaises(ValueError):
            rrmse([1.0], [1.0], normalization="std")

    def test_si_mentaschi_is_bias_free(self):
        """Si mentaschi is bias free."""

        ref = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(si(ref, ref + 0.7), 0.0)

    def test_linear_fit(self):
        """Linear fit."""

        ref = np.array([0.0, 1.0, 2.0, 3.0])
        slope, intercept = linear_fit(ref, 2.0 * ref + 1.0)
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 1.0)

    def test_percentile_bias(self):
        """Percentile bias."""

        ref = np.arange(101.0)
        self.assertAlmostEqual(percentile_bias(ref, ref + 2.0, q=99), 2.0)

    def test_willmott_bounds(self):
        """Willmott bounds."""

        rng = np.random.default_rng(1)
        d = willmott_d(rng.normal(size=200), rng.normal(size=200))
        self.assertTrue(0.0 <= d <= 1.0)

    def test_degenerate_returns_nan(self):
        """Degenerate returns nan."""

        const = np.ones(5)
        self.assertTrue(np.isnan(r2(const, const + 1)))
        self.assertTrue(np.isnan(pearson_r(const, np.arange(5.0))))
        self.assertTrue(np.isnan(rrmse(np.zeros(3), np.ones(3))))


class TestPairing(unittest.TestCase):
    """Tests for pairing."""

    def test_drops_non_finite_pairs(self):
        """Drops non finite pairs."""

        ref = np.array([1.0, np.nan, 3.0, 4.0, np.inf])
        est = np.array([1.0, 2.0, np.nan, 5.0, 1.0])
        r, e = paired_finite(ref, est)
        np.testing.assert_array_equal(r, [1.0, 4.0])
        np.testing.assert_array_equal(e, [1.0, 5.0])
        self.assertAlmostEqual(bias(ref, est), 0.5)
        self.assertEqual(compute_metrics(ref, est)["n"], 2)

    def test_flattens_2d(self):
        """Flattens 2d."""

        ref = np.arange(6.0).reshape(2, 3)
        self.assertEqual(paired_finite(ref, ref)[0].shape, (6,))

    def test_length_mismatch(self):
        """Length mismatch."""

        with self.assertRaises(ValueError):
            bias([1.0, 2.0], [1.0])

    def test_empty_returns_nan(self):
        """Empty returns nan."""

        m = compute_metrics([np.nan], [1.0])
        self.assertEqual(m["n"], 0)
        self.assertTrue(all(np.isnan(m[k]) for k in DEFAULT_METRICS))


class TestCircularMetrics(unittest.TestCase):
    """Tests for circular metrics."""

    def test_difference_wraps(self):
        """Difference wraps."""

        np.testing.assert_allclose(
            circular_difference([350.0, 10.0], [10.0, 350.0]), [20.0, -20.0]
        )

    def test_mean_wraps(self):
        """Mean wraps."""

        self.assertAlmostEqual(circular_mean([350.0, 10.0]) % 360, 0.0, places=6)

    def test_constant_offset(self):
        """Constant offset."""

        ref = np.linspace(0.0, 360.0, 100, endpoint=False)
        est = (ref + 5.0) % 360.0
        self.assertAlmostEqual(circular_bias(ref, est), 5.0)
        self.assertAlmostEqual(circular_rmse(ref, est), 5.0)
        self.assertGreater(circular_r2(ref, est), 0.9)

    def test_compute_metrics_circular(self):
        """Compute metrics circular."""

        ref = np.array([355.0, 5.0, 15.0])
        est = np.array([5.0, 15.0, 25.0])
        m = compute_metrics(ref, est, ["bias", "rmse", "rrmse", "r"], circular=True)
        self.assertAlmostEqual(m["bias"], 10.0)
        self.assertAlmostEqual(m["rmse"], 10.0)
        self.assertTrue(np.isnan(m["rrmse"]))
        self.assertTrue(np.isnan(m["r"]))

    def test_unknown_metric(self):
        """Unknown metric."""

        with self.assertRaises(ValueError):
            compute_metrics([1.0], [1.0], ["nse"])


if __name__ == "__main__":
    unittest.main()
