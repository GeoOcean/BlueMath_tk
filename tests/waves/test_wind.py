"""Tests for bluemath_tk.waves.wind."""

import unittest

import numpy as np

from bluemath_tk.waves.wind import (
    equivalent_fetch,
    fully_developed_wind,
    growth_hs,
    growth_tp,
    infer_wind,
    wind_sea_weight,
)


class TestWindSeaWeight(unittest.TestCase):
    """Swell-to-wind-sea weight from steepness and spreading."""

    def test_ramps(self):
        """Swell -> 0, young wind sea -> 1, halfway on the steepness ramp -> 0.5."""

        # Hs / L0 with L0 = g Tp^2 / 2pi
        l0 = 9.81 * 8.0**2 / (2 * np.pi)
        hs = np.array([0.015, 0.025, 0.04]) * l0
        np.testing.assert_allclose(wind_sea_weight(hs, 8.0), [0.0, 0.5, 1.0])
        np.testing.assert_allclose(
            wind_sea_weight(hs, 8.0, spr=[30.0, 22.5, 10.0]), [0.0, 0.25, 0.0]
        )

    def test_calm_and_missing(self):
        """No energy gives 0; NaN stays NaN."""

        w = wind_sea_weight([0.0, np.nan], [5.0, 5.0], spr=[30.0, 30.0])
        self.assertEqual(w[0], 0.0)
        self.assertTrue(np.isnan(w[1]))


class TestInferWind(unittest.TestCase):
    """Wind from the wave direction at the fully developed speed."""

    def test_fully_developed_speed(self):
        """U = max(sqrt(g Hs / 0.24), g Tp / 7.69)."""

        np.testing.assert_allclose(
            fully_developed_wind([4.0, 1.0], [9.0, 12.0]),
            [np.sqrt(9.81 * 4.0 / 0.24), 9.81 * 12.0 / 7.69],
        )

    def test_wind_sea_and_swell(self):
        """A steep broad sea gets wind from its direction; swell gets none."""

        u10, u10dir = infer_wind([3.0, 1.5], [7.0, 14.0], [370.0, 300.0], [30.0, 30.0])
        self.assertAlmostEqual(float(u10[0]), float(fully_developed_wind(3.0, 7.0)))
        self.assertEqual(float(u10[1]), 0.0)
        np.testing.assert_allclose(u10dir, [10.0, 300.0])


class TestGrowthCurves(unittest.TestCase):
    """Kahma & Calkoen growth with Pierson-Moskowitz and depth limits."""

    def test_deep_water_growth(self):
        """Hs and Tp follow the power laws and saturate when fully developed."""

        u, x = 10.0, 50e3
        xn = 9.81 * x / u**2
        self.assertAlmostEqual(float(growth_hs(u, x)), 0.00288 * xn**0.45 * u**2 / 9.81)
        self.assertAlmostEqual(float(growth_tp(u, x)), 0.459 * xn**0.27 * u / 9.81)
        self.assertAlmostEqual(float(growth_hs(u, 1e9)), 0.24 * u**2 / 9.81)
        self.assertAlmostEqual(float(growth_tp(u, 1e9)), 7.69 * u / 9.81)

    def test_depth_limit(self):
        """Shallow water caps the growth (Breugem & Holthuijsen)."""

        u, d = 20.0, 10.0
        cap = 0.13 * (9.81 * d / u**2) ** 0.65 * u**2 / 9.81
        self.assertAlmostEqual(float(growth_hs(u, 1e9, depth=d)), cap)
        self.assertLess(float(growth_hs(u, 1e9, depth=d)), float(growth_hs(u, 1e9)))

    def test_equivalent_fetch_round_trip(self):
        """equivalent_fetch inverts growth_hs; inf once fully developed."""

        u, x = 15.0, 80e3
        self.assertAlmostEqual(float(equivalent_fetch(growth_hs(u, x), u)), x, places=3)
        self.assertTrue(np.isinf(equivalent_fetch(0.24 * u**2 / 9.81, u)))


if __name__ == "__main__":
    unittest.main()
