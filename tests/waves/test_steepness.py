"""Tests for bluemath_tk.waves.steepness."""

import unittest

import numpy as np
import pandas as pd

from bluemath_tk.waves.steepness import (
    deep_water_wavelength,
    filter_by_steepness,
    wave_steepness,
)


class TestSteepness(unittest.TestCase):
    """Tests for wave steepness and the plausible sea-state filter."""

    def test_wavelength_and_steepness(self):
        """L0 = g T^2 / 2pi; steepness = Hs / L0."""

        self.assertAlmostEqual(float(deep_water_wavelength(10.0)), 156.13, places=2)
        np.testing.assert_allclose(
            wave_steepness([1.0, 2.0], [10.0, 10.0]),
            [1 / 156.13, 2 / 156.13],
            rtol=1e-4,
        )

    def test_filter_keeps_plausible_rows(self):
        """Steep rows are dropped and the original index is kept."""

        cases = pd.DataFrame({"hs": [1.0, 6.0, 0.5], "tp": [10.0, 4.0, 2.0]})
        kept = filter_by_steepness(cases, max_steepness=0.06)
        self.assertEqual(kept.index.tolist(), [0])
        kept = filter_by_steepness(cases, max_steepness=0.09)
        self.assertEqual(kept.index.tolist(), [0, 2])


if __name__ == "__main__":
    unittest.main()
