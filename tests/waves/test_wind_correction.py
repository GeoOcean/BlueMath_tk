"""Tests for bluemath_tk.waves.hywaves.wind_correction and its reconstruction hook."""

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from bluemath_tk.waves.hywaves.reconstruction import sum_partition_spectra_batched
from bluemath_tk.waves.hywaves.wind_correction import (
    WIND_GOAL,
    WIND_PARTITION,
    WindCorrection,
    correction_features,
    offshore_wind_sea,
    wind_sea_contribution,
)

SITES = ["a", "b"]
TIMES = pd.date_range("2020-01-01", periods=3, freq="h")


def _fitted(k_of: callable) -> WindCorrection:
    """A correction fitted on synthetic pairs with K = k_of(g Hs / U^2)."""

    rng = np.random.default_rng(0)
    n = 1500
    u = rng.uniform(5, 25, n)
    hs0 = rng.uniform(0.3, 6, n)
    hs1 = hs0 * k_of(9.81 * hs0 / u**2)
    return WindCorrection(max_iter=100).fit(
        hs0, hs1, u, rng.uniform(1e4, 1e5, n), 20.0, 25.0, 0.0
    )


def _cube(hs: float = 2.0, weight: float = 1.0) -> xr.Dataset:
    """Two goals, bulk partition; goal 2 is swell (weight 0)."""

    shape = (2, 1, len(TIMES), len(SITES))
    coords = {"goal": [1, 2], "partition": [-1], "time": TIMES, "sites": SITES}
    dims = ("goal", "partition", "time", "sites")
    forcing_dims = ("goal", "partition", "time")
    return xr.Dataset(
        {
            "hs": (dims, np.full(shape, hs)),
            "tp": (dims, np.full(shape, 7.0)),
            "dir": (dims, np.full(shape, 300.0)),
            "spr": (dims, np.full(shape, 25.0)),
            "ws_weight": (forcing_dims, np.stack([np.full((1, 3), weight), np.zeros((1, 3))])),
            "u10": (forcing_dims, np.stack([np.full((1, 3), 15.0), np.zeros((1, 3))])),
            "u10dir": (forcing_dims, np.full((2, 1, 3), 300.0)),
        },
        coords=coords,
    )


def _fetch() -> xr.Dataset:
    dirs = np.arange(0.0, 360.0, 5.0)
    return xr.Dataset(
        {
            "fetch_eff": (("sites", "dir"), np.full((2, len(dirs)), 5e4)),
            "mean_depth": (("sites", "dir"), np.full((2, len(dirs)), 25.0)),
            "site_depth": ("sites", [20.0, 20.0]),
        },
        coords={"sites": SITES, "dir": dirs},
    )


class TestWindCorrection(unittest.TestCase):
    """K model: features, fit/predict and its limits."""

    def test_features(self):
        """Dimensionless groups scale with U^2 / g; cos of the angle last."""

        x = correction_features(2.0, 10.0, 1e4, 20.0, 30.0, 60.0)
        np.testing.assert_allclose(
            x[0], [2 * 9.81 / 100, 1e4 * 9.81 / 100, 20 * 9.81 / 100, 30 * 9.81 / 100, 0.5]
        )

    def test_fit_predict_and_limits(self):
        """Learns K from g Hs / U^2; K = 1 without wind or wind sea."""

        corr = _fitted(lambda h: 1 + 2 * np.clip(0.2 - h, 0, None))
        k = corr.predict(np.array([0.5, 0.5, 0.5, 0.01]), np.array([20.0, 0.5, 20.0, 20.0]), 5e4, 20.0, 25.0, 0.0)
        self.assertAlmostEqual(float(k[0]), 1 + 2 * (0.2 - 9.81 * 0.5 / 400), delta=0.03)
        self.assertEqual(float(k[1]), 1.0)  # no wind
        self.assertEqual(float(k[3]), 1.0)  # no wind sea


class TestWindSeaContribution(unittest.TestCase):
    """The extra contribution carries (K^2 - 1) E_ws from the wind direction."""

    def test_energy_and_direction(self):
        """Only weighted contributions count; hs_add^2 = (K^2 - 1) sum w hs^2."""

        corr = _fitted(lambda h: np.full_like(h, 1.5))
        out = wind_sea_contribution(_cube(), corr, _fetch())
        self.assertEqual(out["goal"].values.tolist(), [WIND_GOAL])
        self.assertEqual(out["partition"].values.tolist(), [WIND_PARTITION])
        k = float(out["k"].mean())
        self.assertAlmostEqual(k, 1.5, delta=0.02)
        np.testing.assert_allclose(out["hs"], 2.0 * np.sqrt(k**2 - 1), rtol=1e-6)
        np.testing.assert_allclose(out["dir"], 300.0)
        np.testing.assert_allclose(out["tp"], 7.0)

    def test_no_wind_sea(self):
        """No weighted contribution: nothing to add."""

        corr = _fitted(lambda h: np.full_like(h, 1.5))
        self.assertIsNone(wind_sea_contribution(_cube(weight=0.0), corr, _fetch()))

    def test_offshore_wind_sea(self):
        """Steep broad seas are wind sea with wind from their direction; swell not."""

        forcing = pd.DataFrame(
            {"hs": [3.0, 1.0], "tp": [7.0, 14.0], "dir": [310.0, 250.0], "spr": [30.0, 15.0]}
        )
        ws = offshore_wind_sea(forcing)
        self.assertEqual(ws["ws_weight"].tolist(), [1.0, 0.0])
        self.assertGreater(ws["u10"].iloc[0], 10.0)
        self.assertEqual(ws["u10"].iloc[1], 0.0)

    def test_reconstruction_adds_the_energy(self):
        """Summed spectra gain (K^2 - 1) E_ws; wind_k is reported; NaN parts dropped."""

        cube = _cube()
        chunks = [cube.sel(goal=[g]) for g in (1, 2)]
        variables = {"hs": "hs"}
        base, _ = sum_partition_spectra_batched(chunks, SITES, 3, 2, variables)
        corr = _fitted(lambda h: np.full_like(h, 1.5))
        wind, _ = sum_partition_spectra_batched(
            chunks, SITES, 3, 2, variables,
            wind_correction={"correction": corr, "fetch": _fetch()},
        )
        k = float(wind["wind_k"].mean())
        expected = np.sqrt(base["hs"] ** 2 + (k**2 - 1) * 2.0**2)
        np.testing.assert_allclose(wind["hs"], expected, rtol=0.02)
        self.assertNotIn("wind_k", base)


if __name__ == "__main__":
    unittest.main()
