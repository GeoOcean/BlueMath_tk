import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

from bluemath_tk.waves.hywaves.offshore import (
    ERA5_TO_HYWAVES,
    EDH_KEY_VARIABLE,
    era5_at_goals,
    jonswap_tm10_ratio,
    open_era5_waves,
    partition_peak_periods,
    peak_period_samples,
)


def _era5_like() -> xr.Dataset:
    """Tiny ERA5-like grid: every variable = 1, widths = 0.5 rad."""

    time = pd.date_range("2013-01-01", periods=6, freq="h")
    lat = np.arange(55.0, 50.5, -0.5)
    lon = np.arange(0.0, 8.0, 0.5)
    shape = (time.size, lat.size, lon.size)
    data = {
        name: (
            ("valid_time", "latitude", "longitude"),
            np.full(shape, 0.5 if name.startswith(("wdw", "dw")) else 1.0),
        )
        for name in set(ERA5_TO_HYWAVES.values())
    }
    data["swh"] = (data["swh"][0], np.broadcast_to(lon, shape).copy())  # hs = longitude
    return xr.Dataset(
        data, coords={"valid_time": time, "latitude": lat, "longitude": lon}
    )


class TestEra5AtGoals(unittest.TestCase):
    """ERA5 waves at goal nodes in HyWaves names."""

    def test_names_units_and_nodes(self):
        """All partitions, widths in degrees, nearest grid point per goal."""

        out = era5_at_goals(
            _era5_like(),
            [2.4, 7.1],
            [51.6, 54.0],
            [1, 2],
            time=slice("2013-01-01 01", None),
        )
        self.assertEqual(set(out.data_vars), set(ERA5_TO_HYWAVES))
        for i in range(4):
            self.assertIn(f"phs{i}", out)
        self.assertEqual(out.sizes["time"], 5)
        self.assertEqual(list(out.goal.values), [1, 2])
        np.testing.assert_allclose(out["hs"].isel(time=0), [2.5, 7.0])
        np.testing.assert_allclose(out["spr"], np.rad2deg(0.5))
        np.testing.assert_allclose(out["pspr2"], np.rad2deg(0.5))
        np.testing.assert_allclose(out["latitude"], [51.5, 54.0])
        self.assertIn("mean wave period", out["ptp1"].attrs["long_name"].lower())

    def test_key_required(self):
        """Without a key (argument or environment) opening fails early."""

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertNotIn(EDH_KEY_VARIABLE, os.environ)
            with self.assertRaises(ValueError):
                open_era5_waves()


def _partitions(phs: list[list[float]]) -> xr.Dataset:
    """Two goals x hours; tp = 11, ptp{i} = 10 (Tm-1,0); hs from the partitions."""

    phs = np.asarray(phs, dtype=float)  # (partition, time)
    ds = xr.Dataset({f"phs{i}": ("time", h) for i, h in enumerate(phs)})
    ds = ds.expand_dims(goal=[1, 2])
    ds["hs"] = np.sqrt(sum(ds[f"phs{i}"] ** 2 for i in range(len(phs))))
    ds["tp"] = xr.full_like(ds["hs"], 11.0)
    for i in range(len(phs)):
        ds[f"ptp{i}"] = xr.full_like(ds["hs"], 10.0)
    return ds


class TestPeakPeriods(unittest.TestCase):
    """Partition mean periods (Tm-1,0) to peak periods."""

    def test_jonswap_ratio(self):
        """Pierson-Moskowitz Tm-1,0 / Tp = 0.857; narrower spectra are closer to 1."""

        self.assertAlmostEqual(jonswap_tm10_ratio(1.0), 0.857, places=3)
        self.assertGreater(jonswap_tm10_ratio(3.3), jonswap_tm10_ratio(1.0))

    def test_samples_only_where_one_partition_dominates(self):
        """Hour 0: sea dominates; hour 1: swell 1; hour 2: mixed (no sample)."""

        ds = _partitions([[3.0, 0.1, 1.0], [0.1, 2.0, 1.0]])
        samples = peak_period_samples(ds, min_fraction=0.9)
        self.assertEqual(samples["partition"].value_counts().to_dict(), {"sea": 2, "swell": 2})
        np.testing.assert_allclose(samples["ratio"], 1.1)

    def test_conversion(self):
        """ptp0 x sea ratio, swells x swell ratio, recorded in the attributes."""

        ds = _partitions([[1.0], [1.0], [1.0]])
        out = partition_peak_periods(ds, {"sea": 1.2, "swell": 1.1})
        np.testing.assert_allclose(out["ptp0"], 12.0)
        np.testing.assert_allclose(out["ptp2"], 11.0)
        np.testing.assert_allclose(ds["ptp0"], 10.0)
        self.assertEqual(out["ptp1"].attrs["tp_tm10_ratio"], 1.1)
        self.assertIn("peak", out["ptp1"].attrs["long_name"].lower())


if __name__ == "__main__":
    unittest.main()
