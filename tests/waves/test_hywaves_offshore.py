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
    open_era5_waves,
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


if __name__ == "__main__":
    unittest.main()
