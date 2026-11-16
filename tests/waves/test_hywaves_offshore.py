import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

from bluemath_tk.waves.hywaves.offshore import (
    ERA5_TO_HYWAVES,
    EDH_KEY_VARIABLE,
    clean_peak_periods,
    codec_at_goals,
    era5_at_goals,
    jonswap_tm10_ratio,
    open_era5_waves,
    partition_peak_periods,
    peak_period_samples,
    scale_partitions_to_bulk,
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



class TestCleanPeakPeriods(unittest.TestCase):
    def test_bulk_replaced_by_largest_partition(self):
        time = pd.date_range("2000-01-01", periods=3, freq="h")
        dims = ("time", "goal")
        ds = xr.Dataset(
            {
                "hs": (dims, [[1.0], [1.0], [1.0]]),
                "tp": (dims, [[10.0], [250.0], [np.nan]]),
                "phs0": (dims, [[0.5], [0.3], [0.9]]),
                "ptp0": (dims, [[5.0], [6.0], [7.0]]),
                "phs1": (dims, [[0.8], [0.9], [0.1]]),
                "ptp1": (dims, [[12.0], [14.0], [30.0]]),
            },
            coords={"time": time, "goal": [1]},
        )
        out = clean_peak_periods(ds, max_tp=25.0)
        np.testing.assert_allclose(out["tp"].values.ravel(), [10.0, 14.0, 7.0])
        self.assertTrue(np.isnan(float(out["ptp1"][2, 0])))
        self.assertEqual(out.attrs["tp_replaced"], 2)
        self.assertEqual(out.attrs["ptp_removed"], 1)


class TestScalePartitionsToBulk(unittest.TestCase):
    def test_partition_energy_matches_bulk(self):
        dims = ("time", "goal")
        ds = xr.Dataset(
            {
                "hs": (dims, [[2.0], [1.0], [1.0]]),
                "phs0": (dims, [[1.0], [0.0], [np.nan]]),
                "phs1": (dims, [[1.5], [0.0], [0.6]]),
                "pdir1": (dims, [[200.0], [10.0], [30.0]]),
            },
            coords={"time": pd.date_range("2000", periods=3, freq="h"), "goal": [1]},
        )
        out = scale_partitions_to_bulk(ds)
        energy = out["phs0"].fillna(0.0) ** 2 + out["phs1"] ** 2
        np.testing.assert_allclose(energy.values[[0, 2], 0], [4.0, 1.0])
        self.assertEqual(float(out["phs1"][1, 0]), 0.0)  # no partition energy: left as is
        self.assertAlmostEqual(float(out["phs0"][0, 0] / out["phs1"][0, 0]), 1.0 / 1.5)
        np.testing.assert_array_equal(out["pdir1"].values, ds["pdir1"].values)
        self.assertAlmostEqual(out.attrs["partition_energy_share"], round((3.25 + 0.36) / 5.0, 4))


class TestCodecAtGoals(unittest.TestCase):
    def test_monthly_files_concatenated_per_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for month in ("01", "02"):
                time = pd.date_range(f"2010-{month}-01", periods=4, freq="h")
                level = np.arange(4)[:, None] + 10.0 * np.arange(5)[None, :]
                ds = xr.Dataset(
                    {"waterlevel": (("time", "stations"), level)},
                    coords={"time": time, "station_x_coordinate": ("stations", np.arange(5.0))},
                )
                paths.append(os.path.join(tmp, f"codec_{month}.nc"))
                ds.to_netcdf(paths[-1])
            out = codec_at_goals(paths[::-1], np.array([3, 1]), np.array([1, 2]))
        self.assertEqual(out["waterlevel"].dims, ("time", "goal"))
        self.assertEqual(out.sizes["time"], 8)
        self.assertTrue(out.indexes["time"].is_monotonic_increasing)
        np.testing.assert_allclose(out["waterlevel"].sel(goal=1).values[:4], [30, 31, 32, 33])
        np.testing.assert_allclose(out["station_x_coordinate"].values, [3.0, 1.0])

if __name__ == "__main__":
    unittest.main()
