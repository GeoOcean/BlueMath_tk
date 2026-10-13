import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from bluemath_tk.waves.altimetry import (
    altimeter_near,
    altimeter_passes,
    collocate,
    haversine_km,
    open_altimeter,
)


def _write_store(path: Path) -> None:
    """Two 1-degree cells (lat 53, lon 3 and 4), obs sorted by cell then time."""

    t0 = np.datetime64("2010-01-01T10:00:00")
    # cell (53, 3): one pass of 5 points near (3.9, 53.5), one far point
    lon_a = np.array([3.90, 3.91, 3.92, 3.93, 3.94, 3.05])
    lat_a = np.array([53.50, 53.51, 53.52, 53.53, 53.54, 53.05])
    time_a = t0 + np.array([0, 1, 2, 3, 4, 3600], dtype="timedelta64[s]")
    # cell (53, 4): a pass 2 h later (one bad flag, one shallow, one fill depth)
    lon_b = np.array([4.05, 4.06, 4.07, 4.08, 4.09])
    lat_b = np.array([53.50, 53.50, 53.50, 53.50, 53.50])
    time_b = t0 + np.array([7200, 7201, 7202, 7203, 7204], dtype="timedelta64[s]")
    obs = {
        "longitude": np.r_[lon_a, lon_b].astype("float32"),
        "latitude": np.r_[lat_a, lat_b].astype("float32"),
        "time": np.r_[time_a, time_b].astype("datetime64[ns]"),
        "hs": np.r_[[2.0, 2.1, 2.2, 1.9, 2.0, 5.0], [3.0, 3.1, 9.0, 2.9, 3.0]].astype(
            "float32"
        ),
        "flag": np.r_[[1, 1, 2, 1, 1, 1], [1, 1, 3, 1, 1]].astype("int8"),
        "depth": np.r_[[-30] * 6, [-30, -30, -30, -5, -32000]].astype("float32"),
        "dist2coast": np.full(11, 40.0, dtype="float32"),
        "satellite": np.r_[[7] * 6, [11] * 5].astype("int8"),
    }
    lat_cell = np.arange(-90, 90, dtype="int16")
    lon_cell = np.arange(0, 360, dtype="int16")
    start = np.zeros((180, 360), dtype="int64")
    count = np.zeros((180, 360), dtype="int32")
    start[53 + 90, 3], count[53 + 90, 3] = 0, 6
    start[53 + 90, 4], count[53 + 90, 4] = 6, 5
    ds = xr.Dataset(
        {
            k: ("obs", v)
            for k, v in obs.items()
            if k not in ("time", "latitude", "longitude")
        },
        coords={
            "time": ("obs", obs["time"]),
            "latitude": ("obs", obs["latitude"]),
            "longitude": ("obs", obs["longitude"]),
            "lat_cell": lat_cell,
            "lon_cell": lon_cell,
        },
    )
    ds["cell_start"] = (("lat_cell", "lon_cell"), start)
    ds["cell_count"] = (("lat_cell", "lon_cell"), count)
    ds.to_zarr(path)


class TestAltimetry(unittest.TestCase):
    """Reading, pass averaging and collocation of altimeter data."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = Path(self.tmp) / "alt.zarr"
        _write_store(self.path)
        self.ds = open_altimeter(str(self.path))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_near_radius_flags_depth(self):
        """Radius, quality flags and depth (fill values as missing) filter obs."""

        obs = altimeter_near(self.ds, 4.0, 53.5, radius_km=20, flags=None)
        self.assertEqual(len(obs), 10)  # the far point (3.05, 53.05) is out
        self.assertTrue((obs["distance_km"] <= 20).all())
        self.assertEqual(int(obs["depth"].isna().sum()), 1)  # -32000 fill value
        obs = altimeter_near(self.ds, 4.0, 53.5, radius_km=20, min_depth=10)
        self.assertEqual(len(obs), 7)  # flag 3, the 5 m and the fill-depth points out
        obs = altimeter_near(self.ds, 4.0, 53.5, radius_km=20, time=slice("2011", None))
        self.assertTrue(obs.empty)

    def test_passes_and_collocation(self):
        """One median per pass; hindcast at the nearest step within tolerance."""

        obs = altimeter_near(self.ds, 4.0, 53.5, radius_km=20, min_depth=10)
        passes = altimeter_passes(obs, min_points=2)
        self.assertEqual(len(passes), 2)
        self.assertAlmostEqual(float(passes["hs"].iloc[0]), 2.0, places=5)
        self.assertEqual(list(passes["n"]), [5, 2])

        time = pd.date_range("2010-01-01", periods=48, freq="h")
        hindcast = xr.Dataset({"hs": ("time", np.arange(48.0))}, coords={"time": time})
        pairs = collocate(hindcast, passes, max_time_diff="30min")
        self.assertEqual(list(pairs["hs"]), [10.0, 12.0])
        self.assertIn("sat_hs", pairs)
        none = collocate(hindcast.isel(time=slice(0, 5)), passes, max_time_diff="30min")
        self.assertTrue(none.empty)

    def test_haversine(self):
        """One degree of latitude is ~111 km."""

        self.assertAlmostEqual(
            float(haversine_km(0.0, 1.0, 0.0, 0.0)), 111.19, places=1
        )


if __name__ == "__main__":
    unittest.main()
