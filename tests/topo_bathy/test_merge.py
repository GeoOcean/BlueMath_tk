from __future__ import annotations

import unittest

import numpy as np
import xarray as xr

from bluemath_tk.topo_bathy.merge import RasterSource, feather_weight, merge_rasters, target_grid


def _raster(value: float, west: float, east: float, south: float, north: float, step: float) -> xr.DataArray:
    x = np.arange(west + step / 2, east, step)
    y = np.arange(north - step / 2, south, -step)
    da = xr.DataArray(np.full((y.size, x.size), value, dtype="float32"), coords={"y": y, "x": x}, dims=("y", "x"))
    return da.rio.write_crs("EPSG:4326")


class TestMerge(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = target_grid((0.0, 0.0, 1.0, 1.0), 0.01)
        self.base = RasterSource("base", _raster(-100.0, -0.5, 1.5, -0.5, 1.5, 0.05))

    def test_base_only(self) -> None:
        ds = merge_rasters([self.base], self.grid)
        np.testing.assert_allclose(ds.elevation.values, -100.0)
        self.assertTrue((ds.source.values == 0).all())

    def test_priority_offset_and_feather(self) -> None:
        local = RasterSource("local", _raster(-50.0, 0.2, 0.8, 0.2, 0.8, 0.002), offset=1.0, feather_m=5_000.0)
        ds = merge_rasters([self.base, local], self.grid)
        centre = ds.elevation.sel(x=0.5, y=0.5, method="nearest")
        self.assertAlmostEqual(float(centre), -49.0, places=4)
        outside = ds.elevation.sel(x=0.1, y=0.5, method="nearest")
        self.assertAlmostEqual(float(outside), -100.0, places=4)
        row = ds.elevation.sel(y=0.5, method="nearest").sel(x=slice(0.2, 0.3)).values
        self.assertTrue(np.all(np.diff(row) >= -1e-4))
        self.assertTrue(-100.0 < row[0] < -49.0)
        self.assertEqual(int(ds.source.sel(x=0.5, y=0.5, method="nearest")), 1)

    def test_nan_in_source_keeps_lower(self) -> None:
        data = _raster(-50.0, 0.0, 1.0, 0.0, 1.0, 0.01)
        data[40:60, 40:60] = np.nan
        ds = merge_rasters([self.base, RasterSource("local", data, feather_m=0.0)], self.grid)
        self.assertAlmostEqual(float(ds.elevation.sel(x=0.5, y=0.5, method="nearest")), -100.0, places=4)

    def test_feather_weight_edges(self) -> None:
        valid = np.ones((11, 11), dtype=bool)
        w = feather_weight(valid, 3.0)
        self.assertAlmostEqual(float(w[0, 5]), 1 / 6, places=5)
        self.assertEqual(float(w[5, 5]), 1.0)


if __name__ == "__main__":
    unittest.main()
