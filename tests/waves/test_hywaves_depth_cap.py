import unittest

import numpy as np
import xarray as xr

from bluemath_tk.waves.hywaves.depth_cap import apply_depth_cap, cap_factor, depth_ceiling

VARIABLES = {"hs": "hs", "tp": "tp", "dir": "dm"}


def _bulk(hs):
    hs = np.asarray(hs, dtype="float32")
    return xr.Dataset(
        {
            "hs": (("time", "sites"), hs),
            "tp": (("time", "sites"), np.full(hs.shape, 9.0, "float32")),
            "dir": (("time", "sites"), np.full(hs.shape, 300.0, "float32")),
        },
        coords={"time": np.arange(hs.shape[0]), "sites": ["a", "b"][: hs.shape[1]]},
    )


class TestDepthCeiling(unittest.TestCase):
    def test_highest_ratio_per_site(self):
        hs = [[1.0, 2.0], [2.0, 1.0]]
        depth = np.array([[5.0, 10.0], [4.0, 10.0]])
        np.testing.assert_allclose(depth_ceiling(hs, depth), [0.5, 0.2])

    def test_shallow_cases_ignored_and_empty_site_nan(self):
        hs = [[0.4, 1.0], [3.0, 1.0]]
        depth = np.array([[0.4, 0.3], [6.0, 0.5]])
        np.testing.assert_allclose(depth_ceiling(hs, depth), [0.5, np.nan])

    def test_depth_broadcasts(self):
        ceiling = depth_ceiling([[1.0, 1.0], [2.0, 1.0]], np.array([4.0, 2.0]) + np.array([[0.0], [0.0]]))
        np.testing.assert_allclose(ceiling, [0.5, 0.5])


class TestCap(unittest.TestCase):
    def test_factor(self):
        hs = xr.DataArray([2.0, 1.0, 0.0, np.nan, 3.0])
        cap = xr.DataArray([1.0, 2.0, 1.0, 1.0, np.nan])
        np.testing.assert_allclose(cap_factor(hs, cap), [0.5, 1.0, 1.0, 1.0, 1.0])

    def test_heights_capped_shape_kept(self):
        bulk = _bulk([[2.0, 1.0], [0.5, np.nan]])
        cap = xr.DataArray([1.0, 1.5], dims="sites", coords={"sites": ["a", "b"]})
        spectra = xr.Dataset({"efth": (("time", "sites", "freq"), np.ones((2, 2, 3), "float32"))})
        out, spec = apply_depth_cap(bulk, cap, VARIABLES, spectra)
        np.testing.assert_allclose(out["hs"], [[1.0, 1.0], [0.5, np.nan]])
        xr.testing.assert_equal(out[["tp", "dir"]], bulk[["tp", "dir"]])
        self.assertEqual(out["hs"].dtype, np.float32)
        np.testing.assert_allclose(spec["efth"][0, 0], 0.25)  # energy x (1 / 2)^2
        np.testing.assert_allclose(spec["efth"][0, 1], 1.0)
        np.testing.assert_allclose(bulk["hs"][0, 0], 2.0)  # input untouched

    def test_needs_hs(self):
        with self.assertRaises(ValueError):
            apply_depth_cap(_bulk([[1.0, 1.0]]), xr.DataArray(1.0), {"tp": "tp"})


if __name__ == "__main__":
    unittest.main()
