import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from bluemath_tk.core.operations import in_nautical_sector
from bluemath_tk.core.plotting.base_plotting import (
    DefaultStaticPlotting,
    decimate_raster,
)
from bluemath_tk.core.plotting.scatter import (
    density_scatter,
    scatter_grid_shape,
    validation_scatter,
    validation_scatter_grid,
)
from bluemath_tk.core.plotting.timeseries import plot_timeseries_comparison


class TestDensityScatter(unittest.TestCase):
    """Tests for density scatter."""

    def test_kde_and_histogram_paths(self):
        """Kde and histogram paths."""

        rng = np.random.default_rng(0)
        for n in (200, 20000):
            x = rng.normal(size=n)
            y = x + rng.normal(scale=0.1, size=n)
            xs, ys, z = density_scatter(x, y, max_kde_points=5000)
            self.assertEqual(xs.size, n)
            self.assertTrue(np.all(np.diff(z) >= 0))

    def test_constant_values_fall_back(self):
        """Constant values fall back."""

        xs, _, z = density_scatter(np.ones(10), np.ones(10))
        self.assertEqual(z.size, 10)


class TestValidationScatter(unittest.TestCase):
    """Tests for validation scatter."""

    def tearDown(self):
        """Close all figures."""

        plt.close("all")

    def test_reference_on_x_estimate_on_y(self):
        """Reference on x estimate on y."""

        ref = np.array([1.0, 2.0, 3.0, 4.0])
        est = np.array([10.0, 20.0, 30.0, 40.0])
        fig, ax = plt.subplots()
        scores = validation_scatter(
            ax, ref, est, "Buoy", "Model", units="m", metrics=["bias"], qq=False
        )
        xs, ys = ax.collections[0].get_offsets().T
        self.assertLessEqual(xs.max(), 4.0)
        self.assertGreaterEqual(ys.min(), 10.0)
        self.assertEqual(ax.get_xlabel(), "Buoy [m]")
        self.assertEqual(ax.get_ylabel(), "Model [m]")
        self.assertAlmostEqual(scores["bias"], 22.5)

    def test_metrics_use_all_points_when_subsampled(self):
        """Metrics use all points when subsampled."""

        rng = np.random.default_rng(0)
        ref = rng.gamma(2.0, 1.0, 5000)
        fig, ax = plt.subplots()
        scores = validation_scatter(ax, ref, ref + 0.2, max_points=500)
        self.assertEqual(scores["n"], 5000)
        self.assertEqual(len(ax.collections[0].get_offsets()), 500)

    def test_circular_and_empty(self):
        """Circular and empty."""

        fig, (ax1, ax2) = plt.subplots(1, 2)
        scores = validation_scatter(ax1, [350.0, 10.0], [10.0, 20.0], circular=True)
        self.assertEqual(ax1.get_xlim(), (0.0, 360.0))
        self.assertAlmostEqual(scores["bias"], 15.0)
        empty = validation_scatter(ax2, [np.nan], [1.0])
        self.assertEqual(empty["n"], 0)

    def test_grid(self):
        """Grid."""

        self.assertEqual(scatter_grid_shape(3), (2, 2))
        rng = np.random.default_rng(0)
        ref = pd.DataFrame(
            {
                "hs": rng.gamma(2, 1, 50),
                "tp": rng.gamma(8, 1, 50),
                "dir": rng.uniform(0, 360, 50),
            }
        )
        est = ref + 0.1
        fig, axes, scores = validation_scatter_grid(ref, est, units={"hs": "m"})
        self.assertEqual(set(scores), {"hs", "tp", "dir"})
        self.assertEqual(axes[2].get_xlim(), (0.0, 360.0))
        self.assertFalse(axes[-1].get_visible())


class TestTimeseries(unittest.TestCase):
    """Tests for timeseries."""

    def test_estimates_aligned_to_reference(self):
        """Estimates aligned to reference."""

        t = pd.date_range("2020-01-01", periods=10, freq="h")
        ref = pd.Series(np.arange(10.0), index=t)
        est = pd.Series(
            np.arange(20.0), index=pd.date_range(t[0], periods=20, freq="h")
        )
        fig, ax = plt.subplots()
        plot_timeseries_comparison(ax, ref, {"Model": est})
        self.assertEqual(len(ax.lines[1].get_xdata()), 10)
        plt.close(fig)


class TestMapPlotting(unittest.TestCase):
    """Tests for the DefaultStaticPlotting map helpers."""

    def setUp(self):
        """Synthetic rioxarray-like raster (x/y dims) from -500 m to +100 m."""

        x = np.linspace(0.0, 1.0, 200)
        y = np.linspace(0.0, 1.0, 150)
        self.raster = xr.DataArray(
            np.add.outer(y, x) * 300.0 - 500.0, coords={"y": y, "x": x}, dims=("y", "x")
        )
        self.plotter = DefaultStaticPlotting()

    def tearDown(self):
        """Close all figures."""

        plt.close("all")

    def test_decimate_raster(self):
        """Large rasters are strided down to the pixel budget."""

        small = decimate_raster(self.raster, 3000)
        self.assertLessEqual(small.size, 3000)
        self.assertIs(decimate_raster(self.raster, 10**6), self.raster)

    def test_bathymetry_value_range_ocean_only(self):
        """Fixed colour range, land masking, imshow and labelled isobaths."""

        fig, ax = plt.subplots()
        self.plotter.plot_bathymetry(
            ax,
            self.raster,
            area=None,
            value_range=(-200, 300),
            max_pixels=5000,
            ocean_only=True,
            method="imshow",
            add_colorbar=False,
            isodepths=[-100, -50],
            isodepths_kwargs={"colors": ["red", "blue"], "legend": True},
        )
        image = ax.images[0]
        self.assertTrue(
            np.isnan(image.get_array()).any() or np.ma.is_masked(image.get_array())
        )
        labels = [line.get_label() for line in ax.lines]
        self.assertEqual(labels, ["100m isobath", "50m isobath"])

    def test_ugrid_mesh(self):
        """UGRID edges (1-based) are drawn as one line collection."""

        mesh = xr.Dataset(
            {
                "mesh2d_node_x": ("node", [0.0, 1.0, 0.0]),
                "mesh2d_node_y": ("node", [0.0, 0.0, 1.0]),
                "mesh2d_edge_nodes": (
                    ("edge", "two"),
                    np.array([[1, 2], [2, 3], [3, 1]]),
                    {"start_index": 1},
                ),
            }
        )
        fig, ax = plt.subplots()
        lines = self.plotter.plot_ugrid_mesh(ax, mesh)
        self.assertEqual(len(lines.get_segments()), 3)
        np.testing.assert_allclose(lines.get_segments()[0], [[0, 0], [1, 0]])

    def test_satellite_needs_cartopy_axes(self):
        """A clear error instead of a broken image on plain axes."""

        fig, ax = plt.subplots()
        with self.assertRaises(TypeError):
            self.plotter.plot_satellite(ax, area=(0, 1, 0, 1))


class TestNauticalSector(unittest.TestCase):
    """Tests for nautical sector."""

    def test_wrapping_sector(self):
        """Wrapping sector."""

        np.testing.assert_array_equal(
            in_nautical_sector(np.array([0.0, 90.0, 355.0, -5.0]), -10, 30),
            [True, False, True, True],
        )

    def test_plain_sector(self):
        """Plain sector."""

        self.assertTrue(in_nautical_sector(150.0, 100, 200))
        self.assertFalse(in_nautical_sector(250.0, 100, 200))


if __name__ == "__main__":
    unittest.main()
