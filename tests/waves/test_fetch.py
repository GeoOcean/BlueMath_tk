"""Tests for bluemath_tk.waves.fetch."""

import unittest

import numpy as np

from bluemath_tk.waves.fetch import effective_fetch, fetch_table


def _square_mesh(size: float = 10e3, n: int = 21):
    """Regular triangulated square [0, size]^2 in metres, depth 10 m."""

    x, y = np.meshgrid(np.linspace(0, size, n), np.linspace(0, size, n))
    idx = np.arange(n * n).reshape(n, n)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    triangles = np.r_[np.c_[a, b, d], np.c_[a, d, c]]
    return x.ravel(), y.ravel(), triangles, np.full(n * n, 10.0)


class TestFetchTable(unittest.TestCase):
    """Rays walked upwind through a square mesh, in metres."""

    def setUp(self):
        self.mesh = _square_mesh()
        west_edge = np.array([[0.0, 0.0], [0.0, 10e3]])
        self.table = fetch_table(
            np.array([[2e3, 5e3], [50e3, 50e3]]),
            *self.mesh,
            directions=np.arange(0.0, 360.0, 10.0),
            step=100.0,
            max_distance=30e3,
            geographic=False,
            open_boundary=west_edge,
            open_tolerance=500.0,
        )

    def test_distance_to_the_edge(self):
        """Wind from the west: 2 km to the west edge; from the east: 8 km."""

        f = self.table.fetch.isel(point=0)
        self.assertAlmostEqual(float(f.sel(dir=270.0)), 2e3, delta=100.0)
        self.assertAlmostEqual(float(f.sel(dir=90.0)), 8e3, delta=100.0)
        self.assertAlmostEqual(float(self.table.mean_depth[0, 0]), 10.0)

    def test_open_boundary_and_outside_points(self):
        """Only rays leaving through the west edge are open; outside points are NaN."""

        exits = self.table.open_exit.isel(point=0)
        self.assertTrue(bool(exits.sel(dir=270.0)))
        self.assertFalse(bool(exits.sel(dir=90.0)))
        self.assertTrue(np.isnan(self.table.fetch.isel(point=1)).all())

    def test_effective_fetch(self):
        """Saville weighting of a uniform fetch field returns sum cos^2 / sum cos times it."""

        table = self.table.copy()
        table["fetch"] = table.fetch * 0 + 1000.0
        eff = effective_fetch(table, half_width=40.0)
        cos = np.cos(np.deg2rad(np.arange(-40, 50, 10)))
        np.testing.assert_allclose(
            eff.isel(point=0), 1000.0 * (cos**2).sum() / cos.sum()
        )


if __name__ == "__main__":
    unittest.main()
