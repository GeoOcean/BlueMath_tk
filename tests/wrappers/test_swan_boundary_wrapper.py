"""Tests for the boundary-forced SWAN wrappers and their utilities (no SWAN run)."""

import os.path as op
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import Delaunay

from bluemath_tk.wrappers.snapwave import boundary_time_series
from bluemath_tk.wrappers.swan.swan_utils import (
    boundary_par_table,
    clockwise_triangles,
    read_adcirc_grid,
    read_swan_table,
    swan_boundary_side,
    write_ccw_adcirc_grid,
)
from bluemath_tk.wrappers.swan.swan_wrapper import (
    SwanDynamicModelWrapper,
    SwanMetaModelWrapper,
)

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "data" / "swan" / "boundary"
GOALS = np.array([[3.0, 52.2], [3.0, 52.5], [3.0, 52.8]])
POINTS = pd.DataFrame({"lon": [3.5, 3.9], "lat": [52.5, 52.5], "name": ["a", "b"]})


def _write_square_grid(path: str, n: int = 11) -> None:
    """ADCIRC grid on lon 3-4, lat 52-53: open boundary on the west side (S->N)."""

    x, y = (
        v.ravel() for v in np.meshgrid(np.linspace(3, 4, n), np.linspace(52, 53, n))
    )
    tri = Delaunay(np.c_[x, y]).simplices
    west = [i for i in np.argsort(y) if np.isclose(x[i], 3)]
    south = [i for i in np.argsort(x) if np.isclose(y[i], 52)]
    east = [i for i in np.argsort(y) if np.isclose(x[i], 4)]
    north = [i for i in np.argsort(-x) if np.isclose(y[i], 53)]
    land = list(dict.fromkeys(south + east + north))
    with open(path, "w") as f:
        f.write(f"square\n{len(tri)} {len(x)}\n")
        for i in range(len(x)):
            f.write(f"{i + 1} {x[i]:.6f} {y[i]:.6f} {2 + 38 * (4 - x[i]):.3f}\n")
        for k, t in enumerate(tri):
            f.write(f"{k + 1} 3 {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
        f.write(f"1 ! open\n{len(west)}\n{len(west)}\n")
        f.write("".join(f"{i + 1}\n" for i in west))
        f.write(f"1 ! land\n{len(land)}\n{len(land)} 0\n")
        f.write("".join(f"{i + 1}\n" for i in land))


class TestSwanBoundaryUtils(unittest.TestCase):
    """ADCIRC grid reading, SWAN boundary side and PAR rows."""

    def setUp(self):
        """Square test grid in a temporary folder."""
        self.test_dir = tempfile.mkdtemp()
        self.grid = op.join(self.test_dir, "grid.grd")
        _write_square_grid(self.grid)

    def tearDown(self):
        """Remove the temporary folder."""
        shutil.rmtree(self.test_dir)

    def test_side_excludes_corners_and_finds_direction(self):
        """South->north along a west open boundary is clockwise; corners are land."""

        grid = read_adcirc_grid(self.grid)
        self.assertEqual(grid["triangles"].shape[1], 3)
        xy, side = swan_boundary_side(grid)
        self.assertEqual(side, "1 CLOCKWISE")
        np.testing.assert_allclose(xy[[0, -1]], [[3.0, 52.1], [3.0, 52.9]])

        grid["open_boundaries"][0] = grid["open_boundaries"][0][::-1]
        xy, side = swan_boundary_side(grid)
        self.assertEqual(side, "1 CCW")
        np.testing.assert_allclose(xy[0], [3.0, 52.9])

    def test_par_rows_padded_and_ordered(self):
        """Nodes snap to boundary vertices; both ends repeat the nearest values."""

        xy, _ = swan_boundary_side(read_adcirc_grid(self.grid))
        rows = boundary_par_table(xy, GOALS, {"hs": [1.0, 0.0, 0.0]})
        np.testing.assert_allclose(rows["len"], [0.0, 0.1, 0.4, 0.7, 0.8])
        self.assertEqual(rows["hs"], [1.0, 1.0, 0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            boundary_par_table(xy, GOALS[::-1], {"hs": [1.0, 0.0, 0.0]})

    def test_clockwise_triangles_are_reoriented(self):
        """Clockwise triangles are found and a counterclockwise copy written."""
        clockwise = op.join(self.test_dir, "clockwise.grd")
        with open(self.grid) as src, open(clockwise, "w") as dst:
            lines = src.read().splitlines()
            n_el, n_v = (int(v) for v in lines[1].split())
            for k, line in enumerate(lines):
                if 2 + n_v <= k < 2 + n_v + n_el:
                    i, n, a, b, c = line.split()
                    line = f"{i} {n} {a} {c} {b}"
                dst.write(line + "\n")
        self.assertTrue(clockwise_triangles(read_adcirc_grid(clockwise)).all())

        out = op.join(self.test_dir, "fixed.grd")
        self.assertEqual(write_ccw_adcirc_grid(clockwise, out), n_el)
        fixed = read_adcirc_grid(out)
        self.assertFalse(clockwise_triangles(fixed).any())
        original = read_adcirc_grid(self.grid)
        np.testing.assert_array_equal(
            fixed["land_boundaries"][0], original["land_boundaries"][0]
        )

    def test_read_table_masks_exception_values(self):
        """Negative (exception) values become NaN; columns get short names."""

        path = op.join(self.test_dir, "out.tab")
        with open(path, "w") as f:
            f.write(" 0.12E+01  0.80E+01  0.27E+03  0.20E+02\n")
            f.write(" 0.00E+00 -0.90E+01 -0.99E+03 -0.90E+01\n")
        table = read_swan_table(path, ["HSIGN", "TPS", "DIR", "DSPR"])
        self.assertEqual(list(table.columns), ["hs", "tp", "dir", "spr"])
        self.assertEqual(table.hs.tolist(), [1.2, 0.0])
        self.assertTrue(table.loc[1, ["tp", "dir", "spr"]].isna().all())


class TestSwanMetaModelWrapper(unittest.TestCase):
    """Case building and postprocessing of the metamodel wrapper."""

    def setUp(self):
        """Two metamodel cases on the square test grid."""
        self.test_dir = tempfile.mkdtemp()
        grid = op.join(self.test_dir, "grid.grd")
        _write_square_grid(grid)
        self.wrapper = SwanMetaModelWrapper(
            templates_dir=str(TEMPLATES_DIR),
            metamodel_parameters={
                "goal": [1, 2],
                "active_node": [1, 2],
                "hs": [1.0, 3.0],
                "tp": [8.0, 12.0],
                "dir": [270.0, 280.0],
                "spr": [20.0, 25.0],
                "wl": [0.0, 1.0],
            },
            fixed_parameters={},
            output_dir=op.join(self.test_dir, "cases"),
            grid_file=grid,
            boundary_nodes=GOALS,
            output_points=POINTS,
            point_dim="sites",
            point_coords={"sites": "name"},
            store_parameters=("hs", "tp", "wl"),
            split_by="goal",
            debug=False,
        )
        self.wrapper.build_cases()

    def tearDown(self):
        """Remove the temporary folder."""
        shutil.rmtree(self.test_dir)

    def test_case_files(self):
        """fort.14 is linked; Hs decays from the active goal to its neighbours."""

        case_dir = self.wrapper.cases_dirs[1]
        self.assertTrue(op.islink(op.join(case_dir, "fort.14")))
        self.assertEqual(len(np.loadtxt(op.join(case_dir, "output_sites.txt"))), 2)
        with open(op.join(case_dir, "INPUT")) as f:
            inp = f.read()
        self.assertIn("BOUNDSPEC SIDE 1 CLOCKWISE VARIABLE PAR", inp)
        par = inp.split("VARIABLE PAR &\n")[1].split("\nPOINTS")[0].splitlines()
        self.assertEqual(
            [line.split()[:3] for line in par],
            [
                ["0.00000000", "0.0000", "12.0000"],
                ["0.10000000", "0.0000", "12.0000"],
                ["0.40000000", "3.0000", "12.0000"],
                ["0.70000000", "0.0000", "12.0000"],
                ["0.80000000", "0.0000", "12.0000"],
            ],
        )
        self.assertIn("SET LEVEL=1.0", inp)
        self.assertIn("TABLE 'sites' NOHEAD 'output_sites.tab' HSIGN TPS DIR DSPR", inp)

    def test_postprocess_split(self):
        """Fake tables are joined on case_num and split per goal."""

        for case_dir in self.wrapper.cases_dirs:
            with open(op.join(case_dir, "output_sites.tab"), "w") as f:
                f.write(" 1.0 8.0 270.0 20.0\n 0.5 8.1 275.0 15.0\n")
        written = self.wrapper.postprocess_cases()
        ds = xr.open_dataset(written[2])
        self.assertEqual(list(ds.sites.values), ["a", "b"])
        self.assertEqual(ds.case_num.values.tolist(), [1])
        self.assertEqual(float(ds.hs_forcing[0]), 3.0)
        self.assertEqual(float(ds.hs.sel(sites="b")[0]), 0.5)
        ds.close()


class TestSwanDynamicModelWrapper(unittest.TestCase):
    """Per-node forcing and water level of the dynamic wrapper."""

    def test_hourly_cases(self):
        """Hourly cases get per-node forcing, the mean water level and a time axis."""
        test_dir = tempfile.mkdtemp()
        try:
            grid = op.join(test_dir, "grid.grd")
            _write_square_grid(grid)
            time = pd.date_range("2013-12-05", periods=2, freq="h")
            forcing = xr.Dataset(
                {
                    v: (("time", "node"), np.full((2, 3), x) + np.arange(2)[:, None])
                    for v, x in dict(
                        hs=1.0, tp=9.0, dir=280.0, spr=20.0, wl=0.5
                    ).items()
                },
                coords={"time": time, "node": [1, 2, 3]},
            )
            wrapper = SwanDynamicModelWrapper(
                templates_dir=str(TEMPLATES_DIR),
                metamodel_parameters=boundary_time_series(
                    forcing, variables=("hs", "tp", "dir", "spr", "wl")
                ),
                fixed_parameters={},
                output_dir=op.join(test_dir, "cases"),
                grid_file=grid,
                boundary_nodes=GOALS,
                output_points=POINTS,
                debug=False,
            )
            wrapper.build_cases()
            with open(op.join(wrapper.cases_dirs[1], "INPUT")) as f:
                inp = f.read()
            self.assertIn("SET LEVEL=1.5", inp)
            self.assertIn("0.40000000 2.0000 10.0000 281.000 21.000 &", inp)
            for case_dir in wrapper.cases_dirs:
                with open(op.join(case_dir, "output_sites.tab"), "w") as f:
                    f.write(" 1.0 8.0 270.0 20.0\n 0.5 8.1 275.0 15.0\n")
            ds = wrapper.postprocess_cases()
            self.assertEqual(list(pd.DatetimeIndex(ds.time.values)), list(time))
            self.assertEqual(dict(ds.hs.sizes), {"time": 2, "points": 2})
        finally:
            shutil.rmtree(test_dir)


    def test_wind_field_and_uniform_wind(self):
        """A wind field is written as u then v blocks, south to north, with its grid."""
        test_dir = tempfile.mkdtemp()
        try:
            grid = op.join(test_dir, "grid.grd")
            _write_square_grid(grid)
            wind = xr.Dataset(
                {
                    "u10": (("time", "latitude", "longitude"), np.full((1, 2, 3), 5.0)),
                    "v10": (
                        ("time", "latitude", "longitude"),
                        np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]),
                    ),
                },
                coords={
                    "time": pd.to_datetime(["2013-12-05"]),
                    "latitude": [53.0, 52.0],  # north first, as in ERA5
                    "longitude": [3.0, 3.5, 4.0],
                },
            )
            params = {
                "tref": ["20131205 000000"],
                "u10": [12.0],
                "u10dir": [300.0],
                **{f"{v}_nodes": [[1.0, 1.0, 1.0]] for v in ("hs", "tp", "dir", "spr")},
            }
            wrapper = SwanDynamicModelWrapper(
                templates_dir=str(TEMPLATES_DIR),
                metamodel_parameters=params,
                fixed_parameters={},
                output_dir=op.join(test_dir, "cases"),
                grid_file=grid,
                boundary_nodes=GOALS,
                output_points=POINTS,
                wind=wind,
                debug=False,
            )
            wrapper.build_cases()
            context = wrapper.cases_context[0]
            self.assertEqual(context["wind_file"], "wind.dat")
            self.assertEqual(
                context["wind_grid"],
                "3.000000 52.000000 0 2 1 0.500000 1.000000",
            )
            self.assertEqual((context["u10"], context["u10dir"]), (12.0, 300.0))
            values = np.loadtxt(op.join(wrapper.cases_dirs[0], "wind.dat"))
            np.testing.assert_array_equal(values[:2], 5.0)  # u block
            np.testing.assert_array_equal(values[2:, 0], [2.0, 1.0])  # v at 52N, then 53N

            # the case plot reads INPUT, wind.dat and the TABLE output back
            import matplotlib

            matplotlib.use("Agg")
            from bluemath_tk.wrappers.swan.swan_plots import load_swan_case, plot_case

            case_dir = wrapper.cases_dirs[0]
            with open(op.join(case_dir, "INPUT"), "a") as f:
                f.write(f"INPGRID WIND REGULAR {context['wind_grid']}\nREADINP WIND 1 'wind.dat' 3 0 FREE\n")
            with open(op.join(case_dir, "output_sites.tab"), "w") as f:
                f.write(" 1.0 8.0 270.0 20.0\n 0.5 8.1 275.0 15.0\n")
            case = load_swan_case(case_dir)
            self.assertEqual(case.boundary.shape[1], 5)
            self.assertEqual(len(case.wind[0]), 6)
            np.testing.assert_allclose(case.table["hs"], [1.0, 0.5])
            fig = plot_case(case_dir, mode="dynamic")
            self.assertGreaterEqual(len(fig.axes), 5)
        finally:
            shutil.rmtree(test_dir)

    def test_grid_copied_without_symlinks(self):
        """Where symlinks are not allowed (e.g. Windows), fort.14 is a copy of the grid."""
        from unittest import mock

        test_dir = tempfile.mkdtemp()
        try:
            grid = op.join(test_dir, "grid.grd")
            _write_square_grid(grid)
            params = {
                "tref": ["20131205 000000"],
                **{f"{v}_nodes": [[1.0, 1.0, 1.0]] for v in ("hs", "tp", "dir", "spr")},
            }
            wrapper = SwanDynamicModelWrapper(
                templates_dir=str(TEMPLATES_DIR),
                metamodel_parameters=params,
                fixed_parameters={},
                output_dir=op.join(test_dir, "cases"),
                grid_file=grid,
                boundary_nodes=GOALS,
                output_points=POINTS,
                debug=False,
            )
            with mock.patch("os.symlink", side_effect=OSError("no symlinks")):
                wrapper.build_cases()
            fort14 = op.join(wrapper.cases_dirs[0], "fort.14")
            self.assertFalse(op.islink(fort14))
            with open(fort14) as a, open(grid) as b:
                self.assertEqual(a.read(), b.read())
        finally:
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    unittest.main()
