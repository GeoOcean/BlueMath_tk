"""Tests for the SnapWave wrapper, its utilities and case plots."""

import os.path as op
import shutil
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from shapely.geometry import box

from bluemath_tk.wrappers.snapwave import (
    SnapWaveDynamicModelWrapper,
    SnapWaveMetaModelWrapper,
    boundary_time_series,
    build_enclosure_polygon,
    read_boundary_nodes,
    read_enclosure_polygon,
    read_forcing_table,
)
from bluemath_tk.wrappers.snapwave.snapwave_plots import active_boundary_node, plot_case

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "data" / "snapwave"
NODES = np.array([[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]])
POINTS = pd.DataFrame({"lon": [0.2, 0.4], "lat": [0.3, 0.5]})


def _fake_his(case_dir: str, n_times: int = 1) -> None:
    """Write a minimal SnapWave history file into *case_dir*."""

    shape = (n_times, len(POINTS))
    ds = xr.Dataset(
        {
            "point_hm0": (("time", "stations"), np.full(shape, 0.5)),
            "point_tp": (("time", "stations"), np.full(shape, 9.0)),
            "point_wavdir": (("time", "stations"), np.full(shape, 300.0)),
            "point_dirspr": (("time", "stations"), np.full(shape, 20.0)),
            "station_x": (("stations",), POINTS["lon"].values),
            "station_y": (("stations",), POINTS["lat"].values),
        },
        coords={"time": np.arange(n_times, dtype=float)},
    )
    ds.to_netcdf(op.join(case_dir, "output_sites.nc"))


class TestSnapWaveMetaModelWrapper(unittest.TestCase):
    """Tests for the stationary (metamodel) wrapper."""

    def setUp(self):
        """Build three cases from the test templates."""

        self.test_dir = tempfile.mkdtemp()
        self.wrapper = SnapWaveMetaModelWrapper(
            templates_dir=str(TEMPLATES_DIR / "metamodel"),
            metamodel_parameters={
                "hs": [1.0, 2.5, 0.5],
                "tp": [8.0, 10.0, 12.0],
                "dir": [280.0, 300.0, 320.0],
                "spr": [20.0, 25.0, 30.0],
                "wl": [0.0, 0.5, -0.5],
                "active_node": [1, 2, 3],
                "save_map": [False, True, False],
            },
            fixed_parameters={"gridfile": "mesh.nc"},
            output_dir=op.join(self.test_dir, "cases"),
            boundary_nodes=NODES,
            enclosure_polygon=build_enclosure_polygon(box(0, 0, 1, 1)),
            output_points=POINTS,
            debug=False,
        )
        self.wrapper.build_cases()

    def tearDown(self):
        """Remove the temporary folder."""

        shutil.rmtree(self.test_dir)
        plt.close("all")

    def test_case_files(self):
        """Boundary, enclosure, points and rendered templates are written."""

        case_dir = self.wrapper.cases_dirs[1]
        for name in ("boundary.txt", "enclosure.txt", "output_sites.txt", "hs.txt"):
            self.assertTrue(op.exists(op.join(case_dir, name)), name)
        np.testing.assert_allclose(read_boundary_nodes(case_dir), NODES)
        self.assertEqual(len(read_enclosure_polygon(case_dir).exterior.coords), 5)

        hs = read_forcing_table(op.join(case_dir, "hs.txt"))
        np.testing.assert_array_equal(hs.iloc[0].values, [0, 2.5, 0])
        tp = read_forcing_table(op.join(case_dir, "tp.txt"))
        np.testing.assert_array_equal(tp.iloc[0].values, [10.0] * 3)
        with open(op.join(case_dir, "snapwave.inp")) as f:
            inp = f.read()
        self.assertIn("map_file       = output_map.nc", inp)
        self.assertIn("gridfile       = mesh.nc", inp)

    def test_postprocess_on_case_num(self):
        """Cases are postprocessed and joined on case_num with short names."""

        for case_dir in self.wrapper.cases_dirs:
            _fake_his(case_dir)
        ds = self.wrapper.postprocess_cases(write_output_nc=False)
        self.assertEqual(ds.sizes["case_num"], 3)
        self.assertIn("hs", ds)
        self.assertNotIn("point_hm0", ds)

    def test_slurm_array(self):
        """The SLURM script reads case_dirs.txt and loads the cluster module."""

        path = self.wrapper.write_slurm_array(
            cpus_per_task=4, extra_sbatch=["--time=10"]
        )
        with open(path) as f:
            script = f.read()
        self.assertIn("#SBATCH --cpus-per-task=4", script)
        self.assertIn("#SBATCH --time=10", script)
        self.assertIn("module load snapwave", script)
        self.assertIn('sed -n "${SLURM_ARRAY_TASK_ID}p"', script)
        with open(op.join(self.wrapper.output_dir, "case_dirs.txt")) as f:
            self.assertEqual(len(f.read().splitlines()), 3)

    def test_monitor_and_plot(self):
        """Finished cases are detected and a metamodel case can be plotted."""

        _fake_his(self.wrapper.cases_dirs[2])
        status = self.wrapper.monitor_cases(value_counts="simple")
        self.assertIn("FINISHED", str(status))
        self.assertEqual(active_boundary_node(self.wrapper.cases_dirs[2]), 2)
        fig = plot_case(self.wrapper.cases_dirs[2], mode="metamodel")
        self.assertIn("bnd3", fig._suptitle.get_text())
        self.assertIn("hs=0.5m @bnd3", " ".join(t.get_text() for t in fig.texts))


class TestSnapWaveOutputOptions(unittest.TestCase):
    """Tests for point metadata, stored parameters and split output."""

    def setUp(self):
        """Two cases per active node, with labelled output points."""

        self.test_dir = tempfile.mkdtemp()
        points = POINTS.assign(name=["a", "b"], depth=[5.0, 10.0])
        self.wrapper = SnapWaveMetaModelWrapper(
            templates_dir=str(TEMPLATES_DIR / "metamodel"),
            metamodel_parameters={
                "hs": [1.0, 2.0, 1.0, 2.0],
                "tp": [8.0, 9.0, 10.0, 11.0],
                "dir": [280.0] * 4,
                "spr": [20.0] * 4,
                "wl": [0.0] * 4,
                "active_node": [1, 1, 2, 2],
            },
            fixed_parameters={"gridfile": "mesh.nc"},
            output_dir=op.join(self.test_dir, "cases"),
            boundary_nodes=NODES[:2],
            enclosure_polygon=build_enclosure_polygon(box(0, 0, 1, 1)),
            output_points=points,
            point_dim="sites",
            point_coords={"sites": "name", "site_depth": "depth"},
            store_parameters=("hs", "tp", "wl"),
            split_by="active_node",
            split_filename="node_{}.nc",
            debug=False,
        )
        self.wrapper.build_cases()
        for case_dir in self.wrapper.cases_dirs:
            _fake_his(case_dir)

    def tearDown(self):
        """Remove the temporary folder."""

        shutil.rmtree(self.test_dir)

    def test_split_files_with_metadata(self):
        """One file per active node, labelled sites, stored forcing."""

        written = self.wrapper.postprocess_cases()
        self.assertEqual(sorted(written), [1, 2])
        ds = xr.open_dataset(written[2])
        self.assertEqual(list(ds.sites.values), ["a", "b"])
        self.assertEqual(float(ds.site_depth.sel(sites="b")), 10.0)
        np.testing.assert_array_equal(ds.site_depth.values, [5.0, 10.0])
        np.testing.assert_array_equal(ds.case_num.values, [2, 3])
        np.testing.assert_array_equal(ds.tp_forcing.values, [10.0, 11.0])
        np.testing.assert_array_equal(ds.hs_forcing.values, [1.0, 2.0])
        self.assertNotIn("station_x", ds.coords)
        self.assertNotIn("active_node", ds.coords)
        self.assertEqual(
            sorted(ds.data_vars),
            ["dir", "hs", "hs_forcing", "spr", "tp", "tp_forcing", "wl_forcing"],
        )
        ds.close()

    def test_numeric_string_nodes_sorted(self):
        """Node ids stored as strings are ordered numerically."""

        forcing = xr.Dataset(
            {
                v: (("time", "goal"), [[2.0, 10.0, 1.0]])
                for v in ("hs", "tp", "dir", "spr")
            },
            coords={"time": pd.date_range("2020", periods=1), "goal": ["2", "10", "1"]},
        )
        out = boundary_time_series(forcing, node_dim="goal")
        self.assertEqual(out["hs_nodes"][0], [1.0, 2.0, 10.0])


class TestSnapWaveDynamic(unittest.TestCase):
    """Tests for the time-series wrapper and boundary forcing."""

    def setUp(self):
        """Temporary folder."""

        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary folder."""

        shutil.rmtree(self.test_dir)
        plt.close("all")

    def test_boundary_time_series_and_cases(self):
        """Per-node forcing reaches the rendered dynamic templates in node order."""

        time = pd.date_range("2020-01-01", periods=2, freq="h")
        forcing = xr.Dataset(
            {
                v: (("time", "node"), np.arange(6.0).reshape(2, 3) + i)
                for i, v in enumerate(("hs", "tp", "dir", "spr"))
            },
            coords={"time": time, "node": [3, 1, 2]},
        )
        params = boundary_time_series(forcing)
        self.assertEqual(params["tref"], ["20200101 000000", "20200101 010000"])
        self.assertEqual(params["hs_nodes"][0], [1.0, 2.0, 0.0])  # sorted by node

        wrapper = SnapWaveDynamicModelWrapper(
            templates_dir=str(TEMPLATES_DIR / "dynamic"),
            metamodel_parameters=params,
            fixed_parameters={"gridfile": "mesh.nc"},
            output_dir=op.join(self.test_dir, "cases"),
            boundary_nodes=NODES,
            enclosure_polygon=build_enclosure_polygon(box(0, 0, 1, 1)),
            output_points=POINTS.to_numpy(),
            debug=False,
        )
        wrapper.build_cases()
        case_dir = wrapper.cases_dirs[1]
        hs = read_forcing_table(op.join(case_dir, "hs.txt"))
        np.testing.assert_array_equal(hs.iloc[0].values, [4.0, 5.0, 3.0])
        wl = read_forcing_table(op.join(case_dir, "wl.txt"))
        np.testing.assert_array_equal(wl.iloc[0].values, [0, 0, 0])
        with open(op.join(case_dir, "snapwave.inp")) as f:
            self.assertIn("tref           = 20200101 010000", f.read())

        for d in wrapper.cases_dirs:
            _fake_his(d)
        joined = wrapper.join_postprocessed_files(
            [
                wrapper.postprocess_case(i, d, {})
                for i, d in enumerate(wrapper.cases_dirs)
            ]
        )
        self.assertEqual(joined.sizes["time"], 2)
        fig = plot_case(case_dir, mode="dynamic")
        self.assertEqual(len(fig.axes) >= 5, True)

    def test_wind_field_samples(self):
        """A wind field is interpolated to each case time and written as samples."""

        lon, lat = np.array([-0.5, 0.5, 1.5]), np.array([-0.5, 1.5])
        # from 350 deg in the west column, from 10 deg elsewhere (crosses North)
        dirs = np.array([[350.0, 10.0, 10.0], [350.0, 10.0, 10.0]])
        u = -10.0 * np.sin(np.deg2rad(dirs))
        v = -10.0 * np.cos(np.deg2rad(dirs))
        wind = xr.Dataset(
            {
                "u10": (("time", "latitude", "longitude"), np.stack([u, 3 * u])),
                "v10": (("time", "latitude", "longitude"), np.stack([v, 3 * v])),
            },
            coords={
                "time": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 02:00"]),
                "latitude": lat,
                "longitude": lon,
            },
        )
        params = {
            "tref": ["20200101 010000"],
            **{f"{v}_nodes": [[1.0, 1.0, 1.0]] for v in ("hs", "tp", "dir", "spr")},
        }
        wrapper = SnapWaveDynamicModelWrapper(
            templates_dir=str(TEMPLATES_DIR / "dynamic"),
            metamodel_parameters=params,
            fixed_parameters={"gridfile": "mesh.nc"},
            output_dir=op.join(self.test_dir, "cases"),
            boundary_nodes=NODES,
            enclosure_polygon=build_enclosure_polygon(box(0, 0, 1, 1)),
            output_points=POINTS.to_numpy(),
            wind=wind,
            debug=False,
        )
        wrapper.build_cases()
        self.assertEqual(wrapper.cases_context[0]["u10"], "u10.txt")
        speed = np.loadtxt(op.join(wrapper.cases_dirs[0], "u10.txt"))
        direction = np.loadtxt(op.join(wrapper.cases_dirs[0], "u10dir.txt"))
        np.testing.assert_allclose(speed[:, 2], 20.0)  # halfway between 10 and 30
        np.testing.assert_allclose(speed[:, :2], np.c_[np.tile(lon, 2), np.repeat(lat, 3)])
        # continuous across North: 350 is written as -10 next to 10
        np.testing.assert_allclose(direction[:, 2], [-10.0, 10.0, 10.0] * 2, atol=1e-6)

    def test_missing_variables(self):
        """Forcing without a required variable is rejected."""

        with self.assertRaises(ValueError):
            boundary_time_series(xr.Dataset({"hs": (("time", "node"), [[1.0]])}))


if __name__ == "__main__":
    unittest.main()
