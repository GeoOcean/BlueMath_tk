import unittest

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

from bluemath_tk.waves.hywaves import plots

GOALS = {
    "1": {"lon": 0.0, "lat": 0.0, "angles": [-45.0, 45.0]},
    "2": {"lon": 0.5, "lat": 0.0, "angles": [180.0, 300.0]},
}


class TestHyWavesPlots(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_goal_colors_never_repeat(self):
        colors = plots.goal_colors(list(range(30)))
        self.assertEqual(len(set(colors.values())), 30)

    def test_goal_sectors_and_direction_ranges(self):
        rng = np.random.default_rng(0)
        cases = pd.DataFrame(
            {
                "goal": np.repeat([1, 2], 50),
                "dir": np.r_[rng.uniform(-45, 45, 50) % 360, rng.uniform(180, 300, 50)],
                "tp": rng.uniform(5, 12, 100),
            }
        )
        colors = plots.goal_colors([1, 2])
        fig, (ax_map, ax_bars) = plt.subplots(2, 1)
        ax_map.set(xlim=(-1, 1.5), ylim=(-1, 1))
        plots.plot_goal_sectors(ax_map, GOALS, colors, directions=cases, max_arrows=5)
        self.assertEqual(len(ax_map.patches), 2)
        self.assertEqual({t.get_text() for t in ax_map.texts if t.get_text()}, {"G1", "G2"})
        self.assertFalse(plots.plot_direction_ranges(ax_bars, cases, GOALS, colors))
        self.assertTrue(plots.plot_direction_ranges(ax_bars, cases, GOALS, colors, max_marks=10))

    def test_circular_spans_jump_the_gap(self):
        self.assertEqual(plots._circular_spans(np.array([350.0, 355.0, 5.0, 10.0])), [(350.0, 360.0), (0.0, 10.0)])
        self.assertEqual(plots._sector_spans(-45.0, 45.0), [(315.0, 360.0), (0.0, 45.0)])

    def test_mda_sweep(self):
        table = pd.DataFrame(
            {
                "metamodel": ["buoys"] * 4,
                "mda_candidate": [50, 100, 50, 100],
                "variable": ["hs", "hs", "dir", "dir"],
                "bias": [0.1, 0.05, 1.0, 0.5],
                "rmse": [0.3, 0.2, 10.0, 8.0],
                "rrmse": [0.2, 0.1, np.nan, np.nan],
                "r2": [0.9, 0.95, 0.8, 0.85],
            }
        )
        fig = plots.plot_mda_sweep(table, ["metamodel"], ["hs", "dir"], units={"hs": "m", "dir": "deg"})
        self.assertEqual(len(fig.axes), 8)

    def test_dynamic_forcing_without_wind_any_dim_order(self):
        times = pd.date_range("2020-01-01", periods=24, freq="h")
        offshore = xr.Dataset(
            {v: (("goal", "time"), np.full((2, 24), x)) for v, x in (("hs", 2.0), ("tp", 8.0), ("dir", 270.0), ("spr", 30.0))},
            coords={"goal": [1, 2], "time": times},
        )
        level = xr.DataArray(np.zeros((24, 2)), dims=("time", "goal"), coords={"time": times, "goal": [1, 2]})
        aoi = gpd.GeoDataFrame(geometry=[box(-1, -1, 1.5, 1)], crs="EPSG:4326")
        fig = plots.plot_dynamic_forcing(aoi, GOALS, offshore, water_level=level)
        self.assertEqual(len(fig.axes), 5)

    def test_wind_k_comparison(self):
        rng = np.random.default_rng(1)
        k = 1 + rng.uniform(0, 0.5, 200)
        df = pd.DataFrame(
            {"k_runs": k, "k_fitted": k + 0.01, "k_formula_deep": k * 1.1,
             "k_formula_depth": k * 1.05, "hs_ws_u2": rng.uniform(0.01, 0.3, 200)}
        )
        self.assertEqual(len(plots.plot_wind_k_comparison(df).axes), 2)

    def test_peak_period_ratio(self):
        df = pd.DataFrame({"partition": ["sea"] * 50 + ["swell"] * 50,
                           "ratio": np.linspace(1.0, 1.3, 100)})
        fig = plots.plot_peak_period_ratio(df, {"JONSWAP 3.3": 1.107})
        self.assertEqual(len(fig.axes), 2)

    def test_wind_case_comparison(self):
        rng = np.random.default_rng(0)
        hs0 = rng.uniform(0.5, 3.0, (2, 30))
        case = {"hs": [3.0, 5.0], "tp": [7.0, 9.0], "dir": [300.0, 200.0], "u10": [15.0, 25.0],
                "u10dir": [320.0, 180.0], "wind_wave_angle": [20.0, -20.0], "ws_weight": [1.0, 0.6],
                "hs_ws_u2": [0.13, 0.05]}
        ds = xr.Dataset(
            {"hs0": (("case", "sites"), hs0), "hs1": (("case", "sites"), hs0 * 1.2),
             "hs_corrected": (("case", "sites"), hs0 * 1.15)},
            coords={"case": [3, 7], "longitude": ("sites", np.linspace(3, 6, 30)),
                    "latitude": ("sites", np.linspace(51, 53, 30)),
                    **{k: ("case", v) for k, v in case.items()}},
        )
        fig = plots.plot_wind_case_comparison(ds, size=2)
        self.assertEqual(len(fig.axes), 2 * 4 + 2 * 2)  # 4 panels + 2 colorbars per case

    def test_k_response(self):
        x = {"u10": np.linspace(1, 40, 5), "angle": np.linspace(-90, 90, 5), "hs_ws_u2": np.geomspace(0.002, 0.6, 5)}
        ds = xr.Dataset(
            {f"k_{name}": (("sites", name), np.full((2, 5), 1.2)) for name in x},
            coords={"sites": ["a", "b"], **x, "depth": ("sites", [8.0, 20.0]), "fetch": ("sites", [4e4, 2e4])},
            attrs={"wind_dir": 300.0, "hs_ws": 2.0, "u10": 15.0, "angle": 0.0},
        )
        fig = plots.plot_k_response(ds, {"u10": (3.0, 35.0), "hs_ws_u2": (0.004, 0.29)})
        self.assertEqual(len(fig.axes), 3)

    def test_wind_sea_event(self):
        time = pd.date_range("2013-12-05", periods=12, freq="h")
        ws = xr.Dataset({name: ("time", np.linspace(0.1, 2.0, 12)) for name in ("hs_ws", "u10", "hs_ws_u2", "angle", "k")},
                        coords={"time": time})
        hs = {"Buoy": ws["hs_ws"] * 2, "model": ws["hs_ws"] * 1.8}
        fig = plots.plot_wind_sea_event(ws, hs, {"hs_ws_u2": (0.004, 0.29), "angle": (-60.0, 60.0)})
        self.assertEqual(len(fig.axes), 6)  # 5 rows + the wind twin axis

    def test_event_maps(self):
        coords = {"sites": ["a", "b", "c"], "site_longitude": ("sites", [3.0, 4.0, 5.0]),
                  "site_latitude": ("sites", [51.0, 52.0, 53.0])}
        ref = xr.DataArray([1.0, 2.0, 3.0], dims="sites", coords=coords)
        fig = plots.plot_event_maps(ref, {"m1": ref * 1.1, "m2": ref * 0.9}, k=ref * 0 + 1.2)
        self.assertEqual(len(fig.axes), 4 + 3)  # 4 maps + 3 colorbars


if __name__ == "__main__":
    unittest.main()
