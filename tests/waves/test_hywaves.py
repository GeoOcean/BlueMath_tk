"""Tests for bluemath_tk.waves.hywaves (goals, sectors, output points)."""

import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point, box

from bluemath_tk.core.operations import (
    in_nautical_sector,
    math_sector_to_nautical_clockwise,
    nautical_sector_to_wedge_angles,
    nautical_sector_wedge_coords,
)
from bluemath_tk.waves.hywaves import metamodel, output_points, reconstruction, sectors

UTM = "EPSG:32631"


class TestNauticalSectors(unittest.TestCase):
    """Tests for the nautical sector helpers."""

    def test_math_to_nautical(self):
        """East-to-North (math) is North-to-East (nautical); wraps through North."""

        self.assertEqual(math_sector_to_nautical_clockwise(0.0, 90.0), [0.0, 90.0])
        self.assertEqual(math_sector_to_nautical_clockwise(80.0, 110.0), [-20.0, 10.0])

    def test_wedge_angles_and_coords(self):
        """A North-centred sector gives a closed wedge around +y."""

        theta1, theta2 = nautical_sector_to_wedge_angles(-20.0, 20.0)
        self.assertEqual((theta1, theta2), (70.0, 110.0))
        ring = nautical_sector_wedge_coords(0.0, 0.0, -20.0, 20.0, 1.0)
        self.assertEqual(ring[0], ring[-1])
        self.assertTrue(all(y > 0.9 for _, y in ring[1:-1]))


class TestSectors(unittest.TestCase):
    """Tests for goals and AoI-aligned sectors."""

    def setUp(self):
        """Build a 1x1 degree AoI with a north coast (land), open sea elsewhere."""

        self.aoi = gpd.GeoDataFrame(
            geometry=[box(2.0, 52.0, 3.0, 53.0)], crs="EPSG:4326"
        )
        self.land = gpd.GeoDataFrame(
            geometry=[box(1.5, 52.8, 3.5, 53.5)], crs="EPSG:4326"
        )

    def test_ring_targets_and_order(self):
        """Targets fall in the sea part of the ring and are numbered 1..N."""

        ring, _ = sectors.build_ring(self.aoi, self.land, 5000.0, UTM)
        targets = sectors.random_targets_in_ring(
            ring, self.land, min_spacing_m=20000.0, work_crs=UTM, max_batches=5
        )
        self.assertGreater(len(targets), 2)
        land = self.land.union_all()
        self.assertFalse(any(land.contains(p) for p in targets.geometry))
        ordered = sectors.sort_targets_along_aoi(targets, self.aoi, work_crs=UTM)
        self.assertEqual(list(ordered.goal_id), list(range(1, len(targets) + 1)))

    def test_sector_faces_open_sea(self):
        """A goal on the southern edge looks south (waves coming from ~180)."""

        ring_sectors = sectors.AoIRingSectors.from_aoi(self.aoi, UTM)
        south = gpd.GeoSeries([Point(2.5, 52.0)], crs="EPSG:4326").to_crs(UTM).iloc[0]
        left, right = ring_sectors.sector_at_boundary(south)["angles"]
        self.assertLess(left, right)
        self.assertTrue(in_nautical_sector(180.0, left, right))
        self.assertFalse(in_nautical_sector(0.0, left, right))

    def test_goals_dict(self):
        """Each goal gets target, sector and nearest catalog node."""

        targets = gpd.GeoDataFrame(
            {"goal_id": [1]}, geometry=[Point(2.5, 52.02)], crs="EPSG:4326"
        )
        catalog = gpd.GeoDataFrame(
            {"fid": [7, 8], "longitude": [2.5, 9.0], "latitude": [51.9, 50.0]},
            geometry=[Point(2.5, 51.9), Point(9.0, 50.0)],
            crs="EPSG:4326",
        )
        goals = sectors.build_goals_dict(
            targets,
            {"whacs": catalog},
            sectors.AoIRingSectors.from_aoi(self.aoi, UTM),
            work_crs=UTM,
        )
        self.assertEqual(goals["1"]["hindcast"]["whacs"]["fid"], 7)
        self.assertEqual(set(goals["1"]), {"target", "angles", "hindcast"})


class TestOutputPoints(unittest.TestCase):
    """Tests for contour and mesh output points."""

    def test_isobath_points(self):
        """Points along the -10 m contour of a linear slope lie at x = 0.5."""

        x = np.linspace(0.0, 1.0, 51)
        y = np.linspace(0.0, 1.0, 41)
        da = xr.DataArray(
            np.tile(-20.0 * x, (y.size, 1)), coords={"y": y, "x": x}, dims=("y", "x")
        )
        pts = output_points.get_isobath_points(da, -10.0, spacing=0.1)
        np.testing.assert_allclose(pts[:, 0], 0.5, atol=1e-6)
        self.assertGreaterEqual(len(pts), 10)
        clipped = output_points.get_isobath_points(
            da, -10.0, aoi_polygon=box(2.0, 2.0, 3.0, 3.0)
        )
        self.assertEqual(clipped.shape, (0, 2))

    def test_resample_polyline(self):
        """Uniform spacing along a straight line, end point included."""

        out = output_points.resample_polyline(np.array([[0.0, 0.0], [1.0, 0.0]]), 0.25)
        np.testing.assert_allclose(out[:, 0], [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_mesh_nodes_in_polygon(self):
        """Only (wet) nodes inside the polygon are returned."""

        mesh = xr.Dataset(
            {
                "mesh2d_node_x": ("n", [0.5, 0.6, 2.0]),
                "mesh2d_node_y": ("n", [0.5, 0.6, 2.0]),
                "mesh2d_node_z": ("n", [-3.0, 1.0, -5.0]),
            }
        )
        nodes = output_points.get_mesh_nodes_in_polygon(box(0, 0, 1, 1), mesh)
        self.assertEqual([n["node_index"] for n in nodes], [0, 1])
        wet = output_points.get_mesh_nodes_in_polygon(
            box(0, 0, 1, 1), mesh, wet_only=True
        )
        self.assertEqual([n["node_index"] for n in wet], [0])


def _synthetic_cases(n_cases: int = 60, n_sites: int = 5, seed: int = 0) -> xr.Dataset:
    """SnapWave-like cases: hs/dir/spr fields that depend smoothly on the forcing."""

    rng = np.random.default_rng(seed)
    tp = rng.uniform(4, 16, n_cases)
    wdir = rng.uniform(250, 340, n_cases)
    spr = rng.uniform(10, 40, n_cases)
    wl = rng.uniform(-1, 1, n_cases)
    gain = np.linspace(0.3, 0.9, n_sites)
    hs = np.clip(gain[None, :] * (0.5 + tp[:, None] / 32) + 0.05 * wl[:, None], 0, None)
    hs[:, 0] = np.nan  # a dry site, dropped by the PCA NaN threshold
    sites = [f"s{i}" for i in range(n_sites)]
    return xr.Dataset(
        {
            "hs": (("case_num", "sites"), hs),
            "dir": (("case_num", "sites"), np.repeat(wdir[:, None], n_sites, 1) - 5),
            "spr": (("case_num", "sites"), np.repeat(spr[:, None], n_sites, 1) * 0.8),
            "tp_forcing": (("case_num",), tp),
            "dir_forcing": (("case_num",), wdir),
            "spr_forcing": (("case_num",), spr),
            "wl_forcing": (("case_num",), wl),
        },
        coords={"case_num": np.arange(n_cases), "sites": sites},
    )


VARS = {
    "hs": {"vars_to_stack": ["hs"], "pca_variance": 0.999},
    "tp": "raw",
    "dir": {"vars_to_stack": ["dir_u", "dir_v"], "pca_variance": 0.999},
    "spr": {"vars_to_stack": ["spr"], "pca_variance": 0.999},
}


class TestMetamodel(unittest.TestCase):
    """Tests for the per-goal MDA -> PCA -> GP metamodel."""

    def setUp(self):
        """Synthetic training cases and their forcing table."""

        self.cases = metamodel.add_direction_components(_synthetic_cases())
        self.forcing = self.cases[
            ["tp_forcing", "dir_forcing", "spr_forcing", "wl_forcing"]
        ].to_dataframe()
        self.forcing.columns = ["tp", "dir", "spr", "wl"]

    def test_site_filter(self):
        """Sites breaking a {var}_max rule are listed and dropped."""

        bad = metamodel.detect_removed_sites(self.cases, {"hs_max": 0.75})
        self.assertIn("s4", bad)
        self.assertNotIn("s1", bad)
        kept = metamodel.drop_sites(self.cases, bad)
        self.assertEqual(kept.sizes["sites"], 5 - len(bad))

    def test_fit_and_predict(self):
        """Fit on MDA centroids, predict held-out cases with floors applied."""

        mda, train_idx, test_idx = metamodel.fit_mda(self.forcing, 30, ["dir"])
        self.assertEqual(len(train_idx) + len(test_idx), 60)
        targets = metamodel.pca_targets(VARS)
        self.assertEqual([t[0] for t in targets], ["hs", "dir", "spr"])
        train = self.cases.isel(case_num=train_idx)
        pcas, summary = metamodel.fit_pcas(train, targets)
        self.assertNotIn("s0", summary["pca_sites_kept"]["hs"])
        gp = metamodel.fit_gp(
            mda.centroids[["tp", "dir", "spr", "wl"]], pcas, ["dir"], epochs=200
        )
        sites = summary["pca_sites_kept"]["hs"]
        test_forcing = self.forcing.iloc[test_idx]
        pred = metamodel.predict_fields(gp, pcas, test_forcing, VARS, sites)
        self.assertEqual(
            dict(pred.sizes), {"case_num": len(test_idx), "sites": len(sites)}
        )
        self.assertGreaterEqual(float(pred.hs.min()), metamodel.MIN_HS)
        self.assertGreaterEqual(
            float(pred.spr.min()), metamodel.MIN_DIRECTIONAL_SPREAD_DEG
        )
        np.testing.assert_allclose(pred.tp.isel(sites=0), test_forcing.tp)
        true_hs = self.cases.hs.isel(case_num=test_idx).sel(sites=sites)
        self.assertLess(float(np.abs(pred.hs - true_hs.values).max()), 0.05)

    def test_goal_forcing_and_valid_steps(self):
        """Partition forcing is renamed; empty partitions and off-sector steps drop."""

        time = pd.date_range("2020", periods=3, freq="h")
        goal = xr.Dataset(
            {
                "phs1": ("time", [1.0, 0.0, 2.0]),
                "ptp1": ("time", [8.0, 9.0, 10.0]),
                "pdir1": ("time", [300.0, 300.0, 90.0]),
                "pspr1": ("time", [20.0, 20.0, 20.0]),
            },
            coords={"time": time},
        )
        df, hs = metamodel.goal_forcing(goal, partition=1, wl=0.5)
        self.assertEqual(list(df.columns), ["tp", "dir", "spr", "wl"])
        self.assertEqual(float(df.wl.iloc[0]), 0.5)
        valid = metamodel.valid_timesteps(df, hs, 1, sector=(250.0, 350.0))
        self.assertEqual(valid.tolist(), [True, False, False])


class TestReconstruction(unittest.TestCase):
    """Tests for the linear spectral summation."""

    def _cube(self, hs_values):
        time = pd.date_range("2020", periods=2, freq="h")
        shape = (len(hs_values), 1, 2, 2)
        return xr.Dataset(
            {
                "hs": (
                    ("goal", "partition", "time", "sites"),
                    np.broadcast_to(np.array(hs_values)[:, None, None, None], shape),
                ),
                "tp": (("goal", "partition", "time", "sites"), np.full(shape, 10.0)),
                "dir": (("goal", "partition", "time", "sites"), np.full(shape, 300.0)),
                "spr": (("goal", "partition", "time", "sites"), np.full(shape, 20.0)),
            },
            coords={
                "goal": np.arange(1, len(hs_values) + 1),
                "partition": [-1],
                "time": time,
                "sites": ["a", "b"],
            },
        )

    def test_energy_adds_up(self):
        """Two goals with the same spectrum shape: Hs adds in quadrature."""

        result, spectra = reconstruction.sum_partition_spectra(
            self._cube([1.0, 1.0]),
            ["a", "b"],
            {"hs": "hs", "dpm": "dpm"},
            save_spectra=True,
        )
        np.testing.assert_allclose(result.hs, np.sqrt(2.0), rtol=0.02)
        self.assertIn("efth", spectra)
        single, _ = reconstruction.sum_partition_spectra(
            self._cube([1.0]), ["a"], {"hs": "hs"}
        )
        np.testing.assert_allclose(single.hs, 1.0, rtol=0.02)

    def test_batched_equals_single_call(self):
        """Site batching gives the same result as one call."""

        cube = self._cube([1.0, 2.0])
        chunks = [cube.isel(goal=[i]) for i in range(2)]
        batched, _ = reconstruction.sum_partition_spectra_batched(
            chunks, ["a", "b"], 2, 2, {"hs": "hs"}, budget_bytes=1
        )
        single, _ = reconstruction.sum_partition_spectra(cube, ["a", "b"], {"hs": "hs"})
        xr.testing.assert_allclose(batched, single)

    def test_partitions_and_batch_size(self):
        """Partition ids per mode, and a batch size that shrinks with the problem."""

        self.assertEqual(
            reconstruction.partition_ids(xr.Dataset(), spectral=False), [-1]
        )
        with self.assertRaises(ValueError):
            reconstruction.partition_ids(xr.Dataset(), spectral=True)
        big = reconstruction.site_batch_size(24, 4)
        self.assertGreater(big, reconstruction.site_batch_size(24 * 30, 40))
        self.assertEqual(
            reconstruction.site_batch_size(10**6, 10**3, budget_bytes=1), 1
        )


if __name__ == "__main__":
    unittest.main()
