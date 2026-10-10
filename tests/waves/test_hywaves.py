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
        """Points along the -10 m contour of a projected slope (x/y in metres)."""

        x = np.linspace(0.0, 50_000.0, 51)
        y = np.linspace(0.0, 40_000.0, 41)
        da = xr.DataArray(
            np.tile(-20.0 * x / x.max(), (y.size, 1)),
            coords={"y": y, "x": x},
            dims=("y", "x"),
        )
        self.assertFalse(output_points.is_geographic(da))
        pts = output_points.get_isobath_points(da, -10.0, spacing_m=4000.0)
        np.testing.assert_allclose(pts[:, 0], 25_000.0, atol=1e-6)
        np.testing.assert_allclose(np.diff(pts[:-1, 1]), 4000.0)
        clipped = output_points.get_isobath_points(
            da, -10.0, aoi_polygon=box(0.0, 0.0, 1000.0, 1000.0)
        )
        self.assertEqual(clipped.shape, (0, 2))

    def test_resample_polyline(self):
        """Uniform spacing along a straight line, end point included."""

        out = output_points.resample_polyline(
            np.array([[0.0, 0.0], [1000.0, 0.0]]), 250.0, geographic=False
        )
        np.testing.assert_allclose(out[:, 0], [0.0, 250.0, 500.0, 750.0, 1000.0])

    def _lonlat_slope(self, bumps: bool = False):
        """Lon/lat raster at 52N: depth increasing eastwards, optional pits."""

        lon = np.linspace(4.0, 5.0, 201)
        lat = np.linspace(52.0, 52.5, 101)
        z = np.tile(-40.0 * (lon - 4.0), (lat.size, 1))
        if bumps:  # small pits (about 1 km) deeper than -10 m on the shallow side
            for lo, la in ((4.12, 52.1), (4.12, 52.3), (4.15, 52.4)):
                z -= 30.0 * np.exp(
                    -(
                        ((lon[None, :] - lo) / 0.006) ** 2
                        + ((lat[:, None] - la) / 0.004) ** 2
                    )
                )
        return xr.DataArray(z, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))

    def test_spacing_in_metres_on_lonlat(self):
        """Points along a meridian contour are ~2 km apart (great circle)."""

        da = self._lonlat_slope()
        self.assertTrue(output_points.is_geographic(da))
        pts = output_points.get_isobath_points(da, -10.0, spacing_m=2000.0)
        gaps = [
            output_points.polyline_length_m(pts[i : i + 2]) for i in range(len(pts) - 2)
        ]
        np.testing.assert_allclose(gaps, 2000.0, rtol=1e-3)

    def test_smoothing_removes_small_loops(self):
        """Pits add closed -10 m contours; smoothing or min length removes them."""

        da = self._lonlat_slope(bumps=True)
        raw = output_points.get_isobath_points(da, -10.0, spacing_m=500.0)
        clean = output_points.get_isobath_points(
            da, -10.0, spacing_m=500.0, smooth_m=2000.0
        )
        filtered = output_points.get_isobath_points(
            da, -10.0, spacing_m=500.0, min_length_m=20_000.0
        )
        self.assertTrue((raw[:, 0] < 4.2).any())  # points around the pits
        self.assertFalse((clean[:, 0] < 4.2).any())
        self.assertFalse((filtered[:, 0] < 4.2).any())
        np.testing.assert_allclose(filtered[:, 0], 4.25, atol=1e-6)

    def test_smooth_raster_keeps_nans(self):
        """NaN cells stay NaN and do not leak into their neighbours."""

        da = self._lonlat_slope()
        da[:, :50] = np.nan
        smooth = output_points.smooth_raster(da, 1000.0)
        self.assertTrue(np.isnan(smooth[:, :50]).all())
        self.assertTrue(np.isfinite(smooth[:, 50:]).all())
        np.testing.assert_allclose(
            smooth[:, 100], da[:, 100], atol=0.05
        )  # linear field

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
    """SnapWave-like cases: hs/dir/spr fields that depend smoothly on the forcing.

    Site ``hs`` grows with the offshore Hs and is depth-limited (capped) at the
    last site, as breaking would do.
    """

    rng = np.random.default_rng(seed)
    hs_off = rng.uniform(0.5, 4, n_cases)
    tp = rng.uniform(4, 16, n_cases)
    wdir = rng.uniform(250, 340, n_cases)
    spr = rng.uniform(10, 40, n_cases)
    wl = rng.uniform(-1, 1, n_cases)
    gain = np.linspace(0.3, 0.9, n_sites)
    hs = gain[None, :] * hs_off[:, None] * (0.5 + tp[:, None] / 32)
    hs = np.clip(hs + 0.05 * wl[:, None], 0, None)
    hs[:, -1] = np.minimum(hs[:, -1], 1.5)
    hs[:, 0] = np.nan  # a dry site, dropped by the PCA NaN threshold
    sites = [f"s{i}" for i in range(n_sites)]
    return xr.Dataset(
        {
            "hs": (("case_num", "sites"), hs),
            "dir": (("case_num", "sites"), np.repeat(wdir[:, None], n_sites, 1) - 5),
            "spr": (("case_num", "sites"), np.repeat(spr[:, None], n_sites, 1) * 0.8),
            "hs_forcing": (("case_num",), hs_off),
            "tp_forcing": (("case_num",), tp),
            "dir_forcing": (("case_num",), wdir),
            "spr_forcing": (("case_num",), spr),
            "wl_forcing": (("case_num",), wl),
        },
        coords={"case_num": np.arange(n_cases), "sites": sites},
    )


INPUTS = ["hs", "tp", "dir", "spr", "wl"]
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
            ["hs_forcing", "tp_forcing", "dir_forcing", "spr_forcing", "wl_forcing"]
        ].to_dataframe()
        self.forcing.columns = INPUTS

    def test_site_filter(self):
        """Sites breaking a {var}_max or {var}_ratio_max rule are dropped."""

        bad = metamodel.detect_removed_sites(self.cases, {"hs_max": 1.6})
        self.assertIn("s3", bad)
        self.assertNotIn("s4", bad)  # depth-limited at 1.5 m
        kept = metamodel.drop_sites(self.cases, bad)
        self.assertEqual(kept.sizes["sites"], 5 - len(bad))

        cases = self.cases.copy(deep=True)
        cases["hs"][3, 2] = 50 * float(cases.hs_forcing[3])  # numerical blow-up
        bad = metamodel.detect_removed_sites(cases, {"hs_ratio_max": 3.0})
        self.assertEqual(bad, ["s2"])

    def test_fit_and_predict(self):
        """Fit on MDA centroids, predict held-out cases with floors applied."""

        mda, train_idx, test_idx = metamodel.fit_mda(self.forcing, 30, ["dir"])
        self.assertEqual(len(train_idx) + len(test_idx), 60)
        targets = metamodel.pca_targets(VARS)
        self.assertEqual([t[0] for t in targets], ["hs", "dir", "spr"])
        train = self.cases.isel(case_num=train_idx)
        pcas, summary = metamodel.fit_pcas(train, targets)
        self.assertNotIn("s0", summary["pca_sites_kept"]["hs"])
        gp = metamodel.fit_gp(mda.centroids[INPUTS], pcas, ["dir"], epochs=200)
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
        self.assertLess(float(np.abs(pred.hs - true_hs.values).max()), 0.1)

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
        df = metamodel.goal_forcing(goal, partition=1, wl=0.5)
        self.assertEqual(list(df.columns), INPUTS)
        self.assertEqual(float(df.wl.iloc[0]), 0.5)
        valid = metamodel.valid_timesteps(df, sector=(250.0, 350.0))
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
