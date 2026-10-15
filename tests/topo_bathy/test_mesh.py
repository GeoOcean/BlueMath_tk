"""Tests for bluemath_tk.topo_bathy.mesh and the output-site gate."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from bluemath_tk.topo_bathy.mesh import (
    Mesh,
    depth_at_points,
    mask_raster_to_mesh,
    mesh_to_raster,
    read_mesh,
)
from bluemath_tk.waves.hywaves.output_points import (
    SITE_CAN_DRY,
    SITE_OK,
    SITE_OUTSIDE,
    site_status,
    spatial_subsample,
)


def _sloping_grid(n: int = 11) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nodes of [0, 1]^2 (degrees), index grid, elevation -20 x (deeper east)."""

    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    idx = np.arange(n * n).reshape(n, n)
    return x.ravel(), y.ravel(), idx, -20.0 * x.ravel()


def _quad_mesh() -> Mesh:
    """Quads on the west half, triangles on the east half, padded with -1."""

    x, y, idx, z = _sloping_grid()
    a, b = idx[:-1, :-1], idx[:-1, 1:]
    c, d = idx[1:, :-1], idx[1:, 1:]
    west = (slice(None), slice(0, 5))
    east = (slice(None), slice(5, None))
    quads = np.c_[a[west].ravel(), b[west].ravel(), d[west].ravel(), c[west].ravel()]
    tris = np.r_[
        np.c_[a[east].ravel(), b[east].ravel(), d[east].ravel()],
        np.c_[a[east].ravel(), d[east].ravel(), c[east].ravel()],
    ]
    faces = np.r_[quads, np.c_[tris, np.full(len(tris), -1)]]
    return Mesh(x=x, y=y, faces=faces, z=z)


class TestMesh(unittest.TestCase):
    """Depth at points and rasters from a mixed quad/triangle mesh."""

    def setUp(self):
        """Build the test data."""

        self.mesh = _quad_mesh()

    def test_triangles(self):
        """Each quad becomes two triangles; padded triangles stay one."""

        self.assertEqual(self.mesh.triangles.shape, (50 * 2 + 100, 3))
        self.assertGreaterEqual(self.mesh.triangles.min(), 0)

    def test_depth_at_points(self):
        """Linear depth inside the mesh (positive down), NaN outside."""

        depth = depth_at_points(self.mesh, [0.25, 0.75, 1.5], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(depth[:2], [5.0, 15.0])
        self.assertTrue(np.isnan(depth[2]))

    def test_mesh_to_raster(self):
        """Raster of the mesh elevation, north up, NaN outside the mesh."""

        da = mesh_to_raster(self.mesh, resolution=0.05, bounds=(-0.5, 0.0, 1.0, 1.0))
        self.assertEqual(da.dims, ("lat", "lon"))
        self.assertTrue(bool((np.diff(da.lat) < 0).all()))
        row = da.isel(lat=10)
        self.assertTrue(bool(row.sel(lon=slice(None, 0)).isnull().all()))
        inside = row.sel(lon=slice(0, None))
        np.testing.assert_allclose(inside, -20.0 * inside.lon, atol=1e-9)

    def test_mask_raster_to_mesh(self):
        """A larger raster keeps its values only inside the mesh."""

        lon = np.linspace(-0.5, 1.5, 41)
        lat = np.linspace(1.0, 0.0, 21)
        da = xr.DataArray(
            np.full((21, 41), -5.0),
            coords={"lat": lat, "lon": lon},
            dims=("lat", "lon"),
        )
        masked = mask_raster_to_mesh(da, self.mesh)
        inside = (da.lon >= 0) & (da.lon <= 1)
        self.assertTrue(bool(masked.where(~inside).isnull().all()))
        self.assertTrue(
            bool((masked.isel(lat=slice(1, -1)).where(inside, -5.0) == -5.0).all())
        )


class TestReadMesh(unittest.TestCase):
    """UGRID (with fill values) and ADCIRC give the same mesh."""

    def setUp(self):
        """Build the test data."""

        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.mesh = _quad_mesh()

    def tearDown(self):
        """Remove the temporary folder."""

        self.tmp.cleanup()

    def test_ugrid_quadtree(self):
        """1-based face nodes with fill values, elevation positive up."""

        faces = np.where(self.mesh.faces >= 0, self.mesh.faces + 1, -999)
        ds = xr.Dataset(
            {
                "mesh2d_node_x": ("n", self.mesh.x, {"units": "degrees_east"}),
                "mesh2d_node_y": ("n", self.mesh.y),
                "mesh2d_node_z": ("n", self.mesh.z),
                "mesh2d_face_nodes": (
                    ("f", "m"),
                    faces,
                    {"start_index": 1, "_FillValue": -999},
                ),
            }
        )
        path = self.dir / "mesh.nc"
        ds.to_netcdf(path)
        mesh = read_mesh(path)
        np.testing.assert_array_equal(mesh.faces, self.mesh.faces)
        self.assertTrue(mesh.geographic)
        np.testing.assert_allclose(depth_at_points(mesh, [0.5], [0.5]), [10.0])

    def test_adcirc(self):
        """ADCIRC depth is positive down in the file."""

        tri = self.mesh.triangles
        lines = ["EPSG:4326", f"{len(tri)} {len(self.mesh.x)}"]
        lines += [
            f"{i + 1} {x} {y} {-z}"
            for i, (x, y, z) in enumerate(zip(self.mesh.x, self.mesh.y, self.mesh.z))
        ]
        lines += [
            f"{i + 1} 3 {a + 1} {b + 1} {c + 1}" for i, (a, b, c) in enumerate(tri)
        ]
        lines += ["0", "0", "0", "0"]
        path = self.dir / "fort.14"
        path.write_text("\n".join(lines) + "\n")
        mesh = read_mesh(path)
        self.assertIsNone(mesh.open_boundary)
        np.testing.assert_allclose(mesh.z, self.mesh.z)
        np.testing.assert_allclose(depth_at_points(mesh, [0.5], [0.5]), [10.0])

    def test_unknown_format(self):
        """Other extensions are rejected."""

        with self.assertRaises(ValueError):
            read_mesh(self.dir / "mesh.msh")


class TestSiteStatus(unittest.TestCase):
    """Sites must be inside the mesh and wet at the lowest water level."""

    def test_status(self):
        """Deep, dry-at-low-water, just-wet and outside sites."""

        depth = np.array([10.0, 3.0, 3.7, np.nan])
        status = site_status(depth, wl_min=-3.5, min_depth=0.1)
        self.assertEqual(
            status.tolist(), [SITE_OK, SITE_CAN_DRY, SITE_OK, SITE_OUTSIDE]
        )
        self.assertEqual(site_status([0.05])[0], SITE_CAN_DRY)


class TestSpatialSubsample(unittest.TestCase):
    """A fraction of the points, spread over the whole area."""

    def setUp(self):
        """Build the test data."""

        rng = np.random.default_rng(1)
        # dense cluster in one corner of a sparse square (metres)
        dense = rng.uniform(0, 1000, (4000, 2))
        sparse = rng.uniform(0, 10000, (1000, 2))
        self.xy = np.r_[dense, sparse]

    def test_size_and_coverage(self):
        """About the requested size, and spread over the area."""

        idx = spatial_subsample(*self.xy.T, fraction=0.05, geographic=False)
        self.assertLess(abs(len(idx) - 250), 25)
        kept = self.xy[idx]
        in_dense = (kept < 1000).all(axis=1).mean()
        self.assertLess(in_dense, 0.2)  # 80 % of the input, a few % of the area
        quadrants = np.unique((kept // 5000).astype(int), axis=0)
        self.assertEqual(len(quadrants), 4)

    def test_reproducible(self):
        """Same seed, same subset; fraction bounds are checked."""

        a = spatial_subsample(*self.xy.T, fraction=0.1, seed=3, geographic=False)
        b = spatial_subsample(*self.xy.T, fraction=0.1, seed=3, geographic=False)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(spatial_subsample(*self.xy.T, fraction=1.0)), len(self.xy))
        with self.assertRaises(ValueError):
            spatial_subsample(*self.xy.T, fraction=0.0)


if __name__ == "__main__":
    unittest.main()
