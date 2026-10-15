"""
Unstructured model meshes: one reader, depth at points, and a raster of the mesh.

A wave model runs on its own mesh, which need not cover the same area as the
bathymetry raster used to draw contours. :func:`read_mesh` reads the usual
formats into one :class:`Mesh` (elevation, positive up, as GEBCO):

- UGRID netCDF (``mesh2d_*``): triangles, quads or a mix (e.g. a D-Flow FM
  quadtree), with fill values for the missing nodes of smaller faces;
- ADCIRC ``fort.14`` / ``.grd`` (depth positive down in the file), used by SWAN.

:func:`depth_at_points` interpolates the mesh depth linearly inside its
triangles (NaN outside the mesh), :func:`mesh_to_raster` samples the mesh on
a regular grid (NaN outside), and :func:`mask_raster_to_mesh` blanks a raster
outside the mesh, so contours traced on it stay inside the model domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


@dataclass
class Mesh:
    """
    An unstructured mesh in one convention.

    Attributes
    ----------
    x, y : np.ndarray
        Node coordinates (lon/lat degrees or projected metres).
    faces : np.ndarray
        ``(n_faces, max_nodes)`` 0-based node indices, ``-1`` where a face has
        fewer nodes (triangles in a quad mesh).
    z : np.ndarray
        Node elevation (m, positive up; negative below the reference level).
    geographic : bool
        True for lon/lat coordinates.
    open_boundary : np.ndarray or None
        ``(n, 2)`` polyline of the open (sea) boundary when the format has one
        (ADCIRC), else None.
    """

    x: np.ndarray
    y: np.ndarray
    faces: np.ndarray
    z: np.ndarray
    geographic: bool = True
    open_boundary: np.ndarray | None = None

    @property
    def depth(self) -> np.ndarray:
        """Node depth (m, positive down)."""

        return -self.z

    @property
    def triangles(self) -> np.ndarray:
        """``(n, 3)`` triangles: faces with more than three nodes split as fans."""

        faces = np.asarray(self.faces, dtype=int)
        tris = []
        for k in range(2, faces.shape[1]):
            sel = faces[:, k] >= 0
            tris.append(faces[sel][:, [0, k - 1, k]])

        return np.vstack(tris)

    def triangulation(self):
        """Build a :class:`matplotlib.tri.Triangulation` of :attr:`triangles`."""

        import matplotlib.tri as mtri

        return mtri.Triangulation(self.x, self.y, self.triangles)


def _looks_geographic(x: np.ndarray, y: np.ndarray) -> bool:
    return bool(np.nanmax(np.abs(x)) <= 360 and np.nanmax(np.abs(y)) <= 90)


def _read_ugrid(path: Path) -> Mesh:
    with xr.open_dataset(path) as ds:
        faces_var = ds["mesh2d_face_nodes"]
        start = int(faces_var.attrs.get("start_index", 0))
        values = np.asarray(faces_var.values, dtype=float)
        missing = ~np.isfinite(values) | (values < start)
        faces = np.where(missing, -1, values - start).astype(int)
        x = ds["mesh2d_node_x"].values.astype(float)
        y = ds["mesh2d_node_y"].values.astype(float)
        z = ds["mesh2d_node_z"].values.astype(float)
        units = str(ds["mesh2d_node_x"].attrs.get("units", ""))
    geographic = "degree" in units or _looks_geographic(x, y)

    return Mesh(x=x, y=y, faces=faces, z=z, geographic=geographic)


def _read_adcirc(path: Path) -> Mesh:
    from ..wrappers.swan.swan_utils import read_adcirc_grid, swan_boundary_side

    grid = read_adcirc_grid(str(path))
    v = grid["vertices"]
    open_boundary = swan_boundary_side(grid)[0] if grid["open_boundaries"] else None

    return Mesh(
        x=v[:, 0],
        y=v[:, 1],
        faces=np.asarray(grid["triangles"], dtype=int),
        z=-v[:, 2],
        geographic=_looks_geographic(v[:, 0], v[:, 1]),
        open_boundary=open_boundary,
    )


def read_mesh(path: str | Path) -> Mesh:
    """
    Read a model mesh: UGRID netCDF (``.nc``) or ADCIRC (``fort.14``, ``.grd``).

    Parameters
    ----------
    path : str or Path
        Mesh file.

    Returns
    -------
    Mesh
        Nodes, faces (0-based) and elevation (positive up).

    Raises
    ------
    ValueError
        If the format is not recognised.
    """

    path = Path(path)
    if path.suffix == ".nc":
        return _read_ugrid(path)
    if path.suffix in (".grd", ".14") or path.name == "fort.14":
        return _read_adcirc(path)

    raise ValueError(f"Unknown mesh format: {path.name} (expected .nc, fort.14, .grd)")


def depth_at_points(mesh: Mesh, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Mesh depth at points, linear inside the triangles.

    Parameters
    ----------
    mesh : Mesh
        Model mesh.
    x, y : np.ndarray
        Point coordinates, in the mesh coordinates.

    Returns
    -------
    np.ndarray
        Depth (m, positive down) at each point; NaN outside the mesh.
    """

    import matplotlib.tri as mtri

    interp = mtri.LinearTriInterpolator(mesh.triangulation(), mesh.depth)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    return np.asarray(interp(x, y).filled(np.nan), dtype=float)


def _median_edge(mesh: Mesh) -> float:
    tri = mesh.triangles
    xy = np.c_[mesh.x, mesh.y]

    return float(np.median(np.hypot(*(xy[tri[:, 1]] - xy[tri[:, 0]]).T)))


def _raster(
    values: np.ndarray, x: np.ndarray, y: np.ndarray, geographic: bool
) -> xr.DataArray:
    xname, yname = ("lon", "lat") if geographic else ("x", "y")
    da = xr.DataArray(
        values,
        dims=(yname, xname),
        coords={yname: y, xname: x},
        name="elevation",
        attrs={"units": "m", "positive": "up"},
    )
    if geographic:
        try:
            da = da.rio.set_spatial_dims(x_dim=xname, y_dim=yname).rio.write_crs(
                "EPSG:4326"
            )
        except AttributeError:  # rioxarray not installed
            pass

    return da


def mesh_to_raster(
    mesh: Mesh,
    resolution: float | None = None,
    bounds: tuple[float, float, float, float] | None = None,
) -> xr.DataArray:
    """
    Sample a mesh's elevation on a regular grid.

    Parameters
    ----------
    mesh : Mesh
        Model mesh.
    resolution : float, optional
        Cell size, in the mesh coordinates. Default: the median edge length
        of the mesh.
    bounds : tuple, optional
        ``(xmin, ymin, xmax, ymax)``. Default: the mesh extent.

    Returns
    -------
    xr.DataArray
        Elevation (m, positive up) on ``(lat, lon)`` (geographic, with an
        EPSG:4326 CRS when rioxarray is installed) or ``(y, x)``, north up;
        NaN outside the mesh. ``.rio.to_raster(path)`` writes a GeoTIFF.
    """

    import matplotlib.tri as mtri

    if resolution is None:
        resolution = _median_edge(mesh)
    xmin, ymin, xmax, ymax = bounds or (
        mesh.x.min(),
        mesh.y.min(),
        mesh.x.max(),
        mesh.y.max(),
    )
    x = np.arange(xmin + resolution / 2, xmax, resolution)
    y = np.arange(ymax - resolution / 2, ymin, -resolution)
    interp = mtri.LinearTriInterpolator(mesh.triangulation(), mesh.z)
    xx, yy = np.meshgrid(x, y)
    values = np.asarray(interp(xx, yy).filled(np.nan), dtype=float)

    return _raster(values, x, y, mesh.geographic)


def mask_raster_to_mesh(da: xr.DataArray, mesh: Mesh) -> xr.DataArray:
    """
    Set a raster to NaN outside a mesh.

    Contours traced on the result (and smoothing with
    :func:`bluemath_tk.waves.hywaves.output_points.smooth_raster`, which
    respects NaNs) then stay inside the model domain.

    Parameters
    ----------
    da : xr.DataArray
        2-D raster on ``(y, x)``, ``(lat, lon)`` or ``(latitude, longitude)``,
        in the mesh coordinates.
    mesh : Mesh
        Model mesh.

    Returns
    -------
    xr.DataArray
        A copy, NaN where the cell centre is outside the mesh.
    """

    from ..waves.hywaves.output_points import xy_coord_names

    xname, yname = xy_coord_names(da)
    xx, yy = np.meshgrid(da[xname].values, da[yname].values)
    inside = mesh.triangulation().get_trifinder()(xx, yy) >= 0
    if da.dims.index(xname) == 0:
        inside = inside.T

    return da.where(inside)
