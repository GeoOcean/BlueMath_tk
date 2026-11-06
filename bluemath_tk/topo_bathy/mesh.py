"""
Unstructured model meshes: one reader, depth at points, and a raster of the mesh.

A wave model runs on its own mesh, which need not cover the same area as the
bathymetry raster used to draw contours. :func:`read_mesh` reads the usual
formats into one :class:`Mesh` (elevation, positive up, as GEBCO):

- UGRID netCDF (``mesh2d_*``): triangles, quads or a mix (e.g. a D-Flow FM
  quadtree), with fill values for the missing nodes of smaller faces;
- ADCIRC ``fort.14`` / ``.grd`` (depth positive down in the file), used by SWAN.

:func:`depth_at_points` interpolates the mesh depth linearly inside its
triangles (NaN outside the mesh), :func:`open_boundary_nodes` finds the open
(sea) part of the mesh outline from points on it, :func:`mesh_to_raster` samples the mesh on
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


def outline_rings(mesh: Mesh) -> list[np.ndarray]:
    """
    Closed outlines of the mesh: node indices in order along each ring.

    Parameters
    ----------
    mesh : Mesh
        Model mesh.

    Returns
    -------
    list of np.ndarray
        One array of node indices per ring (outer boundary and islands).
    """

    tri = mesh.triangles
    edges = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    neighbours: dict[int, list[int]] = {}
    for a, b in unique[counts == 1]:
        neighbours.setdefault(int(a), []).append(int(b))
        neighbours.setdefault(int(b), []).append(int(a))

    rings, seen = [], set()
    for start in neighbours:
        if start in seen:
            continue
        ring, previous, current = [start], None, start
        seen.add(start)
        while True:
            following = next(n for n in neighbours[current] if n != previous)
            if following == start or following in seen:
                break
            ring.append(following)
            seen.add(following)
            previous, current = current, following
        rings.append(np.asarray(ring))

    return rings


def _turn_angle(xy: np.ndarray, before: int, node: int, after: int) -> float:
    """Absolute change of direction (deg) at *node* walking before -> node -> after."""

    v1, v2 = xy[node] - xy[before], xy[after] - xy[node]
    cross = v1[0] * v2[1] - v1[1] * v2[0]

    return abs(float(np.degrees(np.arctan2(cross, v1 @ v2))))


def open_boundary_nodes(
    mesh: Mesh, points: np.ndarray, max_turn: float = 45.0
) -> np.ndarray:
    """
    Open (sea) boundary of a mesh: the outline through *points*, out to the coast.

    The outline arc from the first to the last point (passing every point, in
    order) is extended at both ends along the outline over two corners
    sharper than *max_turn*: past the first (the end of the offshore boundary,
    where the lateral boundary starts) and up to the second (where the
    lateral boundary meets the coast). So the first and last points must sit
    on the offshore boundary, at most at its corners (e.g. HyWaves goals).

    Parameters
    ----------
    mesh : Mesh
        Model mesh.
    points : np.ndarray
        ``(n, 2)`` points on the open boundary, in order along it.
    max_turn : float, optional
        Change of direction (deg) that ends the boundary. Default is 45.

    Returns
    -------
    np.ndarray
        Node indices of the open boundary, in order along it.

    Raises
    ------
    ValueError
        If the points are not in order along one outline.
    """

    points = np.asarray(points, dtype=float).reshape(-1, 2)
    scale = np.cos(np.radians(np.nanmean(mesh.y))) if mesh.geographic else 1.0
    nodes_xy = np.column_stack([mesh.x * scale, mesh.y])
    points_xy = points * [scale, 1.0]

    rings = outline_rings(mesh)
    ring_of = np.concatenate([np.full(len(r), k) for k, r in enumerate(rings)])
    place = np.concatenate([np.arange(len(r)) for r in rings])
    outline_xy = nodes_xy[np.concatenate(rings)]
    nearest = [int(np.argmin(((outline_xy - p) ** 2).sum(1))) for p in points_xy]
    if len(set(ring_of[nearest])) != 1:
        raise ValueError("the points are on different mesh outlines")
    ring = rings[ring_of[nearest[0]]]
    xy = nodes_xy[ring]
    n = len(ring)
    position = [int(place[i]) for i in nearest]

    step = 1
    if [(p - position[0]) % n for p in position] != sorted((p - position[0]) % n for p in position):
        step = -1
        if [(position[0] - p) % n for p in position] != sorted((position[0] - p) % n for p in position):
            raise ValueError("the points are not in order along the mesh outline")
    arc = [position[0]]
    while arc[-1] != position[-1]:
        arc.append((arc[-1] + step) % n)

    def extend(end: int, inner: int, direction: int) -> list[int]:
        added, previous, current, corners = [], inner, end, 0
        while len(arc) + len(added) < n:
            following = (current + direction) % n
            if previous != current and _turn_angle(xy, previous, current, following) > max_turn:
                corners += 1
                if corners == 2:
                    break
            added.append(following)
            previous, current = current, following
        return added

    head = extend(arc[0], arc[1] if len(arc) > 1 else arc[0], -step)
    tail = extend(arc[-1], arc[-2] if len(arc) > 1 else arc[-1], step)

    return ring[np.asarray(head[::-1] + arc + tail)]


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
