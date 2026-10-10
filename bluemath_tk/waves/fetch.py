"""
Fetch over a triangular mesh: how far the wind blows over water to reach a point.

For each point and wind direction, a ray is walked upwind (toward the direction
the wind comes from) in fixed steps until it leaves the mesh. That distance is
the fetch inside the model domain. The table also keeps the mean depth along
the ray and, given the open (sea) boundary, whether the ray left the domain
through it or hit land.

:func:`effective_fetch` adds Saville's effective fetch: a cosine-weighted
mean of the fetches within a sector around each direction, which is less
sensitive than a single ray to islands, headlands and narrow openings.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from ..core.constants import EARTH_RADIUS

#: Metres per degree of latitude.
METRES_PER_DEGREE = np.deg2rad(1.0) * EARTH_RADIUS * 1e3


def _local_metres(
    x: np.ndarray, y: np.ndarray, lat0: float, geographic: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates in metres (equirectangular around *lat0* if geographic)."""

    if not geographic:
        return x, y

    return (
        x * METRES_PER_DEGREE * np.cos(np.deg2rad(lat0)),
        y * METRES_PER_DEGREE,
    )


def _densify(polyline: np.ndarray, spacing: float) -> np.ndarray:
    """Vertices of *polyline* plus points every *spacing* along each segment."""

    out = [polyline[:1]]
    for a, b in zip(polyline[:-1], polyline[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / spacing)))
        out.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))

    return np.vstack(out)


def fetch_table(
    points: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    triangles: np.ndarray,
    node_depth: np.ndarray,
    directions: np.ndarray | None = None,
    step: float = 250.0,
    max_distance: float = 500e3,
    geographic: bool = True,
    open_boundary: np.ndarray | None = None,
    open_tolerance: float = 5e3,
) -> xr.Dataset:
    """
    Fetch inside a triangular mesh for every point and wind direction.

    Parameters
    ----------
    points : np.ndarray
        ``(n, 2)`` point coordinates (x/lon, y/lat).
    node_x, node_y : np.ndarray
        Mesh node coordinates.
    triangles : np.ndarray
        ``(n_faces, 3)`` 0-based node indices.
    node_depth : np.ndarray
        Depth at the nodes (m, positive down).
    directions : np.ndarray, optional
        Wind directions (deg, nautical, coming from). Default every 5 deg.
    step : float, optional
        Ray step (m): the fetch resolution. Default is 250 m.
    max_distance : float, optional
        Longest ray (m). Default is 500 km.
    geographic : bool, optional
        Coordinates are lon/lat degrees (default) or metres.
    open_boundary : np.ndarray, optional
        ``(m, 2)`` polyline along the open (sea) boundary, e.g. the goals in
        order or the open-boundary node string of the grid. When given, rays
        that leave the mesh within *open_tolerance* of it are flagged.
    open_tolerance : float, optional
        Distance (m) from *open_boundary* within which a ray exit counts as
        open sea. Default is 5 km.

    Returns
    -------
    xr.Dataset
        On ``(point, dir)``: ``fetch`` (m; ``max_distance`` if the ray never
        leaves the mesh, NaN for points outside it), ``mean_depth`` (m, depth
        averaged along the ray, the point included) and, with
        *open_boundary*, ``open_exit`` (bool). ``point_x``, ``point_y`` and
        ``point_depth`` are coordinates on ``point``.
    """

    import matplotlib.tri as mtri
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if directions is None:
        directions = np.arange(0.0, 360.0, 5.0)
    directions = np.asarray(directions, dtype=float)
    tri = mtri.Triangulation(node_x, node_y, np.asarray(triangles, dtype=int))
    finder = tri.get_trifinder()
    depth_at = mtri.LinearTriInterpolator(tri, np.asarray(node_depth, dtype=float))

    px, py = points[:, 0], points[:, 1]
    point_depth = np.asarray(depth_at(px, py).filled(np.nan))
    inside_mesh = finder(px, py) >= 0
    distances = np.arange(step, max_distance + step, step)
    n_steps = len(distances)
    lat0 = float(np.nanmean(py)) if geographic else 0.0

    tree = None
    if open_boundary is not None:
        ob = np.asarray(open_boundary, dtype=float).reshape(-1, 2)
        obx, oby = _local_metres(ob[:, 0], ob[:, 1], lat0, geographic)
        tree = cKDTree(_densify(np.c_[obx, oby], step))

    if geographic:
        dx_unit = 1.0 / (METRES_PER_DEGREE * np.cos(np.deg2rad(py)))
        dy_unit = np.full_like(py, 1.0 / METRES_PER_DEGREE)
    else:
        dx_unit = dy_unit = np.ones_like(px)

    shape = (len(points), len(directions))
    fetch = np.full(shape, np.nan)
    mean_depth = np.full(shape, np.nan)
    open_exit = np.zeros(shape, dtype=bool)
    rows = np.arange(len(points))
    for j, theta in enumerate(directions):
        # coming from theta: walk toward the bearing theta
        sx = np.sin(np.deg2rad(theta)) * distances
        sy = np.cos(np.deg2rad(theta)) * distances
        rx = px[:, None] + sx[None] * dx_unit[:, None]
        ry = py[:, None] + sy[None] * dy_unit[:, None]
        inside = (finder(rx.ravel(), ry.ravel()) >= 0).reshape(rx.shape)
        n_in = np.where(inside.all(axis=1), n_steps, np.argmin(inside, axis=1))
        fetch[:, j] = n_in * step

        ray_depth = np.asarray(depth_at(rx.ravel(), ry.ravel()).filled(np.nan))
        ray_depth = ray_depth.reshape(rx.shape)
        ray_depth[np.arange(n_steps)[None] >= n_in[:, None]] = np.nan
        with warnings.catch_warnings():
            # points outside the mesh have no depth at all
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_depth[:, j] = np.nanmean(
                np.c_[point_depth, np.clip(ray_depth, 0.0, None)], axis=1
            )

        if tree is not None:
            last = np.clip(n_in - 1, 0, n_steps - 1)
            ex = np.where(n_in > 0, rx[rows, last], px)
            ey = np.where(n_in > 0, ry[rows, last], py)
            exm, eym = _local_metres(ex, ey, lat0, geographic)
            gap, _ = tree.query(np.c_[exm, eym])
            open_exit[:, j] = (gap <= open_tolerance) & (n_in < n_steps)

    fetch[~inside_mesh] = np.nan
    mean_depth[~inside_mesh] = np.nan
    data = {
        "fetch": (("point", "dir"), fetch, {"units": "m"}),
        "mean_depth": (("point", "dir"), mean_depth, {"units": "m"}),
    }
    if tree is not None:
        open_exit[~inside_mesh] = False
        data["open_exit"] = (("point", "dir"), open_exit)

    return xr.Dataset(
        data,
        coords={
            "dir": ("dir", directions, {"units": "deg, nautical, coming from"}),
            "point_x": ("point", px),
            "point_y": ("point", py),
            "point_depth": ("point", point_depth, {"units": "m"}),
        },
        attrs={"step": step, "max_distance": max_distance},
    )


def effective_fetch(
    table: xr.Dataset, half_width: float = 45.0, power: int = 2
) -> xr.DataArray:
    """
    Saville's effective fetch: ``sum F_i cos^p a_i / sum cos a_i`` within ``+-half_width``.

    Parameters
    ----------
    table : xr.Dataset
        Output of :func:`fetch_table`, with directions evenly spaced over
        the full circle.
    half_width : float, optional
        Half sector width (deg). Default is 45.
    power : int, optional
        Cosine power on the fetches. Default is 2 (Saville).

    Returns
    -------
    xr.DataArray
        Effective fetch (m) on ``(point, dir)``.
    """

    dirs = table["dir"].values
    spacing = 360.0 / len(dirs)
    if not np.allclose(np.mod(np.diff(dirs), 360.0), spacing):
        raise ValueError("effective_fetch needs evenly spaced directions over 360 deg")

    k = int(np.floor(half_width / spacing))
    offsets = np.arange(-k, k + 1)
    cos = np.cos(np.deg2rad(offsets * spacing))
    fetch = table["fetch"].values
    shifted = np.stack(
        [np.roll(fetch, -int(o), axis=1) for o in offsets], axis=-1
    )  # shifted[..., m][:, j] = fetch at direction j + offsets[m]
    eff = (shifted * cos**power).sum(axis=-1) / cos.sum()

    return xr.DataArray(
        eff, dims=("point", "dir"), coords=table["fetch"].coords, attrs={"units": "m"}
    ).rename("fetch_eff")
