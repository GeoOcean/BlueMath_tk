"""
HyWaves output points: where nearshore waves are computed and reconstructed.

Typical sources are points along depth contours of a bathymetry raster
(:func:`get_isobath_points`) and model-mesh nodes inside a polygon (thinned
evenly with :func:`spatial_subsample`), plus any observation sites (buoys).
Every site is then checked against the model mesh (:func:`site_status`: inside
it and never dry; depths from :func:`bluemath_tk.topo_bathy.mesh.depth_at_points`).
Sites are decided once, before any run: a site is kept only if its mesh depth
plus the lowest water level of the cases stays above the models' minimum
depth. Every kept site is then wet in every case, so ``hs = 0`` at a site is a
result (the goal sends no energy there), not a dry or failed point; see
:func:`bluemath_tk.waves.hywaves.metamodel.mask_zero_energy`.

Raw contours of a high-resolution bathymetry follow every small sandbank and
pit, which puts points in odd places. :func:`get_isobath_points` can smooth
the bathymetry first (:func:`smooth_raster`, Gaussian width in metres), drop
short contour pieces, and space points evenly in metres along the result
(:func:`polyline_length_m`, :func:`resample_polyline`). Geographic (lon/lat)
rasters are measured with great-circle distances.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import shapely
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon

from ...core.constants import EARTH_RADIUS

_X_NAMES = ("x", "lon", "longitude")
_Y_NAMES = ("y", "lat", "latitude")
EARTH_RADIUS_M = EARTH_RADIUS * 1000.0


def xy_coord_names(da: xr.DataArray) -> tuple[str, str]:
    """
    Horizontal coordinate names of a 2-D raster.

    Parameters
    ----------
    da : xr.DataArray
        Raster with ``x/y``, ``lon/lat`` or ``longitude/latitude`` coordinates.

    Returns
    -------
    tuple[str, str]
        ``(x_name, y_name)``.

    Raises
    ------
    KeyError
        If no recognised coordinate pair is found.
    """

    x = next((n for n in _X_NAMES if n in da.coords), None)
    y = next((n for n in _Y_NAMES if n in da.coords), None)
    if x is None or y is None:
        raise KeyError(
            f"Could not find x/y coords in {tuple(da.coords)}; "
            f"expected one of {_X_NAMES} and {_Y_NAMES}."
        )

    return x, y


def is_geographic(da: xr.DataArray) -> bool:
    """
    Whether a raster is in longitude/latitude degrees.

    Uses the rioxarray CRS when there is one, else the coordinate names
    (``lon``/``longitude``) or, for ``x``/``y``, value ranges within +-360/+-90.

    Parameters
    ----------
    da : xr.DataArray
        2-D raster.

    Returns
    -------
    bool
        True for geographic coordinates.
    """

    try:
        crs = da.rio.crs
    except Exception:
        crs = None
    if crs is not None:
        return bool(crs.is_geographic)
    xname, yname = xy_coord_names(da)
    if xname in ("lon", "longitude"):
        return True
    x, y = np.asarray(da[xname]), np.asarray(da[yname])

    return bool(np.nanmax(np.abs(x)) <= 360 and np.nanmax(np.abs(y)) <= 90)


def _segment_lengths(pts: np.ndarray, geographic: bool) -> np.ndarray:
    """Length of each polyline edge: metres (great circle) or raster units."""

    if not geographic:
        return np.linalg.norm(np.diff(pts, axis=0), axis=1)
    lon, lat = np.radians(pts[:, 0]), np.radians(pts[:, 1])
    a = (
        np.sin(np.diff(lat) / 2) ** 2
        + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2
    )

    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def polyline_length_m(pts: np.ndarray, geographic: bool = True) -> float:
    """
    Length of a polyline in metres.

    Parameters
    ----------
    pts : np.ndarray
        ``(N, 2)`` vertices, lon/lat degrees or projected metres.
    geographic : bool, optional
        Lon/lat input (great-circle distances). Default is True.

    Returns
    -------
    float
        Total length.
    """

    return float(_segment_lengths(np.asarray(pts, dtype=float), geographic).sum())


def resample_polyline(
    pts: np.ndarray, spacing_m: float, geographic: bool = True
) -> np.ndarray:
    """
    Resample a polyline every *spacing_m* metres of arc length.

    Parameters
    ----------
    pts : np.ndarray
        ``(N, 2)`` vertices: lon/lat degrees, or projected metres.
    spacing_m : float
        Distance between output points, in metres.
    geographic : bool, optional
        *pts* are lon/lat (great-circle arc length). Default is True.

    Returns
    -------
    np.ndarray
        ``(M, 2)`` points, including the start and (nearly) the end; empty when
        the polyline has zero length.
    """

    pts = np.asarray(pts, dtype=float)
    seg_len = _segment_lengths(pts, geographic)
    total = float(seg_len.sum())
    if total <= 0.0:
        return np.empty((0, 2), dtype=float)

    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    ds = np.arange(0.0, total, float(spacing_m))
    if ds.size == 0 or (total - ds[-1]) > 0.25 * spacing_m:
        ds = np.append(ds, total)

    out = np.empty((ds.size, 2), dtype=float)
    for k, d in enumerate(ds):
        i = int(np.searchsorted(cum, d, side="right") - 1)
        i = max(0, min(i, len(seg_len) - 1))
        t = 0.0 if seg_len[i] == 0 else (d - cum[i]) / seg_len[i]
        out[k] = pts[i] * (1.0 - t) + pts[i + 1] * t

    return out


def smooth_raster(da: xr.DataArray, sigma_m: float) -> xr.DataArray:
    """
    Gaussian-smooth a raster, with the filter width in metres and NaNs respected.

    NaN cells (e.g. outside a model domain) neither contribute to nor receive
    values: the filter is normalised by the smoothed valid-data mask, and the
    original NaNs are restored, so contours do not drift into missing areas.

    Parameters
    ----------
    da : xr.DataArray
        2-D raster on ``(y, x)``, lon/lat degrees or projected metres.
    sigma_m : float
        Gaussian standard deviation in metres (converted per axis to cells;
        for lon/lat rasters at the raster's mean latitude).

    Returns
    -------
    xr.DataArray
        Smoothed copy of *da*.
    """

    from scipy.ndimage import gaussian_filter

    xname, yname = xy_coord_names(da)
    x, y = np.asarray(da[xname], dtype=float), np.asarray(da[yname], dtype=float)
    dx, dy = abs(float(np.median(np.diff(x)))), abs(float(np.median(np.diff(y))))
    if is_geographic(da):
        metres_per_deg = np.pi / 180.0 * EARTH_RADIUS_M
        dy *= metres_per_deg
        dx *= metres_per_deg * np.cos(np.radians(float(np.nanmean(y))))
    sigma = (sigma_m / dy, sigma_m / dx)
    if da.dims.index(xname) == 0:
        sigma = sigma[::-1]

    values = np.asarray(da.values, dtype=float)
    valid = np.isfinite(values)
    weight = gaussian_filter(valid.astype(float), sigma, mode="nearest")
    smooth = gaussian_filter(np.where(valid, values, 0.0), sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = np.where(valid & (weight > 0), smooth / weight, np.nan)

    return da.copy(data=smooth)


def get_isobath_points(
    da: xr.DataArray,
    level: float,
    spacing_m: float | None = None,
    smooth_m: float | None = None,
    min_length_m: float | None = None,
    aoi_polygon: Polygon | MultiPolygon | None = None,
) -> np.ndarray:
    """
    Points along one depth contour of a bathymetry raster.

    Order of operations: smooth the raster (*smooth_m*), contour it, keep the
    pieces touching *aoi_polygon* and at least *min_length_m* long, then
    resample each piece every *spacing_m* metres.

    Parameters
    ----------
    da : xr.DataArray
        Topobathy raster (elevation, negative below sea level), dims ``(y, x)``.
    level : float
        Contour level, e.g. ``-10`` for the 10 m isobath.
    spacing_m : float, optional
        Resample each contour piece every *spacing_m* metres of arc length.
        Default is None (contour vertices as they are).
    smooth_m : float, optional
        Gaussian smoothing of the raster before contouring, in metres (see
        :func:`smooth_raster`). Removes small wiggles and tiny closed
        contours around sandbanks and pits. Default is None (no smoothing).
    min_length_m : float, optional
        Drop contour pieces shorter than this, in metres. Default is None.
    aoi_polygon : Polygon or MultiPolygon, optional
        Keep only contour pieces with at least one vertex inside it.

    Returns
    -------
    np.ndarray
        ``(N, 2)`` ``[x, y]`` points; empty when the contour does not exist.

    Raises
    ------
    ValueError
        If the raster shape does not match ``(len(y), len(x))``.
    """

    geographic = is_geographic(da)
    if smooth_m:
        da = smooth_raster(da, smooth_m)

    xname, yname = xy_coord_names(da)
    x = da[xname].values
    y = da[yname].values
    z = np.asarray(da.values, dtype=float)
    if z.shape != (y.size, x.size):
        raise ValueError(
            f"Expected da.shape == (len({yname}), len({xname})); "
            f"got {z.shape} vs ({y.size}, {x.size}). "
            f"Try da.transpose({yname!r}, {xname!r})."
        )

    fig, ax = plt.subplots()
    try:
        cs = ax.contour(x, y, z, levels=[float(level)])
        segments = [np.asarray(p) for p in cs.allsegs[0] if len(p) >= 2]
    finally:
        plt.close(fig)

    if aoi_polygon is not None:
        segments = [
            s
            for s in segments
            if shapely.contains_xy(aoi_polygon, s[:, 0], s[:, 1]).any()
        ]
    if min_length_m:
        segments = [
            s for s in segments if polyline_length_m(s, geographic) >= min_length_m
        ]
    if spacing_m:
        segments = [resample_polyline(s, spacing_m, geographic) for s in segments]
    segments = [s for s in segments if s.shape[0] > 0]

    return np.vstack(segments) if segments else np.empty((0, 2), dtype=float)


def spatial_subsample(
    x: np.ndarray,
    y: np.ndarray,
    fraction: float,
    seed: int = 0,
    geographic: bool = True,
) -> np.ndarray:
    """
    Pick a random subset of points that still covers their whole area.

    Stratified sampling: the area is cut into equal square cells, sized so that
    about ``fraction * len(x)`` cells hold points, and one point is drawn at
    random from each of those cells. Dense clusters (e.g. a finer mesh near
    the coast) are thinned more than sparse areas, so the subset spreads
    evenly over the area instead of following the point density.

    Parameters
    ----------
    x, y : np.ndarray
        Point coordinates (lon/lat degrees or projected metres).
    fraction : float
        Share of the points to keep, in ``(0, 1]``. The result has about
        ``fraction * len(x)`` points (one per occupied cell).
    seed : int, optional
        Random seed. Default is 0.
    geographic : bool, optional
        Lon/lat input (cells are square in metres). Default is True.

    Returns
    -------
    np.ndarray
        Sorted indices of the kept points.

    Raises
    ------
    ValueError
        If *fraction* is not in ``(0, 1]``.
    """

    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = x.size
    target = max(1, int(round(fraction * n)))
    if target >= n:
        return np.arange(n)
    if geographic:
        lat0 = np.radians(np.mean(y))
        x = np.radians(x) * EARTH_RADIUS_M * np.cos(lat0)
        y = np.radians(y) * EARTH_RADIUS_M

    def cells(size: float) -> np.ndarray:
        ij = np.floor(np.c_[x - x.min(), y - y.min()] / size).astype(np.int64)
        return np.unique(ij, axis=0, return_inverse=True)[1].ravel()

    lo, hi = 1e-9, float(max(np.ptp(x), np.ptp(y))) + 1.0
    for _ in range(60):  # occupied cells decrease with the cell size
        mid = np.sqrt(lo * hi)
        if cells(mid).max() + 1 > target:
            lo = mid
        else:
            hi = mid
    cell = cells(
        lo
        if abs(cells(lo).max() + 1 - target) < abs(cells(hi).max() + 1 - target)
        else hi
    )

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    _, first = np.unique(cell[order], return_index=True)

    return np.sort(order[first])


#: :func:`site_status` labels.
SITE_OK = "ok"
SITE_OUTSIDE = "outside_mesh"
SITE_CAN_DRY = "can_dry"


def site_status(
    depth: np.ndarray, wl_min: float = 0.0, min_depth: float = 0.1
) -> np.ndarray:
    """
    Whether each output site can be used: inside the mesh and always wet.

    A site is kept only if the model mesh covers it and its water depth stays
    above *min_depth* at the lowest water level the cases can have, so every
    run computes it and ``hs = 0`` there always means "no energy arrives".

    Parameters
    ----------
    depth : np.ndarray
        Mesh depth at the sites (m, positive down; NaN outside the mesh, see
        :func:`bluemath_tk.topo_bathy.mesh.depth_at_points`).
    wl_min : float, optional
        Lowest water level of the cases (m, e.g. the lower ``wl`` bound of the
        metamodel). Default is 0.
    min_depth : float, optional
        Minimum depth the model computes (SnapWave ``hmin``, SWAN ``DEPMIN``).
        Default is 0.1.

    Returns
    -------
    np.ndarray
        :data:`SITE_OK`, :data:`SITE_OUTSIDE` or :data:`SITE_CAN_DRY` per site.
    """

    depth = np.asarray(depth, dtype=float)
    status = np.full(depth.shape, SITE_OK, dtype=object)
    status[depth + float(wl_min) < float(min_depth)] = SITE_CAN_DRY
    status[~np.isfinite(depth)] = SITE_OUTSIDE

    return status
