"""
HyWaves output points: where nearshore waves are computed and reconstructed.

Typical sources are points along depth contours of a bathymetry raster
(:func:`get_isobath_points`) and unstructured-mesh nodes inside a polygon
(:func:`get_mesh_nodes_in_polygon`), plus any observation sites (buoys).

Raw contours of a high-resolution bathymetry follow every small sandbank and
pit, which puts points in odd places. :func:`get_isobath_points` can smooth
the bathymetry first (:func:`smooth_raster`, Gaussian width in metres), drop
short contour pieces, and space points evenly in metres along the result
(:func:`polyline_length_m`, :func:`resample_polyline`). Geographic (lon/lat)
rasters are measured with great-circle distances.
"""

from __future__ import annotations

from typing import Any

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


def get_mesh_nodes_in_polygon(
    polygon: Polygon | MultiPolygon,
    mesh: xr.Dataset,
    wet_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Unstructured-mesh nodes inside a polygon.

    Parameters
    ----------
    polygon : Polygon or MultiPolygon
        Area, in the mesh coordinates.
    mesh : xr.Dataset
        UGRID mesh with ``mesh2d_node_x``, ``mesh2d_node_y``, ``mesh2d_node_z``.
    wet_only : bool, optional
        Keep only nodes with negative bed level. Default is False.

    Returns
    -------
    list[dict]
        One ``{"lon", "lat", "depth", "node_index"}`` dict per node.
    """

    lon = mesh["mesh2d_node_x"].values
    lat = mesh["mesh2d_node_y"].values
    z = mesh["mesh2d_node_z"].values
    mask = shapely.contains_xy(polygon, lon, lat)
    if wet_only:
        mask &= z < 0.0

    return [
        {
            "lon": float(lon[i]),
            "lat": float(lat[i]),
            "depth": float(z[i]),
            "node_index": int(i),
        }
        for i in np.flatnonzero(mask)
    ]
