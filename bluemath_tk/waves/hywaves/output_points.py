"""
HyWaves output points: where nearshore waves are computed and reconstructed.

Typical sources are points along depth contours of a bathymetry raster
(:func:`get_isobath_points`) and unstructured-mesh nodes inside a polygon
(:func:`get_mesh_nodes_in_polygon`), plus any observation sites (buoys).
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shapely
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon

_X_NAMES = ("x", "lon", "longitude")
_Y_NAMES = ("y", "lat", "latitude")


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


def resample_polyline(pts: np.ndarray, spacing: float) -> np.ndarray:
    """
    Resample a polyline at approximately uniform arc length.

    Parameters
    ----------
    pts : np.ndarray
        ``(N, 2)`` vertices.
    spacing : float
        Target spacing between output points (units of *pts*).

    Returns
    -------
    np.ndarray
        ``(M, 2)`` points, including the start and (nearly) the end; empty when
        the polyline has zero length.
    """

    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())
    if total <= 0.0:
        return np.empty((0, 2), dtype=float)

    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    ds = np.arange(0.0, total, float(spacing))
    if ds.size == 0 or (total - ds[-1]) > 0.25 * spacing:
        ds = np.append(ds, total)

    out = np.empty((ds.size, 2), dtype=float)
    for k, d in enumerate(ds):
        i = int(np.searchsorted(cum, d, side="right") - 1)
        i = max(0, min(i, len(seg_len) - 1))
        t = 0.0 if seg_len[i] == 0 else (d - cum[i]) / seg_len[i]
        out[k] = pts[i] * (1.0 - t) + pts[i + 1] * t

    return out


def get_isobath_points(
    da: xr.DataArray,
    level: float,
    spacing: float | None = None,
    aoi_polygon: Polygon | MultiPolygon | None = None,
) -> np.ndarray:
    """
    Points along one depth contour of a bathymetry raster.

    Parameters
    ----------
    da : xr.DataArray
        Topobathy raster (elevation, negative below sea level), dims ``(y, x)``.
    level : float
        Contour level, e.g. ``-10`` for the 10 m isobath.
    spacing : float, optional
        Resample each contour segment at this spacing (raster units, e.g.
        degrees). Default is None (contour vertices as they are).
    aoi_polygon : Polygon or MultiPolygon, optional
        Keep only contour segments with at least one vertex inside it.

    Returns
    -------
    np.ndarray
        ``(N, 2)`` ``[x, y]`` points; empty when the contour does not exist.

    Raises
    ------
    ValueError
        If the raster shape does not match ``(len(y), len(x))``.
    """

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
    if spacing is not None and spacing > 0:
        segments = [resample_polyline(s, spacing) for s in segments]
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
