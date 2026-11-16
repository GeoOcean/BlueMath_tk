"""
Merge topobathy rasters of different extent, resolution and vertical datum.

A case often has a global grid (GEBCO) and a few better local products (a
national multibeam compilation, a coastal DEM). :func:`merge_rasters` writes
them on one regular lon/lat grid in priority order: each source, shifted to the
common datum by its ``offset`` and resampled to the target grid (averaged when
finer, bilinear when coarser), replaces what is below it. The seam is
feathered: the source weight grows linearly from 0 at the edge of its valid
data to 1 at ``feather_m`` inside it, so no step is left where two products
disagree. The ``source`` map gives the source of highest weight per cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.enums import Resampling
from scipy.ndimage import distance_transform_edt

EARTH_DEG_M = 111_320.0


@dataclass
class RasterSource:
    """
    One topobathy source for :func:`merge_rasters`.

    Parameters
    ----------
    name : str
        Label (kept in the provenance attributes).
    data : xr.DataArray
        Elevation (m, positive up) on two spatial dims with a CRS set
        (``data.rio.crs``); NaN where the source has no data.
    offset : float, optional
        Added to *data* to reach the common vertical datum, e.g. +0.39 for a
        DEM referenced to MHW when MHW is 0.39 m above MSL. Default 0.
    feather_m : float, optional
        Width (m) of the seam inside the source where its weight grows from 0
        to 1. Default 2000.
    """

    name: str
    data: xr.DataArray
    offset: float = 0.0
    feather_m: float = 2000.0


def target_grid(bounds: tuple[float, float, float, float], resolution_deg: float) -> xr.DataArray:
    """
    Empty lon/lat (EPSG:4326) grid, cell centres, latitude decreasing.

    Parameters
    ----------
    bounds : tuple of float
        ``(west, south, east, north)`` in degrees.
    resolution_deg : float
        Cell size in degrees.

    Returns
    -------
    xr.DataArray
        NaN float32 grid on ``(y, x)`` with the CRS and transform set.
    """

    west, south, east, north = bounds
    x = np.arange(west + resolution_deg / 2, east, resolution_deg)
    y = np.arange(north - resolution_deg / 2, south, -resolution_deg)
    grid = xr.DataArray(
        np.full((y.size, x.size), np.nan, dtype="float32"), coords={"y": y, "x": x}, dims=("y", "x")
    )
    return grid.rio.write_crs("EPSG:4326").rio.write_transform(grid.rio.transform(recalc=True))


def _resolution_deg(da: xr.DataArray) -> float:
    rx, ry = (abs(r) for r in da.rio.resolution())
    if da.rio.crs.is_geographic:
        return min(rx, ry)
    return min(rx, ry) / EARTH_DEG_M


def resample_to(source: xr.DataArray, grid: xr.DataArray) -> xr.DataArray:
    """
    Resample *source* onto *grid*: average when finer, bilinear when coarser.

    Parameters
    ----------
    source : xr.DataArray
        Raster with a CRS.
    grid : xr.DataArray
        Target grid (:func:`target_grid`).

    Returns
    -------
    xr.DataArray
        float32 on the grid, NaN where the source has no data.
    """

    finer = _resolution_deg(source) < _resolution_deg(grid)
    src = source.astype("float32").rio.write_nodata(np.nan, encoded=False)
    out = src.rio.reproject_match(
        grid, resampling=Resampling.average if finer else Resampling.bilinear, nodata=np.nan
    )
    return out.assign_coords(x=grid.x, y=grid.y)


def feather_weight(valid: np.ndarray, width_px: float) -> np.ndarray:
    """
    Weight 0 at the edge of *valid* (and outside it), 1 at *width_px* inside.

    The array border counts as an edge, so a source that ends at its own
    rectangle is feathered there too.
    """

    if width_px <= 0:
        return valid.astype("float32")
    padded = np.pad(valid, 1, constant_values=False)
    distance = distance_transform_edt(padded)[1:-1, 1:-1]
    return np.clip((distance - 0.5) / width_px, 0.0, 1.0).astype("float32") * valid


def merge_rasters(sources: list[RasterSource], grid: xr.DataArray) -> xr.Dataset:
    """
    Merge topobathy sources on *grid* in priority order, with feathered seams.

    Parameters
    ----------
    sources : list of RasterSource
        Lowest priority first. The first source should cover the whole grid
        (e.g. GEBCO); it is not feathered.
    grid : xr.DataArray
        Target grid (:func:`target_grid`).

    Returns
    -------
    xr.Dataset
        ``elevation`` (float32, m, positive up) and ``source`` (uint8 index
        into ``sources``, the source of highest weight), on the grid.

    Raises
    ------
    ValueError
        If cells are left without data.
    """

    cell_m = _resolution_deg(grid) * EARTH_DEG_M
    elevation = np.full(grid.shape, np.nan, dtype="float32")
    best = np.zeros(grid.shape, dtype="float32")
    source = np.zeros(grid.shape, dtype="uint8")
    for index, item in enumerate(sources):
        values = resample_to(item.data, grid).values + np.float32(item.offset)
        valid = np.isfinite(values)
        weight = valid.astype("float32") if index == 0 else feather_weight(valid, item.feather_m / cell_m)
        fill = valid & ~np.isfinite(elevation)
        elevation[fill] = values[fill]
        blend = valid & ~fill
        elevation[blend] += weight[blend] * (values[blend] - elevation[blend])
        wins = weight >= np.maximum(best, 0.5) if index else valid
        source[wins] = index
        best = np.maximum(best, weight)
    if np.isnan(elevation).any():
        raise ValueError(f"{int(np.isnan(elevation).sum())} cells have no source")

    attrs = {f"source_{i}": f"{s.name} (offset {s.offset:+g} m, feather {s.feather_m:g} m)" for i, s in enumerate(sources)}
    ds = xr.Dataset(
        {
            "elevation": (("y", "x"), elevation, {"units": "m", "positive": "up", **attrs}),
            "source": (("y", "x"), source, {"long_name": "source of highest weight", **attrs}),
        },
        coords={"y": grid.y, "x": grid.x},
    )
    return ds.rio.write_crs("EPSG:4326")
