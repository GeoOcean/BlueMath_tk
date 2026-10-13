"""
Offshore hindcasts at the HyWaves goals, in the HyWaves variable names.

Every offshore source is stored on ``(time, goal)`` with the same names, so
the rest of the method (dynamic runs, reconstruction, satellite calibration)
does not depend on where it came from:

- bulk ``hs`` (m), ``tp`` (s), ``dir`` (deg, nautical, coming from),
  ``spr`` (deg);
- partitions ``phs{i}``, ``ptp{i}``, ``pdir{i}``, ``pspr{i}``, partition 0
  the wind sea and 1, 2, ... the swells.

This module reads ERA5 ocean waves from the Earth Data Hub zarr
(:data:`EDH_ERA5_WAVES_URL`, hourly, 0.5 degree, 1940 onwards) at the goal
nodes. ERA5 has no partition peak periods: ``ptp{i}`` are the partition mean
periods (``mpww``, ``mwp1..3``, mean period Tm-1,0), shorter than the peak
period (about Tp / 1.1 for a JONSWAP spectrum); the attributes say so. The
three swells share the total-swell directional width (``dwps``).
"""

from __future__ import annotations

import os

import dask
import numpy as np
import xarray as xr

#: Earth Data Hub (DestinE) ERA5 single-level ocean-wave variables.
EDH_ERA5_WAVES_URL = (
    "https://data.earthdatahub.destine.eu/era5/era5-single-levels-ocean-waves-v0.zarr"
)

#: Environment variable holding the Earth Data Hub key.
EDH_KEY_VARIABLE = "EDH_API_KEY"

#: HyWaves name -> ERA5 short name.
ERA5_TO_HYWAVES: dict[str, str] = {
    "hs": "swh",
    "tp": "pp1d",
    "dir": "mwd",
    "spr": "wdw",
    "phs0": "shww",
    "ptp0": "mpww",
    "pdir0": "mdww",
    "pspr0": "dwww",
    **{
        name: f"{era5}{i}"
        for i in (1, 2, 3)
        for name, era5 in ((f"phs{i}", "swh"), (f"ptp{i}", "mwp"), (f"pdir{i}", "mwd"))
    },
    **{f"pspr{i}": "dwps" for i in (1, 2, 3)},
}

_UNITS = {"hs": "m", "tp": "s", "dir": "degree", "spr": "degree"}
_LONG_NAMES = {
    "hs": "Significant wave height",
    "tp": "Peak wave period",
    "dir": "Mean wave direction (coming from)",
    "spr": "Directional spread",
}
_PART_NAMES = {
    "phs": ("Significant wave height", "m"),
    "ptp": ("Mean wave period Tm-1,0 (ERA5 has no partition peak period)", "s"),
    "pdir": ("Mean wave direction (coming from)", "degree"),
    "pspr": ("Directional spread", "degree"),
}


def open_era5_waves(
    key: str | None = None, url: str = EDH_ERA5_WAVES_URL
) -> xr.Dataset:
    """
    Open the ERA5 ocean-wave zarr of Earth Data Hub, lazily.

    Parameters
    ----------
    key : str, optional
        Earth Data Hub key. Default: the ``EDH_API_KEY`` environment variable
        (keep keys out of notebooks and repositories).
    url : str, optional
        Store URL (without credentials). Default :data:`EDH_ERA5_WAVES_URL`.

    Returns
    -------
    xr.Dataset
        ERA5 waves on ``(valid_time, latitude, longitude)``.

    Raises
    ------
    ValueError
        If no key is given or found in the environment.
    """

    key = key or os.environ.get(EDH_KEY_VARIABLE)
    if not key:
        raise ValueError(
            f"No Earth Data Hub key: pass key= or set the {EDH_KEY_VARIABLE} environment variable"
        )
    scheme, rest = url.split("://", 1)

    return xr.open_dataset(f"{scheme}://edh:{key}@{rest}", chunks={}, engine="zarr")


def era5_at_goals(
    ds: xr.Dataset,
    lon: np.ndarray,
    lat: np.ndarray,
    goals: np.ndarray,
    time: slice | None = None,
    num_workers: int = 16,
) -> xr.Dataset:
    """
    ERA5 waves at the goal nodes, in HyWaves names, on ``(time, goal)``.

    Parameters
    ----------
    ds : xr.Dataset
        Output of :func:`open_era5_waves`.
    lon, lat : np.ndarray
        Node of each goal (degrees; the nearest ERA5 grid point is taken).
    goals : np.ndarray
        Goal ids (the ``goal`` coordinate).
    time : slice, optional
        Time window, e.g. ``slice("1979", "2023")``. Default: everything.
    num_workers : int, optional
        Parallel chunk reads. Default is 16.

    Returns
    -------
    xr.Dataset
        ``hs``, ``tp``, ``dir``, ``spr`` and ``phs{i}``, ``ptp{i}``,
        ``pdir{i}``, ``pspr{i}`` (i = 0 wind sea, 1-3 swells), loaded, with
        the grid point of each goal as ``longitude`` / ``latitude``
        coordinates. Directional widths are converted from radians to
        degrees.
    """

    lon = xr.DataArray(np.mod(np.asarray(lon, dtype=float), 360.0), dims="goal")
    lat = xr.DataArray(np.asarray(lat, dtype=float), dims="goal")
    era5_names = sorted(set(ERA5_TO_HYWAVES.values()))
    points = ds[era5_names].sel(longitude=lon, latitude=lat, method="nearest")
    if time is not None:
        points = points.sel(valid_time=time)
    with dask.config.set(scheduler="threads", num_workers=num_workers):
        points = points.compute()

    out = xr.Dataset(
        {
            name: points[era5]
            * (180.0 / np.pi if name.rstrip("0123456789").endswith("spr") else 1.0)
            for name, era5 in ERA5_TO_HYWAVES.items()
        }
    )
    out = out.rename(valid_time="time").assign_coords(goal=np.asarray(goals))
    out = out.assign_coords(
        longitude=("goal", ((points["longitude"].values + 180.0) % 360.0) - 180.0),
        latitude=("goal", points["latitude"].values),
    )
    for name in out.data_vars:
        stem = name.rstrip("0123456789")
        if stem in _PART_NAMES:
            long_name, units = _PART_NAMES[stem]
            label = "wind sea" if name.endswith("0") else f"swell {name[-1]}"
            out[name].attrs = {"long_name": f"{long_name}, {label}", "units": units}
        else:
            out[name].attrs = {"long_name": _LONG_NAMES[name], "units": _UNITS[name]}
        out[name].attrs["era5_name"] = ERA5_TO_HYWAVES[name]
    out.attrs = {
        "source": "ERA5 single levels, ocean waves (ECMWF), via Earth Data Hub",
        "url": EDH_ERA5_WAVES_URL,
        "partitions": "0 = wind sea, 1-3 = swell partitions",
        "note": "ptp* are partition mean periods (Tm-1,0), not peak periods; "
        "the swells share the total-swell directional width (dwps)",
    }

    return out
