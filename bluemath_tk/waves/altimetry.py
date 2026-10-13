"""
Satellite altimeter significant wave height near a point, ready to compare
with a wave hindcast.

The altimeter store is a zarr of along-track observations on one ``obs``
dimension, sorted by 1-degree cell and then time, with a cell index
(``cell_start``, ``cell_count`` on ``lat_cell``, ``lon_cell``), so the
observations near a point are read as a few contiguous slices: the layout of
the IMOS multi-mission product (``imos_altimeter_1985_2020.zarr``):

- ``hs`` (calibrated Ku-band Hs), ``time``, ``latitude``, ``longitude``
  (0-360), ``flag`` (1 good, 2 probably good, 3 correctable), ``depth``
  (negative in the ocean), ``dist2coast`` (km), ``satellite`` (mission code).

Typical use, for one hindcast node::

    ds = open_altimeter(path)
    obs = altimeter_near(ds, lon, lat, radius_km=50, min_depth=20)
    passes = altimeter_passes(obs)
    pairs = collocate(hindcast_node, passes)   # hindcast at the pass times

Along-track 1 Hz observations are strongly correlated, so they are first
reduced to one value per satellite pass (:func:`altimeter_passes`) before any
comparison or fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

#: Variables read from the altimeter store.
ALTIMETER_VARIABLES: tuple[str, ...] = (
    "time",
    "latitude",
    "longitude",
    "hs",
    "flag",
    "depth",
    "dist2coast",
    "satellite",
)

EARTH_RADIUS_KM = 6371.0

#: Deeper "depths" (m) are fill values and read as missing.
MAX_OCEAN_DEPTH = 11000.0


def open_altimeter(path: str) -> xr.Dataset:
    """
    Open a cell-indexed altimeter zarr (local path or URL), lazily.

    Parameters
    ----------
    path : str
        Zarr store path or URL (``https://``, ``s3://``, ... through fsspec).

    Returns
    -------
    xr.Dataset
        The store, with ``cell_start`` / ``cell_count`` loaded.

    Raises
    ------
    ValueError
        If the store has no cell index.
    """

    ds = xr.open_zarr(path)
    if "cell_start" not in ds or "cell_count" not in ds:
        raise ValueError(
            f"{path} has no cell index (cell_start, cell_count on lat_cell, lon_cell)"
        )
    ds["cell_start"] = ds["cell_start"].load()
    ds["cell_count"] = ds["cell_count"].load()

    return ds


def haversine_km(
    lon1: np.ndarray | float, lat1: np.ndarray | float, lon2: float, lat2: float
) -> np.ndarray:
    """Great-circle distance (km) from ``(lon1, lat1)`` to ``(lon2, lat2)``."""

    lon1, lat1 = np.deg2rad(lon1), np.deg2rad(lat1)
    lon2, lat2 = np.deg2rad(lon2), np.deg2rad(lat2)
    a = (
        np.sin((lat1 - lat2) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon1 - lon2) / 2) ** 2
    )

    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _cells_around(lon: float, lat: float, radius_km: float) -> list[tuple[int, int]]:
    """``(lat_cell, lon_cell)`` of the 1-degree cells a circle can touch."""

    dlat = np.rad2deg(radius_km / EARTH_RADIUS_KM)
    lat0, lat1 = max(lat - dlat, -90.0), min(lat + dlat, 89.999)
    coslat = max(np.cos(np.deg2rad(max(abs(lat0), abs(lat1)))), 1e-6)
    dlon = min(dlat / coslat, 180.0)
    lats = range(int(np.floor(lat0)), int(np.floor(lat1)) + 1)
    lons = {int(np.floor(x)) % 360 for x in np.arange(lon - dlon, lon + dlon + 1, 1.0)}
    lons |= {int(np.floor(lon + dlon)) % 360}

    return [(la, lo) for la in lats for lo in sorted(lons)]


def altimeter_near(
    ds: xr.Dataset,
    lon: float,
    lat: float,
    radius_km: float = 50.0,
    *,
    time: slice | None = None,
    flags: tuple[int, ...] = (1, 2),
    min_depth: float | None = None,
    min_dist2coast: float | None = None,
    variables: tuple[str, ...] = ALTIMETER_VARIABLES,
) -> pd.DataFrame:
    """
    Altimeter observations within *radius_km* of a point.

    Parameters
    ----------
    ds : xr.Dataset
        Output of :func:`open_altimeter`.
    lon, lat : float
        Point (degrees; longitude in -180..180 or 0..360).
    radius_km : float, optional
        Search radius. Default is 50 km.
    time : slice, optional
        Time window, e.g. ``slice("2000", "2020")``.
    flags : tuple of int, optional
        Quality flags kept. Default is (1, 2): good and probably good.
    min_depth : float, optional
        Keep only observations in water deeper than this (m, positive): the
        altimeter footprint degrades in shallow water and near the coast.
        Observations without a valid depth are then dropped.
    min_dist2coast : float, optional
        Keep only observations farther than this from the coast (km).
        Observations without a distance (e.g. HY-2A) are dropped.
    variables : tuple of str, optional
        Variables read. Default is :data:`ALTIMETER_VARIABLES`.

    Returns
    -------
    pd.DataFrame
        One row per observation, with ``lon`` (-180..180) and ``distance_km``
        to the point, sorted by time.
    """

    present = [v for v in variables if v in ds.variables]
    chunks = []
    for la, lo in _cells_around(lon, lat, radius_km):
        sel = {"lat_cell": la, "lon_cell": lo}
        n = int(ds["cell_count"].sel(sel))
        if n == 0:
            continue
        start = int(ds["cell_start"].sel(sel))
        chunks.append(ds[present].isel(obs=slice(start, start + n)))
    if not chunks:
        return pd.DataFrame(columns=[*present, "lon", "distance_km"])

    obs = xr.concat(chunks, dim="obs").load().to_dataframe().reset_index(drop=True)
    obs["lon"] = (obs["longitude"] + 180.0) % 360.0 - 180.0
    obs["distance_km"] = haversine_km(
        obs["lon"].to_numpy(), obs["latitude"].to_numpy(), lon, lat
    )
    if "depth" in obs:
        # a few tracks carry fill values far below any ocean depth
        obs.loc[obs["depth"] < -MAX_OCEAN_DEPTH, "depth"] = np.nan
    keep = obs["distance_km"] <= radius_km
    if flags is not None and "flag" in obs:
        keep &= obs["flag"].isin(flags)
    if min_depth is not None and "depth" in obs:
        keep &= obs["depth"] <= -abs(min_depth)
    if min_dist2coast is not None and "dist2coast" in obs:
        keep &= obs["dist2coast"] >= min_dist2coast
    if time is not None:
        t = pd.to_datetime(obs["time"])
        if time.start is not None:
            keep &= t >= pd.Timestamp(time.start)
        if time.stop is not None:
            stop = pd.Timestamp(time.stop)
            if len(str(time.stop)) <= 4:  # a bare year includes the whole year
                stop = stop + pd.DateOffset(years=1)
            keep &= t < stop

    return obs[keep].sort_values("time").reset_index(drop=True)


def altimeter_passes(
    obs: pd.DataFrame, max_gap: str = "5min", min_points: int = 3
) -> pd.DataFrame:
    """
    One value per satellite pass: the along-track median of the observations.

    Consecutive observations of the same satellite less than *max_gap* apart
    belong to one pass (a track crosses a 50 km circle in about 15 s).

    Parameters
    ----------
    obs : pd.DataFrame
        Output of :func:`altimeter_near`.
    max_gap : str, optional
        Largest time gap inside a pass. Default is "5min".
    min_points : int, optional
        Passes with fewer observations are dropped. Default is 3.

    Returns
    -------
    pd.DataFrame
        Indexed by the mean pass ``time``: ``hs`` (median), ``hs_std``,
        ``n``, ``lon``, ``lat``, ``distance_km``, ``depth``, ``satellite``.
    """

    columns = ["hs", "hs_std", "n", "lon", "lat", "distance_km", "depth", "satellite"]
    if obs.empty:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="time"))

    obs = obs.sort_values(["satellite", "time"]).reset_index(drop=True)
    t = pd.to_datetime(obs["time"])
    new = (obs["satellite"].diff() != 0) | (t.diff() > pd.Timedelta(max_gap))
    pass_id = new.cumsum()
    grouped = obs.assign(time=t).groupby(pass_id)
    passes = pd.DataFrame(
        {
            "time": grouped["time"].apply(lambda x: x.mean()),
            "hs": grouped["hs"].median(),
            "hs_std": grouped["hs"].std(),
            "n": grouped["hs"].size(),
            "lon": grouped["lon"].mean(),
            "lat": grouped["latitude"].mean(),
            "distance_km": grouped["distance_km"].mean(),
            "depth": grouped["depth"].median() if "depth" in obs else np.nan,
            "satellite": grouped["satellite"].first(),
        }
    )
    passes = passes[passes["n"] >= min_points]

    return passes.set_index("time").sort_index()[columns]


def collocate(
    hindcast: xr.Dataset | pd.DataFrame,
    passes: pd.DataFrame,
    max_time_diff: str = "30min",
) -> pd.DataFrame:
    """
    Hindcast values at the satellite pass times (nearest time step).

    The nearest step is used rather than interpolating, so directions and
    partitions stay consistent; with an hourly hindcast the default 30 min
    tolerance keeps every pass.

    Parameters
    ----------
    hindcast : xr.Dataset or pd.DataFrame
        Time series at one node (``time`` dimension or a datetime index),
        e.g. ``hs``, ``dir`` and partitions ``phs0``, ``pdir0``, ...
    passes : pd.DataFrame
        Output of :func:`altimeter_passes`.
    max_time_diff : str, optional
        Largest time difference kept. Default is "30min".

    Returns
    -------
    pd.DataFrame
        Indexed by pass time: the hindcast variables and the pass columns
        prefixed ``sat_`` (``sat_hs`` is the reference Hs).
    """

    if isinstance(hindcast, xr.Dataset):
        hindcast = hindcast.to_dataframe()
    hindcast = hindcast.sort_index()
    times = hindcast.index.to_numpy()
    if passes.empty or times.size == 0:
        return pd.DataFrame(
            columns=[*hindcast.columns, *(f"sat_{c}" for c in passes.columns)]
        )

    targets = passes.index.to_numpy()
    right = np.clip(np.searchsorted(times, targets), 0, times.size - 1)
    left = np.clip(right - 1, 0, times.size - 1)
    nearest = np.where(
        np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right
    )
    ok = (
        np.abs(times[nearest] - targets) <= pd.Timedelta(max_time_diff).to_timedelta64()
    )

    out = hindcast.iloc[nearest[ok]].reset_index(drop=True)
    sat = passes[ok].add_prefix("sat_").reset_index()
    out = pd.concat([sat, out], axis=1).set_index("time")

    return out
