"""
HyWaves wind correction: local wind growth added to the reconstructed wind sea.

The metamodel cases run without wind, so the reconstruction misses what the
local wind adds while the waves cross the domain. The correction is applied
once per time step and site, after the goals are predicted and before their
spectra are summed (see :func:`wind_sea_contribution`):

1. Each goal x partition contribution gets a wind-sea weight ``w`` from its
   *offshore* forcing (:func:`~bluemath_tk.waves.wind.wind_sea_weight`), and
   the wind it implies (:func:`~bluemath_tk.waves.wind.infer_wind`).
2. The wind-sea energy at the site, ``E_ws = sum w hs^2``, is summed once over
   the goals (no goal counted twice).
3. The wind at the site is the data (e.g. ERA5) or the ``w hs^2``-weighted
   inferred wind of the contributions.
4. ``K = Hs(with wind) / Hs(without wind)`` comes from :class:`WindCorrection`,
   fitted on pairs of full-domain runs with and without wind, on
   dimensionless features of the wind sea, the fetch and the depth
   (:func:`correction_features`).
5. One extra contribution carries the missing energy, ``(K^2 - 1) E_ws``,
   from the wind direction; it is summed with the others as a spectrum.

The fetch at each site and direction comes from
:func:`bluemath_tk.waves.fetch.fetch_table`.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping

import numpy as np
import pandas as pd
import xarray as xr

from ...core.constants import GRAVITY

#: Partition id of the wind-sea contribution added by the correction.
WIND_PARTITION = -2
#: Goal id of the wind-sea contribution (it belongs to no goal).
WIND_GOAL = 0
#: Features of :func:`correction_features`, in order.
FEATURES = ("hs_ws", "fetch", "depth", "mean_depth", "cos_angle")


def correction_features(
    hs_ws: np.ndarray,
    u10: np.ndarray,
    fetch: np.ndarray,
    depth: np.ndarray,
    mean_depth: np.ndarray,
    angle: np.ndarray,
) -> np.ndarray:
    """
    Dimensionless features of the wind correction.

    ``g Hs_ws / U^2`` (how developed the wind sea already is), ``g F / U^2``
    (fetch left), ``g d / U^2`` at the site and along the fetch (depth
    limitation) and the cosine of the wind-wave angle.

    Parameters
    ----------
    hs_ws : np.ndarray
        Wind-sea Hs at the site without wind (m).
    u10 : np.ndarray
        Wind speed (m/s), positive.
    fetch : np.ndarray
        Fetch along the wind direction (m).
    depth : np.ndarray
        Depth at the site (m).
    mean_depth : np.ndarray
        Mean depth along the fetch (m).
    angle : np.ndarray
        Wind direction minus wind-sea direction (deg).

    Returns
    -------
    np.ndarray
        ``(n, 5)`` features, in the order of :data:`FEATURES`.
    """

    u2 = np.asarray(u10, dtype=float) ** 2 / GRAVITY
    cols = [
        np.asarray(hs_ws, dtype=float) / u2,
        np.asarray(fetch, dtype=float) / u2,
        np.asarray(depth, dtype=float) / u2,
        np.asarray(mean_depth, dtype=float) / u2,
        np.cos(np.deg2rad(np.asarray(angle, dtype=float))),
    ]

    return np.column_stack([np.ravel(c) for c in np.broadcast_arrays(*cols)])


class WindCorrection:
    """
    ``K = Hs(with wind) / Hs(without wind)`` from :func:`correction_features`.

    A gradient-boosting regression of ``log K`` (scikit-learn), fitted on
    pairs of the same full-domain cases run with and without wind. ``K``
    scales the wind-sea part of the energy only (see :meth:`fit`).

    Attributes
    ----------
    model : sklearn.ensemble.HistGradientBoostingRegressor
        Fitted regressor of ``log K``.
    u10_min : float
        Below this wind speed (m/s) ``K = 1``.
    hs_min : float
        Below this wind-sea Hs (m) ``K = 1``.
    k_range : tuple of float
        Bounds on the predicted ``K``.
    """

    def __init__(
        self,
        u10_min: float = 1.0,
        hs_min: float = 0.05,
        k_range: tuple[float, float] = (0.5, 4.0),
        **regressor_kwargs,
    ) -> None:
        """
        Parameters
        ----------
        u10_min : float, optional
            Wind speed (m/s) below which there is no correction. Default 1.
        hs_min : float, optional
            Wind-sea Hs (m) below which there is no correction. Default 0.05.
        k_range : tuple of float, optional
            Bounds on the predicted ``K``. Default ``(0.5, 4.0)``.
        **regressor_kwargs
            Passed to ``HistGradientBoostingRegressor`` (default
            ``max_iter=400, learning_rate=0.05``).
        """

        from sklearn.ensemble import HistGradientBoostingRegressor

        self.u10_min = float(u10_min)
        self.hs_min = float(hs_min)
        self.k_range = (float(k_range[0]), float(k_range[1]))
        self.model = HistGradientBoostingRegressor(
            **{"max_iter": 400, "learning_rate": 0.05, **regressor_kwargs}
        )
        self.n_samples = 0

    def _valid(self, hs_ws: np.ndarray, u10: np.ndarray) -> np.ndarray:
        return (np.asarray(u10) >= self.u10_min) & (np.asarray(hs_ws) >= self.hs_min)

    def fit(
        self,
        hs_without: np.ndarray,
        hs_with: np.ndarray,
        u10: np.ndarray,
        fetch: np.ndarray,
        depth: np.ndarray,
        mean_depth: np.ndarray,
        angle: np.ndarray,
        weight: float | np.ndarray = 1.0,
    ) -> WindCorrection:
        """
        Fit on paired runs: site Hs without and with wind, and their conditions.

        With a wind-sea weight ``w`` of the case forcing, only ``w hs^2`` of
        the energy is wind sea, as in the reconstruction: the feature is
        ``Hs_ws = sqrt(w) hs_without`` and the target the ``K`` of that part,
        ``K^2 = 1 + (hs_with^2 - hs_without^2) / (w hs_without^2)``. Pairs
        with NaNs, no wind (``u10 < u10_min``) or no wind sea
        (``Hs_ws < hs_min``) are left out.

        Parameters
        ----------
        hs_without, hs_with : np.ndarray
            Site Hs (m) of the run without and with wind.
        u10, fetch, depth, mean_depth, angle : np.ndarray
            See :func:`correction_features`.
        weight : float or np.ndarray, optional
            Wind-sea weight of each pair's forcing, in ``(0, 1]``. Default 1.

        Returns
        -------
        WindCorrection
            Self.
        """

        shape = np.broadcast_shapes(
            *(np.shape(a) for a in (hs_without, hs_with, u10, fetch, depth, mean_depth, angle, weight))
        )
        hs0 = np.ravel(np.broadcast_to(hs_without, shape)).astype(float)
        hs1 = np.ravel(np.broadcast_to(hs_with, shape)).astype(float)
        w = np.ravel(np.broadcast_to(weight, shape)).astype(float)
        u = np.ravel(np.broadcast_to(u10, shape)).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            hs_ws = np.sqrt(w) * hs0
            k2 = 1.0 + (hs1**2 - hs0**2) / (w * hs0**2)
        x = correction_features(
            hs_ws,
            u,
            *(np.ravel(np.broadcast_to(a, shape)) for a in (fetch, depth, mean_depth, angle)),
        )
        ok = (
            np.isfinite(x).all(axis=1)
            & np.isfinite(k2)
            & (w > 0)
            & self._valid(hs_ws, u)
        )
        if not ok.any():
            raise ValueError("no valid pairs to fit the wind correction")
        k = np.sqrt(np.clip(k2[ok], self.k_range[0] ** 2, self.k_range[1] ** 2))
        self.model.fit(x[ok], np.log(k))
        self.n_samples = int(ok.sum())

        return self

    def predict(
        self,
        hs_ws: np.ndarray,
        u10: np.ndarray,
        fetch: np.ndarray,
        depth: np.ndarray,
        mean_depth: np.ndarray,
        angle: np.ndarray,
    ) -> np.ndarray:
        """
        ``K`` for each input (1 without wind, wind sea or valid features).

        Parameters
        ----------
        hs_ws, u10, fetch, depth, mean_depth, angle : np.ndarray
            See :func:`correction_features` (broadcast together).

        Returns
        -------
        np.ndarray
            ``K``, with the broadcast shape of the inputs.
        """

        shape = np.broadcast_shapes(*(np.shape(a) for a in (hs_ws, u10, fetch, depth, mean_depth, angle)))
        x = correction_features(hs_ws, u10, fetch, depth, mean_depth, angle)
        hs = np.ravel(np.broadcast_to(hs_ws, shape))
        u = np.ravel(np.broadcast_to(u10, shape))
        k = np.ones(len(x))
        ok = np.isfinite(x).all(axis=1) & self._valid(hs, u)
        if ok.any():
            k[ok] = np.clip(np.exp(self.model.predict(x[ok])), *self.k_range)

        return k.reshape(shape)

    def save_model(self, path: str) -> None:
        """Pickle the correction (read back with :func:`bluemath_tk.core.io.load_model`)."""

        with open(path, "wb") as f:
            pickle.dump(self, f)


def _fetch_along(table: xr.Dataset, name: str, direction: xr.DataArray) -> xr.DataArray:
    """``table[name]`` on ``(sites, dir)`` at the nearest direction bin of *direction*."""

    dirs = table["dir"].values
    step = 360.0 / len(dirs)
    idx = np.mod(np.rint((direction.values - dirs[0]) / step).astype(int), len(dirs))
    values = table[name].transpose("sites", "dir").values
    site_axis = direction.dims.index("sites")
    rows = np.arange(values.shape[0]).reshape(
        [-1 if a == site_axis else 1 for a in range(direction.ndim)]
    )

    return xr.DataArray(values[rows, idx], dims=direction.dims, coords=direction.coords)


def _vector_mean(direction: xr.DataArray, weight: xr.DataArray, dims: list[str]) -> xr.DataArray:
    """Weighted vector mean of nautical directions (deg) over *dims*."""

    rad = np.deg2rad(direction)
    s = (weight * np.sin(rad)).sum(dims)
    c = (weight * np.cos(rad)).sum(dims)

    return np.mod(np.rad2deg(np.arctan2(s, c)), 360.0)


def wind_sea_contribution(
    cube: xr.Dataset,
    correction: WindCorrection,
    fetch: xr.Dataset,
    wind: xr.Dataset | None = None,
    spr: float = 30.0,
    tp: str = "wind_sea",
) -> xr.Dataset | None:
    """
    The extra contribution of the local wind for one batch of sites.

    Parameters
    ----------
    cube : xr.Dataset
        Contributions on ``(goal, partition, time, sites)``: ``hs``, ``tp``,
        ``dir``, ``spr`` (nearshore) plus, from each contribution's offshore
        forcing, ``ws_weight`` (wind-sea weight) and, without *wind*, ``u10``
        and ``u10dir`` (inferred wind), on ``(goal, partition, time)``.
    correction : WindCorrection
        Fitted ``K`` model.
    fetch : xr.Dataset
        ``fetch_eff`` (or ``fetch``) and ``mean_depth`` on ``(sites, dir)``
        and ``site_depth`` on ``sites`` (see
        :func:`bluemath_tk.waves.fetch.fetch_table`), with the cube's sites.
    wind : xr.Dataset, optional
        ``u10`` and ``u10dir`` on ``(time, sites)`` from data. Default: the
        ``w hs^2``-weighted inferred wind of the contributions.
    spr : float, optional
        Directional spreading of the added wind sea (deg). Default is 30.
    tp : str, optional
        Period of the added wind sea: ``"wind_sea"`` (energy-weighted Tp of
        the wind-sea contributions, default) or ``"growth"`` (Kahma & Calkoen
        period after growing over the fetch).

    Returns
    -------
    xr.Dataset or None
        ``hs``, ``tp``, ``dir``, ``spr`` and ``k`` on
        ``(goal, partition, time, sites)`` with goal :data:`WIND_GOAL` and
        partition :data:`WIND_PARTITION`, ready to merge with *cube*; None
        if nothing in the batch is wind sea.
    """

    parts = ["goal", "partition"]
    weight = cube["ws_weight"].fillna(0.0)
    energy = (weight * cube["hs"].fillna(0.0) ** 2).transpose("time", "sites", ...)
    e_ws = energy.sum(parts)
    if not bool((e_ws > 0).any()):
        return None

    with np.errstate(invalid="ignore", divide="ignore"):
        ws_dir = _vector_mean(cube["dir"].fillna(0.0), energy, parts)
        if wind is None:
            u10 = (energy * cube["u10"].fillna(0.0)).sum(parts) / e_ws
            u10dir = _vector_mean(cube["u10dir"].fillna(0.0), energy * cube["u10"].fillna(0.0), parts)
        else:
            u10 = wind["u10"].sel(time=cube["time"], sites=cube["sites"])
            u10dir = wind["u10dir"].sel(time=cube["time"], sites=cube["sites"])
        u10 = u10.fillna(0.0).transpose("time", "sites")
        u10dir = u10dir.transpose("time", "sites")
        hs_ws = np.sqrt(e_ws)

    fetch = fetch.sel(sites=cube["sites"].values)
    name = "fetch_eff" if "fetch_eff" in fetch else "fetch"
    fetch_u = _fetch_along(fetch, name, u10dir)
    depth_u = _fetch_along(fetch, "mean_depth", u10dir)
    site_depth = fetch["site_depth"].broadcast_like(hs_ws)
    angle = u10dir - ws_dir
    k = correction.predict(
        hs_ws.values, u10.values, fetch_u.values, site_depth.values, depth_u.values, angle.values
    )
    k = xr.DataArray(k, dims=hs_ws.dims, coords=hs_ws.coords)
    hs_add = np.sqrt(np.clip(k**2 - 1.0, 0.0, None) * e_ws)

    if tp == "growth":
        from ..wind import equivalent_fetch, growth_tp

        with np.errstate(invalid="ignore", divide="ignore"):
            grown = equivalent_fetch(hs_ws.values, u10.values) + fetch_u.values
            tp_add = growth_tp(u10.values, np.where(np.isfinite(grown), grown, 1e12), depth_u.values)
        tp_add = xr.DataArray(tp_add, dims=hs_ws.dims, coords=hs_ws.coords)
    elif tp == "wind_sea":
        with np.errstate(invalid="ignore", divide="ignore"):
            tp_add = (energy * cube["tp"].fillna(0.0)).sum(parts) / e_ws
    else:
        raise ValueError(f"tp must be 'wind_sea' or 'growth', got {tp!r}")

    keep = (hs_add > 0) & np.isfinite(tp_add) & (tp_add > 0)
    out = xr.Dataset(
        {
            "hs": hs_add.where(keep),
            "tp": tp_add.where(keep),
            "dir": u10dir.where(keep),
            "spr": xr.full_like(hs_add, float(spr)).where(keep),
            "k": k,
        }
    )

    return out.expand_dims(goal=[WIND_GOAL], partition=[WIND_PARTITION]).transpose(
        "goal", "partition", "time", "sites"
    )


def offshore_wind_sea(
    forcing: pd.DataFrame,
    steepness_range: tuple[float, float] | None = None,
    spr_range: tuple[float, float] | None = None,
    speed_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Wind-sea weight and inferred wind of one goal x partition offshore forcing.

    Parameters
    ----------
    forcing : pd.DataFrame
        ``hs``, ``tp``, ``dir``, ``spr`` per time step (see
        :func:`bluemath_tk.waves.hywaves.metamodel.goal_forcing`).
    steepness_range, spr_range : tuple of float, optional
        Ramps of :func:`~bluemath_tk.waves.wind.wind_sea_weight` (defaults
        there).
    speed_factor : float, optional
        See :func:`~bluemath_tk.waves.wind.infer_wind`.

    Returns
    -------
    pd.DataFrame
        ``ws_weight``, ``u10``, ``u10dir`` indexed like *forcing*.
    """

    from ..wind import WIND_SEA_SPREAD, WIND_SEA_STEEPNESS, infer_wind, wind_sea_weight

    ramps: Mapping[str, tuple[float, float]] = {
        "steepness_range": steepness_range or WIND_SEA_STEEPNESS,
        "spr_range": spr_range or WIND_SEA_SPREAD,
    }
    weight = wind_sea_weight(forcing["hs"], forcing["tp"], forcing["spr"], **ramps)
    u10, u10dir = infer_wind(
        forcing["hs"], forcing["tp"], forcing["dir"], forcing["spr"],
        speed_factor=speed_factor, **ramps,
    )

    return pd.DataFrame(
        {"ws_weight": np.nan_to_num(weight), "u10": u10, "u10dir": u10dir},
        index=forcing.index,
    )
