"""
Wind sea from bulk wave parameters: classification, inferred wind, growth curves.

A sea state is a young wind sea, actively forced by the local wind, when it is
steep and directionally broad; long, narrow-spread seas are swell. The
deep-water steepness ``Hs / L0`` (see :mod:`.steepness`) separates them: about
0.025 is a fully developed sea, above 0.03 a young wind sea, below 0.02 swell.
:func:`wind_sea_weight` turns this into a smooth weight in ``[0, 1]``.

Growth curves are the ones SnapWave's wind source is built on: Kahma & Calkoen
(1992) fetch-limited growth, bounded by the Pierson-Moskowitz fully developed
sea and by the Breugem & Holthuijsen (2007) depth limit. In dimensionless form
(``~`` = scaled by ``g`` and the 10 m wind speed ``U``)::

    g Hs / U^2 = 0.00288 (g X / U^2)^0.45 <= min(0.24, 0.13 (g d / U^2)^0.65)
    g Tp / U   = 0.459   (g X / U^2)^0.27 <= min(7.69, 5.0  (g d / U^2)^0.375)

The fully developed limits give the lowest wind speed that sustains a sea
state (:func:`fully_developed_wind`), used by :func:`infer_wind` when no wind
data are available: the wind blows from the wave direction at that speed.

Wind data (e.g. ERA5 ``u10`` / ``v10``) are read with :func:`wind_field_at`
and :func:`wind_speed_direction`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..core.constants import GRAVITY
from .steepness import wave_steepness

#: Kahma & Calkoen (1992): ``g Hs / U^2 = a (g X / U^2)^b`` and ``g Tp / U = a (g X / U^2)^b``.
KC_HS = (0.00288, 0.45)
KC_TP = (0.459, 0.27)
#: Breugem & Holthuijsen (2007) depth limits, same form with ``g d / U^2``.
BH_HS = (0.13, 0.65)
BH_TP = (5.0, 0.375)
#: Pierson-Moskowitz fully developed sea: ``g Hs / U^2`` and ``g Tp / U``.
FULLY_DEVELOPED_HS = 0.24
FULLY_DEVELOPED_TP = 7.69

#: Steepness (``Hs / L0``) ramp from swell (weight 0) to wind sea (weight 1).
WIND_SEA_STEEPNESS = (0.02, 0.03)
#: Directional spreading (deg) ramp from swell (weight 0) to wind sea (weight 1).
WIND_SEA_SPREAD = (20.0, 25.0)


def _ramp(x: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """Linear ramp from 0 at ``bounds[0]`` to 1 at ``bounds[1]`` (NaN stays NaN)."""

    low, high = bounds
    if high <= low:
        raise ValueError(f"ramp bounds must increase, got {bounds}")

    return np.clip((np.asarray(x, dtype=float) - low) / (high - low), 0.0, 1.0)


def wind_sea_weight(
    hs: float | np.ndarray,
    tp: float | np.ndarray,
    spr: float | np.ndarray | None = None,
    steepness_range: tuple[float, float] = WIND_SEA_STEEPNESS,
    spr_range: tuple[float, float] = WIND_SEA_SPREAD,
) -> np.ndarray:
    """
    How much a sea state is a locally forced wind sea, from 0 (swell) to 1.

    The product of two linear ramps: deep-water steepness ``Hs / L0`` over
    *steepness_range* and directional spreading over *spr_range*. Ramps
    instead of thresholds keep anything built on the weight (an inferred
    wind, a wind-sea correction) continuous in the sea-state parameters.

    Parameters
    ----------
    hs : float or np.ndarray
        Significant wave height (m).
    tp : float or np.ndarray
        Peak period (s).
    spr : float or np.ndarray, optional
        Directional spreading (deg). None skips the spreading ramp.
    steepness_range : tuple of float, optional
        Steepness at weight 0 and 1. Default :data:`WIND_SEA_STEEPNESS`.
    spr_range : tuple of float, optional
        Spreading (deg) at weight 0 and 1. Default :data:`WIND_SEA_SPREAD`.

    Returns
    -------
    np.ndarray
        Weight in ``[0, 1]`` (0 where ``hs <= 0``; NaN where inputs are NaN).
    """

    hs = np.asarray(hs, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = _ramp(wave_steepness(hs, tp), steepness_range)
    if spr is not None:
        weight = weight * _ramp(spr, spr_range)

    return np.where(hs > 0, weight, np.where(np.isnan(hs), np.nan, 0.0))


def fully_developed_wind(
    hs: float | np.ndarray, tp: float | np.ndarray
) -> np.ndarray:
    """
    Lowest 10 m wind speed for which a sea state is still growing.

    Wind input stops once ``g Hs / U^2`` reaches :data:`FULLY_DEVELOPED_HS`
    or ``g Tp / U`` reaches :data:`FULLY_DEVELOPED_TP`, so the sea state is
    fully developed for ``U = max(sqrt(g Hs / 0.24), g Tp / 7.69)``. A younger
    (steeper) sea was raised by a stronger wind: this is a lower bound.

    Parameters
    ----------
    hs : float or np.ndarray
        Significant wave height (m).
    tp : float or np.ndarray
        Peak period (s).

    Returns
    -------
    np.ndarray
        Wind speed (m/s).
    """

    hs = np.clip(np.asarray(hs, dtype=float), 0.0, None)

    return np.maximum(
        np.sqrt(GRAVITY * hs / FULLY_DEVELOPED_HS),
        GRAVITY * np.asarray(tp, dtype=float) / FULLY_DEVELOPED_TP,
    )


def infer_wind(
    hs: float | np.ndarray,
    tp: float | np.ndarray,
    dir: float | np.ndarray,
    spr: float | np.ndarray | None = None,
    speed_factor: float = 1.0,
    steepness_range: tuple[float, float] = WIND_SEA_STEEPNESS,
    spr_range: tuple[float, float] = WIND_SEA_SPREAD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Plausible local wind for a sea state, when no wind data are available.

    The wind blows from the wave direction at ``speed_factor`` times the
    fully developed speed (:func:`fully_developed_wind`), scaled by the
    wind-sea weight (:func:`wind_sea_weight`): swell gets no wind, and the
    speed grows smoothly across the swell-to-wind-sea ramps.

    Parameters
    ----------
    hs : float or np.ndarray
        Significant wave height (m).
    tp : float or np.ndarray
        Peak period (s).
    dir : float or np.ndarray
        Wave direction (deg, nautical, coming from).
    spr : float or np.ndarray, optional
        Directional spreading (deg). None skips the spreading ramp.
    speed_factor : float, optional
        Multiplier on the fully developed speed. Default is 1.
    steepness_range, spr_range : tuple of float, optional
        See :func:`wind_sea_weight`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``u10`` (m/s) and ``u10dir`` (deg, nautical, coming from; the wave
        direction, in ``[0, 360)``).
    """

    weight = wind_sea_weight(hs, tp, spr, steepness_range, spr_range)
    u10 = weight * speed_factor * fully_developed_wind(hs, tp)

    return np.nan_to_num(u10, nan=0.0), np.mod(np.asarray(dir, dtype=float), 360.0)


def _depth_limit(
    u10: np.ndarray, depth: float | np.ndarray | None, coef: tuple[float, float]
) -> np.ndarray | float:
    """Breugem & Holthuijsen dimensionless limit, or ``inf`` without a depth."""

    if depth is None:
        return np.inf
    depth = np.clip(np.asarray(depth, dtype=float), 0.0, None)

    return coef[0] * (GRAVITY * depth / u10**2) ** coef[1]


def growth_hs(
    u10: float | np.ndarray,
    fetch: float | np.ndarray,
    depth: float | np.ndarray | None = None,
) -> np.ndarray:
    """
    Fetch-limited significant wave height of a wind sea grown from calm.

    Parameters
    ----------
    u10 : float or np.ndarray
        10 m wind speed (m/s), positive.
    fetch : float or np.ndarray
        Fetch (m).
    depth : float or np.ndarray, optional
        Water depth (m) for the depth limit. None is deep water.

    Returns
    -------
    np.ndarray
        Hs (m).
    """

    u10 = np.asarray(u10, dtype=float)
    x = GRAVITY * np.clip(np.asarray(fetch, dtype=float), 0.0, None) / u10**2
    hs = np.minimum(KC_HS[0] * x ** KC_HS[1], FULLY_DEVELOPED_HS)
    hs = np.minimum(hs, _depth_limit(u10, depth, BH_HS))

    return hs * u10**2 / GRAVITY


def growth_tp(
    u10: float | np.ndarray,
    fetch: float | np.ndarray,
    depth: float | np.ndarray | None = None,
) -> np.ndarray:
    """
    Fetch-limited peak period of a wind sea grown from calm.

    Parameters
    ----------
    u10 : float or np.ndarray
        10 m wind speed (m/s), positive.
    fetch : float or np.ndarray
        Fetch (m).
    depth : float or np.ndarray, optional
        Water depth (m) for the depth limit. None is deep water.

    Returns
    -------
    np.ndarray
        Tp (s).
    """

    u10 = np.asarray(u10, dtype=float)
    x = GRAVITY * np.clip(np.asarray(fetch, dtype=float), 0.0, None) / u10**2
    tp = np.minimum(KC_TP[0] * x ** KC_TP[1], FULLY_DEVELOPED_TP)
    tp = np.minimum(tp, _depth_limit(u10, depth, BH_TP))

    return tp * u10 / GRAVITY


def equivalent_fetch(
    hs: float | np.ndarray, u10: float | np.ndarray
) -> np.ndarray:
    """
    Fetch over which the wind *u10* grows a sea from calm to *hs* (deep water).

    Inverse of :func:`growth_hs` without limits; ``inf`` when *hs* is at or
    above the fully developed height for *u10*.

    Parameters
    ----------
    hs : float or np.ndarray
        Significant wave height (m).
    u10 : float or np.ndarray
        10 m wind speed (m/s), positive.

    Returns
    -------
    np.ndarray
        Fetch (m).
    """

    u10 = np.asarray(u10, dtype=float)
    hs_n = GRAVITY * np.clip(np.asarray(hs, dtype=float), 0.0, None) / u10**2
    x = (hs_n / KC_HS[0]) ** (1.0 / KC_HS[1])

    return np.where(hs_n >= FULLY_DEVELOPED_HS, np.inf, x * u10**2 / GRAVITY)


def wind_speed_direction(
    u10: float | np.ndarray, v10: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Speed and nautical direction (coming from) of 10 m wind components.

    Parameters
    ----------
    u10, v10 : float or np.ndarray
        Eastward and northward wind (m/s), e.g. ERA5 ``u10`` / ``v10``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Speed (m/s) and direction (deg, nautical, coming from, ``[0, 360)``).
    """

    u10, v10 = np.asarray(u10, dtype=float), np.asarray(v10, dtype=float)

    return np.hypot(u10, v10), np.mod(270.0 - np.rad2deg(np.arctan2(v10, u10)), 360.0)


def wind_field_at(wind: xr.Dataset, time: str | np.datetime64) -> xr.Dataset:
    """
    Wind field at one time, linear in time between the records.

    Parameters
    ----------
    wind : xr.Dataset
        ``u10`` and ``v10`` (m/s) on ``(time, latitude, longitude)``, e.g.
        ERA5 single levels with ``valid_time`` renamed to ``time``.
    time : str or np.datetime64
        Requested time, within the records.

    Returns
    -------
    xr.Dataset
        ``u10`` and ``v10`` on ``(latitude, longitude)``.

    Raises
    ------
    ValueError
        If *time* is outside the wind records.
    """

    t = np.datetime64(pd.Timestamp(time))
    times = wind["time"].values
    if t < times.min() or t > times.max():
        raise ValueError(f"wind has no record around {t} ({times.min()} - {times.max()})")

    wind = wind[["u10", "v10"]]
    if t in times:
        return wind.sel(time=t).drop_vars("time")

    return wind.interp(time=t).drop_vars("time")
