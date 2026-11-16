"""
Project: BlueMath_tk
Sub-Module: waves.hywaves.depth_cap
Author: GeoOcean Research Group, Universidad de Cantabria
Repository: https://github.com/GeoOcean/BlueMath_tk.git
Status: Under development (Working)

Depth cap of the linear sum.

Each goal is trained alone, and its waves break in the nearshore model, so at
every site the Hs / h of one goal stays under a ceiling set by depth-induced
breaking. The linear sum adds energy that a run with every goal would break:
in the biggest seas the sum goes above that ceiling at shallow sites. The cap
brings it back:

    cap = ceiling x h,    ceiling = max over the training cases of Hs / h

with h the water depth of the hour (mesh depth + water level). Hs becomes
``min(Hs, cap)``; the summed spectrum is scaled by the same energy factor
``(cap / Hs)^2``, so its shape (Tp, direction, spreading) is kept. The ceiling
needs only the training cases: nothing is fitted on dynamic runs.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

#: wavespectra stats that scale with the square root of the energy.
HEIGHT_STATS = frozenset({"hs", "hmax"})
#: Depths (m) at or below which a case does not count for the ceiling (nearly dry: Hs / h blows up).
MIN_DEPTH = 0.5


def depth_ceiling(hs: ArrayLike, depth: ArrayLike, min_depth: float = MIN_DEPTH) -> np.ndarray:
    """
    Per site, the highest Hs / h over the training cases.

    Parameters
    ----------
    hs : array-like
        Hs (m) on ``(case, site)``.
    depth : array-like
        Water depth h (m), broadcastable to ``(case, site)`` (e.g. mesh depth
        + the case water level).
    min_depth : float, optional
        Cases with ``h <= min_depth`` at a site are ignored there.

    Returns
    -------
    np.ndarray
        Ceiling per site; NaN where no case counts.
    """
    hs = np.asarray(hs, dtype=float)
    depth = np.broadcast_to(np.asarray(depth, dtype=float), hs.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(depth > min_depth, hs / depth, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN sites
        return np.nanmax(ratio, axis=0)


def cap_factor(hs: xr.DataArray, cap: xr.DataArray) -> xr.DataArray:
    """
    Hs multiplier ``min(1, cap / Hs)``.

    It is 1 below the cap, where Hs is 0 or NaN, and where the cap is NaN
    (a site without a ceiling is not capped).
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = (cap / hs).where(hs > 0)
    return factor.clip(max=1.0).fillna(1.0)


def apply_depth_cap(
    bulk: xr.Dataset,
    cap: xr.DataArray,
    variables: Mapping[str, str],
    spectra: xr.Dataset | None = None,
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """
    Cap a reconstruction at ``cap`` (m).

    Parameters
    ----------
    bulk : xr.Dataset
        Bulk parameters of the linear sum, on ``(time, sites)``.
    cap : xr.DataArray
        Highest Hs (m), broadcastable to *bulk* (``ceiling x h``).
    variables : mapping
        Output name -> wavespectra stat of *bulk*; the variable with the stat
        ``"hs"`` sets the factor, the :data:`HEIGHT_STATS` variables are
        scaled by it and the others (periods, directions) are kept.
    spectra : xr.Dataset, optional
        Summed ``efth`` of the same sum, scaled by the factor squared.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset | None]
        The capped bulk parameters and spectra.

    Raises
    ------
    ValueError
        If no variable has the stat ``"hs"``.
    """
    names = [name for name, stat in variables.items() if stat == "hs" and name in bulk]
    if not names:
        raise ValueError("The depth cap needs Hs: no variable with the wavespectra stat 'hs'")
    factor = cap_factor(bulk[names[0]], cap)
    out = bulk.copy()
    for name, stat in variables.items():
        if stat in HEIGHT_STATS and name in out:
            out[name] = (out[name] * factor).astype(out[name].dtype)
    if spectra is not None:
        spectra = spectra.copy()
        spectra["efth"] = (spectra["efth"] * factor**2).astype(spectra["efth"].dtype)
    return out, spectra
