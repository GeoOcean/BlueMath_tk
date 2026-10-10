"""
Wave steepness and physically plausible sea states.

The deep-water steepness of a sea state is ``Hs / L0``, with the deep-water
wavelength ``L0 = g Tp^2 / (2 pi)``. Sea states much steeper than about
0.06 do not occur in nature (waves break by whitecapping first), so
combinations sampled independently, e.g. by LHS over Hs and Tp, are filtered
with :func:`filter_by_steepness` to keep only plausible ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.constants import GRAVITY


def deep_water_wavelength(tp: float | np.ndarray) -> float | np.ndarray:
    """
    Deep-water wavelength ``L0 = g Tp^2 / (2 pi)``.

    Parameters
    ----------
    tp : float or np.ndarray
        Wave period (s).

    Returns
    -------
    float or np.ndarray
        Wavelength (m).
    """

    return GRAVITY * np.asarray(tp, dtype=float) ** 2 / (2 * np.pi)


def wave_steepness(
    hs: float | np.ndarray, tp: float | np.ndarray
) -> float | np.ndarray:
    """
    Deep-water wave steepness ``Hs / L0``.

    Parameters
    ----------
    hs : float or np.ndarray
        Significant wave height (m).
    tp : float or np.ndarray
        Peak period (s).

    Returns
    -------
    float or np.ndarray
        Steepness (dimensionless).
    """

    return np.asarray(hs, dtype=float) / deep_water_wavelength(tp)


def filter_by_steepness(
    cases: pd.DataFrame,
    max_steepness: float = 0.06,
    hs: str = "hs",
    tp: str = "tp",
) -> pd.DataFrame:
    """
    Keep the sea states whose steepness ``Hs / L0`` is at most *max_steepness*.

    Parameters
    ----------
    cases : pd.DataFrame
        Sea states, one per row (e.g. an LHS design).
    max_steepness : float, optional
        Largest plausible steepness. Default is 0.06.
    hs, tp : str, optional
        Columns with the significant wave height (m) and peak period (s).
        Default are "hs" and "tp".

    Returns
    -------
    pd.DataFrame
        The plausible rows, with their original index.
    """

    keep = wave_steepness(cases[hs], cases[tp]) <= max_steepness

    return cases[keep]
