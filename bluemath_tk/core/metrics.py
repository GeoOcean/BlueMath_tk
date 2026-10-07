"""
Paired-sample validation metrics.

Convention
----------
Every metric takes ``(reference, estimate)``, in that order:

- ``reference`` is what you trust (observations, a buoy, a high-fidelity model
  run, ...). It goes on the x-axis of a validation scatter.
- ``estimate`` is what you are evaluating (a hindcast, a metamodel, a
  prediction, ...). It goes on the y-axis.

Which is which depends on what is being compared, so callers must set it
explicitly. Signed metrics follow ``estimate - reference``: a positive bias
means the estimate **overestimates** the reference.

Non-finite pairs (``NaN`` / ``inf`` on either side) are dropped before any
metric is computed; an empty pairing returns ``NaN``.

Directional variables (degrees) have circular counterparts
(:func:`circular_bias`, :func:`circular_rmse`, :func:`circular_mae`,
:func:`circular_r2`), all based on the shortest signed angular difference.
:func:`compute_metrics` dispatches to the right family with ``circular=True``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

ArrayLike = np.ndarray | Iterable[float]

#: Metrics returned by :func:`compute_metrics` when none are requested.
DEFAULT_METRICS: tuple[str, ...] = ("bias", "rmse", "rrmse", "si", "r", "r2")

#: Short display labels, e.g. for a validation scatter annotation.
METRIC_LABELS: dict[str, str] = {
    "n": "n",
    "bias": "BIAS",
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "rrmse": "RRMSE",
    "si": "SI",
    "hh": "HH",
    "r": "r",
    "r2": "R²",
    "willmott_d": "d",
    "slope": "slope",
    "intercept": "intercept",
}


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def paired_finite(
    reference: ArrayLike, estimate: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten two paired arrays and keep only the pairs finite on both sides.

    Parameters
    ----------
    reference : array-like
        Reference values.
    estimate : array-like
        Estimated values, paired element-wise with *reference*.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(reference, estimate)`` as flat float arrays without non-finite pairs.

    Raises
    ------
    ValueError
        If *reference* and *estimate* do not have the same number of elements.
    """

    ref = np.asarray(reference, dtype=float).ravel()
    est = np.asarray(estimate, dtype=float).ravel()
    if ref.size != est.size:
        raise ValueError(
            f"reference and estimate must have the same length "
            f"({ref.size} != {est.size})"
        )
    mask = np.isfinite(ref) & np.isfinite(est)

    return ref[mask], est[mask]


# ---------------------------------------------------------------------------
# Linear metrics
# ---------------------------------------------------------------------------


def bias(reference: ArrayLike, estimate: ArrayLike) -> float:
    """
    Mean error, ``mean(estimate - reference)``.

    Positive means the estimate overestimates the reference.

    Parameters
    ----------
    reference : array-like
        Reference values.
    estimate : array-like
        Estimated values.

    Returns
    -------
    float
        The bias, in the units of the data.
    """

    ref, est = paired_finite(reference, estimate)
    if ref.size == 0:
        return np.nan

    return float(np.mean(est - ref))


def mae(reference: ArrayLike, estimate: ArrayLike) -> float:
    """
    Mean Absolute Error, ``mean(|estimate - reference|)``.

    Parameters
    ----------
    reference : array-like
        Reference values.
    estimate : array-like
        Estimated values.

    Returns
    -------
    float
        Mean absolute error.
    """

    ref, est = paired_finite(reference, estimate)
    if ref.size == 0:
        return np.nan

    return float(np.mean(np.abs(est - ref)))


def mse(reference: ArrayLike, estimate: ArrayLike) -> float:
    """
    Mean Squared Error, ``mean((estimate - reference)**2)``.

    Parameters
    ----------
    reference : array-like
        Reference values.
    estimate : array-like
        Estimated values.

    Returns
    -------
    float
        Mean squared error.
    """

    ref, est = paired_finite(reference, estimate)
    if ref.size == 0:
        return np.nan

    return float(np.mean((est - ref) ** 2))


def rmse(reference: ArrayLike, estimate: ArrayLike) -> float:
    """
    Root Mean Squared Error.

    Parameters
    ----------
    reference : array-like
        Reference values.
    estimate : array-like
        Estimated values.

    Returns
    -------
    float
        Root mean squared error.
    """

    return float(np.sqrt(mse(reference, estimate)))


def rrmse(
    reference: ArrayLike, estimate: ArrayLike, normalization: str = "mean"
) -> float:
    """
    Relative RMSE: RMSE divided by a scale of the reference (dimensionless).

    Parameters
    ----------
    reference : array-like
        Reference values.
    estimate : array-like
        Estimated values.
    normalization : {"mean", "rms"}, optional
        Scale of the reference used as denominator: its mean (default) or its
        root mean square. ``"rms"`` is safer for variables whose mean can
        approach zero.

    Returns
    -------
    float
        Relative RMSE (multiply by 100 for a percentage); ``NaN`` if the
        denominator is zero.

    Raises
    ------
    ValueError
        If *normalization* is not ``"mean"`` or ``"rms"``.
    """

    if normalization not in ("mean", "rms"):
        raise ValueError(
            f"normalization must be 'mean' or 'rms', got {normalization!r}"
        )
    ref, est = paired_finite(reference, estimate)
    if ref.size == 0:
        return np.nan
    if normalization == "mean":
        scale = np.mean(ref)
    else:
        scale = np.sqrt(np.mean(ref**2))
    if scale == 0:
        return np.nan

    return float(np.sqrt(np.mean((est - ref) ** 2)) / scale)


def si(reference: ArrayLike, estimate: ArrayLike) -> float:
    """
    Scatter Index, bias-free form of Mentaschi et al. (2013).

    ``sqrt(sum(((est - mean(est)) - (ref - mean(ref)))**2) / sum(ref**2))``

    Parameters
    ----------
    reference : array-like
        Reference values.
    estimate : array-like
        Estimated values.

    Returns
    -------
    float
        The scatter index (dimensionless).

    References
    ----------
    Mentaschi, L., Besio, G., Cassola, F., Mazzino, A. (2013). Problems in
    RMSE-based wave model validations. Ocean Modelling, 72, 53-58.
    """

    ref, est = paired_finite(reference, estimate)
    if ref.size == 0:
        return np.nan
    denom = np.sum(ref**2)
    if denom == 0:
        return np.nan

    return float(
        np.sqrt(np.sum(((est - est.mean()) - (ref - ref.mean())) ** 2) / denom)
    )


def hh(reference: ArrayLike, estimate: ArrayLike) -> float:
    """
    Hanna & Heinold (1985) index, ``sqrt(sum((est - ref)**2) / sum(est * ref))``.

    Recommended by Mentaschi et al. (2013) over RMSE and SI for wave model
    validation, as it does not reward models that underestimate the scatter.

    Parameters
    ----------
    reference : array-like
        Reference values.
    estimate : array-like
        Estimated values.

    Returns
    -------
    float
        The HH index (dimensionless).
    """

    ref, est = paired_finite(reference, estimate)
    if ref.size == 0:
        return np.nan
    denom = np.sum(est * ref)
    if denom <= 0:
        return np.nan

    return float(np.sqrt(np.sum((est - ref) ** 2) / denom))


def pearson_r(reference: ArrayLike, estimate: ArrayLike) -> float:
    """
    Pearson correlation coefficient.

    Parameters
    ----------
    reference : array-like
        Reference values.
    estimate : array-like
        Estimated values.

    Returns
    -------
    float
        Correlation in ``[-1, 1]``; ``NaN`` if either side has zero variance.
    """

    ref, est = paired_finite(reference, estimate)
    if ref.size < 2 or np.std(ref) == 0 or np.std(est) == 0:
        return np.nan

    return float(np.corrcoef(ref, est)[0, 1])


def r2(reference: ArrayLike, estimate: ArrayLike) -> float:
    """
    Coefficient of determination, ``1 - SS_res / SS_tot(reference)``.

    Normalised by the variance of the **reference**, so it is not symmetric:
    swapping the arguments changes the result. It can be negative when the
    estimate is worse than the reference mean.

    Parameters
    ----------
    reference : array-like
        Reference values.
    estimate : array-like
        Estimated values.

    Returns
    -------
    float
        The R² score; ``NaN`` if the reference has zero variance.
    """

    ref, est = paired_finite(reference, estimate)
    if ref.size == 0:
        return np.nan
    ss_tot = np.sum((ref - ref.mean()) ** 2)
    if ss_tot == 0:
        return np.nan

    return float(1.0 - np.sum((est - ref) ** 2) / ss_tot)


def willmott_d(reference: ArrayLike, estimate: ArrayLike) -> float:
    """
    Willmott (1981) index of agreement, in ``[0, 1]`` (1 = perfect).

    Parameters
    ----------
    reference : array-like
        Reference values.
    estimate : array-like
        Estimated values.

    Returns
    -------
    float
        Index of agreement.
    """

    ref, est = paired_finite(reference, estimate)
    if ref.size == 0:
        return np.nan
    ref_mean = ref.mean()
    denom = np.sum((np.abs(est - ref_mean) + np.abs(ref - ref_mean)) ** 2)
    if denom == 0:
        return np.nan

    return float(1.0 - np.sum((est - ref) ** 2) / denom)


def linear_fit(reference: ArrayLike, estimate: ArrayLike) -> tuple[float, float]:
    """
    Least-squares line ``estimate = slope * reference + intercept``.

    Parameters
    ----------
    reference : array-like
        Reference values (x-axis).
    estimate : array-like
        Estimated values (y-axis).

    Returns
    -------
    tuple[float, float]
        ``(slope, intercept)``; ``NaN`` if the reference has zero variance.
    """

    ref, est = paired_finite(reference, estimate)
    if ref.size < 2 or np.std(ref) == 0:
        return np.nan, np.nan
    slope, intercept = np.polyfit(ref, est, 1)

    return float(slope), float(intercept)


def percentile_bias(
    reference: ArrayLike, estimate: ArrayLike, q: float = 95.0
) -> float:
    """
    Difference of percentiles, ``P_q(estimate) - P_q(reference)``.

    Unlike :func:`bias`, this compares the two distributions rather than
    paired errors, which is what matters for extremes (e.g. ``q=99`` for
    storm Hs). Only finite pairs are used, so both percentiles come from the
    same samples.

    Parameters
    ----------
    reference : array-like
        Reference values.
    estimate : array-like
        Estimated values.
    q : float, optional
        Percentile in ``[0, 100]``. Default is 95.

    Returns
    -------
    float
        Percentile difference; positive means the estimate's upper tail is
        too high.
    """

    ref, est = paired_finite(reference, estimate)
    if ref.size == 0:
        return np.nan

    return float(np.percentile(est, q) - np.percentile(ref, q))


# ---------------------------------------------------------------------------
# Circular metrics (degrees)
# ---------------------------------------------------------------------------


def circular_difference(reference: ArrayLike, estimate: ArrayLike) -> np.ndarray:
    """
    Shortest signed angular difference ``estimate - reference`` in degrees.

    Parameters
    ----------
    reference : array-like
        Reference directions in degrees.
    estimate : array-like
        Estimated directions in degrees.

    Returns
    -------
    np.ndarray
        Differences in ``[-180, 180)``; positive means the estimate is
        clockwise of the reference (for nautical directions).
    """

    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)

    return (est - ref + 180.0) % 360.0 - 180.0


def circular_mean(angles: ArrayLike) -> float:
    """
    Mean direction in degrees, in ``[0, 360)``.

    Parameters
    ----------
    angles : array-like
        Directions in degrees; non-finite values are ignored.

    Returns
    -------
    float
        Circular mean.
    """

    rad = np.deg2rad(np.asarray(angles, dtype=float).ravel())
    rad = rad[np.isfinite(rad)]
    if rad.size == 0:
        return np.nan

    return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)


def circular_bias(reference: ArrayLike, estimate: ArrayLike) -> float:
    """
    Mean shortest angular difference ``estimate - reference`` (degrees).

    Parameters
    ----------
    reference : array-like
        Reference directions in degrees.
    estimate : array-like
        Estimated directions in degrees.

    Returns
    -------
    float
        Circular bias.
    """

    ref, est = paired_finite(reference, estimate)
    if ref.size == 0:
        return np.nan

    return float(np.mean(circular_difference(ref, est)))


def circular_mae(reference: ArrayLike, estimate: ArrayLike) -> float:
    """
    Mean absolute shortest angular difference (degrees).

    Parameters
    ----------
    reference : array-like
        Reference directions in degrees.
    estimate : array-like
        Estimated directions in degrees.

    Returns
    -------
    float
        Circular MAE.
    """

    ref, est = paired_finite(reference, estimate)
    if ref.size == 0:
        return np.nan

    return float(np.mean(np.abs(circular_difference(ref, est))))


def circular_rmse(reference: ArrayLike, estimate: ArrayLike) -> float:
    """
    Root mean squared shortest angular difference (degrees).

    Parameters
    ----------
    reference : array-like
        Reference directions in degrees.
    estimate : array-like
        Estimated directions in degrees.

    Returns
    -------
    float
        Circular RMSE.
    """

    ref, est = paired_finite(reference, estimate)
    if ref.size == 0:
        return np.nan

    return float(np.sqrt(np.mean(circular_difference(ref, est) ** 2)))


def circular_r2(reference: ArrayLike, estimate: ArrayLike) -> float:
    """
    R² for directions, using shortest angular deviations.

    ``1 - sum(d(ref, est)**2) / sum(d(ref, circular_mean(ref))**2)``, with
    ``d`` the shortest angular difference.

    Parameters
    ----------
    reference : array-like
        Reference directions in degrees.
    estimate : array-like
        Estimated directions in degrees.

    Returns
    -------
    float
        Circular R²; ``NaN`` if the reference has no angular spread.
    """

    ref, est = paired_finite(reference, estimate)
    if ref.size == 0:
        return np.nan
    ss_res = np.sum(circular_difference(ref, est) ** 2)
    ss_tot = np.sum(circular_difference(circular_mean(ref), ref) ** 2)
    if ss_tot == 0:
        return np.nan

    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_LINEAR: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "bias": bias,
    "mae": mae,
    "mse": mse,
    "rmse": rmse,
    "rrmse": rrmse,
    "si": si,
    "hh": hh,
    "r": pearson_r,
    "r2": r2,
    "willmott_d": willmott_d,
    "slope": lambda ref, est: linear_fit(ref, est)[0],
    "intercept": lambda ref, est: linear_fit(ref, est)[1],
}

_CIRCULAR: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "bias": circular_bias,
    "mae": circular_mae,
    "mse": lambda ref, est: float(np.mean(circular_difference(ref, est) ** 2)),
    "rmse": circular_rmse,
    "r2": circular_r2,
}

#: Every metric name :func:`compute_metrics` understands.
AVAILABLE_METRICS: tuple[str, ...] = tuple(_LINEAR)


def compute_metrics(
    reference: ArrayLike,
    estimate: ArrayLike,
    metrics: Iterable[str] | None = None,
    *,
    circular: bool = False,
    rrmse_normalization: str = "mean",
) -> dict[str, float]:
    """
    Compute several metrics at once on the finite pairs of two arrays.

    Parameters
    ----------
    reference : array-like
        Reference values (observations, high-fidelity model, ...).
    estimate : array-like
        Estimated values being evaluated.
    metrics : iterable of str, optional
        Metric names from :data:`AVAILABLE_METRICS`. Default is
        :data:`DEFAULT_METRICS`.
    circular : bool, optional
        Treat values as directions in degrees. Metrics without a circular
        definition (``rrmse``, ``si``, ``hh``, ``r``, ...) are returned as
        ``NaN`` so tables keep the same columns. Default is False.
    rrmse_normalization : {"mean", "rms"}, optional
        Passed to :func:`rrmse`. Default is "mean".

    Returns
    -------
    dict[str, float]
        ``{"n": number_of_finite_pairs, <metric>: value, ...}``.

    Raises
    ------
    ValueError
        If an unknown metric name is requested.

    Examples
    --------
    >>> compute_metrics([1.0, 2.0, 3.0], [1.5, 2.5, 3.5], ["bias", "rmse"])
    {'n': 3, 'bias': 0.5, 'rmse': 0.5}
    """

    names = list(DEFAULT_METRICS if metrics is None else metrics)
    unknown = [m for m in names if m not in _LINEAR]
    if unknown:
        raise ValueError(
            f"Unknown metric(s) {unknown}; available: {list(AVAILABLE_METRICS)}"
        )

    ref, est = paired_finite(reference, estimate)
    out: dict[str, float] = {"n": int(ref.size)}
    for name in names:
        if ref.size == 0:
            out[name] = np.nan
        elif circular:
            func = _CIRCULAR.get(name)
            out[name] = func(ref, est) if func is not None else np.nan
        elif name == "rrmse":
            out[name] = rrmse(ref, est, normalization=rrmse_normalization)
        else:
            out[name] = _LINEAR[name](ref, est)

    return out
