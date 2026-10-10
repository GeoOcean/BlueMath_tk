"""
Scatter plots, including reference-vs-estimate validation scatters.

Validation scatters follow the :mod:`bluemath_tk.core.metrics` convention:
the **reference** goes on the x-axis and the **estimate** on the y-axis, and
the annotated metrics are computed as ``metric(reference, estimate)`` (a
positive bias means the estimate overestimates).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde, probplot

from ..metrics import METRIC_LABELS, compute_metrics, linear_fit, paired_finite
from .base_plotting import DefaultStaticPlotting
from .colors import default_colors

#: Metrics annotated by :func:`validation_scatter` by default.
DEFAULT_SCATTER_METRICS: tuple[str, ...] = ("bias", "rmse", "rrmse", "r2")


def density_scatter(
    x: np.ndarray,
    y: np.ndarray,
    max_kde_points: int = 5000,
    bins: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute point densities for a density-coloured scatter plot.

    Small samples use a gaussian KDE. Above *max_kde_points* (KDE evaluation is
    O(n²)) the density is the count of the 2-D histogram bin each point falls
    in, which is O(n) and visually equivalent for large samples.

    Parameters
    ----------
    x : np.ndarray
        X values for the scatter plot.
    y : np.ndarray
        Y values for the scatter plot.
    max_kde_points : int, optional
        Largest sample evaluated with a KDE. Default is 5000.
    bins : int, optional
        Bins per axis for the histogram density. Default is 100.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(x, y, density)`` sorted by increasing density, so the densest
        points are drawn last (on top).
    """

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")

    z = None
    if x.size <= max_kde_points:
        try:
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
        except (np.linalg.LinAlgError, ValueError):
            z = None  # degenerate sample (e.g. constant values): use histogram
    if z is None:
        counts, xedges, yedges = np.histogram2d(x, y, bins=bins)
        ix = np.clip(np.digitize(x, xedges) - 1, 0, counts.shape[0] - 1)
        iy = np.clip(np.digitize(y, yedges) - 1, 0, counts.shape[1] - 1)
        z = counts[ix, iy]

    idx = z.argsort()

    return x[idx], y[idx], z[idx]


def _format_metrics_box(metrics: Mapping[str, float], fmt: str) -> str:
    """Multi-line ``LABEL = value`` text for a metrics annotation."""

    lines = []
    for name, value in metrics.items():
        label = METRIC_LABELS.get(name, name)
        if name == "n":
            lines.append(f"{label} = {int(value)}")
        else:
            lines.append(f"{label} = {value:{fmt}}")

    return "\n".join(lines)


def _axis_limits(ref: np.ndarray, est: np.ndarray) -> tuple[float, float]:
    """Square limits covering both arrays; start at 0 for non-negative data."""

    lo = float(min(ref.min(), est.min()))
    hi = float(max(ref.max(), est.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lo = 0.0 if lo >= 0 else lo - pad

    return lo, hi + pad


def validation_scatter(
    ax: Axes,
    reference: np.ndarray,
    estimate: np.ndarray,
    reference_label: str = "Reference",
    estimate_label: str = "Estimate",
    title: str | None = None,
    *,
    units: str | None = None,
    circular: bool = False,
    metrics: Iterable[str] | None = DEFAULT_SCATTER_METRICS,
    show_n: bool = True,
    limits: tuple[float, float] | None = None,
    qq: bool = True,
    fit_line: bool = False,
    cmap: str = "rainbow",
    s: float = 5,
    max_points: int | None = 20000,
    seed: int = 0,
    fmt: str = ".3f",
    rrmse_normalization: str = "mean",
) -> dict[str, float]:
    """
    Density scatter of an estimate against its reference, with metrics.

    The reference goes on the x-axis and the estimate on the y-axis; points
    above the 1:1 line are overestimations. Metrics are computed on **all**
    finite pairs, even when the drawn points are subsampled.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    reference : np.ndarray
        Reference values (observations, high-fidelity model, ...): x-axis.
    estimate : np.ndarray
        Estimated values being evaluated: y-axis.
    reference_label : str, optional
        X-axis label. Default is "Reference".
    estimate_label : str, optional
        Y-axis label. Default is "Estimate".
    title : str, optional
        Axes title. Default is None (no title).
    units : str, optional
        Appended to both axis labels as `` [units]``. Default is None.
    circular : bool, optional
        Treat values as directions in degrees: limits fixed to ``[0, 360]``,
        circular metrics, no Q-Q line. Default is False.
    metrics : iterable of str, optional
        Metric names (see :data:`bluemath_tk.core.metrics.AVAILABLE_METRICS`)
        annotated on the plot; ``None`` or empty for no annotation. Default
        is :data:`DEFAULT_SCATTER_METRICS`.
    show_n : bool, optional
        Also annotate the number of pairs. Default is True.
    limits : tuple[float, float], optional
        Shared ``(min, max)`` for both axes. Default is computed from data.
    qq : bool, optional
        Overlay a Q-Q line (reference vs estimate quantiles). Default is True.
    fit_line : bool, optional
        Overlay the least-squares fit ``estimate = a * reference + b``.
        Default is False.
    cmap : str, optional
        Colormap for the point density. Default is "rainbow".
    s : float, optional
        Marker size. Default is 5.
    max_points : int, optional
        Randomly subsample to this many points for drawing only. ``None``
        draws everything. Default is 20000.
    seed : int, optional
        Seed for the drawing subsample. Default is 0.
    fmt : str, optional
        Number format for the annotation. Default is ".3f".
    rrmse_normalization : {"mean", "rms"}, optional
        Passed to :func:`bluemath_tk.core.metrics.rrmse`. Default is "mean".

    Returns
    -------
    dict[str, float]
        The metrics computed on all finite pairs (always includes ``n``).
    """

    ref, est = paired_finite(reference, estimate)
    if circular:
        ref, est = np.mod(ref, 360.0), np.mod(est, 360.0)

    suffix = f" [{units}]" if units else ""
    ax.set_xlabel(f"{reference_label}{suffix}")
    ax.set_ylabel(f"{estimate_label}{suffix}")
    if title is not None:
        ax.set_title(title)

    names = list(metrics or [])
    scores = compute_metrics(
        ref,
        est,
        names,
        circular=circular,
        rrmse_normalization=rrmse_normalization,
    )
    if ref.size == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        return scores

    # draw (possibly subsampled) density scatter
    ref_plot, est_plot = ref, est
    if max_points is not None and ref.size > max_points:
        idx = np.random.default_rng(seed).choice(ref.size, max_points, replace=False)
        ref_plot, est_plot = ref[idx], est[idx]
    xs, ys, z = density_scatter(ref_plot, est_plot)
    ax.scatter(xs, ys, c=z, s=s, cmap=cmap, edgecolors="none")

    # 1:1 line and limits
    if limits is None:
        limits = (0.0, 360.0) if circular else _axis_limits(ref, est)
    lo, hi = limits
    ax.plot([lo, hi], [lo, hi], "-r", lw=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    if qq and not circular:
        xq = probplot(ref, dist="norm")[0][1]
        yq = probplot(est, dist="norm")[0][1]
        ax.plot(xq, yq, "o", markersize=0.5, color="k", label="Q-Q plot")
    if fit_line and not circular:
        slope, intercept = linear_fit(ref, est)
        if np.isfinite(slope):
            ax.plot(
                [lo, hi],
                [slope * lo + intercept, slope * hi + intercept],
                "--",
                color="grey",
                lw=1.0,
                label=f"y = {slope:.2f}x {intercept:+.2f}",
            )

    shown = {k: v for k, v in scores.items() if k != "n"}
    if show_n:
        shown["n"] = scores["n"]
    if shown:
        props = dict(
            boxstyle="round", facecolor="w", edgecolor="grey", linewidth=0.8, alpha=0.6
        )
        ax.text(
            0.04,
            0.96,
            _format_metrics_box(shown, fmt),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=props,
        )

    return scores


def scatter_grid_shape(n: int) -> tuple[int, int]:
    """
    Return ``(nrows, ncols)`` for *n* panels: up to 2x2, then roughly square.

    Parameters
    ----------
    n : int
        Number of panels.

    Returns
    -------
    tuple[int, int]
        Grid shape.
    """

    if n <= 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n <= 4:
        return 2, 2
    ncols = min(4, int(np.ceil(np.sqrt(n))))

    return int(np.ceil(n / ncols)), ncols


def validation_scatter_grid(
    reference: Mapping[str, np.ndarray] | pd.DataFrame,
    estimate: Mapping[str, np.ndarray] | pd.DataFrame,
    variables: Sequence[str] | None = None,
    reference_label: str = "Reference",
    estimate_label: str = "Estimate",
    *,
    units: Mapping[str, str] | None = None,
    circular_variables: Iterable[str] = ("dir",),
    titles: Mapping[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
    **kwargs,
) -> tuple[Figure, np.ndarray, dict[str, dict[str, float]]]:
    """
    One :func:`validation_scatter` panel per variable.

    Parameters
    ----------
    reference : mapping or DataFrame
        Variable name -> reference values.
    estimate : mapping or DataFrame
        Variable name -> estimated values, paired with *reference*.
    variables : sequence of str, optional
        Variables to plot, in order. Default is every variable in both.
    reference_label : str, optional
        X-axis label prefix. Default is "Reference".
    estimate_label : str, optional
        Y-axis label prefix. Default is "Estimate".
    units : mapping, optional
        Variable name -> units for the axis labels.
    circular_variables : iterable of str, optional
        Variables treated as directions in degrees. Default is ``("dir",)``.
    titles : mapping, optional
        Variable name -> panel title. Default is the variable name.
    figsize : tuple[float, float], optional
        Figure size. Default scales with the grid.
    suptitle : str, optional
        Figure title.
    **kwargs
        Forwarded to :func:`validation_scatter`.

    Returns
    -------
    tuple[Figure, np.ndarray, dict]
        Figure, flat array of axes, and ``{variable: metrics}``.
    """

    if variables is None:
        variables = [v for v in reference if v in estimate]
    variables = list(variables)
    units = units or {}
    titles = titles or {}
    circular_variables = set(circular_variables)

    nrows, ncols = scatter_grid_shape(len(variables))
    if figsize is None:
        figsize = (4.2 * ncols, 3.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    scores: dict[str, dict[str, float]] = {}
    for ax, var in zip(axes, variables):
        scores[var] = validation_scatter(
            ax,
            np.asarray(reference[var]),
            np.asarray(estimate[var]),
            f"{reference_label} {var}",
            f"{estimate_label} {var}",
            titles.get(var, var),
            units=units.get(var),
            circular=var in circular_variables,
            **kwargs,
        )
    for ax in axes[len(variables) :]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()

    return fig, axes, scores


def plot_scatters_in_triangle(
    dataframes: list[pd.DataFrame],
    data_colors: list[str] = None,
    **kwargs,
) -> tuple[Figure, np.ndarray]:
    """
    Plot scatter plots of the dataframes with axes in a triangle arrangement.

    Parameters
    ----------
    dataframes : List[pd.DataFrame]
        List of dataframes to plot. Each dataframe should contain the same columns.
    data_colors : Optional[List[str]], optional
        List of colors for the dataframes. If None, uses default_colors.
    **kwargs : dict, optional
        Additional keyword arguments for the scatter plot. These will be passed to
        matplotlib.pyplot.scatter. Common parameters include:
        - s : float, marker size
        - alpha : float, transparency
        - marker : str, marker style

    Returns
    -------
    Tuple[Figure, np.ndarray]
        A tuple containing:
        - Figure object
        - 2D array of Axes objects

    Raises
    ------
    ValueError
        If the variables in the first dataframe are not present in all other dataframes.
    """

    if data_colors is None:
        data_colors = default_colors

    # Get the number and names of variables from the first dataframe
    variables_names = list(dataframes[0].columns)
    num_variables = len(variables_names)

    # Check variables names are in all dataframes
    for df in dataframes:
        if not all(v in df.columns for v in variables_names):
            raise ValueError(
                f"Variables {variables_names} are not in dataframe {df.columns}."
            )

    # Create figure and axes
    default_static_plot = DefaultStaticPlotting()
    fig, axes = default_static_plot.get_subplots(
        nrows=num_variables - 1,
        ncols=num_variables - 1,
        sharex=False,
        sharey=False,
    )
    if isinstance(axes, Axes):
        axes = np.array([[axes]])

    for c1, v1 in enumerate(variables_names[1:]):
        for c2, v2 in enumerate(variables_names[:-1]):
            for idf, df in enumerate(dataframes):
                default_static_plot.plot_scatter(
                    ax=axes[c2, c1],
                    x=df[v1],
                    y=df[v2],
                    c=data_colors[idf],
                    alpha=0.6,
                    **kwargs,
                )
            if c1 == c2:
                axes[c2, c1].set_xlabel(variables_names[c1 + 1])
                axes[c2, c1].set_ylabel(variables_names[c2])
            elif c1 > c2:
                axes[c2, c1].xaxis.set_ticklabels([])
                axes[c2, c1].yaxis.set_ticklabels([])
            else:
                fig.delaxes(axes[c2, c1])

    return fig, axes
