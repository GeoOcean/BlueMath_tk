"""Time-series plots comparing several estimates against one reference."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import xarray as xr
from matplotlib.axes import Axes

SeriesLike = pd.Series | xr.DataArray


def _to_series(data: SeriesLike) -> pd.Series:
    """1-D time-indexed :class:`pandas.Series` from a Series or DataArray."""

    if isinstance(data, xr.DataArray):
        return data.squeeze(drop=True).to_series()

    return data


def plot_timeseries_comparison(
    ax: Axes,
    reference: SeriesLike,
    estimates: Mapping[str, SeriesLike],
    reference_label: str = "Reference",
    *,
    colors: Mapping[str, str] | None = None,
    reference_color: str = "black",
    circular: bool = False,
    ylabel: str | None = None,
    crop_to_reference: bool = True,
    legend: bool = True,
) -> Axes:
    """
    Overlay one or more estimates on a reference time series.

    Each estimate is restricted to the reference's time stamps (inner join),
    so all lines cover the same period. Directional variables are drawn as
    markers, since lines across the 0/360 wrap are misleading.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    reference : pd.Series or xr.DataArray
        Reference series (e.g. buoy observations), indexed by time.
    estimates : mapping
        Label -> estimated series (e.g. models), indexed by time.
    reference_label : str, optional
        Legend label of the reference. Default is "Reference".
    colors : mapping, optional
        Label -> colour for the estimates. Default is the matplotlib cycle.
    reference_color : str, optional
        Colour of the reference. Default is "black".
    circular : bool, optional
        Directions in degrees: markers instead of lines, y-limits
        ``[0, 360]``. Default is False.
    ylabel : str, optional
        Y-axis label.
    crop_to_reference : bool, optional
        Limit the x-axis to the span where the reference has finite data.
        Default is True.
    legend : bool, optional
        Draw a legend. Default is True.

    Returns
    -------
    Axes
        The axes plotted on.
    """

    colors = colors or {}
    ref = _to_series(reference)

    if circular:
        ax.scatter(
            ref.index,
            ref.values % 360.0,
            s=8,
            color=reference_color,
            alpha=0.75,
            zorder=3,
            label=reference_label,
        )
    else:
        ax.plot(
            ref.index, ref.values, color=reference_color, lw=1.2, label=reference_label
        )

    for i, (label, est) in enumerate(estimates.items()):
        est = _to_series(est).reindex(ref.index)
        color = colors.get(label, f"C{i}")
        if circular:
            ax.scatter(
                est.index,
                est.values % 360.0,
                s=6,
                color=color,
                alpha=0.65,
                zorder=2,
                label=label,
            )
        else:
            ax.plot(est.index, est.values, color=color, lw=1.0, alpha=0.9, label=label)

    if circular:
        ax.set_ylim(0.0, 360.0)
    if crop_to_reference:
        valid = ref.dropna()
        if not valid.empty and valid.index[0] != valid.index[-1]:
            ax.set_xlim(valid.index[0], valid.index[-1])
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(loc="upper right", fontsize=8)

    return ax
