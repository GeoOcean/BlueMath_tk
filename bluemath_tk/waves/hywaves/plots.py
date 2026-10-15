"""
Project: BlueMath_tk
Sub-Module: waves.hywaves.plots
Author: GeoOcean Research Group, Universidad de Cantabria
Repository: https://github.com/GeoOcean/BlueMath_tk.git
Status: Under development (Working)

Figures of the HyWaves method:

- goals: colours (:func:`goal_colors`), sectors with the sampled or observed
  wave directions on a map (:func:`plot_goal_sectors`) and per goal
  (:func:`plot_direction_ranges`);
- the met-ocean forcing of dynamic runs (:func:`plot_dynamic_forcing`);
- the wind correction's K (:func:`plot_wind_k_comparison`);
- the MDA-size sweep of the holdout metrics (:func:`plot_mda_sweep`).

Goals are mappings ``{goal_id: {"lon", "lat", "angles": [left, right]}}``,
directions nautical, coming-from, clockwise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Wedge

from ...core.metrics import METRIC_LABELS
from ...core.operations import nautical_sector_to_wedge_angles
from ...core.plotting.labels import place_labels

Goals = Mapping[str, Mapping[str, Any]]


def goal_colors(goal_ids: Sequence[Any]) -> dict[Any, tuple]:
    """
    One colour per goal from viridis, by rank (never repeats, unlike tab10).

    Goals are numbered along the domain boundary, so the colour is also a
    position cue.
    """

    positions = np.linspace(0.08, 0.92, len(goal_ids)) if len(goal_ids) > 1 else [0.5]
    cmap = plt.get_cmap("viridis")
    return {gid: cmap(pos) for gid, pos in zip(goal_ids, positions)}


def plot_goal_sectors(
    ax: Axes,
    goals: Goals,
    colors: Mapping[int, Any],
    *,
    directions: pd.DataFrame | None = None,
    radius: float = 0.35,
    arrow_length: float = 0.15,
    max_arrows: int = 25,
) -> None:
    """
    Goal sectors as wedges, goal stars with non-overlapping labels, wave arrows.

    Parameters
    ----------
    ax : Axes
        Map axes with its final limits (labels are placed on the rendered figure).
    goals : mapping
        ``{goal_id: {"lon", "lat", "angles"}}``.
    colors : mapping
        Colour by integer goal id (see :func:`goal_colors`).
    directions : pd.DataFrame, optional
        ``goal`` and ``dir`` columns: wave-from directions drawn as arrows
        into each goal (a random subsample of *max_arrows*).
    radius : float, optional
        Wedge radius (degrees).
    arrow_length : float, optional
        Arrow length (degrees).
    max_arrows : int, optional
        Arrows per goal.
    """

    keys = sorted(goals, key=int)
    crowded = len(keys) > 15
    for key in keys:
        goal = goals[key]
        color = colors[int(key)]
        theta1, theta2 = nautical_sector_to_wedge_angles(*goal["angles"])
        ax.add_patch(Wedge((goal["lon"], goal["lat"]), radius, theta1, theta2, facecolor=color,
                           edgecolor=color, alpha=0.15, linewidth=0.8, zorder=4))
        ax.plot(goal["lon"], goal["lat"], marker="*", color=color, markersize=9 if crowded else 12, zorder=6)
        if directions is None:
            continue
        sub = directions[directions["goal"] == int(key)]
        if len(sub) > max_arrows:
            sub = sub.sample(max_arrows, random_state=0)
        theta = np.deg2rad(sub["dir"].to_numpy(float))
        for dx, dy in zip(np.sin(theta) * arrow_length, np.cos(theta) * arrow_length):
            ax.annotate(
                "", xy=(goal["lon"], goal["lat"]), xytext=(goal["lon"] + dx, goal["lat"] + dy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, alpha=0.55, shrinkA=0,
                                shrinkB=2, mutation_scale=8),
                zorder=5,
            )
    place_labels(
        ax,
        [goals[k]["lon"] for k in keys],
        [goals[k]["lat"] for k in keys],
        [f"G{k}" for k in keys],
        leader_colors=[colors[int(k)] for k in keys],
        fontsize=7.5 if crowded else 9,
        weight="bold",
        zorder=7,
        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
    )


def _sector_spans(left: float, right: float) -> list[tuple[float, float]]:
    """One or two ``(start, end)`` spans on [0, 360) of a nautical sector."""
    left_n, right_n = float(np.mod(left, 360.0)), float(np.mod(right, 360.0))
    if left < 0 or left_n > right_n:
        return [(left_n, 360.0), (0.0, right_n)]
    return [(left_n, right_n)]


def _circular_spans(dirs: np.ndarray) -> list[tuple[float, float]]:
    """One or two spans on [0, 360) tightly bounding *dirs* (jumps the largest gap)."""
    vals = np.sort(np.mod(np.asarray(dirs, dtype=float), 360.0))
    if vals.size <= 1:
        v = float(vals[0]) if vals.size else 0.0
        return [(v, v)]
    gaps = np.append(np.diff(vals), vals[0] + 360.0 - vals[-1])
    i = int(np.argmax(gaps))
    if i == len(vals) - 1:
        return [(float(vals[0]), float(vals[-1]))]
    start, end = float(vals[i + 1]), float(vals[i])
    return [(start, 360.0), (0.0, end)]


def plot_direction_ranges(
    ax: Axes,
    cases: pd.DataFrame,
    goals: Goals,
    colors: Mapping[int, Any],
    *,
    bracket: str = "observed",
    max_marks: int = 200,
    title: str = "Wave directions per goal",
) -> bool:
    """
    One row per goal: its sector (light bar) and the directions of *cases*.

    Up to *max_marks* cases per goal every direction is a tick with a solid
    range bar; above that the row is a density strip (alpha = density).

    Parameters
    ----------
    ax : Axes
        Target axes.
    cases : pd.DataFrame
        ``goal``, ``dir`` and ``tp`` columns (one row per case or time step).
    goals : mapping
        ``{goal_id: {"angles"}}``.
    colors : mapping
        Colour by integer goal id.
    bracket : {"observed", "sector"}, optional
        What the crisp outline bounds: the observed interval (designed cases)
        or the configured sector (hindcast, whose directions may spill out).
    max_marks : int, optional
        Cases per goal above which rows become density strips.
    title : str, optional
        Axes title.

    Returns
    -------
    bool
        Whether the density-strip style was used (for a legend text).
    """

    goal_ids = sorted(cases["goal"].unique())
    dense = bool((cases.groupby("goal").size() > max_marks).any())
    for y, gid in enumerate(goal_ids):
        color = colors[gid]
        sub = cases[cases["goal"] == gid]
        dirs = sub["dir"].to_numpy(float)
        sector = _sector_spans(*goals[str(gid)]["angles"])
        for x0, x1 in sector:
            ax.barh(y, x1 - x0, left=x0, height=0.55, color=color, alpha=0.18, edgecolor=color, linewidth=0.6)
        spans = sector if bracket == "sector" else _circular_spans(dirs)
        if dense:
            counts, _ = np.histogram(np.mod(dirs, 360.0), bins=120, range=(0, 360))
            rgba = np.zeros((1, 120, 4))
            rgba[0, :, :3] = mcolors.to_rgb(color)
            rgba[0, :, 3] = counts / max(counts.max(), 1) * 0.85
            ax.imshow(rgba, extent=(0, 360, y - 0.31, y + 0.31), aspect="auto",
                      interpolation="nearest", zorder=2)
            edge = tuple(c * 0.55 for c in mcolors.to_rgb(color))
            for x0, x1 in spans:
                ax.barh(y, x1 - x0, left=x0, height=0.62, fill=False, edgecolor=edge, linewidth=1.4, zorder=3)
        else:
            for x0, x1 in spans:
                ax.barh(y, x1 - x0, left=x0, height=0.28, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
            ax.scatter(dirs, np.full(len(dirs), y), marker="|", s=90, color=color, linewidths=1.2, zorder=4)
        ax.text(362, y, f"tp {sub['tp'].min():.1f}-{sub['tp'].max():.1f}s", va="center", fontsize=8, color="#333333")
    ax.set_yticks(range(len(goal_ids)), [f"G{g}" for g in goal_ids])
    ax.set_xlim(0, 395)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xlabel("wave-from direction (° nautical)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    return dense

#: Metrics drawn by :func:`plot_mda_sweep`, one row each.
SWEEP_METRICS: tuple[str, ...] = ("bias", "rmse", "rrmse", "r2")


def plot_mda_sweep(
    table: pd.DataFrame,
    series: Sequence[str],
    variables: Sequence[str],
    *,
    metrics: Sequence[str] = SWEEP_METRICS,
    units: Mapping[str, str] | None = None,
    colors: Mapping[str, str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Holdout metrics against the MDA size: one panel per metric and variable.

    Parameters
    ----------
    table : pd.DataFrame
        One row per series, ``mda_candidate`` and ``variable``, with the
        *metrics* columns (e.g. pooled with
        :func:`bluemath_tk.core.metrics.pool_metrics`).
    series : sequence of str
        Columns whose values make one line; the legend joins them.
    variables : sequence of str
        Variables, one column of panels each.
    metrics : sequence of str, optional
        Metrics, one row of panels each (those without values are skipped).
    units : mapping, optional
        Units per variable for the axis labels.
    colors : mapping, optional
        Line colour by the last *series* value (e.g. a group name); others
        follow the default colour cycle.
    title : str, optional
        Figure title.
    figsize : tuple, optional
        Default scales with the panel grid.

    Returns
    -------
    Figure
        The panel grid, with one shared legend.
    """

    units, colors = units or {}, colors or {}
    metrics = [m for m in metrics if m in table and table[m].notna().any()]
    nrows, ncols = len(metrics), len(variables)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize or (4.2 * ncols, 3.2 * nrows),
        sharex=True,
        squeeze=False,
    )
    for j, var in enumerate(variables):
        sub = table[table["variable"] == var]
        for key, line in sub.groupby(list(series), sort=False):
            key = key if isinstance(key, tuple) else (key,)
            line = line.sort_values("mda_candidate")
            for i, metric in enumerate(metrics):
                axes[i, j].plot(
                    line["mda_candidate"],
                    line[metric],
                    "o-",
                    lw=1.5,
                    ms=4,
                    color=colors.get(str(key[-1])),
                    label=" / ".join(str(k) for k in key),
                )
        for i, metric in enumerate(metrics):
            ax = axes[i, j]
            unit = "" if metric in ("rrmse", "r2", "r", "si") else f" [{units.get(var, '')}]"
            ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)}{unit}")
            if metric == "r2":
                ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
        axes[0, j].set_title(var)
        axes[-1, j].set_xlabel("MDA train cases")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(4, len(labels)),
            bbox_to_anchor=(0.5, 1.04),
            fontsize=8,
        )
    if title:
        fig.suptitle(title, fontsize=12, y=1.08 if handles else 1.0)
    fig.tight_layout()
    return fig



def plot_dynamic_forcing(
    aoi: Any,
    goals: Goals,
    offshore: xr.Dataset,
    *,
    wind: xr.Dataset | None = None,
    water_level: xr.DataArray | None = None,
    angle_range: tuple[float, float] | None = None,
    wind_name: str | None = None,
    water_level_name: str | None = None,
    title: str = "Dynamic met-ocean forcing",
) -> Figure:
    """
    Wind and water level given to dynamic runs, next to the offshore waves.

    Top: mean wind over the period (shading: mean speed; arrows: mean
    vector, short where the direction keeps changing) and the wind at the
    storm hour (highest goal-mean Hs) with the goals' wave directions.
    Bottom: wind rose of the domain-mean wind; wind-wave angle at the goals
    for hours with wind sea, with *angle_range* shaded; water-level envelope
    over the goals.

    Parameters
    ----------
    aoi : geopandas.GeoDataFrame
        Area of interest (outline and extent).
    goals : mapping
        ``{goal_id: {"lon", "lat"}}``.
    offshore : xr.Dataset
        ``hs``, ``tp``, ``dir``, ``spr`` on ``(time, goal)``.
    wind : xr.Dataset, optional
        ``u10``/``v10`` on ``(time, latitude, longitude)``.
    water_level : xr.DataArray, optional
        Water level on ``(time, goal)``.
    angle_range : tuple of float, optional
        Wind-wave angles (deg) a wind correction was trained on.
    wind_name, water_level_name : str, optional
        Source names for the titles.
    title : str, optional
        Figure title.

    Returns
    -------
    Figure
    """

    from ...core.plotting.wind import plot_wind_arrows, plot_wind_rose
    from ..wind import wind_sea_weight, wind_speed_direction

    offshore = offshore.transpose("time", "goal", ...)
    times = pd.DatetimeIndex(offshore["time"].values)
    goal_ids = [int(g) for g in offshore["goal"].values]
    goal_xy = np.array([[goals[str(g)]["lon"], goals[str(g)]["lat"]] for g in goal_ids])
    storm = times[int(np.nanargmax(offshore["hs"].mean("goal").values))]

    fig = plt.figure(figsize=(16, 10.5), constrained_layout=True)
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1.25, 1.0])
    ax_mean, ax_storm = fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:])
    ax_rose = fig.add_subplot(gs[1, :2], projection="polar")
    ax_angle, ax_wl = fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:])
    fig.suptitle(title, fontsize=14, fontweight="bold")

    aoi = aoi.to_crs(4326)
    x0, y0, x1, y1 = aoi.total_bounds

    def frame(ax: Axes, panel_title: str) -> None:
        aoi.boundary.plot(ax=ax, color="black", lw=1.0, zorder=4)
        ax.scatter(*goal_xy.T, s=25, c="white", edgecolors="black", zorder=7)
        for g, (x, y) in zip(goal_ids, goal_xy):
            ax.annotate(f"G{g}", (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7, zorder=8)
        ax.set(xlim=(x0 - 0.15, x1 + 0.15), ylim=(y0 - 0.15, y1 + 0.15))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(panel_title, fontsize=11)
        ax.tick_params(labelsize=8)

    def empty(ax: Axes, text: str) -> None:
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    if wind is not None:
        wind = wind.sel(time=slice(times.min(), times.max()))
        lon2, lat2 = np.meshgrid(wind["longitude"].values, wind["latitude"].values)
        speed = np.hypot(wind["u10"], wind["v10"])
        vmax = max(float(np.nanmax(speed.max("time").values)), 1.0)
        mean_speed = speed.mean("time").values
        s_mean, d_mean = wind_speed_direction(wind["u10"].mean("time").values, wind["v10"].mean("time").values)
        mesh = ax_mean.pcolormesh(lon2, lat2, mean_speed, cmap="YlGnBu", vmin=0, vmax=vmax * 0.6, shading="auto")
        plt.colorbar(mesh, ax=ax_mean, fraction=0.03, label="mean u10 (m/s)")
        plot_wind_arrows(ax_mean, lon2, lat2, s_mean, d_mean, max_arrows=400, key_speed=5.0, scale=80.0, width=0.003)
        frame(ax_mean, f"Mean wind {times.min():%Y-%m} → {times.max():%Y-%m} "
                       f"(arrows: mean vector; steadiness {np.nanmean(s_mean) / np.nanmean(mean_speed):.2f})")

        at = wind.sel(time=storm, method="nearest")
        s_storm, d_storm = wind_speed_direction(at["u10"].values, at["v10"].values)
        mesh = ax_storm.pcolormesh(lon2, lat2, s_storm, cmap="YlOrRd", vmin=0, vmax=vmax, shading="auto")
        plt.colorbar(mesh, ax=ax_storm, fraction=0.03, label="u10 (m/s)")
        plot_wind_arrows(ax_storm, lon2, lat2, s_storm, d_storm, max_arrows=400, scale=280.0, width=0.003)
        off = offshore.sel(time=storm)
        for (x, y), d in zip(goal_xy, off["dir"].values):
            th = np.deg2rad(d)
            ax_storm.annotate("", xy=(x, y), xytext=(x + 0.25 * np.sin(th), y + 0.25 * np.cos(th)),
                              arrowprops=dict(arrowstyle="-|>", color="#08306b", lw=2, mutation_scale=14), zorder=8)
        frame(ax_storm, f"Storm hour {storm:%Y-%m-%d %H:%M} (max goal-mean Hs "
                        f"{float(off['hs'].mean()):.1f} m; blue = wave dir)")

        dm_speed, dm_dir = wind_speed_direction(
            wind["u10"].mean(["latitude", "longitude"]).values, wind["v10"].mean(["latitude", "longitude"]).values
        )
        plot_wind_rose(ax_rose, dm_speed, dm_dir)
        ax_rose.set_title(f"Domain-mean wind ({wind_name or 'u10'})", fontsize=11, pad=12)
        ax_rose.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

        at_goals = wind.interp(
            latitude=xr.DataArray(goal_xy[:, 1], dims="goal"),
            longitude=xr.DataArray(goal_xy[:, 0], dims="goal"),
        ).reindex(time=times, method="nearest", tolerance=pd.Timedelta("1h"))
        w_speed, w_dir = wind_speed_direction(at_goals["u10"].values, at_goals["v10"].values)
        weight = wind_sea_weight(offshore["hs"].values, offshore["tp"].values, offshore["spr"].values)
        angle = np.mod(w_dir - offshore["dir"].values + 180.0, 360.0) - 180.0
        sel = (weight > 0) & np.isfinite(angle) & (w_speed > 1)
        ax_angle.hist(angle[sel], bins=np.arange(-180, 181, 10), color="#4c72b0", edgecolor="white")
        panel = "Wind − wave direction, wind-sea hours"
        if angle_range is not None:
            ax_angle.axvspan(*angle_range, color="#55a868", alpha=0.18, label="wind-correction training range")
            outside = np.mean((angle[sel] < angle_range[0]) | (angle[sel] > angle_range[1])) * 100 if sel.any() else 0
            ax_angle.legend(fontsize=8, loc="upper left")
            panel += f" ({outside:.0f}% outside)"
        ax_angle.set(title=panel, xlabel="angle (deg; + = wind clockwise of waves)", ylabel="goal-hours", xlim=(-180, 180))
    else:
        for ax in (ax_mean, ax_storm, ax_angle, ax_rose):
            empty(ax, "no wind forcing")

    if water_level is not None:
        wl = water_level.sel(goal=goal_ids).reindex(time=times).load()
        ax_wl.fill_between(times, wl.min("goal"), wl.max("goal"), color="#9ecae1", label="goal range", lw=0)
        ax_wl.plot(times, wl.mean("goal"), color="#08519c", lw=0.6, label="goal mean")
        missing = wl.isnull().all("goal").values
        edges = np.flatnonzero(np.diff(np.r_[0, missing.astype(int), 0]))
        for k, (a, b) in enumerate(zip(edges[::2], edges[1::2])):
            ax_wl.axvspan(times[a], times[b - 1], color="#d62728", alpha=0.12, label="no data (wl = 0)" if k == 0 else None)
        ax_wl.set(title=f"Water level ({water_level_name or 'per goal'})", ylabel="m")
        ax_wl.legend(fontsize=8, loc="upper right")
        locator = mdates.AutoDateLocator(maxticks=6)
        ax_wl.xaxis.set_major_locator(locator)
        ax_wl.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    else:
        empty(ax_wl, "no water-level forcing")
    return fig


def plot_wind_k_comparison(df: pd.DataFrame, title: str = "How K is estimated") -> Figure:
    """
    The wind correction's ``K``: fitted model and growth formulas against the runs.

    Left: each estimate against the runs' ``K`` (reference, x-axis), with
    RMSE and bias. Right: median ``K`` against how developed the wind sea
    already is (``g Hs_ws / U^2``; 0.24 = fully developed), in bins of at
    least 30 site-cases.

    Parameters
    ----------
    df : pd.DataFrame
        ``k_runs``, ``k_fitted``, ``k_formula_deep``, ``k_formula_depth`` and
        ``hs_ws_u2`` per site-case.
    title : str, optional
        Figure title.

    Returns
    -------
    Figure
    """

    estimates = {
        "k_fitted": ("fitted model", "#1f77b4"),
        "k_formula_deep": ("formula, deep water", "#ff7f0e"),
        "k_formula_depth": ("formula, depth-limited", "#2ca02c"),
    }
    fig, (ax_sc, ax_bin) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    ref = df["k_runs"].to_numpy()
    top = float(np.nanpercentile(ref, 99.5))
    for col, (label, color) in estimates.items():
        est = df[col].to_numpy()
        ok = np.isfinite(ref) & np.isfinite(est)
        err = est[ok] - ref[ok]
        ax_sc.scatter(ref[ok], est[ok], s=4, alpha=0.25, color=color, rasterized=True,
                      label=f"{label}: RMSE {np.sqrt(np.mean(err**2)):.3f}, bias {np.mean(err):+.3f}")
    ax_sc.plot([1, top], [1, top], "k--", lw=1)
    ax_sc.set(xlim=(0.9, top), ylim=(0.9, top), xlabel="K from the runs (with / without wind)",
              ylabel="K estimated", title=f"Held-out cases · {len(df):,} site-cases")
    ax_sc.legend(fontsize=9, markerscale=4, loc="upper left")

    bins = np.geomspace(max(float(df["hs_ws_u2"].min()), 1e-3), float(df["hs_ws_u2"].max()), 16)
    mid = np.sqrt(bins[:-1] * bins[1:])
    grouped = df.groupby(pd.cut(df["hs_ws_u2"], bins), observed=False)
    enough = (grouped.size() >= 30).to_numpy()
    medians = grouped[["k_runs", *estimates]].median()[enough]
    ax_bin.plot(mid[enough], medians["k_runs"], "k-o", ms=4, label="runs (reference)")
    for col, (label, color) in estimates.items():
        ax_bin.plot(mid[enough], medians[col], "-o", ms=3, color=color, label=label)
    ax_bin.axvline(0.24, color="0.5", ls=":", lw=1)
    ax_bin.text(0.24, ax_bin.get_ylim()[1], " fully developed", va="top", fontsize=8, color="0.4")
    ax_bin.set(xscale="log", xlabel="g Hs_ws / U²  (how developed the wind sea already is)",
               ylabel="median K", title="K vs development of the wind sea")
    ax_bin.legend(fontsize=9)
    return fig


__all__ = [
    "SWEEP_METRICS",
    "goal_colors",
    "plot_direction_ranges",
    "plot_dynamic_forcing",
    "plot_goal_sectors",
    "plot_mda_sweep",
    "plot_wind_k_comparison",
]
