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


def plot_peak_period_ratio(
    samples: pd.DataFrame, theory: Mapping[str, float], title: str = "Tp / Tm-1,0"
) -> Figure:
    """
    Measured Tp / Tm-1,0 per partition kind, against spectral shapes.

    Parameters
    ----------
    samples : pd.DataFrame
        ``partition`` and ``ratio`` (e.g. ``offshore.peak_period_samples``).
    theory : Mapping[str, float]
        Label -> ratio of reference spectra (e.g. JONSWAP for a few gamma),
        drawn as vertical lines.
    title : str, optional
        Figure title.

    Returns
    -------
    Figure
    """

    kinds = list(dict.fromkeys(samples["partition"]))
    fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 4), sharey=True,
                             constrained_layout=True, squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    bins = np.linspace(0.9, 1.5, 61)
    colors = plt.get_cmap("Greys")(np.linspace(0.4, 0.9, len(theory)))
    for ax, kind in zip(axes[0], kinds):
        ratio = samples.loc[samples["partition"] == kind, "ratio"]
        ax.hist(ratio, bins=bins, density=True, color="#1f77b4", alpha=0.6)
        ax.axvline(ratio.median(), color="#d62728", lw=2, label=f"median {ratio.median():.3f}")
        for (label, value), color in zip(theory.items(), colors):
            ax.axvline(value, color=color, ls="--", lw=1.2, label=f"{label} {value:.3f}")
        ax.set(title=f"{kind} · {len(ratio):,} hours", xlabel="Tp / Tm-1,0", ylabel="density")
        ax.legend(fontsize=9)
    return fig


def _direction_arrow(ax: Axes, direction: float, xy: tuple[float, float], color: str, label: str) -> None:
    """Arrow towards where waves / wind go, from a nautical coming-from *direction* (axes fraction)."""
    rad = np.deg2rad(direction)
    dx, dy = -0.1 * np.sin(rad), -0.1 * np.cos(rad)
    start = (xy[0] - dx / 2, xy[1] - dy / 2)
    ax.annotate("", xy=(start[0] + dx, start[1] + dy), xytext=start, xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2})
    ax.text(xy[0], xy[1] - 0.1, label, transform=ax.transAxes, ha="center", va="top", fontsize=8, color=color)


def plot_wind_case_comparison(ds: xr.Dataset, size: float = 4) -> Figure:
    """
    Training cases with and without wind, and the wind correction, at every site.

    One row per case: what the wind adds in the runs (with - without wind),
    what the correction adds (corrected - without wind), on the same scale;
    their difference (corrected - with wind) with its bias and RMSE; and the
    run with wind (x, reference) against the run without wind and the
    corrected one. Arrows: waves (black) and wind (red), towards where they
    go.

    Parameters
    ----------
    ds : xr.Dataset
        ``hs0``, ``hs1``, ``hs_corrected`` on ``(case, sites)``, with site
        ``longitude`` / ``latitude`` and case ``hs``, ``tp``, ``u10``,
        ``dir``, ``u10dir``, ``wind_wave_angle``, ``ws_weight``,
        ``hs_ws_u2`` coordinates.
    size : float, optional
        Panel size (inches). Default is 4.

    Returns
    -------
    Figure
    """

    n = ds.sizes["case"]
    fig, axes = plt.subplots(n, 4, figsize=(4 * size, n * size), squeeze=False, constrained_layout=True)
    lon, lat = ds["longitude"].values, ds["latitude"].values
    aspect = 1.0 / np.cos(np.deg2rad(np.nanmean(lat)))
    for row, case in zip(axes, ds["case"].values):
        c = ds.sel(case=case)
        added = {"wind added, runs": (c["hs1"] - c["hs0"]).values,
                 "wind added, correction": (c["hs_corrected"] - c["hs0"]).values}
        lim = max(float(np.nanpercentile(np.abs(np.r_[tuple(added.values())]), 99)), 0.05)
        for ax, (label, value) in zip(row, added.items()):
            sc = ax.scatter(lon, lat, c=value, s=2, cmap="RdBu_r", vmin=-lim, vmax=lim, rasterized=True)
            ax.set_title(f"{label} (median {np.nanmedian(value):+.2f} m)", fontsize=10)
        fig.colorbar(sc, ax=row[:2], label="Hs difference (m)", shrink=0.8)
        err = (c["hs_corrected"] - c["hs1"]).values
        sc = row[2].scatter(lon, lat, c=err, s=2, cmap="RdBu_r", vmin=-lim, vmax=lim, rasterized=True)
        fig.colorbar(sc, ax=row[2], label="corrected - with wind (m)", shrink=0.8)
        row[2].set_title(f"bias {np.nanmean(err):+.2f} m, RMSE {np.sqrt(np.nanmean(err**2)):.2f} m", fontsize=10)
        for ax in row[:3]:
            ax.set_aspect(aspect)
            ax.tick_params(labelsize=7)
        _direction_arrow(row[0], float(c["dir"]), (0.12, 0.88), "black", "waves")
        _direction_arrow(row[0], float(c["u10dir"]), (0.30, 0.88), "#d62728", "wind")
        row[0].set_ylabel(
            f"case {case}: Hs {float(c['hs']):.1f} m, Tp {float(c['tp']):.1f} s\n"
            f"U {float(c['u10']):.1f} m/s, angle {float(c['wind_wave_angle']):+.0f}°, "
            f"w {float(c['ws_weight']):.2f}, gHs/U² {float(c['hs_ws_u2']):.2f}",
            fontsize=9,
        )
        ax = row[3]
        ref = c["hs1"].values
        ax.scatter(ref, c["hs0"].values, s=2, color="0.6", alpha=0.5, label="without wind", rasterized=True)
        ax.scatter(ref, c["hs_corrected"].values, s=2, color="#1f77b4", alpha=0.5, label="corrected", rasterized=True)
        top = float(np.nanmax(np.r_[ref, c["hs_corrected"].values]))
        ax.plot([0, top], [0, top], "k--", lw=1)
        ax.set(xlim=(0, top), ylim=(0, top), xlabel="Hs with wind (m)", ylabel="Hs (m)", title="all sites")
        ax.legend(fontsize=8, markerscale=4, loc="upper left")
    return fig


def plot_k_response(
    ds: xr.Dataset, ranges: Mapping[str, tuple[float, float]] | None = None, title: str = "K response"
) -> Figure:
    """
    ``K`` of the wind correction against each input, one line per site.

    Parameters
    ----------
    ds : xr.Dataset
        ``k_u10``, ``k_angle``, ``k_hs_ws_u2`` on ``(sites, <input>)``, site
        ``depth`` and ``fetch`` (m) coordinates and the base state
        (``wind_dir``, ``hs_ws``, ``u10``, ``angle``) in the attributes.
    ranges : Mapping[str, tuple of float], optional
        ``u10``, ``angle``, ``hs_ws_u2`` -> range of the training cases;
        outside it is shaded (the model keeps its edge value there).
    title : str, optional
        Figure title.

    Returns
    -------
    Figure
    """

    panels = {
        "u10": ("wind speed U10 (m/s)", f"Hs_ws {ds.attrs['hs_ws']:g} m, angle {ds.attrs['angle']:g}°"),
        "angle": ("wind - wave angle (°)", f"Hs_ws {ds.attrs['hs_ws']:g} m, U {ds.attrs['u10']:g} m/s"),
        "hs_ws_u2": ("g Hs_ws / U²  (how developed the wind sea is)",
                     f"U {ds.attrs['u10']:g} m/s, angle {ds.attrs['angle']:g}°"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True, constrained_layout=True)
    fig.suptitle(f"{title} · wind from {ds.attrs['wind_dir']:g}°", fontsize=14, fontweight="bold")
    for ax, (name, (xlabel, base)) in zip(axes, panels.items()):
        x = ds[name].values
        for site in ds["sites"].values:
            s = ds.sel(sites=site)
            ax.plot(x, s[f"k_{name}"], lw=2,
                    label=f"{site} ({float(s['depth']):.0f} m, fetch {float(s['fetch']) / 1e3:.0f} km)")
        if ranges is not None and name in ranges:
            low, high = ranges[name]
            ax.axvspan(x.min(), low, color="0.85", zorder=0)
            ax.axvspan(high, x.max(), color="0.85", zorder=0, label="outside training")
        if name == "hs_ws_u2":
            ax.set_xscale("log")
            ax.axvline(0.24, color="0.4", ls=":", lw=1)
            ax.text(0.24, 1.0, " fully developed", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color="0.4")
        ax.axhline(1, color="k", lw=0.8)
        ax.set(xlabel=xlabel, xlim=(x.min(), x.max()), title=base)
    axes[0].set_ylabel("K = Hs with / without wind")
    axes[0].legend(fontsize=8, loc="upper left")
    return fig


def plot_wind_sea_event(
    ws: xr.Dataset,
    hs: Mapping[str, xr.DataArray],
    ranges: Mapping[str, tuple[float, float]] | None = None,
    title: str = "",
) -> Figure:
    """
    The wind correction at one site through an event.

    Rows: Hs of every source; the wind sea the correction sees (``hs_ws``)
    and the wind (``u10``); how developed the wind sea is
    (``g Hs_ws / U^2``); the wind-wave angle; ``K``. Shaded: outside the
    training cases (*ranges*).

    Parameters
    ----------
    ws : xr.Dataset
        ``hs_ws``, ``u10``, ``hs_ws_u2``, ``angle``, ``k`` on ``time`` (one
        site of ``wind_sea_inputs``).
    hs : Mapping[str, xr.DataArray]
        Label -> Hs on ``time`` (e.g. buoy, dynamic run, metamodel without and
        with the correction).
    ranges : Mapping[str, tuple of float], optional
        ``hs_ws_u2``, ``angle`` -> training range.
    title : str, optional
        Figure title.

    Returns
    -------
    Figure
    """

    fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [2, 1, 1, 1, 1]})
    fig.suptitle(title, fontsize=14, fontweight="bold")
    times = ws["time"].values
    ax = axes[0]
    for label, series in hs.items():
        marker = "o" if label.lower().startswith("buoy") else None
        ax.plot(series["time"], series, lw=1.5, marker=marker, ms=2.5, ls="-" if marker is None else "none", label=label)
    ax.set(ylabel="Hs (m)")
    ax.legend(fontsize=9, ncol=2)
    ax = axes[1]
    ax.plot(times, ws["hs_ws"], color="#2ca02c", lw=1.5, label="wind sea Hs_ws (m)")
    ax.set_ylabel("Hs_ws (m)", color="#2ca02c")
    twin = ax.twinx()
    twin.plot(times, ws["u10"], color="#d62728", lw=1.2, label="U10")
    twin.set_ylabel("U10 (m/s)", color="#d62728")
    for ax, name, label, log in ((axes[2], "hs_ws_u2", "g Hs_ws / U²", True), (axes[3], "angle", "wind - wave (°)", False)):
        values = ws[name].values if name != "angle" else (ws[name].values + 180.0) % 360.0 - 180.0
        ax.plot(times, values, color="k", lw=1.2)
        if ranges is not None and name in ranges:
            low, high = ranges[name]
            ax.axhspan(low, high, color="#1f77b4", alpha=0.12, label="training range")
            ax.legend(fontsize=8, loc="upper left")
        if log:
            ax.set_yscale("log")
            ax.axhline(0.24, color="0.4", ls=":", lw=1)
        ax.set_ylabel(label)
    axes[4].plot(times, ws["k"], color="#9467bd", lw=1.5)
    axes[4].axhline(1, color="k", lw=0.8)
    axes[4].set_ylabel("K")
    axes[4].xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    return fig


def plot_event_maps(
    reference: xr.DataArray,
    estimates: Mapping[str, xr.DataArray],
    k: xr.DataArray | None = None,
    reference_label: str = "reference",
    title: str = "",
    size: float = 4.5,
) -> Figure:
    """
    Hs at every site at one time: a reference, each estimate minus it, and ``K``.

    Parameters
    ----------
    reference : xr.DataArray
        Hs on ``sites`` with ``site_longitude`` / ``site_latitude``.
    estimates : Mapping[str, xr.DataArray]
        Label -> Hs on the same sites.
    k : xr.DataArray, optional
        ``K`` of the wind correction on the same sites.
    reference_label : str, optional
        Name of the reference.
    title : str, optional
        Figure title.
    size : float, optional
        Panel size (inches).

    Returns
    -------
    Figure
    """

    n = 1 + len(estimates) + int(k is not None)
    fig, axes = plt.subplots(1, n, figsize=(n * size, size * 1.15), constrained_layout=True, squeeze=False)
    axes = axes[0]
    fig.suptitle(title, fontsize=14, fontweight="bold")
    lon, lat = reference["site_longitude"].values, reference["site_latitude"].values
    sc = axes[0].scatter(lon, lat, c=reference.values, s=3, cmap="viridis", vmin=0, rasterized=True)
    fig.colorbar(sc, ax=axes[0], label="Hs (m)", shrink=0.8)
    axes[0].set_title(f"{reference_label} (median {float(reference.median()):.2f} m)", fontsize=10)
    diffs = {label: (est.sel(sites=reference["sites"]) - reference).values for label, est in estimates.items()}
    lim = max(float(np.nanpercentile(np.abs(np.concatenate(list(diffs.values()))), 99)), 0.05) if diffs else 1.0
    for ax, (label, diff) in zip(axes[1:], diffs.items()):
        sc = ax.scatter(lon, lat, c=diff, s=3, cmap="RdBu_r", vmin=-lim, vmax=lim, rasterized=True)
        ax.set_title(f"{label} - {reference_label}\nbias {np.nanmean(diff):+.2f}, RMSE {np.sqrt(np.nanmean(diff**2)):.2f} m",
                     fontsize=10)
    if diffs:
        fig.colorbar(sc, ax=axes[1 : 1 + len(diffs)], label="Hs difference (m)", shrink=0.8)
    if k is not None:
        sc = axes[-1].scatter(lon, lat, c=k.sel(sites=reference["sites"]).values, s=3, cmap="magma_r", vmin=1, rasterized=True)
        fig.colorbar(sc, ax=axes[-1], label="K", shrink=0.8)
        axes[-1].set_title("K of the wind correction", fontsize=10)
    aspect = 1.0 / np.cos(np.deg2rad(np.nanmean(lat)))
    for ax in axes:
        ax.set_aspect(aspect)
        ax.tick_params(labelsize=7)
    return fig


__all__ = [
    "SWEEP_METRICS",
    "goal_colors",
    "plot_direction_ranges",
    "plot_dynamic_forcing",
    "plot_event_maps",
    "plot_goal_sectors",
    "plot_k_response",
    "plot_mda_sweep",
    "plot_peak_period_ratio",
    "plot_wind_case_comparison",
    "plot_wind_k_comparison",
    "plot_wind_sea_event",
]
