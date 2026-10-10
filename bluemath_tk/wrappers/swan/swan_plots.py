"""
Diagnostic plots of finished boundary-forced SWAN cases, read from a case folder.

Same layout as :mod:`bluemath_tk.wrappers.snapwave.snapwave_plots`: a
full-width domain panel on top (grid outline with the open boundary, the
``BOUNDSPEC ... VARIABLE PAR`` points coloured by their Hs with direction
arrows, the wind, the output points) and a 2x2 grid of output-point panels
(``hs``, ``tp``, ``dir``, ``spr`` from the ``TABLE`` output) below. Cases are
those of :class:`~bluemath_tk.wrappers.swan.swan_wrapper.SwanMetaModelWrapper`
and :class:`~bluemath_tk.wrappers.swan.swan_wrapper.SwanDynamicModelWrapper`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ..snapwave.snapwave_plots import (
    DEFAULT_CASE_PLOT_STYLE,
    CasePlotStyle,
    _add_colorbar,
    _color_limits,
    _draw_dir_arrow,
)
from .swan_utils import (
    boundary_distances,
    read_adcirc_grid,
    read_swan_table,
    swan_boundary_side,
)

Extent = tuple[float, float, float, float]


@dataclass
class SwanCase:
    """What a SWAN case folder holds, parsed from its ``INPUT``."""

    case_dir: Path
    level: float | None
    boundary: np.ndarray | None  # (n, 5): len, hs, tp, dir, spr
    points: np.ndarray | None  # (n, 2)
    table: dict[str, np.ndarray] | None
    wind: tuple[np.ndarray, ...] | None  # lon, lat, speed, dir


def _read_wind_field(path: Path, grid: str) -> tuple[np.ndarray, ...]:
    """``READINP WIND ... 3`` file (u block, v block, rows south to north)."""

    from ...waves.wind import wind_speed_direction

    xp, yp, _, mx, my, dx, dy = (float(v) for v in grid.split()[:7])
    nx, ny = int(mx) + 1, int(my) + 1
    values = np.loadtxt(path).reshape(2, ny, nx)
    lon, lat = np.meshgrid(xp + dx * np.arange(nx), yp + dy * np.arange(ny))
    speed, direction = wind_speed_direction(values[0], values[1])

    return lon.ravel(), lat.ravel(), speed.ravel(), direction.ravel()


def load_swan_case(case_dir: str | Path, extent: Extent | None = None) -> SwanCase:
    """
    Parse a SWAN case folder: water level, boundary rows, points, table, wind.

    Parameters
    ----------
    case_dir : str or Path
        Case folder.
    extent : tuple, optional
        ``(xmin, xmax, ymin, ymax)`` used to spread a uniform ``WIND`` for plotting.

    Returns
    -------
    SwanCase
        Parsed contents; missing parts are None.
    """

    case_dir = Path(case_dir)
    text = (case_dir / "INPUT").read_text()
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("$")]

    level = re.search(r"LEVEL\s*=\s*([-\d.eE+]+)", text)
    rows = []
    in_par = False
    for line in lines:
        if line.upper().startswith("BOUNDSPEC"):
            in_par = True
            continue
        if in_par:
            values = line.rstrip("&").split()
            try:
                rows.append([float(v) for v in values[:5]])
            except ValueError:
                in_par = False
            if not line.endswith("&"):
                in_par = False

    points = table = None
    m = re.search(r"POINTS\s+'[^']*'\s+FILE\s+'([^']+)'", text)
    if m and (case_dir / m.group(1)).is_file():
        points = np.loadtxt(case_dir / m.group(1)).reshape(-1, 2)
    m = re.search(r"TABLE\s+'[^']*'\s+\w+\s+'([^']+)'\s+([A-Z ]+)", text)
    if m and (case_dir / m.group(1)).is_file():
        df = read_swan_table(str(case_dir / m.group(1)), m.group(2).split())
        table = {c: df[c].to_numpy(float) for c in df}

    wind = None
    grid = re.search(r"INPGRID\s+WIND\s+REGULAR\s+([-\d.\s]+)", text)
    readinp = re.search(r"READINP\s+WIND\s+[\d.]+\s+'([^']+)'", text)
    uniform = re.search(r"^WIND\s+([-\d.]+)\s+([-\d.]+)", text, re.MULTILINE)
    if grid and readinp and (case_dir / readinp.group(1)).is_file():
        wind = _read_wind_field(case_dir / readinp.group(1), grid.group(1))
    elif uniform and float(uniform.group(1)) > 0 and extent is not None:
        lon, lat = np.meshgrid(
            np.linspace(extent[0], extent[1], 8)[1:-1], np.linspace(extent[2], extent[3], 6)[1:-1]
        )
        wind = (lon.ravel(), lat.ravel(), np.full(lon.size, float(uniform.group(1))),
                np.full(lon.size, float(uniform.group(2))))

    return SwanCase(
        case_dir=case_dir,
        level=float(level.group(1)) if level else None,
        boundary=np.asarray(rows) if rows else None,
        points=points,
        table=table,
        wind=wind,
    )


def _boundary_xy(side_xy: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Positions along the open boundary at ``[len]`` distances."""

    dist = boundary_distances(side_xy)

    return np.c_[np.interp(lengths, dist, side_xy[:, 0]), np.interp(lengths, dist, side_xy[:, 1])]


def _plot_site_panel(
    ax: Axes, points: np.ndarray, values: np.ndarray, var: str, mode: str,
    style: CasePlotStyle, extent: Extent,
) -> None:
    ok = np.isfinite(values)
    vmin, vmax = _color_limits(values)
    sc = ax.scatter(
        points[ok, 0], points[ok, 1], c=values[ok], s=style.site_marker_size,
        cmap=style.cmap_for(var, mode), vmin=vmin, vmax=vmax,
        edgecolors="white", linewidths=0.25, zorder=3,
    )
    ax.set_title(var, fontsize=11)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8)
    _add_colorbar(ax, sc, vmin, vmax, fraction=0.05)


def plot_case(
    case_dir: str | Path,
    mode: str = "dynamic",
    style: CasePlotStyle | None = None,
    title: str | None = None,
    show_satellite: bool = False,
    satellite_source: str = "arcgis",
) -> Figure:
    """
    Diagnostic figure of a finished (or only built) SWAN case.

    Parameters
    ----------
    case_dir : str or Path
        SWAN case folder (``INPUT``, ``fort.14``, outputs).
    mode : {"metamodel", "dynamic"}, optional
        Case flavour: one active boundary point, or all forced. Default "dynamic".
    style : CasePlotStyle, optional
        Colormaps and marker size. Default the SnapWave case style.
    title : str, optional
        Figure title.
    show_satellite, satellite_source : bool, str, optional
        Draw a satellite image under the panels.

    Returns
    -------
    Figure
        The figure.
    """

    from ...core.plotting.base_plotting import DefaultStaticPlotting
    from ...core.plotting.wind import plot_wind_arrows
    from ..snapwave.snapwave_plots import wind_summary

    if mode not in ("metamodel", "dynamic"):
        raise ValueError(f"mode must be 'metamodel' or 'dynamic', got {mode!r}")
    style = style or DEFAULT_CASE_PLOT_STYLE
    case_dir = Path(case_dir)
    grid = read_adcirc_grid(str(case_dir / "fort.14"))
    side_xy, _ = swan_boundary_side(grid)
    x, y = grid["vertices"][:, 0], grid["vertices"][:, 1]
    pad = 0.08
    extent = (x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad)
    case = load_swan_case(case_dir, extent)
    static = DefaultStaticPlotting()

    fig = plt.figure(figsize=(11, 11))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.25, 1.0, 1.0], hspace=0.32, wspace=0.22)
    axes_kw = {"projection": ccrs.PlateCarree()} if show_satellite else {}
    ax_map = fig.add_subplot(gs[0, :], **axes_kw)

    if show_satellite:
        static.plot_satellite(ax_map, area=extent, source=satellite_source, zorder=0)
    for b in grid["land_boundaries"]:
        ax_map.plot(x[b], y[b], color="#555555", lw=0.6, zorder=2)
    ax_map.plot(*side_xy.T, color="#d62728", lw=1.6, label="open boundary", zorder=3)

    if case.boundary is not None:
        bxy = _boundary_xy(side_xy, case.boundary[:, 0])
        hs, dirs = case.boundary[:, 1], case.boundary[:, 3]
        if mode == "metamodel":
            ax_map.scatter(*bxy.T, c="#888888", s=24, edgecolors="white", linewidths=0.4,
                           zorder=5, label="boundary points")
            k = int(np.argmax(hs))
            ax_map.scatter(*bxy[k], c="#d62728", s=120, marker="*", edgecolors="black",
                           linewidths=0.5, zorder=6, label="active boundary")
            _draw_dir_arrow(ax_map, *bxy[k], dirs[k], extent)
        else:
            peak = np.nanmax(hs)
            norm = hs / peak if peak > 0 else hs
            ax_map.scatter(*bxy.T, c=plt.cm.viridis(norm), s=20 + 80 * norm, edgecolors="white",
                           linewidths=0.4, zorder=5, label="boundary points")
            for (lon, lat), h, d in zip(bxy, hs, dirs):
                if h > 0:
                    _draw_dir_arrow(ax_map, lon, lat, d, extent)
    if case.wind is not None:
        plot_wind_arrows(ax_map, *case.wind)
    if case.points is not None:
        ax_map.scatter(*case.points.T, s=8, c="white", edgecolors="black", linewidths=0.2,
                       alpha=0.7, label="output sites", zorder=8)
    ax_map.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax_map.set_title("domain overview", fontsize=11)
    ax_map.set_xlim(extent[0], extent[1])
    ax_map.set_ylim(extent[2], extent[3])
    ax_map.set_aspect("equal", adjustable="datalim")

    for i, var in enumerate(style.top_vars[:4]):
        ax = fig.add_subplot(gs[1 + i // 2, i % 2], **axes_kw)
        if case.table is not None and var in case.table and case.points is not None:
            if show_satellite:
                static.plot_satellite(ax, area=extent, source=satellite_source, zorder=0)
            _plot_site_panel(ax, case.points, case.table[var], var, mode, style, extent)
        else:
            ax.text(0.5, 0.5, f"{var}: no TABLE output", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()

    parts = []
    if case.boundary is not None:
        b = case.boundary
        if mode == "metamodel":
            k = int(np.argmax(b[:, 1]))
            parts.append(f"hs={b[k, 1]:.3g}m · tp={b[k, 2]:.3g}s · dir={b[k, 3]:.0f}° · spr={b[k, 4]:.0f}°")
        else:
            for j, (var, unit) in enumerate((("hs", "m"), ("tp", "s"), ("dir", "°"), ("spr", "°")), start=1):
                parts.append(f"{var}=[{np.nanmin(b[:, j]):.3g}, {np.nanmax(b[:, j]):.3g}]{unit}")
    if case.level is not None:
        parts.append(f"wl={case.level:.3g}m")
    if case.wind is not None:
        parts.append(wind_summary(case.wind))
    parts.append("output=yes" if case.table is not None else "output=no")
    fig.suptitle(title or f"SWAN {mode} · case {case_dir.name}", fontsize=14,
                 fontweight="bold", y=0.98)
    fig.text(0.5, 0.94, "  ·  ".join(parts), ha="center", va="top", fontsize=10,
             color="#444444", wrap=True)
    fig.subplots_adjust(top=0.875)

    return fig
