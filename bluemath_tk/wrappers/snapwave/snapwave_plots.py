"""
Diagnostic plots of finished SnapWave cases, read directly from a case folder.

Layout: a full-width domain panel on top (mesh output from ``map_file`` when
present, enclosure, boundary nodes with their forcing, observation points)
and a 2x2 grid of observation-point panels (``hs``, ``tp``, ``dir``, ``spr``)
below.

Two modes, matching the wrappers:

- ``"metamodel"``: a stationary case with unit Hs at one active boundary node
  (read from ``hs.txt``), highlighted with its wave direction.
- ``"dynamic"``: a time-series case; every boundary node is coloured and
  sized by its forcing Hs, with one direction arrow per active node.

Backgrounds are pluggable: pass ``background(ax, extent, panel=...)`` to draw
a project basemap; by default a satellite image is drawn when requested, and
the mesh edges when there is no map output.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.tri as mtri
import numpy as np
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from shapely.geometry import Polygon

from ...core.plotting.base_plotting import DefaultStaticPlotting
from .snapwave_utils import (
    FORCING_FILES,
    parse_snapwave_inp,
    read_boundary_nodes,
    read_enclosure_polygon,
    read_forcing_table,
)

Extent = tuple[float, float, float, float]
Background = Callable[..., None]

#: Native SnapWave names accepted for each short variable name.
WAVE_VAR_ALIASES: dict[str, tuple[str, ...]] = {
    "hs": ("point_hm0", "hm0", "Hs", "hm"),
    "tp": ("point_tp", "tp", "Tm"),
    "dir": ("point_wavdir", "wd", "wavdir", "dir", "ThetaM"),
    "spr": ("point_dirspr", "dirspr", "spr"),
}

#: Default observation-point panels (2x2 under the domain panel).
TOP_VARS: tuple[str, ...] = ("hs", "tp", "dir", "spr")

_STATIC = DefaultStaticPlotting()


@dataclass
class CasePlotStyle:
    """Colormaps, limits and marker size for SnapWave case plots."""

    cmaps: dict[str, str] = field(
        default_factory=lambda: {
            "hs": "viridis",
            "tp": "plasma",
            "dir": "twilight_shifted",
            "spr": "cividis",
        }
    )
    metamodel_cmaps: dict[str, str] = field(
        default_factory=lambda: {
            "hs": "Blues_r",
            "tp": "plasma",
            "dir": "twilight_shifted",
            "spr": "cividis",
        }
    )
    metamodel_vlims: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"hs": (0.0, 1.0)}
    )
    site_marker_size: float = 28
    top_vars: tuple[str, ...] = TOP_VARS

    def cmap_for(self, var: str, mode: str = "dynamic") -> str:
        """
        Colormap name for *var* in *mode*.

        Parameters
        ----------
        var : str
            Short variable name.
        mode : str, optional
            "metamodel" or "dynamic". Default is "dynamic".

        Returns
        -------
        str
            Colormap name.
        """

        if mode == "metamodel":
            return self.metamodel_cmaps.get(var, self.cmaps.get(var, "viridis"))

        return self.cmaps.get(var, "viridis")


DEFAULT_CASE_PLOT_STYLE = CasePlotStyle()


@dataclass
class CaseContext:
    """Paths and settings found in a SnapWave case folder."""

    case_dir: Path
    snapwave_inp: dict[str, str]
    gridfile: Path | None
    map_file: Path | None
    his_file: Path | None


def load_case_context(case_dir: str | Path) -> CaseContext:
    """
    Read ``snapwave.inp`` and locate the outputs of a case folder.

    Parameters
    ----------
    case_dir : str or Path
        SnapWave case folder.

    Returns
    -------
    CaseContext
        Paths; missing optional files are None.
    """

    case_dir = Path(case_dir)
    inp = parse_snapwave_inp(case_dir / "snapwave.inp")
    grid = Path(inp["gridfile"]) if inp.get("gridfile") else None
    map_file = None
    for name in (inp.get("map_file", "").strip(), "output_map.nc"):
        if name and (case_dir / name).exists():
            map_file = case_dir / name
            break
    his = case_dir / inp.get("his_file", "output_sites.nc")

    return CaseContext(
        case_dir=case_dir,
        snapwave_inp=inp,
        gridfile=grid if grid is not None and grid.exists() else None,
        map_file=map_file,
        his_file=his if his.exists() else None,
    )


def resolve_wave_variable(
    ds: xr.Dataset,
    name: str,
    time_index: int = 0,
    keep_dim: str | None = None,
) -> tuple[str, xr.DataArray]:
    """
    Find a wave variable in a SnapWave output dataset by its short name.

    Parameters
    ----------
    ds : xr.Dataset
        SnapWave history or map output.
    name : str
        ``hs``, ``tp``, ``dir`` or ``spr``.
    time_index : int, optional
        Time step to select. Default is 0.
    keep_dim : str, optional
        Dimension not to squeeze (e.g. ``nmesh2d_node`` for map output).

    Returns
    -------
    tuple[str, xr.DataArray]
        Native variable name and its values.

    Raises
    ------
    KeyError
        If *name* is unknown or none of its aliases is in *ds*.
    """

    key = name.strip().lower()
    if key not in WAVE_VAR_ALIASES:
        raise KeyError(
            f"Unknown variable {name!r}; expected {sorted(WAVE_VAR_ALIASES)}"
        )
    candidates = (key, *WAVE_VAR_ALIASES[key])
    var_name = next((c for c in candidates if c in ds.data_vars), None)
    if var_name is None:
        raise KeyError(f"{name!r} not found; tried {candidates}")

    da = ds[var_name]
    if "time" in da.dims:
        da = da.isel(time=time_index)
    extra = [d for d in da.dims if d != keep_dim]
    if da.ndim > 1 and len(extra) == 1 and keep_dim in da.dims:
        da = da.squeeze(extra[0], drop=True)

    return var_name, da


def map_triangulation(map_ds: xr.Dataset) -> mtri.Triangulation:
    """
    Matplotlib triangulation of a SnapWave map mesh (quads split in two).

    Triangles touching a node with undefined bed level are masked.

    Parameters
    ----------
    map_ds : xr.Dataset
        Map output with ``mesh2d_node_x/y`` and 1-based ``mesh2d_face_nodes``.

    Returns
    -------
    mtri.Triangulation
        The triangulation.
    """

    faces = map_ds["mesh2d_face_nodes"].values
    n0, n1, n2 = faces[:, 0], faces[:, 1], faces[:, 2]
    n3 = faces[:, 3] if faces.shape[1] > 3 else np.full_like(n0, np.nan)
    tris = []
    for tri in (np.stack([n0, n1, n2], axis=1), np.stack([n0, n2, n3], axis=1)):
        tris.append(tri[~np.isnan(tri).any(axis=1)].astype(np.int64) - 1)
    triangles = np.vstack([t for t in tris if t.size])

    triangulation = mtri.Triangulation(
        map_ds["mesh2d_node_x"].values.astype(float),
        map_ds["mesh2d_node_y"].values.astype(float),
        triangles,
    )
    if "mesh2d_node_z" in map_ds:
        bad = np.isnan(map_ds["mesh2d_node_z"].values.astype(float))
        triangulation.set_mask(bad[triangles].any(axis=1))

    return triangulation


def _forcing_row(case_dir: Path, var: str, time_index: int) -> tuple[float, np.ndarray]:
    """Time and per-node values of a forcing file (index clamped to its length)."""

    table = read_forcing_table(case_dir / FORCING_FILES[var])
    idx = min(time_index, len(table) - 1)

    return float(table.index[idx]), table.iloc[idx].to_numpy(float)


def active_boundary_node(case_dir: str | Path, time_index: int = 0) -> int:
    """
    0-based index of a metamodel case's active boundary node (largest Hs).

    Parameters
    ----------
    case_dir : str or Path
        SnapWave case folder.
    time_index : int, optional
        Forcing row. Default is 0.

    Returns
    -------
    int
        Node index.

    Raises
    ------
    ValueError
        If no node has positive Hs.
    """

    _, hs = _forcing_row(Path(case_dir), "hs", time_index)
    idx = int(np.argmax(hs))
    if hs[idx] <= 0:
        raise ValueError("No active boundary node (hs row has no positive entry)")

    return idx


def _color_limits(
    var: str, values: np.ndarray, mode: str, style: CasePlotStyle
) -> tuple[float, float]:
    """Return fixed metamodel limits, else the 2nd-98th percentile."""

    if mode == "metamodel" and var in style.metamodel_vlims:
        return style.metamodel_vlims[var]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = (float(v) for v in np.percentile(finite, [2, 98]))
    scale = max(abs(vmin), abs(vmax), 1.0)
    if vmax - vmin < max(1e-6, 1e-4 * scale):
        mid = 0.5 * (vmin + vmax)
        pad = max(0.5, 0.05 * abs(mid))
        return mid - pad, mid + pad

    return vmin, vmax


def _add_colorbar(ax: Axes, mappable, vmin: float, vmax: float, fraction: float):
    """Colorbar with a few ticks and two decimals."""

    cb = plt.colorbar(mappable, ax=ax, fraction=fraction, pad=0.02)
    cb.locator = mticker.MaxNLocator(nbins=5 if vmax - vmin < 1 else 6)
    cb.formatter = mticker.FormatStrFormatter("%.2f")
    cb.update_ticks()


def _default_background(
    show_satellite: bool, satellite_source: str, gridfile: Path | None
) -> Background:
    """Satellite image when requested; mesh edges on a domain panel without map."""

    def background(ax: Axes, extent: Extent, panel: str, has_map: bool) -> None:
        if show_satellite:
            _STATIC.plot_satellite(ax, area=extent, source=satellite_source, zorder=0)
        if panel == "domain" and not has_map and gridfile is not None:
            _STATIC.plot_ugrid_mesh(ax, gridfile)

    return background


def _extent(enclosure: Polygon, pad: float = 0.08) -> Extent:
    """Return a padded ``(xmin, xmax, ymin, ymax)`` around the enclosure."""

    minx, miny, maxx, maxy = enclosure.bounds

    return minx - pad, maxx + pad, miny - pad, maxy + pad


def _draw_dir_arrow(ax: Axes, lon: float, lat: float, dir_deg: float, extent: Extent):
    """Arrow pointing at ``(lon, lat)`` from the nautical wave-from direction."""

    theta = np.deg2rad(float(dir_deg))
    length = 0.08 * max(extent[1] - extent[0], extent[3] - extent[2])
    ax.annotate(
        "",
        xy=(lon, lat),
        xytext=(lon + np.sin(theta) * length, lat + np.cos(theta) * length),
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            lw=2.0,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=22,
        ),
        zorder=7,
    )


def _plot_domain_panel(
    ax: Axes,
    ctx: CaseContext,
    map_ds: xr.Dataset | None,
    his_ds: xr.Dataset | None,
    time_index: int,
    mode: str,
    map_variable: str,
    style: CasePlotStyle,
    extent: Extent,
    background: Background,
) -> None:
    """Domain panel: background or map output, enclosure, nodes, points."""

    background(ax, extent, panel="domain", has_map=map_ds is not None)
    if map_ds is not None:
        _, values = resolve_wave_variable(
            map_ds, map_variable, time_index=time_index, keep_dim="nmesh2d_node"
        )
        vals = np.asarray(values.values, dtype=float)
        if np.isfinite(vals).any():
            vmin, vmax = _color_limits(map_variable, vals, mode, style)
            mesh = ax.tripcolor(
                map_triangulation(map_ds),
                vals,
                cmap=style.cmap_for(map_variable, mode),
                vmin=vmin,
                vmax=vmax,
                shading="flat",
                zorder=1,
                **(
                    {"transform": ccrs.PlateCarree()} if isinstance(ax, GeoAxes) else {}
                ),
            )
            _add_colorbar(ax, mesh, vmin, vmax, fraction=0.025)
        ax.set_title(f"domain map · {map_variable}", fontsize=11)
    else:
        ax.set_title("domain overview", fontsize=11)

    enclosure = read_enclosure_polygon(ctx.case_dir)
    ex, ey = enclosure.exterior.xy
    ax.plot(ex, ey, "--", color="#d62728", lw=1.3, label="enclosure", zorder=4)

    nodes = read_boundary_nodes(ctx.case_dir)
    _, hs = _forcing_row(ctx.case_dir, "hs", time_index)
    _, dirs = _forcing_row(ctx.case_dir, "dir", time_index)
    if mode == "metamodel":
        ax.scatter(
            *nodes.T,
            c="#888888",
            s=28,
            edgecolors="white",
            linewidths=0.4,
            zorder=5,
            label="boundary nodes",
        )
        active = active_boundary_node(ctx.case_dir, time_index)
        ax.scatter(
            *nodes[active],
            c="#d62728",
            s=120,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
            label="active boundary",
        )
        _draw_dir_arrow(ax, *nodes[active], dirs[active], extent)
    else:
        peak = np.nanmax(hs)
        hs_norm = hs / peak if peak > 0 else hs
        ax.scatter(
            *nodes.T,
            c=plt.cm.viridis(hs_norm),
            s=20 + 80 * hs_norm,
            edgecolors="white",
            linewidths=0.4,
            zorder=5,
            label="boundary nodes",
        )
        for (lon, lat), h, d in zip(nodes, hs, dirs):
            if h > 0:
                _draw_dir_arrow(ax, lon, lat, d, extent)

    if his_ds is not None:
        ax.scatter(
            his_ds["station_x"].values,
            his_ds["station_y"].values,
            s=8,
            c="white",
            edgecolors="black",
            linewidths=0.2,
            alpha=0.7,
            label="output sites",
            zorder=8,
        )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)


def _plot_site_panel(
    ax: Axes,
    his_ds: xr.Dataset,
    var: str,
    time_index: int,
    mode: str,
    style: CasePlotStyle,
    extent: Extent,
    background: Background,
) -> None:
    """One observation-point panel; raises KeyError if *var* is not in output."""

    _, values = resolve_wave_variable(his_ds, var, time_index=time_index)
    background(ax, extent, panel="site", has_map=False)
    vals = np.asarray(values.values, dtype=float)
    ok = np.isfinite(vals)
    vmin, vmax = _color_limits(var, vals, mode, style)
    sc = ax.scatter(
        his_ds["station_x"].values.astype(float)[ok],
        his_ds["station_y"].values.astype(float)[ok],
        c=vals[ok],
        s=style.site_marker_size,
        cmap=style.cmap_for(var, mode),
        vmin=vmin,
        vmax=vmax,
        edgecolors="white",
        linewidths=0.25,
        zorder=3,
    )
    ax.set_title(var, fontsize=11)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8)
    _add_colorbar(ax, sc, vmin, vmax, fraction=0.05)


def _subtitle(ctx: CaseContext, mode: str, time_index: int, has_map: bool) -> str:
    """One-line forcing summary."""

    parts: list[str] = []
    if mode == "metamodel":
        parts.append(f"hs=1 @bnd{active_boundary_node(ctx.case_dir, time_index) + 1}")
        for var, unit in (("tp", "s"), ("dir", "°"), ("spr", "°"), ("wl", "m")):
            if (ctx.case_dir / FORCING_FILES[var]).exists():
                _, vals = _forcing_row(ctx.case_dir, var, time_index)
                parts.append(f"{var}={vals[0]:.3g}{unit}")
    else:
        for var in ("hs", "tp", "dir", "spr", "wl"):
            if (ctx.case_dir / FORCING_FILES[var]).exists():
                t, vals = _forcing_row(ctx.case_dir, var, time_index)
                parts.append(
                    f"{var}=[{np.nanmin(vals):.3g}, {np.nanmax(vals):.3g}] @t={t:g}s"
                )
    parts.append("map=yes" if has_map else "map=no")

    return "  ·  ".join(parts)


def plot_case(
    case_dir: str | Path,
    mode: str = "dynamic",
    time_index: int = 0,
    show_map: bool = True,
    map_variable: str = "hs",
    style: CasePlotStyle | None = None,
    title: str | None = None,
    show_satellite: bool = False,
    satellite_source: str = "arcgis",
    background: Background | None = None,
) -> Figure:
    """
    Diagnostic figure of a finished SnapWave case.

    Parameters
    ----------
    case_dir : str or Path
        SnapWave case folder.
    mode : {"metamodel", "dynamic"}, optional
        Case flavour (see module docstring). Default is "dynamic".
    time_index : int, optional
        Forcing/output time step. Default is 0.
    show_map : bool, optional
        Shade the ``map_file`` output when the case wrote one. Default True.
    map_variable : str, optional
        Variable shaded on the mesh. Default is "hs".
    style : CasePlotStyle, optional
        Colormaps and limits. Default is :data:`DEFAULT_CASE_PLOT_STYLE`.
    title : str, optional
        Figure title. Default names the mode, active node and case.
    show_satellite : bool, optional
        Use cartopy axes and draw a satellite image (default background).
    satellite_source : str, optional
        Tile source for the default background. Default is "arcgis".
    background : callable, optional
        ``background(ax, extent, panel=..., has_map=...)`` drawing a basemap,
        with ``panel`` "domain" or "site". Replaces the default background.

    Returns
    -------
    Figure
        The figure.
    """

    if mode not in ("metamodel", "dynamic"):
        raise ValueError(f"mode must be 'metamodel' or 'dynamic', got {mode!r}")
    style = style or DEFAULT_CASE_PLOT_STYLE
    ctx = load_case_context(case_dir)
    if background is None:
        background = _default_background(show_satellite, satellite_source, ctx.gridfile)
    his_ds = xr.open_dataset(ctx.his_file) if ctx.his_file else None
    map_ds = xr.open_dataset(ctx.map_file) if show_map and ctx.map_file else None
    extent = _extent(read_enclosure_polygon(ctx.case_dir))

    if title is None:
        title = f"SnapWave {mode} · case {ctx.case_dir.name}"
        if mode == "metamodel":
            node = active_boundary_node(ctx.case_dir, time_index) + 1
            title = f"SnapWave {mode} · bnd{node} · case {ctx.case_dir.name}"

    fig = plt.figure(figsize=(11, 11))
    gs = GridSpec(
        3, 2, figure=fig, height_ratios=[1.25, 1.0, 1.0], hspace=0.32, wspace=0.22
    )
    axes_kw = {"projection": ccrs.PlateCarree()} if show_satellite else {}
    ax_map = fig.add_subplot(gs[0, :], **axes_kw)

    for i, var in enumerate(style.top_vars[:4]):
        ax = fig.add_subplot(gs[1 + i // 2, i % 2], **(axes_kw if his_ds else {}))
        message = (
            None
            if his_ds is not None
            else f"{ctx.snapwave_inp.get('his_file', 'his_file')} missing"
        )
        if his_ds is not None:
            try:
                _plot_site_panel(
                    ax, his_ds, var, time_index, mode, style, extent, background
                )
            except KeyError:
                message = f"{var} not in output"
        if message:
            ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

    _plot_domain_panel(
        ax_map,
        ctx,
        map_ds,
        his_ds,
        time_index,
        mode,
        map_variable,
        style,
        extent,
        background,
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.94,
        _subtitle(ctx, mode, time_index, map_ds is not None),
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )
    fig.subplots_adjust(top=0.90)
    ax_map.set_xlim(extent[0], extent[1])
    ax_map.set_ylim(extent[2], extent[3])
    ax_map.set_aspect("equal", adjustable="datalim")

    for ds in (his_ds, map_ds):
        if ds is not None:
            ds.close()

    return fig
