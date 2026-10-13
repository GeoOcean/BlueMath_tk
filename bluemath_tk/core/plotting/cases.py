"""
Shared pieces of the numerical-case diagnostic plots.

Used by :mod:`bluemath_tk.wrappers.snapwave.snapwave_plots` and
:mod:`bluemath_tk.wrappers.swan.swan_plots`, so a case plot looks the same
whatever model ran it: :class:`CasePlotStyle` (colormaps per variable and
mode), colour limits and colorbars, the wave-direction arrow and the wind
summary line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.axes import Axes

from .wind import wind_components

#: ``(xmin, xmax, ymin, ymax)`` in plot coordinates.
Extent = tuple[float, float, float, float]

#: Default observation-point panels (2x2 under the domain panel).
TOP_VARS: tuple[str, ...] = ("hs", "tp", "dir", "spr")


@dataclass
class CasePlotStyle:
    """Colormaps and marker size for numerical-case plots (SnapWave, SWAN)."""

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


def color_limits(values: np.ndarray) -> tuple[float, float]:
    """Return the 2nd-98th percentile of the finite values."""

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


def add_colorbar(ax: Axes, mappable, vmin: float, vmax: float, fraction: float):
    """Colorbar with a few ticks and two decimals."""

    cb = plt.colorbar(mappable, ax=ax, fraction=fraction, pad=0.02)
    cb.locator = mticker.MaxNLocator(nbins=5 if vmax - vmin < 1 else 6)
    cb.formatter = mticker.FormatStrFormatter("%.2f")
    cb.update_ticks()


def draw_dir_arrow(ax: Axes, lon: float, lat: float, dir_deg: float, extent: Extent):
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


def wind_summary(wind: tuple[np.ndarray, ...] | None) -> str | None:
    """``u10=[min, max] m/s from <mean dir>°`` of a ``(lon, lat, speed, dir)`` tuple."""

    if wind is None:
        return None
    _, _, speed, direction = wind
    u, v = wind_components(speed, direction)
    mean_dir = np.mod(np.rad2deg(np.arctan2(-np.nanmean(u), -np.nanmean(v))), 360.0)

    return (
        f"u10=[{np.nanmin(speed):.3g}, {np.nanmax(speed):.3g}] m/s from {mean_dir:.0f}°"
    )
