"""
Wind plotting: arrows from speed and nautical direction, and a wind rose.

Directions are nautical, the direction the wind comes **from**, clockwise from
North; arrows point where the wind blows **to**.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.quiver import Quiver

#: Speed bins (m/s) of :func:`plot_wind_rose`.
ROSE_SPEED_BINS = (0.0, 5.0, 10.0, 15.0, 20.0, np.inf)


def wind_components(
    speed: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Eastward and northward components of a wind given as speed and nautical direction.

    Parameters
    ----------
    speed : np.ndarray
        Speed (m/s).
    direction : np.ndarray
        Direction the wind comes from (deg, nautical).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``u`` and ``v`` (m/s), pointing where the wind blows to.
    """

    rad = np.deg2rad(np.asarray(direction, dtype=float))
    speed = np.asarray(speed, dtype=float)

    return -speed * np.sin(rad), -speed * np.cos(rad)


def plot_wind_arrows(
    ax: Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    speed: np.ndarray,
    direction: np.ndarray,
    *,
    max_arrows: int = 150,
    color: str | None = "#1f3a5f",
    cmap: str = "YlOrRd",
    clim: tuple[float, float] | None = None,
    scale: float | None = None,
    key_speed: float | None = 10.0,
    zorder: int = 6,
    **quiver_kwargs,
) -> Quiver:
    """
    Wind arrows at scattered or gridded points, subsampled to about *max_arrows*.

    Parameters
    ----------
    ax : Axes
        Target axes (lon/lat data coordinates).
    lon, lat : np.ndarray
        Point coordinates (any matching shape; flattened).
    speed : np.ndarray
        Speed (m/s).
    direction : np.ndarray
        Direction the wind comes from (deg, nautical).
    max_arrows : int, optional
        Upper bound on the number of arrows (evenly strided). Default 150.
    color : str or None, optional
        Single arrow colour; None colours the arrows by speed with *cmap*.
    cmap : str, optional
        Colormap when *color* is None. Default "YlOrRd".
    clim : tuple of float, optional
        Speed limits of the colormap.
    scale : float, optional
        ``quiver`` scale (m/s per axes width); default draws 20 m/s as a
        twentieth of the panel width.
    key_speed : float or None, optional
        Speed of the reference arrow drawn in the corner; None skips it.
    zorder : int, optional
        Drawing order. Default 6.
    **quiver_kwargs
        Passed to ``ax.quiver``.

    Returns
    -------
    Quiver
        The arrows.
    """

    lon, lat = np.ravel(lon), np.ravel(lat)
    speed, direction = np.ravel(speed), np.ravel(direction)
    ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(speed) & np.isfinite(direction)
    idx = np.flatnonzero(ok)
    if len(idx) > max_arrows:
        idx = idx[:: int(np.ceil(len(idx) / max_arrows))]
    u, v = wind_components(speed[idx], direction[idx])
    kwargs = {
        "angles": "uv",
        "scale_units": "width",
        "scale": scale if scale is not None else 20.0 * 20,
        "width": 0.0022,
        "alpha": 0.85,
        "zorder": zorder,
        **quiver_kwargs,
    }
    if color is None:
        q = ax.quiver(lon[idx], lat[idx], u, v, speed[idx], cmap=cmap, clim=clim, **kwargs)
    else:
        q = ax.quiver(lon[idx], lat[idx], u, v, color=color, **kwargs)
    if key_speed:
        ax.quiverkey(
            q, 0.08, 0.04, key_speed, f"{key_speed:g} m/s", labelpos="E",
            coordinates="axes", fontproperties={"size": 8}, zorder=zorder + 1,
        )

    return q


def plot_wind_rose(
    ax: Axes,
    speed: np.ndarray,
    direction: np.ndarray,
    *,
    sectors: int = 16,
    speed_bins: tuple[float, ...] = ROSE_SPEED_BINS,
    cmap: str = "viridis",
) -> None:
    """
    Wind rose: frequency (%) per direction sector, stacked by speed bin.

    Parameters
    ----------
    ax : Axes
        Polar axes (``projection="polar"``); set to nautical orientation here.
    speed : np.ndarray
        Speed (m/s).
    direction : np.ndarray
        Direction the wind comes from (deg, nautical).
    sectors : int, optional
        Number of direction sectors. Default 16.
    speed_bins : tuple of float, optional
        Speed bin edges (m/s). Default :data:`ROSE_SPEED_BINS`.
    cmap : str, optional
        Colormap of the speed bins. Default "viridis".
    """

    import matplotlib.pyplot as plt

    speed, direction = np.ravel(speed), np.ravel(direction)
    ok = np.isfinite(speed) & np.isfinite(direction)
    speed, direction = speed[ok], np.mod(direction[ok], 360.0)
    width = 360.0 / sectors
    sector = np.floor(np.mod(direction + width / 2, 360.0) / width).astype(int)
    theta = np.deg2rad(np.arange(sectors) * width)
    bottom = np.zeros(sectors)
    colors = plt.get_cmap(cmap)(np.linspace(0.1, 0.95, len(speed_bins) - 1))
    for k, (low, high) in enumerate(zip(speed_bins[:-1], speed_bins[1:])):
        in_bin = (speed >= low) & (speed < high)
        freq = np.bincount(sector[in_bin], minlength=sectors) / max(len(speed), 1) * 100
        label = f"{low:g}–{high:g}" if np.isfinite(high) else f"> {low:g}"
        ax.bar(theta, freq, width=np.deg2rad(width) * 0.95, bottom=bottom,
               color=colors[k], edgecolor="white", linewidth=0.3, label=f"{label} m/s")
        bottom += freq
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda t, _: f"{t:g}%"))
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=8)
