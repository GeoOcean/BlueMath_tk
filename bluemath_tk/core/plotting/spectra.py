"""
Polar wave-spectrum plots for comparing several sources side by side.

wavespectra's ``.spec.plot()`` cannot draw on an existing subplot: it always
builds its own polar axes (see ``wavespectra.plot.WavePlot.__call__``), which
xarray rejects together with a caller-supplied ``ax=``. So each spectrum is
rendered to its own image, and the images are then composed, e.g. a buoy
spectrum next to two reconstructions of the same moment.

Requires the optional ``wavespectra`` dependency (``bluemath-tk[waves]``).
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import xarray as xr
from matplotlib import pyplot as plt
from PIL import Image

DEFAULT_FIGSIZE = (5.5, 4.8)
DEFAULT_CMAP = "magma"  # sequential: spectral energy density is non-negative


def _require_wavespectra() -> None:
    """Import wavespectra so the ``.spec`` accessor is registered."""

    try:
        import wavespectra  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Spectra plots need wavespectra: pip install 'bluemath-tk[waves]'"
        ) from exc


def _figure_to_image(fig, **savefig_kwargs) -> Image.Image:
    """Rasterise and close *fig*."""

    buf = BytesIO()
    fig.savefig(buf, format="png", **savefig_kwargs)
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)


def render_polar_spectrum(
    da: xr.DataArray,
    title: str,
    *,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    cmap: str = DEFAULT_CMAP,
    dpi: int = 130,
    **plot_kwargs: Any,
) -> Image.Image:
    """
    Render one polar wave spectrum (``da.spec.plot()``) to an image.

    Parameters
    ----------
    da : xr.DataArray
        A 2-D (freq, dir) spectrum, or anything ``.spec.plot()`` accepts
        (e.g. a time-averaged one).
    title : str
        Axes title.
    figsize : tuple[float, float], optional
        Figure size in inches.
    cmap : str, optional
        Colormap. Default is "magma".
    dpi : int, optional
        Raster resolution. Default is 130.
    **plot_kwargs
        Forwarded to ``da.spec.plot()`` (e.g. ``normalised``, ``levels``,
        ``rmax``).

    Returns
    -------
    PIL.Image.Image
        Rendered figure, cropped tight, on a white background.
    """

    _require_wavespectra()
    fig = plt.figure(figsize=figsize, facecolor="white")
    da.load().spec.plot(cmap=cmap, add_colorbar=True, **plot_kwargs)
    plt.gca().set_title(title, fontsize=11, fontweight="bold", pad=10)

    return _figure_to_image(fig, dpi=dpi, bbox_inches="tight", facecolor="white")


def render_spectrum_bubble(
    da: xr.DataArray,
    *,
    figsize: tuple[float, float] = (1.8, 1.8),
    cmap: str = DEFAULT_CMAP,
    levels: int = 16,
    dpi: int = 130,
) -> Image.Image:
    """
    Render one polar spectrum as a small transparent bubble for a map.

    No title, ticks or colorbar: meant to be stamped onto a basemap at its
    site's coordinates (``matplotlib.offsetbox.AnnotationBbox`` +
    ``OffsetImage``), e.g. many reconstructed spectra across a domain.

    Parameters
    ----------
    da : xr.DataArray
        A 2-D (freq, dir) spectrum.
    figsize : tuple[float, float], optional
        Figure size in inches. Default is (1.8, 1.8).
    cmap : str, optional
        Colormap. Default is "magma".
    levels : int, optional
        Number of contour levels. Default is 16.
    dpi : int, optional
        Raster resolution. Default is 130.

    Returns
    -------
    PIL.Image.Image
        Rendered bubble with a transparent background.
    """

    _require_wavespectra()
    fig = plt.figure(figsize=figsize)
    da.load().spec.plot(cmap=cmap, levels=levels, normalised=True, add_colorbar=False)
    ax = plt.gca()
    ax.set_title("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.35, linewidth=0.5, color="white")

    return _figure_to_image(
        fig, dpi=dpi, bbox_inches="tight", pad_inches=0.02, transparent=True
    )


def stack_images_vertical(images: list[Image.Image], *, pad: int = 12) -> Image.Image:
    """
    Stack images top to bottom, centred, with white padding between them.

    Parameters
    ----------
    images : list[PIL.Image.Image]
        Images in reading order.
    pad : int, optional
        Padding in pixels. Default is 12.

    Returns
    -------
    PIL.Image.Image
        The composed image.
    """

    width = max(im.width for im in images)
    height = sum(im.height for im in images) + pad * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for im in images:
        canvas.paste(im, ((width - im.width) // 2, y))
        y += im.height + pad

    return canvas


def stack_images_horizontal(images: list[Image.Image], *, pad: int = 14) -> Image.Image:
    """
    Stack images left to right, centred, with white padding between them.

    Parameters
    ----------
    images : list[PIL.Image.Image]
        Images in reading order.
    pad : int, optional
        Padding in pixels. Default is 14.

    Returns
    -------
    PIL.Image.Image
        The composed image.
    """

    width = sum(im.width for im in images) + pad * (len(images) - 1)
    height = max(im.height for im in images)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for im in images:
        canvas.paste(im, (x, (height - im.height) // 2))
        x += im.width + pad

    return canvas


def plot_spectrum_comparison(
    specs: list[tuple[str, xr.DataArray]],
    *,
    orientation: str = "horizontal",
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    cmap: str = DEFAULT_CMAP,
    dpi: int = 130,
    **plot_kwargs: Any,
) -> Image.Image:
    """
    Render several spectra of the same moment, laid out for comparison.

    One panel per ``(label, spectrum)`` pair, in the order given (e.g. always
    reference first: ``[("Buoy", ...), ("Bulk", ...), ("Spectral", ...)]``),
    so every figure has the same, directly comparable layout.

    Parameters
    ----------
    specs : list[tuple[str, xr.DataArray]]
        ``(panel_label, spectrum)`` pairs, in reading order.
    orientation : {"horizontal", "vertical"}, optional
        Panels left to right (default) or top to bottom.
    figsize : tuple[float, float], optional
        Size of each panel, in inches.
    cmap : str, optional
        Colormap. Default is "magma".
    dpi : int, optional
        Raster resolution. Default is 130.
    **plot_kwargs
        Forwarded to :func:`render_polar_spectrum` for every panel.

    Returns
    -------
    PIL.Image.Image
        The composed image (show it with IPython's ``display()``, or
        ``.save(path)`` it).

    Raises
    ------
    ValueError
        If *orientation* is not "horizontal" or "vertical".
    """

    if orientation not in ("horizontal", "vertical"):
        raise ValueError(
            f"orientation must be 'horizontal' or 'vertical', got {orientation!r}"
        )
    images = [
        render_polar_spectrum(
            da, label, figsize=figsize, cmap=cmap, dpi=dpi, **plot_kwargs
        )
        for label, da in specs
    ]
    if orientation == "horizontal":
        return stack_images_horizontal(images)

    return stack_images_vertical(images)
