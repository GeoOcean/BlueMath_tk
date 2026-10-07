from abc import ABC, abstractmethod
from functools import lru_cache
from os import PathLike
from typing import Dict, List, Tuple, Union

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.collections import LineCollection

from ...config.paths import PATHS
from .colors import hex_colors_land, hex_colors_water, hex_colors_water_transition
from .satellite import get_satellite_image
from .utils import format_ticks, join_colormaps, nice_ticks, single_colormap


@lru_cache(maxsize=32)
def _cached_satellite_image(source: str, area: Tuple[float, float, float, float]):
    """
    Memoised :func:`get_satellite_image`.

    Multi-panel figures request the same ``(source, area)`` tiles several times.
    """

    return get_satellite_image(source=source, area=area)


def decimate_raster(da: xr.DataArray, max_pixels: int) -> xr.DataArray:
    """
    Stride a 2-D raster down to at most *max_pixels* cells for display.

    Source rasters (e.g. GEBCO tiffs) can hold 100M+ cells; rendering them at
    native resolution costs several GB for a figure a few thousand pixels
    wide, with no visible gain.

    Parameters
    ----------
    da : xr.DataArray
        2-D raster.
    max_pixels : int
        Pixel budget.

    Returns
    -------
    xr.DataArray
        The raster, strided equally along both dimensions if needed.
    """

    total = int(np.prod([da.sizes[d] for d in da.dims]))
    if total <= max_pixels:
        return da
    step = int(np.ceil((total / max_pixels) ** 0.5))

    return da.isel({d: slice(None, None, step) for d in da.dims})


class BasePlotting(ABC):
    """
    Abstract base class for handling default plotting functionalities across the project.
    """

    def __init__(self):
        pass

    @abstractmethod
    def plot_line(self, x, y):
        """
        Abstract method for plotting a line.
        Should be implemented by subclasses.
        """

        pass

    @abstractmethod
    def plot_scatter(self, x, y):
        """
        Abstract method for plotting a scatter plot.
        Should be implemented by subclasses.
        """

        pass


class DefaultStaticPlotting(BasePlotting):
    """
    Concrete implementation of BasePlotting with static plotting behaviors.
    """

    # Class-level dictionary for default settings
    templates = {
        "default": {
            "line": {
                "color": "blue",
                "line_style": "-",
            },
            "scatter": {
                "color": "red",
                "size": 10,
                "marker": "o",
            },
            "bathymetry": {
                "cmap": "albita_ocean",
            },
        }
    }

    def __init__(self, template: str = "default") -> None:
        """
        Initialize an instance of the DefaultStaticPlotting class.

        Parameters
        ----------
        template : str
            The template to use for the plotting settings. Default is "default".

        Notes
        -----
        - If no keyword arguments are provided, the default template is used.
        - If a keyword argument is provided, it will override the corresponding default setting.
        - Any other provided keyword arguments will be set as instance attributes.
        """

        super().__init__()
        # Update instance attributes with either default template or passed-in values / template
        for key, value in self.templates.get(template, "default").items():
            setattr(self, f"{key}_defaults", value)

    def get_subplots(self, **kwargs):
        fig, ax = plt.subplots(**kwargs)
        return fig, ax

    def get_subplot(self, figsize, **kwargs):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(**kwargs)
        return fig, ax

    def plot_line(self, ax: plt.Axes, **kwargs):
        c = kwargs.pop("c", self.line_defaults.get("color"))
        ls = kwargs.pop("ls", self.line_defaults.get("line_style"))
        ax.plot(
            c=c,
            ls=ls,
            **kwargs,
        )

    def plot_scatter(self, ax: plt.Axes, **kwargs):
        c = kwargs.pop("c", self.scatter_defaults.get("color"))
        s = kwargs.pop("s", self.scatter_defaults.get("size"))
        marker = kwargs.pop("marker", self.scatter_defaults.get("marker"))
        ax.scatter(
            c=c,
            s=s,
            marker=marker,
            **kwargs,
        )

    def plot_bathymetry(
        self,
        ax: plt.Axes,
        source: str,
        area: Tuple[float, float, float, float],
        var_name: str = "elevation",
        transition_depth: float = None,
        isodepths: Union[float, List[float]] = None,
        isodepths_labels: bool = True,
        isodepths_kwargs: Dict = None,
        value_range: Tuple[float, float] = None,
        max_pixels: int = None,
        ocean_only: bool = False,
        method: str = "pcolormesh",
        **kwargs,
    ) -> None:
        """
        Plot a bathymetry map from a bathymetry dataset stored in the PATHS dictionary.

        Parameters
        ----------
        ax: plt.Axes
            The axes on which to plot the data.
        source: str
            The source of the bathymetry data. Must be a key in the PATHS dictionary.
        area: Tuple[float, float, float, float]
            The area of the bathymetry data in the format (lon_min, lon_max, lat_min, lat_max).
        transition_depth: float
            Depth (negative, in data units) at which the ocean colormap turns from
            blue to beige. Only used with the "albita_ocean" colormap. Default is
            None, which spreads the ocean colours evenly over the depth range.
        isodepths: Union[float, List[float]]
            Depth / elevation values (in data units, so depths are negative) at which
            to draw contour lines on top of the map. Default is None (no contours).
        isodepths_labels: bool
            Whether to annotate the contour lines with their value. Default is True.
        isodepths_kwargs: Dict
            Additional keyword arguments passed to the contour call (e.g. colors,
            linewidths, linestyles), and, under the "labels_kwargs" key, a dictionary
            of arguments passed to ax.clabel().
        value_range: Tuple[float, float]
            Fixed (min, max) of the colour scale, e.g. (-200, 300) to resolve the
            shelf instead of spreading colours over the deep ocean. Values beyond it
            take the end colours. Default is None (data range).
        max_pixels: int
            Downsample the raster to at most this many cells before plotting (see
            decimate_raster). Default is None (no downsampling).
        ocean_only: bool
            Mask land (values >= 0), e.g. to keep a satellite image visible on land
            underneath. Default is False.
        method: str
            xarray plotting method, "pcolormesh" (default) or "imshow". "imshow" is
            much faster for large regular rasters.
        **kwargs
            Additional keyword arguments passed to the xr.Dataset.plot() function.
        """

        if isinstance(source, str):
            if source not in PATHS:
                raise ValueError(f"Source '{source}' not found in PATHS.")

            else:
                bathymetry_ds = (
                    xr.open_dataset(PATHS[source])
                    .sel(lon=slice(area[0], area[1]), lat=slice(area[2], area[3]))[var_name]
                )

        elif isinstance(source, xr.Dataset):
            bathymetry_ds = source[var_name].sel(lon=slice(area[0], area[1]), lat=slice(area[2], area[3]))

        elif isinstance(source, xr.DataArray):
            bathymetry_ds = source

        if max_pixels is not None:
            bathymetry_ds = decimate_raster(bathymetry_ds, max_pixels)
        isodepths_source = bathymetry_ds
        if ocean_only:
            bathymetry_ds = bathymetry_ds.where(bathymetry_ds < 0)
        plot_func = getattr(bathymetry_ds.plot, method)

        cmap = kwargs.pop("cmap", self.bathymetry_defaults.get("cmap"))
        if cmap == "albita_ocean":
            if value_range is not None:
                vmin, vmax = map(float, value_range)
            else:
                vmin = float(bathymetry_ds.min())
                vmax = float(bathymetry_ds.max())
            anchor = (
                None
                if transition_depth is None
                else (hex_colors_water_transition, transition_depth)
            )

            # Only keep the halves of the colormap the data actually covers, so that
            # e.g. an all-ocean map does not carry an unused land ramp in its colorbar
            if vmax <= 0.0:
                cmap, norm = single_colormap(
                    cmap=hex_colors_water,
                    value_range=(vmin, vmax),
                    anchor=anchor,
                )
            elif vmin >= 0.0:
                cmap, norm = single_colormap(
                    cmap=hex_colors_land,
                    value_range=(vmin, vmax),
                )
            else:
                cmap, norm = join_colormaps(
                    cmap1=hex_colors_water,
                    cmap2=hex_colors_land,
                    value_range1=(vmin, 0.0),
                    value_range2=(0.0, vmax),
                    anchor1=anchor,
                )

            # The norm is boundary-based, so ask for a colorbar that stays linear in
            # data space and is labelled with round values instead of raw boundaries
            ticks = None
            if kwargs.get("add_colorbar", True):
                cbar_kwargs = dict(kwargs.pop("cbar_kwargs", None) or {})
                cbar_kwargs.setdefault("spacing", "proportional")
                if "ticks" not in cbar_kwargs:
                    ticks = nice_ticks((vmin, vmax), include=0.0)
                    cbar_kwargs["ticks"] = ticks
                kwargs["cbar_kwargs"] = cbar_kwargs

            p = plot_func(ax=ax, cmap=cmap, norm=norm, **kwargs)
            if hasattr(p, "colorbar") and p.colorbar is not None:
                # Hide minor ticks on colorbar
                p.colorbar.minorticks_off()
                if ticks is not None:
                    # End ticks sit at the exact data limits, so round their labels
                    p.colorbar.set_ticklabels(format_ticks(ticks))
        else:
            if value_range is not None:
                kwargs.setdefault("vmin", value_range[0])
                kwargs.setdefault("vmax", value_range[1])
            plot_func(ax=ax, cmap=cmap, **kwargs)

        if isodepths is not None:
            self.plot_isodepths(
                ax=ax,
                bathymetry=isodepths_source,
                isodepths=isodepths,
                labels=isodepths_labels,
                transform=kwargs.get("transform"),
                **(isodepths_kwargs or {}),
            )

    @staticmethod
    def plot_isodepths(
        ax: plt.Axes,
        bathymetry: xr.DataArray,
        isodepths: Union[float, List[float]],
        labels: bool = True,
        transform=None,
        legend: bool = False,
        **kwargs,
    ):
        """
        Draw isodepth contour lines on top of a bathymetry map.

        Parameters
        ----------
        ax: plt.Axes
            The axes on which to plot the contours.
        bathymetry: xr.DataArray
            The bathymetry data to contour.
        isodepths: Union[float, List[float]]
            Depth / elevation values (in data units, so depths are negative).
        labels: bool
            Whether to annotate the contour lines with their value. Default is True.
        transform
            Cartopy transform of the data, if any.
        legend: bool
            Add one invisible proxy line per level, labelled e.g. "10m isobath", so
            the contours appear in a later ax.legend() call. Default is False.
        **kwargs
            Additional keyword arguments passed to the contour call. "colors" may
            be one colour or one per level (in ascending level order). The
            "labels_kwargs" key holds a dictionary of arguments passed to ax.clabel().

        Returns
        -------
        matplotlib.contour.QuadContourSet
            The contour set, so that it can be further customized.
        """

        levels = sorted(np.atleast_1d(isodepths).astype(float))
        labels_kwargs = kwargs.pop("labels_kwargs", {})
        if transform is not None:
            kwargs.setdefault("transform", transform)
        colors = kwargs.pop("colors", "black")
        linewidths = kwargs.pop("linewidths", 0.5)

        contours = bathymetry.plot.contour(
            ax=ax,
            levels=levels,
            colors=colors,
            linewidths=linewidths,
            # Solid by default, as matplotlib dashes negative levels (i.e. all depths)
            linestyles=kwargs.pop("linestyles", "solid"),
            add_colorbar=False,
            add_labels=False,
            **kwargs,
        )
        if labels:
            ax.clabel(
                contours,
                **{
                    "fmt": "%g",
                    "inline": True,
                    "fontsize": 7,
                    **labels_kwargs,
                },
            )
        if legend:
            level_colors = (
                [colors] * len(levels) if isinstance(colors, str) else list(colors)
            )
            for level, color in zip(levels, level_colors):
                ax.plot(
                    [], [], color=color, linewidth=1.2, label=f"{abs(level):g}m isobath"
                )

        return contours

    def plot_satellite(
        self,
        ax: plt.Axes,
        area: Tuple[float, float, float, float],
        source: str = "arcgis",
        **kwargs,
    ) -> None:
        """
        Downloads and displays a satellite/raster map for the given bounding box.

        Parameters
        ----------
        ax: plt.Axes
            The axes on which to plot the data.
        source: str
            The source of the satellite data.
        area: Tuple[float, float, float, float]
            The area of the satellite data.
        **kwargs
            Additional keyword arguments passed to the plotting function.
        """

        if not isinstance(ax, GeoAxes):
            raise TypeError(
                "plot_satellite needs cartopy axes "
                "(e.g. subplot_kw={'projection': ccrs.PlateCarree()})"
            )
        # Tiles are cached per (source, area): figures with several panels over the
        # same region fetch them once.
        map_img, extent = _cached_satellite_image(source, tuple(map(float, area)))
        ax.set_extent(area, crs=ccrs.PlateCarree())
        ax.imshow(
            map_img,
            extent=extent,
            transform=ccrs.Mercator.GOOGLE,
            **kwargs,
        )


    @staticmethod
    def plot_ugrid_mesh(
        ax: plt.Axes,
        mesh: Union[str, PathLike, xr.Dataset],
        color: str = "#fbbf24",
        linewidth: float = 0.3,
        alpha: float = 0.55,
        **kwargs,
    ) -> LineCollection:
        """
        Draw the edges of a UGRID mesh (e.g. a Delft3D-FM / SnapWave grid).

        At a whole-domain zoom individual triangles are far below one pixel, so
        the mesh reads as a tint over its footprint; a saturated colour keeps that
        footprint visible over satellite imagery.

        Parameters
        ----------
        ax: plt.Axes
            The axes on which to draw (plain or cartopy).
        mesh: Union[str, PathLike, xr.Dataset]
            Mesh netCDF path or dataset with ``mesh2d_node_x``, ``mesh2d_node_y``
            (lon/lat) and ``mesh2d_edge_nodes`` (``start_index`` attribute honoured).
        color, linewidth, alpha
            Line style.
        **kwargs
            Additional keyword arguments passed to LineCollection.

        Returns
        -------
        LineCollection
            The drawn edges.
        """

        ds = mesh if isinstance(mesh, xr.Dataset) else xr.open_dataset(mesh)
        try:
            node_x = ds["mesh2d_node_x"].values
            node_y = ds["mesh2d_node_y"].values
            edge_var = ds["mesh2d_edge_nodes"]
            edges = edge_var.values.astype(int) - int(
                edge_var.attrs.get("start_index", 0)
            )
        finally:
            if ds is not mesh:
                ds.close()

        segments = np.stack(
            [
                np.column_stack([node_x[edges[:, 0]], node_y[edges[:, 0]]]),
                np.column_stack([node_x[edges[:, 1]], node_y[edges[:, 1]]]),
            ],
            axis=1,
        )
        kwargs.setdefault("zorder", 2)
        collection = LineCollection(
            segments, colors=color, linewidths=linewidth, alpha=alpha, **kwargs
        )
        if isinstance(ax, GeoAxes):
            # add_collection is not wrapped by cartopy to default the transform
            collection.set_transform(ccrs.PlateCarree())
        # Data limits are updated, but the view is left alone: the mesh is usually
        # drawn over a map whose extent is already set (call ax.autoscale_view()
        # for a standalone plot).
        ax.add_collection(collection)

        return collection


class DefaultInteractivePlotting(BasePlotting):
    """
    Concrete implementation of BasePlotting with interactive plotting behaviors.
    """

    def __init__(self):
        super().__init__()

    def plot_line(self, x, y):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines", line=dict(color=self.default_line_color))
        )
        fig.update_layout(
            title="Interactive Line Plot", xaxis_title="X-axis", yaxis_title="Y-axis"
        )
        fig.show()

    def plot_scatter(self, x, y):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="markers", marker=dict(color=self.default_scatter_color)
            )
        )
        fig.update_layout(
            title="Interactive Scatter Plot", xaxis_title="X-axis", yaxis_title="Y-axis"
        )
        fig.show()

    def plot_map(self, markers=None):
        fig = go.Figure(
            go.Scattermapbox(
                lat=[marker[0] for marker in markers] if markers else [],
                lon=[marker[1] for marker in markers] if markers else [],
                mode="markers",
                marker=go.scattermapbox.Marker(size=10, color="red"),
            )
        )
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(
                    lat=self.default_map_center[0], lon=self.default_map_center[1]
                ),
                zoom=self.default_map_zoom_start,
            ),
            title="Interactive Map with Plotly",
        )
        fig.show()
