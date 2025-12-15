from abc import ABC, abstractmethod
from typing import Tuple

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import xarray as xr

from ...config.paths import PATHS
from .colors import hex_colors_land, hex_colors_water
from .satellite import get_satellite_image
from .utils import join_colormaps


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
        **kwargs
            Additional keyword arguments passed to the xr.Dataset.plot() function.
        """

        if source not in PATHS:
            raise ValueError(f"Source {source} not found in PATHS")
        else:
            bathymetry_ds = (
                xr.open_dataset(PATHS[source])
                .sel(lon=slice(area[0], area[1]), lat=slice(area[2], area[3]))
                .elevation
            )

        cmap = kwargs.pop("cmap", self.bathymetry_defaults.get("cmap"))
        if cmap == "albita_ocean":
            cmap, norm = join_colormaps(
                cmap1=hex_colors_water,
                cmap2=hex_colors_land,
                value_range1=(bathymetry_ds.min(), 0.0),
                value_range2=(0.0, bathymetry_ds.max()),
            )
            p = bathymetry_ds.plot(ax=ax, cmap=cmap, norm=norm, **kwargs)
            # Hide minor ticks on colorbar
            if hasattr(p, "colorbar") and p.colorbar is not None:
                p.colorbar.minorticks_off()
        else:
            bathymetry_ds.plot(ax=ax, cmap=cmap, **kwargs)

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

        map_img, extent = get_satellite_image(
            source=source,
            area=area,
        )
        ax.set_extent(area)
        ax.imshow(
            map_img,
            extent=extent,
            transform=ccrs.Mercator.GOOGLE,
            **kwargs,
        )


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
