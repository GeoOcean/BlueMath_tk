from abc import ABC, abstractmethod
from typing import Union
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import plotly.graph_objects as go
import cartopy.crs as ccrs


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

    @abstractmethod
    def plot_map(self, markers=None):
        """
        Abstract method for plotting a map.
        Should be implemented by subclasses.
        """

        pass

    def get_list_of_colors_for_colormap(
        self, cmap: Union[str, Colormap], num_colors: int
    ) -> list:
        """
        Get a list of colors from a colormap.

        Parameters
        ----------
        cmap : str or Colormap
            The colormap to use.
        num_colors : int
            The number of colors to generate.

        Returns
        -------
        list
            A list of colors generated from the colormap.
        """

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        return [cmap(i) for i in range(0, 256, 256 // num_colors)]


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

    def plot_line(self, ax, **kwargs):
        c = kwargs.get("c", self.line_defaults.get("color"))
        kwargs.pop("c", None)
        ls = kwargs.get("ls", self.line_defaults.get("line_style"))
        kwargs.pop("ls", None)
        ax.plot(
            c=c,
            ls=ls,
            **kwargs,
        )
        self.set_grid(ax)

    def plot_scatter(self, ax, **kwargs):
        c = kwargs.get("c", self.scatter_defaults.get("color"))
        kwargs.pop("c", None)
        s = kwargs.get("s", self.scatter_defaults.get("size"))
        kwargs.pop("s", None)
        marker = kwargs.get("marker", self.scatter_defaults.get("marker"))
        kwargs.pop("marker", None)
        ax.scatter(
            c=c,
            s=s,
            marker=marker,
            **kwargs,
        )
        self.set_grid(ax)

    def plot_pie(self, ax, **kwargs):
        ax.pie(**kwargs)

    def plot_map(self, ax, **kwargs):
        ax.set_global()
        ax.coastlines()

    def set_title(self, ax, title="Plot Title"):
        """
        Sets the title for a given axis.
        """
        ax.set_title(title)

    def set_xlim(self, ax, xmin, xmax):
        """
        Sets the x-axis limits for a given axis.
        """
        ax.set_xlim(xmin, xmax)

    def set_ylim(self, ax, ymin, ymax):
        """
        Sets the y-axis limits for a given axis.
        """
        ax.set_ylim(ymin, ymax)

    def set_xlabel(self, ax, xlabel="X-axis"):
        """
        Sets the x-axis label for a given axis.
        """
        ax.set_xlabel(xlabel)

    def set_ylabel(self, ax, ylabel="Y-axis"):
        """
        Sets the y-axis label for a given axis.
        """
        ax.set_ylabel(ylabel)

    def set_grid(self, ax, grid=True):
        """
        Sets the grid for a given axis.
        """
        ax.grid(grid)


if __name__ == "__main__":
    static = DefaultStaticPlotting()
    fig, ax = static.get_subplots()
    static.plot_line(ax, x=[1, 2, 3], y=[4, 5, 6])


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
