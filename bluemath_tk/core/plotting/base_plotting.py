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
        self.default_line_color = "blue"
        self.default_line_style = "-"
        self.default_scatter_color = "red"
        self.default_scatter_size = 10
        self.default_marker = "o"

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
            "line_color": "blue",
            "line_style": "-",
            "scatter_color": "red",
            "scatter_size": 10,
            "marker": "o",
        }
    }

    def __init__(self, **kwargs):
        """
        Initialize an instance of the DefaultStaticPlotting class.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to override default settings.
            Valid keys are:
            - template: str, name of the template to use, defaults to "default"
            - line_color: str, color of the line, defaults to blue
            - line_style: str, style of the line, defaults to solid line
            - scatter_color: str, color of the scatter plot, defaults to red
            - scatter_size: int, size of the scatter plot, defaults to 10
            - marker: str, marker of the scatter plot, defaults to circle

        Returns
        -------
        None

        Notes
        -----
        - If no keyword arguments are provided, the default template is used.
        - If a keyword argument is provided, it will override the corresponding default setting.
        - Any other provided keyword arguments will be set as instance attributes.
        """
        super().__init__()
        # Update instance attributes with either default template or passed-in values / template
        for key, value in self.templates.get(
            kwargs.get("template", "default"), self.templates.get("default")
        ).items():
            setattr(self, key, kwargs.get(key, value))
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_subplots(self, **kwargs):
        fig, ax = plt.subplots(**kwargs)
        return fig, ax

    def plot_line(self, ax, **kwargs):
        ax.plot(**kwargs)
        self.set_grid(ax)

    def plot_scatter(self, ax, **kwargs):
        ax.scatter(**kwargs)
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
    plotting = DefaultStaticPlotting()
    plotting.get_subplots(figsize=(10, 6))


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
