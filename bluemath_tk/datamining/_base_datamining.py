from abc import abstractmethod
from typing import Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ..core.models import BlueMathModel
from ..core.plotting.base_plotting import DefaultStaticPlotting


class BaseSampling(BlueMathModel):
    """
    Base class for all sampling BlueMath models.
    This class provides the basic structure for all sampling models.

    Methods
    -------
    generate(*args, **kwargs)
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def generate(self, *args, **kwargs) -> pd.DataFrame:
        """
        Generates samples.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        pd.DataFrame
            The generated samples.
        """

        return pd.DataFrame()

    def plot_generated_data(
        self,
        data_color: str = "blue",
        **kwargs,
    ) -> Tuple[plt.figure, plt.axes]:
        """
        Plots the generated data on a scatter plot matrix.

        Parameters
        ----------
        data_color : str, optional
            Color for the data points. Default is "blue".
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the scatter plot function.

        Returns
        -------
        fig : plt.figure
            The figure object containing the plot.
        axes : plt.axes
            Array of axes objects for the subplots.

        Raises
        ------
        ValueError
            If the data is empty.
        """

        if not self.data.empty:
            variables_names = list(self.data.columns)
            num_variables = len(variables_names)
        else:
            raise ValueError("Data must be a non-empty DataFrame with columns to plot.")

        # Create figure and axes
        default_static_plot = DefaultStaticPlotting()
        fig, axes = default_static_plot.get_subplots(
            nrows=num_variables - 1,
            ncols=num_variables - 1,
            sharex=False,
            sharey=False,
        )

        for c1, v1 in enumerate(variables_names[1:]):
            for c2, v2 in enumerate(variables_names[:-1]):
                default_static_plot.plot_scatter(
                    ax=axes[c2, c1],
                    x=self.data[v1],
                    y=self.data[v2],
                    c=data_color,
                    s=kwargs.get("s", default_static_plot.default_scatter_size),
                    alpha=kwargs.get("alpha", 0.7),
                )
                if c1 == c2:
                    axes[c2, c1].set_xlabel(variables_names[c1 + 1])
                    axes[c2, c1].set_ylabel(variables_names[c2])
                elif c1 > c2:
                    axes[c2, c1].xaxis.set_ticklabels([])
                    axes[c2, c1].yaxis.set_ticklabels([])
                else:
                    fig.delaxes(axes[c2, c1])

        return fig, axes


class BaseClustering(BlueMathModel):
    """
    Base class for all clustering BlueMath models.
    This class provides the basic structure for all clustering models.

    Methods
    -------
    fit(*args, **kwargs)
    predict(*args, **kwargs)
    fit_predict(*args, **kwargs)
    plot_selected_data(data_color, centroids_color, **kwargs)
    """

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Fits the model to the data.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """

        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        Predicts the clusters for the provided data.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """

        pass

    @abstractmethod
    def fit_predict(self, *args, **kwargs):
        """
        Fits the model to the data and predicts the clusters.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """

        pass

    def plot_selected_centroids(
        self,
        data_color: str = "blue",
        centroids_color: str = "red",
        **kwargs,
    ) -> Tuple[plt.figure, plt.axes]:
        """
        Plots data and selected centroids on a scatter plot matrix.

        Parameters
        ----------
        data_color : str, optional
            Color for the data points. Default is "blue".
        centroids_color : str, optional
            Color for the centroid points. Default is "red".
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the scatter plot function.

        Returns
        -------
        fig : plt.figure
            The figure object containing the plot.
        axes : plt.axes
            Array of axes objects for the subplots.

        Raises
        ------
        ValueError
            If the data and centroids do not have the same number of columns or if the columns are empty.
        """

        if (
            list(self.data.columns) == list(self.centroids.columns)
            and list(self.data.columns) != []
        ):
            variables_names = list(self.data.columns)
            num_variables = len(variables_names)
        else:
            raise ValueError(
                "Data and centroids must have the same number of columns > 0."
            )

        # Create figure and axes
        default_static_plot = DefaultStaticPlotting()
        fig, axes = default_static_plot.get_subplots(
            nrows=num_variables - 1,
            ncols=num_variables - 1,
            sharex=False,
            sharey=False,
        )

        for c1, v1 in enumerate(variables_names[1:]):
            for c2, v2 in enumerate(variables_names[:-1]):
                default_static_plot.plot_scatter(
                    ax=axes[c2, c1],
                    x=self.data[v1],
                    y=self.data[v2],
                    c=data_color,
                    s=kwargs.get("s", default_static_plot.default_scatter_size),
                    alpha=kwargs.get("alpha", 0.7),
                )
                if self.centroids is not None:
                    default_static_plot.plot_scatter(
                        ax=axes[c2, c1],
                        x=self.centroids[v1],
                        y=self.centroids[v2],
                        c=centroids_color,
                        s=kwargs.get("s", default_static_plot.default_scatter_size),
                        alpha=kwargs.get("alpha", 0.9),
                    )
                if c1 == c2:
                    axes[c2, c1].set_xlabel(variables_names[c1 + 1])
                    axes[c2, c1].set_ylabel(variables_names[c2])
                elif c1 > c2:
                    axes[c2, c1].xaxis.set_ticklabels([])
                    axes[c2, c1].yaxis.set_ticklabels([])
                else:
                    fig.delaxes(axes[c2, c1])

        return fig, axes

    def plot_data_as_clusters(
        self,
        data: pd.DataFrame,
        closest_centroids: np.ndarray,
        **kwargs,
    ) -> Tuple[plt.figure, plt.axes]:
        """
        Plots data as closest clusters.

        Parameters
        ----------
        data : pd.DataFrame
            The data to plot.
        closest_centroids : np.ndarray
            The closest centroids.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the scatter plot function.

        Returns
        -------
        fig : plt.figure
            The figure object containing the plot.
        axes : plt.axes
            The axes object for the plot.
        """

        if (
            not data.empty
            and list(self.data.columns) != []
            and closest_centroids.size > 0
        ):
            variables_names = list(data.columns)
            num_variables = len(variables_names)
        else:
            raise ValueError(
                "Data must have columns and closest centroids must have values."
            )

        # Create figure and axes
        default_static_plot = DefaultStaticPlotting()
        fig, axes = default_static_plot.get_subplots(
            nrows=num_variables - 1,
            ncols=num_variables - 1,
            sharex=False,
            sharey=False,
        )

        # Gets colors for clusters and append to each closest centroid
        colors_for_clusters = default_static_plot.get_list_of_colors_for_colormap(
            cmap="viridis", num_colors=self.centroids.shape[0]
        )
        closest_centroids_colors = [colors_for_clusters[i] for i in closest_centroids]

        for c1, v1 in enumerate(variables_names[1:]):
            for c2, v2 in enumerate(variables_names[:-1]):
                default_static_plot.plot_scatter(
                    ax=axes[c2, c1],
                    x=data[v1],
                    y=data[v2],
                    c=closest_centroids_colors,
                    s=kwargs.get("s", default_static_plot.default_scatter_size),
                    alpha=kwargs.get("alpha", 0.7),
                )
                if c1 == c2:
                    axes[c2, c1].set_xlabel(variables_names[c1 + 1])
                    axes[c2, c1].set_ylabel(variables_names[c2])
                elif c1 > c2:
                    axes[c2, c1].xaxis.set_ticklabels([])
                    axes[c2, c1].yaxis.set_ticklabels([])
                else:
                    fig.delaxes(axes[c2, c1])

        return fig, axes


class BaseReduction(BlueMathModel):
    """
    Base class for all dimensionality reduction BlueMath models.
    This class provides the basic structure for all dimensionality reduction models.
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
