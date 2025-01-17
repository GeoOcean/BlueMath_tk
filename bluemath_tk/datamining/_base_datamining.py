from abc import abstractmethod
from typing import Tuple, List
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
    generate : pd.DataFrame
        Generates samples.
    plot_generated_data : Tuple[plt.figure, plt.axes]
        Plots the generated data on a scatter plot matrix.
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
        plt.figure
            The figure object containing the plot.
        plt.axes
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
    fit : None
        Fits the model to the data.
    predict : pd.DataFrame
        Predicts the clusters for the provided data.
    fit_predict : pd.DataFrame
        Fits the model to the data and predicts the clusters.
    plot_selected_centroids : Tuple[plt.figure, plt.axes]
        Plots data and selected centroids on a scatter plot matrix.
    plot_data_as_clusters : Tuple[plt.figure, plt.axes]
        Plots data as nearest clusters.
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
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
    def predict(self, *args, **kwargs) -> pd.DataFrame:
        """
        Predicts the clusters for the provided data.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        pd.DataFrame
            The predicted clusters.
        """

        return pd.DataFrame()

    @abstractmethod
    def fit_predict(self, *args, **kwargs) -> pd.DataFrame:
        """
        Fits the model to the data and predicts the clusters.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        pd.DataFrame
            The predicted clusters.
        """

        return pd.DataFrame()

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
        plt.figure
            The figure object containing the plot.
        plt.axes
            Array of axes objects for the subplots.

        Raises
        ------
        ValueError
            If the data and centroids do not have the same number of columns or if the columns are empty.
        """

        if (
            len(self.data.columns) == len(self.centroids.columns)
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
                    for i in range(self.centroids.shape[0]):
                        axes[c2, c1].text(
                            self.centroids[v1][i],
                            self.centroids[v2][i],
                            str(i + 1),
                            fontsize=kwargs.get("fontsize", 12),
                            fontweight=kwargs.get("fontweight", "bold"),
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
        nearest_centroids: np.ndarray,
        **kwargs,
    ) -> Tuple[plt.figure, plt.axes]:
        """
        Plots data as nearest clusters.

        Parameters
        ----------
        data : pd.DataFrame
            The data to plot.
        nearest_centroids : np.ndarray
            The nearest centroids.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the scatter plot function.

        Returns
        -------
        plt.figure
            The figure object containing the plot.
        plt.axes
            The axes object for the plot.
        """

        if (
            not data.empty
            and list(self.data.columns) != []
            and nearest_centroids.size > 0
        ):
            variables_names = list(data.columns)
            num_variables = len(variables_names)
        else:
            raise ValueError(
                "Data must have columns and nearest centroids must have values."
            )

        # Create figure and axes
        default_static_plot = DefaultStaticPlotting()
        fig, axes = default_static_plot.get_subplots(
            nrows=num_variables - 1,
            ncols=num_variables - 1,
            sharex=False,
            sharey=False,
        )

        # Gets colors for clusters and append to each nearest centroid
        colors_for_clusters = default_static_plot.get_list_of_colors_for_colormap(
            cmap="jet", num_colors=self.centroids.shape[0]
        )
        nearest_centroids_colors = [colors_for_clusters[i] for i in nearest_centroids]

        for c1, v1 in enumerate(variables_names[1:]):
            for c2, v2 in enumerate(variables_names[:-1]):
                default_static_plot.plot_scatter(
                    ax=axes[c2, c1],
                    x=data[v1],
                    y=data[v2],
                    c=nearest_centroids_colors,
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


class ClusteringComparator:
    """
    Class for comparing clustering models.
    """

    def __init__(self, list_of_models: List[BaseClustering]) -> None:
        """
        Initializes the ClusteringComparator class.
        """

        self.list_of_models = list_of_models

    def fit(
        self,
        data: pd.DataFrame,
        directional_variables: List[str] = [],
        custom_scale_factor: dict = {},
    ) -> None:
        """
        Fits the clustering models.
        """

        for model in self.list_of_models:
            if model.__class__.__name__ == "SOM":
                model.fit(
                    data=data,
                    directional_variables=directional_variables,
                )
            else:
                model.fit(
                    data=data,
                    directional_variables=directional_variables,
                    custom_scale_factor=custom_scale_factor,
                )

    def plot_selected_centroids(self) -> None:
        """
        Plots the selected centroids for the clustering models.
        """

        for model in self.list_of_models:
            fig, axes = model.plot_selected_centroids()
            fig.suptitle(f"Selected centroids for {model.__class__.__name__}")

    def plot_data_as_clusters(self, data: pd.DataFrame) -> None:
        """
        Plots the data as clusters for the clustering models.
        """

        for model in self.list_of_models:
            nearest_centroids, _ = model.predict(data=data)
            fig, axes = model.plot_data_as_clusters(
                data=data, nearest_centroids=nearest_centroids
            )
            fig.suptitle(f"Data as clusters for {model.__class__.__name__}")
