from abc import abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.models import BlueMathModel
from ..core.plotting.base_plotting import DefaultStaticPlotting
from ..core.plotting.scatter import plot_scatters_in_triangle
from ..core.plotting.utils import get_list_of_colors_for_colormap


class BaseSampling(BlueMathModel):
    """
    Base class for all sampling BlueMath models.
    This class provides the basic structure for all sampling models.
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
    ) -> Tuple[Figure, Axes]:
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
        Figure
            The figure object containing the plot.
        Axes
            Array of axes objects for the subplots.

        Raises
        ------
        ValueError
            If the data is empty.
        """

        if self.data.empty:
            raise ValueError("Data must be a non-empty DataFrame with columns to plot.")

        fig, axes = plot_scatters_in_triangle(
            dataframes=[self.data],
            data_colors=[data_color],
            **kwargs,
        )

        return fig, axes


class BaseClustering(BlueMathModel):
    """
    Base class for all clustering BlueMath models.
    This class provides the basic structure for all clustering models.
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

        self._exclude_attributes = [
            "_data",
            "_normalized_data",
            "_data_to_fit",
        ]

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame,
        directional_variables: List[str] = [],
        custom_scale_factor: dict = {},
        normalize_data: bool = False,
    ) -> None:
        """
        Preprocess some data to be used in the fit of children classes.
        """

        self._data = data.copy()
        self.directional_variables = directional_variables.copy()
        for directional_variable in self.directional_variables:
            u_comp, v_comp = self.get_uv_components(
                x_deg=self.data[directional_variable].values
            )
            self.data[f"{directional_variable}_u"] = u_comp
            self.data[f"{directional_variable}_v"] = v_comp
        self.data_variables = list(self.data.columns)

        # Get just the data to be used in the training
        self._data_to_fit = self.data.copy()
        for directional_variable in self.directional_variables:
            self.data_to_fit.drop(columns=[directional_variable], inplace=True)
        self.fitting_variables = list(self.data_to_fit.columns)

        if normalize_data:
            self.custom_scale_factor = custom_scale_factor.copy()
        else:
            self.logger.info(
                "Normalization is disabled. Set normalize_data to True to enable normalization."
            )
            self.custom_scale_factor = {
                fitting_variable: (0, 1) for fitting_variable in self.fitting_variables
            }
        # Normalize data using custom min max scaler
        self._normalized_data, self.scale_factor = self.normalize(
            data=self.data_to_fit, custom_scale_factor=self.custom_scale_factor
        )

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess some data to be used in the predict of children classes.
        """

        data = data.copy()  # Avoid modifying the original data to predict
        for directional_variable in self.directional_variables:
            u_comp, v_comp = self.get_uv_components(
                x_deg=data[directional_variable].values
            )
            data[f"{directional_variable}_u"] = u_comp
            data[f"{directional_variable}_v"] = v_comp
            data.drop(columns=[directional_variable], inplace=True)
        normalized_data, _ = self.normalize(
            data=data, custom_scale_factor=self.scale_factor
        )

        return normalized_data

    def plot_selected_centroids(
        self,
        data_color: str = "blue",
        centroids_color: str = "red",
        plot_text: bool = False,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plots data and selected centroids on a scatter plot matrix.

        Parameters
        ----------
        data_color : str, optional
            Color for the data points. Default is "blue".
        centroids_color : str, optional
            Color for the centroid points. Default is "red".
        plot_text : bool, optional
            Whether to display text labels for centroids. Default is False.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the scatter plot function.

        Returns
        -------
        Figure
            The figure object containing the plot.
        Axes
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
        else:
            raise ValueError(
                "Data and centroids must have the same number of columns > 0."
            )

        fig, axes = plot_scatters_in_triangle(
            dataframes=[self.data, self.centroids],
            data_colors=[data_color, centroids_color],
            **kwargs,
        )
        if plot_text:
            for c1, v1 in enumerate(variables_names[1:]):
                for c2, v2 in enumerate(variables_names[:-1]):
                    for i in range(self.centroids.shape[0]):
                        axes[c2, c1].text(
                            self.centroids[v1][i],
                            self.centroids[v2][i],
                            str(i + 1),
                            fontsize=12,
                            fontweight="bold",
                        )

        return fig, axes

    def plot_data_as_clusters(
        self,
        data: pd.DataFrame,
        nearest_centroids: np.ndarray,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
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
        Figure
            The figure object containing the plot.
        Axes
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
        if isinstance(axes, Axes):
            axes = np.array([[axes]])

        # Gets colors for clusters and append to each nearest centroid
        colors_for_clusters = get_list_of_colors_for_colormap(
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
                    alpha=0.9,
                    **kwargs,
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
    def transform(self, *args, **kwargs) -> xr.Dataset:
        """
        Transforms the data using the fitted model.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        xr.Dataset
            The transformed data.
        """

        return xr.Dataset()

    @abstractmethod
    def fit_transform(self, *args, **kwargs) -> xr.Dataset:
        """
        Fits the model to the data and transforms it.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        xr.Dataset
            The transformed data.
        """

        return xr.Dataset()

    @abstractmethod
    def inverse_transform(self, *args, **kwargs) -> xr.Dataset:
        """
        Inversely transforms the data using the fitted model.

        Parameters
        ----------
        *args : list
            Positional arguments.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        xr.Dataset
            The inversely transformed data.
        """

        return xr.Dataset()


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
