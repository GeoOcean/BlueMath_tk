from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minisom import MiniSom

from ..core.decorators import validate_data_som
from ..core.plotting.base_plotting import DefaultStaticPlotting
from ._base_datamining import BaseClustering


class SOMError(Exception):
    """
    Custom exception for SOM class.
    """

    def __init__(self, message: str = "SOM error occurred."):
        self.message = message
        super().__init__(self.message)


class SOM(BaseClustering):
    """
    Self-Organizing Maps (SOM) class.

    This class performs the Self-Organizing Map algorithm on a given dataframe.

    Attributes
    ----------
    som_shape : Tuple[int, int]
        The shape of the SOM.
    num_dimensions : int
        The number of dimensions of the input data.
    data_variables : List[str]
        A list with all data variables.
    directional_variables : List[str]
        A list with directional variables.
    fitting_variables : List[str]
        A list with fitting variables.
    custom_scale_factor : dict
        A dictionary of custom scale factors.
    scale_factor : dict
        A dictionary of scale factors (after normalizing the data).
    centroids : pd.DataFrame
        The selected centroids.
    normalized_centroids : pd.DataFrame
        The selected normalized centroids.
    is_fitted : bool
        A flag to check if the SOM model is fitted.

    Notes
    -----
    - Check MiniSom documentation for more information:
        https://github.com/JustGlowing/minisom

    Examples
    --------
    :: jupyter-execute::

        import numpy as np
        import pandas as pd
        from bluemath_tk.datamining.som import SOM

        data = pd.DataFrame(
            {
                "Hs": np.random.rand(1000) * 7,
                "Tp": np.random.rand(1000) * 20,
                "Dir": np.random.rand(1000) * 360
            }
        )
        som = SOM(som_shape=(3, 3), num_dimensions=4)
        nearest_centroids_idxs, nearest_centroids_df = som.fit_predict(
            data=data,
            directional_variables=["Dir"],
        )

        som.plot_selected_centroids(plot_text=True)
    """

    def __init__(
        self,
        som_shape: Tuple[int, int],
        num_dimensions: int,
        sigma: float = 1,
        learning_rate: float = 0.5,
        decay_function: str = "asymptotic_decay",
        neighborhood_function: str = "gaussian",
        topology: str = "rectangular",
        activation_distance: str = "euclidean",
        random_seed: int = None,
        sigma_decay_function: str = "asymptotic_decay",
    ) -> None:
        """
        Initializes a Self Organizing Maps.

        A rule of thumb to set the size of the grid for a dimensionality
        reduction task is that it should contain 5*sqrt(N) neurons
        where N is the number of samples in the dataset to analyze.

        E.g. if your dataset has 150 samples, 5*sqrt(150) = 61.23
        hence a map 8-by-8 should perform well.

        Parameters
        ----------
        som_shape : tuple
            Shape of the SOM. This should be a tuple with two integers.
        num_dimensions : int
            Number of the elements of the vectors in input.

        For the other parameters, check the MiniSom documentation:
            https://github.com/JustGlowing/minisom/blob/master/minisom.py

        Raises
        ------
        ValueError
            If the SOM shape is not a tuple with two integers.
            Or if the number of dimensions is not an integer.
        """

        super().__init__()
        self.set_logger_name(name=self.__class__.__name__)

        if not isinstance(som_shape, tuple):
            if len(som_shape) != 2:
                raise ValueError("Invalid SOM shape.")
        self.som_shape = som_shape
        if not isinstance(num_dimensions, int):
            raise ValueError("Invalid number of dimensions.")

        self.num_dimensions = num_dimensions
        self.x = self.som_shape[0]
        self.y = self.som_shape[1]
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.decay_function = decay_function
        self.neighborhood_function = neighborhood_function
        self.topology = topology
        self.activation_distance = activation_distance
        self.random_seed = random_seed
        self.sigma_decay_function = sigma_decay_function
        self._som = MiniSom(
            x=self.x,
            y=self.y,
            input_len=self.num_dimensions,
            sigma=self.sigma,
            learning_rate=self.learning_rate,
            decay_function=self.decay_function,
            neighborhood_function=self.neighborhood_function,
            topology=self.topology,
            activation_distance=self.activation_distance,
            random_seed=self.random_seed,
            sigma_decay_function=self.sigma_decay_function,
        )

        self._data: pd.DataFrame = pd.DataFrame()
        self._normalized_data: pd.DataFrame = pd.DataFrame()
        self._data_to_fit: pd.DataFrame = pd.DataFrame()
        self.data_variables: List[str] = []
        self.directional_variables: List[str] = []
        self.fitting_variables: List[str] = []
        self.custom_scale_factor: dict = {}
        self.scale_factor: dict = {}
        self.centroids: pd.DataFrame = pd.DataFrame()
        self.normalized_centroids: pd.DataFrame = pd.DataFrame()
        self.is_fitted: bool = False

    @property
    def som(self) -> MiniSom:
        return self._som

    @som.setter
    def som(self, som_params_dict: dict) -> None:
        """
        Setter for the SOM object.

        Parameters
        ----------
        som_params_dict : dict
            A dictionary with the parameters to set the SOM object.
            The keys should be the same as the parameters of the MiniSom class.
            Example: {"sigma": 1, "learning_rate": 0.5}
        """

        self._som = MiniSom(**som_params_dict)

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns the original data used for clustering.
        """

        return self._data

    @property
    def normalized_data(self) -> pd.DataFrame:
        """
        Returns the normalized data used for clustering.
        """

        return self._normalized_data

    @property
    def data_to_fit(self) -> pd.DataFrame:
        """
        Returns the data used for fitting the K-Means algorithm.
        """

        return self._data_to_fit

    @property
    def distance_map(self) -> np.ndarray:
        """
        Returns the distance map of the SOM.
        """

        return self.som.distance_map().T

    def _get_winner_neurons(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Returns the winner neurons of the given normalized data.
        """

        winner_neurons = np.array([self.som.winner(x) for x in normalized_data]).T
        return np.ravel_multi_index(winner_neurons, self.som_shape)

    def activation_response(self, data: pd.DataFrame = None) -> np.ndarray:
        """
        Returns the activation response of the given data.
        """

        if data is None:
            data = self.normalized_data.copy()
        else:
            data, _ = self.normalize(data=data, scaler=self.scaler)

        return self.som.activation_response(data=data.values)

    def get_centroids_probs_for_labels(
        self, data: pd.DataFrame, labels: List[str]
    ) -> pd.DataFrame:
        """
        Returns the labels map of the given data.
        """

        # TODO: JAVI: Could this method be implemented in more datamining classes?

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
        dict_with_probs = self.som.labels_map(normalized_data.values, labels)

        return pd.DataFrame(dict_with_probs).T.sort_index()

    def plot_centroids_probs_for_labels(
        self, probs_data: pd.DataFrame
    ) -> Tuple[plt.figure, plt.axes]:
        """
        Plots the labels map of the given data.
        """

        default_static_plot = DefaultStaticPlotting()
        fig, axes = default_static_plot.get_subplots(
            nrows=self.som_shape[0],
            ncols=self.som_shape[1],
        )
        for index in probs_data.index:
            default_static_plot.plot_pie(
                ax=axes[*index], x=probs_data.loc[index], labels=probs_data.columns
            )

        return fig, axes

    @validate_data_som
    def fit(
        self,
        data: pd.DataFrame,
        directional_variables: List[str] = [],
        custom_scale_factor: dict = {},
        num_iteration: int = 1000,
        normalize_data: bool = False,
    ) -> None:
        """
        Fits the SOM model to the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the SOM algorithm.
        directional_variables : List[str], optional
            A list of directional variables that will be transformed to u and v components.
            Then, to use custom_scale_factor, you must specify the variables names with the u and v suffixes.
            Example: directional_variables=["Dir"], custom_scale_factor={"Dir_u": [0, 1], "Dir_v": [0, 1]}.
            Default is [].
        custom_scale_factor : dict, optional
            A dictionary specifying custom scale factors for normalization.
            If normalize_data is True, this will be used to normalize the data.
            Example: {"Hs": [0, 10], "Tp": [0, 10]}.
            Default is {}.
        num_iteration : int, optional
            The number of iterations for the SOM fitting.
            Default is 1000.
        normalize_data : bool, optional
            A flag to normalize the data.
            If True, the data will be normalized using the custom_scale_factor.
            Default is False.
        """

        super().fit(
            data=data,
            directional_variables=directional_variables,
            custom_scale_factor=custom_scale_factor,
            normalize_data=normalize_data,
        )

        # Train the SOM model
        self.som.train(data=self.normalized_data.values, num_iteration=num_iteration)

        # Save winner neurons and calculate centroids values
        data_and_winners = self.data.copy()
        data_and_winners["winner_neurons"] = self._get_winner_neurons(
            normalized_data=self.normalized_data.values
        )
        self.normalized_centroids = (
            data_and_winners.groupby("winner_neurons")
            .mean()
            .drop(columns=self.directional_variables)
        )
        self.centroids = self.denormalize(
            normalized_data=self.normalized_centroids, scale_factor=self.scale_factor
        )
        for directional_variable in self.directional_variables:
            self.centroids[directional_variable] = self.get_degrees_from_uv(
                xu=self.centroids[f"{directional_variable}_u"].values,
                xv=self.centroids[f"{directional_variable}_v"].values,
            )

        # Set the fitted flag to True
        self.is_fitted = True

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Predicts the nearest centroid for the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the prediction.

        Returns
        -------
        Tuple[np.ndarray, pd.DataFrame]
            A tuple with the winner neurons and the centroids of the given data.
        """

        if self.is_fitted is False:
            raise SOMError("SOM model is not fitted.")

        normalized_data = super().predict(data=data)

        winner_neurons = self._get_winner_neurons(
            normalized_data=normalized_data.values
        )

        return winner_neurons, self.centroids.iloc[winner_neurons]

    def fit_predict(
        self,
        data: pd.DataFrame,
        directional_variables: List[str] = [],
        custom_scale_factor: dict = {},
        num_iteration: int = 1000,
        normalize_data: bool = False,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Fit the SOM algorithm to the provided data and predict the nearest centroid for each data point.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the SOM algorithm.
        directional_variables : List[str], optional
            A list of directional variables that will be transformed to u and v components.
            Then, to use custom_scale_factor, you must specify the variables names with the u and v suffixes.
            Example: directional_variables=["Dir"], custom_scale_factor={"Dir_u": [0, 1], "Dir_v": [0, 1]}.
            Default is [].
        custom_scale_factor : dict, optional
            A dictionary specifying custom scale factors for normalization.
            If normalize_data is True, this will be used to normalize the data.
            Example: {"Hs": [0, 10], "Tp": [0, 10]}.
            Default is {}.
        num_iteration : int, optional
            The number of iterations for the SOM fitting.
            Default is 1000.
        normalize_data : bool, optional
            A flag to normalize the data.
            If True, the data will be normalized using the custom_scale_factor.
            Default is False.

        Returns
        -------
        Tuple[np.ndarray, pd.DataFrame]
            A tuple containing the winner neurons for each data point and the nearest centroids.
        """

        self.fit(
            data=data,
            directional_variables=directional_variables,
            custom_scale_factor=custom_scale_factor,
            num_iteration=num_iteration,
            normalize_data=normalize_data,
        )

        return self.predict(data=data)
