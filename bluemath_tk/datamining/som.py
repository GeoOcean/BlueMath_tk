from typing import List, Tuple
import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from ..core.plotting.base_plotting import DefaultStaticPlotting
from ..core.decorators import validate_data_som
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
    Self-Organizing Map (SOM) class.

    This class performs the Self-Organizing Map algorithm on a given dataframe.

    Attributes
    ----------
    som_shape : Tuple[int, int]
        The shape of the SOM.
    num_dimensions : int
        The number of dimensions of the input data.
    data : pd.DataFrame
        The input data.
    standarized_data : pd.DataFrame
        The standarized input data.
    data_to_fit : pd.DataFrame
        The data to fit the SOM algorithm.
    data_variables : List[str]
        A list with all data variables.
    directional_variables : List[str]
        A list with directional variables.
    fitting_variables : List[str]
        A list with fitting variables.
    scaler : StandardScaler
        The StandardScaler object.
    centroids : pd.DataFrame
        The selected centroids.
    is_fitted : bool
        A flag to check if the SOM model is fitted.

    Methods
    -------
    activation_response(data)
        Returns the activation response of the given data.
    get_centroids_probs_for_labels(data, labels)
        Returns the labels map of the given data.
    plot_centroids_probs_for_labels(probs_data)
        Plots the labels map of the given data.
    fit(data, directional_variables, num_iteration)
        Fits the SOM model to the provided data.
    predict(data)
        Predicts the nearest centroid for the provided data.
    fit_predict(data, directional_variables, num_iteration)
        Fit the SOM algorithm to the provided data and predict the nearest centroid for each data point.

    Notes
    -----
    - Check MiniSom documentation for more information:
        https://github.com/JustGlowing/minisom

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from bluemath_tk.datamining.som import SOM
    >>> data = pd.DataFrame(
    ...     {
    ...         'Hs': np.random.rand(1000) * 7,
    ...         'Tp': np.random.rand(1000) * 20,
    ...         'Dir': np.random.rand(1000) * 360
    ...     }
    ... )
    >>> som = SOM(som_shape=(3, 3), num_dimensions=4)
    >>> nearest_centroids_idxs, nearest_centroids_df = som.fit_predict(
    ...     data=data,
    ...     directional_variables=['Dir'],
    ... )

    TODO
    -----
    - Add option to normalize data?
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
        self._standarized_data: pd.DataFrame = pd.DataFrame()
        self._data_to_fit: pd.DataFrame = pd.DataFrame()
        self.data_variables: List[str] = []
        self.directional_variables: List[str] = []
        self.fitting_variables: List[str] = []
        self.scaler: StandardScaler = StandardScaler()
        self.centroids: pd.DataFrame = pd.DataFrame()
        self.is_fitted: bool = False

    @property
    def som(self) -> MiniSom:
        return self._som

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def standarized_data(self) -> pd.DataFrame:
        return self._standarized_data

    @property
    def data_to_fit(self) -> pd.DataFrame:
        return self._data_to_fit

    @property
    def distance_map(self) -> np.ndarray:
        """
        Returns the distance map of the SOM.
        """

        return self.som.distance_map().T

    def _get_winner_neurons(self, standarized_data: np.ndarray) -> np.ndarray:
        """
        Returns the winner neurons of the given standarized data.
        """

        winner_neurons = np.array([self.som.winner(x) for x in standarized_data]).T
        return np.ravel_multi_index(winner_neurons, self.som_shape)

    def activation_response(self, data: pd.DataFrame = None) -> np.ndarray:
        """
        Returns the activation response of the given data.
        """

        if data is None:
            data = self.standarized_data.copy()
        else:
            data, _ = self.standarize(data=data, scaler=self.scaler)

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
        standarized_data, _ = self.standarize(data=data, scaler=self.scaler)
        dict_with_probs = self.som.labels_map(standarized_data.values, labels)
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
        num_iteration: int = 1000,
    ) -> None:
        """
        Fits the SOM model to the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the fitting.
        directional_variables : List[str], optional
            A list with the directional variables (will be transformed to u and v).
            Default is [].
        num_iteration : int, optional
            The number of iterations for the SOM fitting.
            Default is 1000.

        Notes
        -----
        - The function assumes that the data is validated by the `validate_data_som`
        decorator before execution.
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

        # Standarize data using the StandardScaler custom method
        self._standarized_data, self.scaler = self.standarize(data=self.data_to_fit)

        # Train the SOM model
        self.som.train(data=self.standarized_data.values, num_iteration=num_iteration)

        # Save winner neurons and calculate centroids values
        data_and_winners = self.data.copy()
        data_and_winners["winner_neurons"] = self._get_winner_neurons(
            standarized_data=self.standarized_data.values
        )
        self.centroids = data_and_winners.groupby("winner_neurons").mean()
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
        data = data.copy()  # Avoid modifying the original data to predict
        for directional_variable in self.directional_variables:
            u_comp, v_comp = self.get_uv_components(
                x_deg=data[directional_variable].values
            )
            data[f"{directional_variable}_u"] = u_comp
            data[f"{directional_variable}_v"] = v_comp
            data.drop(columns=[directional_variable], inplace=True)
        standarized_data, _ = self.standarize(data=data, scaler=self.scaler)
        winner_neurons = self._get_winner_neurons(
            standarized_data=standarized_data.values
        )

        return winner_neurons, self.centroids.iloc[winner_neurons]

    def fit_predict(
        self,
        data: pd.DataFrame,
        directional_variables: List[str] = [],
        num_iteration: int = 1000,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Fit the SOM algorithm to the provided data and predict the nearest centroid for each data point.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the SOM algorithm.
        directional_variables : List[str], optional
            A list of directional variables (will be transformed to u and v).
            Default is [].
        num_iteration : int, optional
            The number of iterations for the SOM fitting.
            Default is 1000.

        Returns
        -------
        Tuple[np.ndarray, pd.DataFrame]
            A tuple containing the winner neurons for each data point and the nearest centroids.
        """

        self.fit(
            data=data,
            directional_variables=directional_variables,
            num_iteration=num_iteration,
        )

        return self.predict(data=data)
