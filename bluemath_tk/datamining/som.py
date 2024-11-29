# TODO: JAVI: This code is not finished.
# https://colab.research.google.com/github/JustGlowing/minisom/blob/master/examples/BasicUsage.ipynb#scrollTo=bBgxPEQJggeK

from typing import List, Tuple
import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
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
        Initializes the SOM class.
        """

        super().__init__()
        self.set_logger_name(name=self.__class__.__name__)
        if not isinstance(som_shape, tuple):
            if len(som_shape) != 2:
                raise SOMError("Invalid SOM shape.")
        self.som_shape = som_shape
        if not isinstance(num_dimensions, int):
            raise SOMError("Invalid number of dimensions.")
        self.num_dimensions = num_dimensions
        self.x = self.som_shape[0]
        self.y = self.som_shape[1]
        self.input_len = self.num_dimensions
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
            input_len=self.input_len,
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
        self.scaler: StandardScaler = StandardScaler()
        self.data_variables: List[str] = []
        self.centroids: pd.DataFrame = pd.DataFrame()

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
            data = self.standarized_data
        else:
            data, _ = self.standarize(data=data, scaler=self.scaler)

        return self.som.activation_response(data=data)

    def labels_map(self, data: pd.DataFrame, labels: List[str]) -> np.ndarray:
        """
        Returns the labels map of the given data.
        """

        # TODO: Check the logic of this method
        standarized_data, _ = self.standarize(data=data, scaler=self.scaler)
        return self.som.labels_map(standarized_data, labels)

    def fit(
        self,
        data: pd.DataFrame,
        num_iteration: int = 1000,
    ) -> None:
        """
        Fits the SOM model to the given data.
        """

        self._data = data.copy()

        # Standarize data using the StandardScaler custom method
        self._standarized_data, self.scaler = self.standarize(data=self.data)

        # Train the SOM model
        self.som.train(data=self.standarized_data, num_iteration=num_iteration)

        # Save winner neurons and calculate centroids values
        data_and_winners = self.data.copy()
        data_and_winners["winner_neurons"] = self._get_winner_neurons(
            standarized_data=self.standarized_data
        )
        self.centroids = data_and_winners.groupby("winner_neurons").mean()

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the cluster of the given data.
        """

        standarized_data, _ = self.standarize(data=data, scaler=self.scaler)
        winner_neurons = self._get_winner_neurons(standarized_data=standarized_data)

        return winner_neurons, self.centroids.iloc[winner_neurons]

    def fit_predict(self, *args, **kwargs):
        return super().fit_predict(*args, **kwargs)
