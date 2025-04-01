from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ..core.decorators import validate_data_kma
from ._base_datamining import BaseClustering


class KMAError(Exception):
    """
    Custom exception for KMA class.
    """

    def __init__(self, message: str = "KMA error occurred."):
        self.message = message
        super().__init__(self.message)


class KMA(BaseClustering):
    """
    K-Means (KMA) class.

    This class performs the K-Means algorithm on a given dataframe.

    Attributes
    ----------
    num_clusters : int
        The number of clusters to use in the K-Means algorithm.
    seed : int
        The random seed to use as initial datapoint.
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
    centroid_real_indices : np.array
        The real indices of the selected centroids.

    Notes
    -----
    - The K-Means algorithm is used to cluster data points into k clusters.
    - The K-Means algorithm is sensitive to the initial centroids.
    - The K-Means algorithm is not suitable for large datasets.

    Examples
    --------
    .. jupyter-execute::

        import numpy as np
        import pandas as pd
        from bluemath_tk.datamining.kma import KMA

        data = pd.DataFrame(
            {
                'Hs': np.random.rand(1000) * 7,
                'Tp': np.random.rand(1000) * 20,
                'Dir': np.random.rand(1000) * 360
            }
        )
        kma = KMA(num_clusters=5)
        nearest_centroids_idxs, nearest_centroids_df = kma.fit_predict(
            data=data,
            directional_variables=['Dir'],
        )

        kma.plot_selected_centroids(plot_text=True)
    """

    def __init__(
        self,
        num_clusters: int,
        seed: int = None,
    ) -> None:
        """
        Initializes the KMA class.

        Parameters
        ----------
        num_clusters : int
            The number of clusters to use in the K-Means algorithm.
            Must be greater than 0.
        seed : int, optional
            The random seed to use as initial datapoint.
            Must be greater or equal to 0 and less than number of datapoints.
            Default is 0.

        Raises
        ------
        ValueError
            If num_clusters is not greater than 0.
            Or if seed is not greater or equal to 0.
        """

        super().__init__()
        self.set_logger_name(name=self.__class__.__name__, console=False)

        if num_clusters > 0:
            self.num_clusters = int(num_clusters)
        else:
            raise ValueError("Variable num_clusters must be > 0")
        if seed is None:
            self.seed = None
        elif seed >= 0:
            self.seed = int(seed)
        else:
            raise ValueError("Variable seed must be >= 0")
        self._kma = KMeans(
            n_clusters=self.num_clusters,
            random_state=self.seed,
        )
        self.logger.info(
            f"KMA object created with {self.num_clusters} clusters and seed {self.seed}."
            "To customize kma, do self.kma = dict(n_clusters=..., random_state=..., etc)"
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
        self.centroid_real_indices: np.array = np.array([])
        self.is_fitted: bool = False

        self._exclude_attributes = [
            "_data",
            "_normalized_data",
            "_data_to_fit",
        ]

    @property
    def kma(self) -> KMeans:
        return self._kma

    @kma.setter
    def kma(self, kwargs) -> None:
        self._kma = KMeans(**kwargs)

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

    @validate_data_kma
    def fit(
        self,
        data: pd.DataFrame,
        directional_variables: List[str] = [],
        custom_scale_factor: dict = {},
        min_number_of_points: int = None,
        max_number_of_iterations: int = 10,
        normalize_data: bool = True,
    ) -> None:
        """
        Fit the K-Means algorithm to the provided data.

        This method initializes centroids for the K-Means algorithm using the
        provided dataframe and custom scale factor.
        It normalizes the data, and returns the calculated centroids.

        TODO: Implement KMA regression guided with variable.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the KMA algorithm.
        directional_variables : List[str], optional
            A list of directional variables (will be transformed to u and v).
            Default is [].
        custom_scale_factor : dict, optional
            A dictionary specifying custom scale factors for normalization.
            Default is {}.
        min_number_of_points : int, optional
            The minimum number of points to consider a cluster.
            Default is None.
        max_number_of_iterations : int, optional
            The maximum number of iterations for the K-Means algorithm.
            This is used when min_number_of_points is not None.
            Default is 10.
        normalize_data : bool, optional
            A flag to normalize the data. Default is True.
        """

        self._data = data.copy()
        self.directional_variables = directional_variables.copy()
        for directional_variable in self.directional_variables:
            u_comp, v_comp = self.get_uv_components(
                x_deg=self.data[directional_variable].values
            )
            self._data[f"{directional_variable}_u"] = u_comp
            self._data[f"{directional_variable}_v"] = v_comp
        self.data_variables = list(self.data.columns)

        # Get just the data to be used in the fitting
        self._data_to_fit = self.data.copy()
        for directional_variable in self.directional_variables:
            self.data_to_fit.drop(columns=[directional_variable], inplace=True)
        self.fitting_variables = list(self.data_to_fit.columns)

        if normalize_data:
            self.custom_scale_factor = custom_scale_factor.copy()
        else:
            self.custom_scale_factor = {
                fitting_variable: (0, 1) for fitting_variable in self.fitting_variables
            }
        # Normalize data using custom min max scaler
        self._normalized_data, self.scale_factor = self.normalize(
            data=self.data_to_fit, custom_scale_factor=self.custom_scale_factor
        )

        # Fit K-Means algorithm
        if min_number_of_points is not None:
            stable_kma_child = False
            number_of_tries = 0
            while not stable_kma_child:
                kma_child = KMeans(n_clusters=self.num_clusters)
                predicted_labels = kma_child.fit_predict(self.normalized_data)
                _unique_labels, counts = np.unique(predicted_labels, return_counts=True)
                if np.all(counts >= min_number_of_points):
                    stable_kma_child = True
                number_of_tries += 1
                if number_of_tries > max_number_of_iterations:
                    raise ValueError(
                        f"Failed to find a stable K-Means configuration after {max_number_of_iterations} attempts."
                        "Change max_number_of_iterations or min_number_of_points."
                    )
            self.logger.info(
                f"Found a stable K-Means configuration after {number_of_tries} attempts."
            )
            self._kma = kma_child
        else:
            self._kma = self.kma.fit(self.normalized_data)

        # Calculate the centroids
        self.centroid_real_indices = self.kma.labels_.copy()
        self.normalized_centroids = pd.DataFrame(
            self.kma.cluster_centers_, columns=self.fitting_variables
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

    def predict(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predict the nearest centroid for the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the prediction.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the nearest centroid index for each data point,
            and the nearest centroids.
        """

        if self.is_fitted is False:
            raise KMAError("KMA model is not fitted.")

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
        y = self.kma.predict(X=normalized_data)

        return pd.DataFrame(
            y, columns=["kma_bmus"], index=data.index
        ), self.centroids.iloc[y]

    def fit_predict(
        self,
        data: pd.DataFrame,
        directional_variables: List[str] = [],
        custom_scale_factor: dict = {},
        min_number_of_points: int = None,
        max_number_of_iterations: int = 10,
        normalize_data: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit the K-Means algorithm to the provided data and predict the nearest centroid
        for each data point.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the KMA algorithm.
        directional_variables : List[str], optional
            A list of directional variables (will be transformed to u and v).
            Default is [].
        custom_scale_factor : dict
            A dictionary specifying custom scale factors for normalization.
            Default is {}.
        min_number_of_points : int, optional
            The minimum number of points to consider a cluster.
            Default is None.
        max_number_of_iterations : int, optional
            The maximum number of iterations for the K-Means algorithm.
            This is used when min_number_of_points is not None.
            Default is 10.
        normalize_data : bool, optional
            A flag to normalize the data. Default is True.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the nearest centroid index for each data point,
            and the nearest centroids.
        """

        self.fit(
            data=data,
            directional_variables=directional_variables,
            custom_scale_factor=custom_scale_factor,
            min_number_of_points=min_number_of_points,
            max_number_of_iterations=max_number_of_iterations,
            normalize_data=normalize_data,
        )

        return self.predict(data=data)
