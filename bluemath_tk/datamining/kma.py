import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.cluster import KMeans
from ._base_datamining import BaseClustering
from ..core.decorators import validate_data_kma


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
        The random seed to use.
    _data : pd.DataFrame
        The input data.
    _normalized_data : pd.DataFrame
        The normalized input data.
    data_variables : List[str]
        A list of all data variables.
    custom_scale_factor : dict
        A dictionary of custom scale factors.
    scale_factor : dict
        A dictionary of scale factors (after normalizing the data).
    centroids : pd.DataFrame
        The selected centroids.
    normalized_centroids : pd.DataFrame
        The selected normalized centroids.
    bmus : np.array
        The cluster assignments for each data point.

    Notes
    -----
    - The K-Means algorithm is used to cluster data points into k clusters.
    - The K-Means algorithm is sensitive to the initial centroids.
    - The K-Means algorithm is not suitable for large datasets.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from bluemath_tk.datamining.kma import KMA
    >>> data = pd.DataFrame(
    ...     {
    ...         'Hs': np.random.rand(1000) * 7,
    ...         'Tp': np.random.rand(1000) * 20,
    ...         'Dir': np.random.rand(1000) * 360
    ...     }
    ... )
    >>> kma = KMA(num_clusters=5)
    >>> kma_centroids, kma_centroids_df = kma.fit(data=data)
    """

    def __init__(self, num_clusters: int, seed: int = 0) -> None:
        """
        Initializes the KMA class.

        Parameters
        ----------
        num_clusters : int
            The number of clusters to use in the K-Means algorithm.
            Must be greater than 0.
        seed : int, optional
            The random seed to use.
            Must be greater or equal to 0.
            Default is 0.

        Raises
        ------
        ValueError
            If num_centers is not greater than 0.
            Or if seed is not greater or equal to 0.
        """

        super().__init__()
        self.set_logger_name(name=self.__class__.__name__)
        if num_clusters > 0:
            self.num_clusters = int(num_clusters)
        else:
            raise ValueError("Variable num_clusters must be > 0")
        if seed >= 0:
            self.seed = int(seed)
        else:
            raise ValueError("Variable seed must be >= 0")
        # TODO: check random_state and n_init
        self._kma = KMeans(
            n_clusters=self.num_clusters, random_state=self.seed, n_init="auto"
        )
        self._data: pd.DataFrame = pd.DataFrame()
        self._normalized_data: pd.DataFrame = pd.DataFrame()
        self.data_variables: List[str] = []
        self.custom_scale_factor: dict = {}
        self.scale_factor: dict = {}
        self.centroids: pd.DataFrame = pd.DataFrame()
        self.normalized_centroids: pd.DataFrame = pd.DataFrame()
        self.bmus: np.array = np.array([])

    @property
    def kma(self) -> KMeans:
        return self._kma

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def normalized_data(self) -> pd.DataFrame:
        return self._normalized_data

    @validate_data_kma
    def fit(
        self,
        data: pd.DataFrame,
        custom_scale_factor: dict = {},
    ) -> None:
        """
        Fit the K-Means algorithm to the provided data.

        This method initializes centroids for the K-Means algorithm using the
        provided dataframe and custom scale factor.
        It normalizes the data, and returns the calculated centroids.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the KMA algorithm.
        custom_scale_factor : dict, optional
            A dictionary specifying custom scale factors for normalization.
            Default is {}.

        Notes
        -----
        - The function assumes that the data is validated by the `validate_data_kma`
        decorator before execution.
        - The method logs the progress of centroid initialization.
        """

        self._data = data.copy()
        self.data_variables = list(self.data.columns)
        self.custom_scale_factor = custom_scale_factor.copy()

        # TODO: add good explanation of fitting
        self.logger.info(
            f"\nkma parameters: {self._data.shape[0]} --> {self.num_clusters}\n"
        )

        # Normalize data using custom min max scaler
        self._normalized_data, self.scale_factor = self.normalize(
            data=self.data, custom_scale_factor=self.custom_scale_factor
        )

        # Fit K-Means algorithm
        kma = self.kma.fit(self.normalized_data)

        # Calculate the centroids
        self.bmus = kma.labels_
        self.normalized_centroids = pd.DataFrame(
            kma.cluster_centers_, columns=self.data_variables
        )
        self.centroids = self.denormalize(
            normalized_data=self.normalized_centroids, scale_factor=self.scale_factor
        )

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Predict the nearest centroid for the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the prediction.

        Returns
        -------
        Tuple[np.ndarray, pd.DataFrame]
            A tuple containing the nearest centroid index for each data point and the nearest centroids.
        """

        normalized_data, _ = self.normalize(
            data=data, custom_scale_factor=self.scale_factor
        )
        y = self.kma.predict(X=normalized_data)

        return y, self.centroids.iloc[y]

    def fit_predict(
        self,
        data: pd.DataFrame,
        custom_scale_factor: dict = {},
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Fit the K-Means algorithm to the provided data and predict the nearest centroid for each data point.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the KMA algorithm.
        custom_scale_factor : dict
            A dictionary specifying custom scale factors for normalization.
            Default is {}.

        Returns
        -------
        Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]
            A tuple containing the nearest centroid index for each data point, and the nearest centroids.
        """

        self.fit(
            data=data,
            custom_scale_factor=custom_scale_factor,
        )
        y, nearest_centroids = self.predict(data=data)

        return y, nearest_centroids
