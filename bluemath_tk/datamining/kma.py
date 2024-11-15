import numpy as np
import pandas as pd
from typing import List
from sklearn.cluster import KMeans
from ..core.models import BlueMathModel
from ..core.decorators import validate_data_kma


class KMAError(Exception):
    """
    Custom exception for KMA class.
    """

    def __init__(self, message: str = "KMA error occurred."):
        self.message = message
        super().__init__(self.message)


class KMA(BlueMathModel):
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
    data_variables : list
        A list of all data variables.
    directional_variables : list
        A list with directional variables.
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
    >>> kma_centroids_df = kma.fit(
    ...     data=data,
    ...     directional_variables=['Dir'],
    ...     custom_scale_factor={'Dir': [0, 360]},
    ... )
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
            Defaults to 0.

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
            # TODO: check random_state and n_init
            self._kma = KMeans(
                n_clusters=self.num_clusters, random_state=seed, n_init="auto"
            )
        else:
            raise ValueError("Variable num_clusters must be > 0")
        if seed >= 0:
            self.seed = int(seed)
        else:
            raise ValueError("Variable seed must be >= 0")
        self._data: pd.DataFrame = pd.DataFrame()
        self._normalized_data: pd.DataFrame = pd.DataFrame()
        self.data_variables: list = []
        self.directional_variables: list = []
        self.custom_scale_factor: dict = {}
        self.scale_factor: dict = {}
        self.centroids: pd.DataFrame = pd.DataFrame()
        self.normalized_centroids: pd.DataFrame = pd.DataFrame()
        self.bmus: np.array = np.array([])

    @validate_data_kma
    def fit(
        self,
        data: pd.DataFrame,
        directional_variables: List[str],
        custom_scale_factor: dict,
    ):
        """
        Fit the K-Means algorithm to the provided data.

        This method initializes centroids for the K-Means algorithm using the
        provided dataframe, directional variables, and custom scale factor.
        It normalizes the data, and returns the calculated centroids.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the KMA algorithm.
        directional_variables : List[str]
            A list of names of the directional variables within the data.
        custom_scale_factor : dict
            A dictionary specifying custom scale factors for normalization.

        Returns
        -------
        pd.DataFrame
            The calculated centroids of the data.

        Notes
        -----
        - The function assumes that the data is validated by the `validate_data_kma`
        decorator before execution.
        - The method logs the progress of centroid initialization.
        """

        self._data = data.copy()
        self.data_variables = list(self._data.columns)
        self.directional_variables = directional_variables
        self.custom_scale_factor = custom_scale_factor

        # TODO: add good explanation of fitting
        self.logger.info(
            f"\nkma parameters: {self._data.shape[0]} --> {self.num_clusters}\n"
        )

        # Normalize data using custom min max scaler
        self._normalized_data, self.scale_factor = self.normalize(
            data=self._data, custom_scale_factor=self.custom_scale_factor
        )

        # Fit K-Means algorithm
        kma = self._kma.fit(self._normalized_data)

        # De-normalize scalar and directional data
        self.bmus = kma.labels_
        self.normalized_centroids = pd.DataFrame(
            kma.cluster_centers_, columns=self.data_variables
        )
        self.centroids = self.denormalize(
            normalized_data=self.normalized_centroids, scale_factor=self.scale_factor
        )

        return self.centroids
