"""
Package: BlueMath_tk
Module: datamining
File: kma.py
Author: GeoOcean Research Group, Universidad de Cantabria
Repository: https://github.com/GeoOcean/BlueMath_tk.git
Status: Under development (Working)
"""

import platform

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression

from ..core.decorators import validate_data_kma
from ._base_datamining import BaseClustering


class KMAError(Exception):
    """
    Custom exception for the KMA class.
    """

    def __init__(self, message: str = "KMA error occurred."):
        self.message = message
        super().__init__(self.message)


class KMA(BaseClustering):
    """
    K-Means Algorithm (KMA) class - generalized to support multiple clustering
    algorithms.

    This class performs centroid-based clustering on a given dataframe.
    Supports k-means, k-medians, and k-medoids (all via pyclustering).

    Examples
    --------
    .. jupyter-execute::

        import numpy as np
        import pandas as pd
        from bluemath_tk.datamining.kma import KMA

        data = pd.DataFrame(
            {
                "Hs": np.random.rand(1000) * 7,
                "Tp": np.random.rand(1000) * 20,
                "Dir": np.random.rand(1000) * 360
            }
        )
        # K-Means (default)
        kma = KMA(num_clusters=5)
        kma.fit(
            data=data,
            directional_variables=["Dir"],
        )

    References
    ----------
    [1] https://github.com/asavvides/pyclustering
    """

    def __init__(
        self,
        num_clusters: int,
        seed: int | None = None,
        algorithm: str = "kmeans",
        distance_metric: str | None = None,
        distance_matrix: np.ndarray | None = None,
    ) -> None:
        """
        Initialize the KMA class.

        Parameters
        ----------
        num_clusters : int
            The number of clusters to use in the clustering algorithm.
            Must be greater than 0.
        seed : int, optional
            The random seed to use as initial datapoint.
            Must be greater or equal to 0 and less than number of datapoints.
            Default is None.
        algorithm : str, optional
            The clustering algorithm to use. Options:
            - 'kmeans': K-Means (default)
            - 'kmedians': K-Medians
            - 'kmedoids': K-Medoids

        distance_metric : str, optional
            Distance metric to use. Options depend on algorithm:
            - For kmeans/kmedians: 'euclidean' (default), 'manhattan',
              'chebyshev'
            - For kmedoids: 'euclidean' (default), 'manhattan', 'chebyshev',
              'minkowski'
            Default is None (uses algorithm-specific default).

        distance_matrix : np.ndarray, optional
            Precomputed distance matrix. Shape (n_samples, n_samples).
            Element (i, j) represents the distance between data points i and j.
            Currently only supported for k-medoids algorithm.
            If provided, the algorithm will use this matrix instead of computing
            distances from raw data. Default is None.
        """

        super().__init__()
        self.set_logger_name(name=self.__class__.__name__)

        initial_msg = f"""
        ---------------------------------------------------------------------------------
        | Initializing KMA object with the following parameters:
        |    - num_clusters: {num_clusters}
        |    - seed: {seed}
        |    - algorithm: {algorithm}
        |    - distance_metric: {distance_metric}
        |    - distance_matrix: {distance_matrix}
        | For more information, please refer to the documentation.
        ---------------------------------------------------------------------------------
        """
        self.logger.info(initial_msg)

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

        # Validate algorithm
        algorithm = algorithm.lower()
        if algorithm not in ["kmeans", "kmedians", "kmedoids"]:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                "Supported: 'kmeans', 'kmedians', 'kmedoids'"
            )

        # Store algorithm info
        self.algorithm_name = algorithm
        self.distance_metric = distance_metric
        self.distance_matrix = distance_matrix

        # Validate distance matrix if provided
        if distance_matrix is not None:
            if algorithm != "kmedoids":
                raise ValueError(
                    f"distance_matrix is only supported for 'kmedoids' algorithm, "
                    f"not '{algorithm}'"
                )
            distance_matrix = np.array(distance_matrix)
            if distance_matrix.ndim != 2:
                raise ValueError("distance_matrix must be 2-dimensional")
            if distance_matrix.shape[0] != distance_matrix.shape[1]:
                raise ValueError("distance_matrix must be square")
            self.distance_matrix = distance_matrix

        # Internal state
        self._model = None
        self._labels = None
        self._cluster_centers = None
        self._init_centers = None

        self.logger.info(
            f"KMA object created with {self.num_clusters} clusters, "
            f"algorithm '{self.algorithm_name}', and seed {self.seed}."
        )

        self._data: pd.DataFrame = pd.DataFrame()
        self._normalized_data: pd.DataFrame = pd.DataFrame()
        self._data_to_fit: pd.DataFrame = pd.DataFrame()
        self.data_variables: list[str] = []
        self.directional_variables: list[str] = []
        self.fitting_variables: list[str] = []
        self.custom_scale_factor: dict = {}
        self.scale_factor: dict = {}
        self.centroids: pd.DataFrame = pd.DataFrame()
        self.normalized_centroids: pd.DataFrame = pd.DataFrame()
        self.centroid_real_indices: np.array = np.array([])
        self.is_fitted: bool = False
        self.regression_guided: dict = {}

    def _get_initial_centers(self, data: list) -> list:
        """
        Get initial centers/medians/medoids for the algorithm.

        Parameters
        ----------
        data : list
            Data in pyclustering format (list of lists) or distance matrix.

        Returns
        -------
        list
            Initial centers/medians/medoids (indices if using distance matrix).
        """

        if self._init_centers is not None:
            # Use provided initial centers
            if isinstance(self._init_centers, pd.DataFrame):
                if (
                    self.distance_matrix is not None
                    or self.algorithm_name == "kmedoids"
                ):
                    # For distance matrix or kmedoids, we need indices,
                    # not actual centers
                    # Find nearest data points to the provided centers
                    if self.distance_matrix is not None:
                        raise NotImplementedError(
                            "MDA initialization with distance matrix not yet supported"
                        )
                    # For kmedoids without distance matrix, find nearest indices
                    init_values = self._init_centers.values
                    data_array = np.array(data)
                    distances = cdist(init_values, data_array)
                    nearest_indices = np.argmin(distances, axis=1)
                    initial_centers = [int(i) for i in nearest_indices]
                else:
                    initial_centers = self._init_centers.values.tolist()
            else:
                if (
                    self.distance_matrix is not None
                    or self.algorithm_name == "kmedoids"
                ):
                    # For distance matrix or kmedoids,
                    # initial_centers should be indices
                    initial_centers = [
                        int(i) for i in np.array(self._init_centers).flatten()
                    ]
                else:
                    initial_centers = np.array(self._init_centers).tolist()
        else:
            # Initialize randomly
            if self.seed is not None:
                np.random.seed(self.seed)
            n_samples = len(data) if isinstance(data, list) else data.shape[0]
            initial_centers_idx = np.random.choice(
                n_samples, size=self.num_clusters, replace=False
            )
            if self.distance_matrix is not None or self.algorithm_name == "kmedoids":
                # For distance matrix or kmedoids, return indices
                initial_centers = [int(i) for i in initial_centers_idx]
            else:
                # For raw data with kmeans/kmedians, return actual data points
                initial_centers = [data[i] for i in initial_centers_idx]

        return initial_centers

    def _create_pyclustering_model(
        self, data: list | np.ndarray, initial_centers: list
    ):
        """
        Create and return the pyclustering model.

        Parameters
        ----------
        data : list or np.ndarray
            Data in pyclustering format (list of lists) or distance matrix.
        initial_centers : list
            Initial centers/medians/medoids (or indices if using distance matrix).

        Returns
        -------
        pyclustering model
            The initialized pyclustering clustering model.
        """

        # Build kwargs for pyclustering
        kwargs = {}
        # Use Python implementation (ccore=False) on macOS to avoid architecture
        # compatibility issues with the native C++ library (x86_64 vs arm64)
        # On other platforms, use the faster C++ implementation (ccore=True, default)
        if platform.system() == "Darwin":  # macOS
            kwargs["ccore"] = False
        if self.distance_metric is not None:
            # Map common metric names to pyclustering format if needed
            # Note: pyclustering's distance metric handling varies by algorithm
            # For simplicity, we'll let pyclustering use defaults
            # Advanced users can modify the model directly if needed
            pass

        # Import and create the appropriate algorithm
        if self.algorithm_name == "kmeans":
            from pyclustering.cluster.kmeans import kmeans

            self._model = kmeans(data, initial_centers, **kwargs)
        elif self.algorithm_name == "kmedians":
            from pyclustering.cluster.kmedians import kmedians

            self._model = kmedians(data, initial_centers, **kwargs)
        elif self.algorithm_name == "kmedoids":
            from pyclustering.cluster.kmedoids import kmedoids

            # For k-medoids, support distance matrix
            if self.distance_matrix is not None:
                # Convert distance matrix to list of lists
                if isinstance(self.distance_matrix, np.ndarray):
                    distance_data = self.distance_matrix.tolist()
                else:
                    distance_data = self.distance_matrix
                self._model = kmedoids(
                    distance_data,
                    initial_centers,
                    data_type="distance_matrix",
                    **kwargs,
                )
            else:
                self._model = kmedoids(data, initial_centers, **kwargs)

        return self._model

    def _fit_clustering(self, normalized_data: pd.DataFrame):
        """
        Fit the clustering algorithm to normalized data.

        Parameters
        ----------
        normalized_data : pd.DataFrame
            Normalized data to fit (ignored if distance_matrix is provided).
        """

        try:
            # Use distance matrix if provided, otherwise use normalized data
            if self.distance_matrix is not None:
                data = self.distance_matrix
            else:
                # Convert to list of lists (pyclustering format)
                data = normalized_data.values.tolist()

            # Get initial centers
            initial_centers = self._get_initial_centers(data)

            # Create and process the model
            self._create_pyclustering_model(data, initial_centers)
            self._model.process()

            # Extract labels
            clusters = self._model.get_clusters()
            n_samples = (
                self.distance_matrix.shape[0]
                if self.distance_matrix is not None
                else len(data)
            )
            self._labels = np.zeros(n_samples, dtype=int)
            for cluster_idx, cluster in enumerate(clusters):
                for point_idx in cluster:
                    self._labels[point_idx] = cluster_idx

            # Extract centers (method name varies by algorithm)
            if self.algorithm_name == "kmeans":
                centers = self._model.get_centers()
            elif self.algorithm_name == "kmedians":
                centers = self._model.get_medians()
            elif self.algorithm_name == "kmedoids":
                # For kmedoids, get_medoids() always returns indices
                medoid_indices = self._model.get_medoids()
                if self.distance_matrix is not None:
                    # For distance matrix, medoids are indices, not actual points
                    # We need to get the actual data points corresponding to medoid
                    # indices
                    # If we have normalized_data, use it to get actual centers
                    if not normalized_data.empty:
                        centers = normalized_data.iloc[medoid_indices].values
                    else:
                        # Can't get actual centers without data, use indices as
                        # placeholder. This shouldn't happen in normal usage
                        raise KMAError(
                            "Cannot extract centers from distance matrix without "
                            "original data. Provide normalized_data when using "
                            "distance_matrix."
                        )
                else:
                    # For kmedoids without distance matrix,
                    # medoid_indices are still indices
                    # Convert to actual data points
                    if not normalized_data.empty:
                        centers = normalized_data.iloc[medoid_indices].values
                    else:
                        # Fallback: use indices (shouldn't happen)
                        centers = np.array(medoid_indices)

            self._cluster_centers = np.array(centers)

        except ImportError:
            raise ImportError(
                "Clustering algorithms require pyclustering. "
                "Install it with: pip install pyclustering"
            )

    def _predict_clustering(self, normalized_data: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for normalized data.

        Parameters
        ----------
        normalized_data : pd.DataFrame
            Normalized data to predict.

        Returns
        -------
        np.ndarray
            Cluster labels.
        """

        if self._model is None:
            raise KMAError("Model must be fitted before prediction.")

        # Convert to numpy array if needed
        if isinstance(normalized_data, pd.DataFrame):
            X = normalized_data.values
        else:
            X = np.array(normalized_data)

        # Calculate distances to each center
        centers = self._cluster_centers

        # Use appropriate distance metric based on algorithm
        if self.algorithm_name == "kmedians":
            # L1 distance for medians
            distances = np.array(
                [np.sum(np.abs(X - center), axis=1) for center in centers]
            )
        else:
            # L2 distance (Euclidean) for means and medoids
            distances = np.array(
                [np.sum((X - center) ** 2, axis=1) for center in centers]
            )

        # Assign to closest center
        labels = np.argmin(distances, axis=0)
        return labels

    @property
    def kma(self) -> object:
        """
        Returns the clustering algorithm object (for advanced usage).

        Returns
        -------
        object
            The pyclustering model object. Can be modified directly for advanced
            usage.
        """
        return self._model

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
        Returns the data used for fitting the clustering algorithm.
        """
        return self._data_to_fit

    @staticmethod
    def add_regression_guided(
        data: pd.DataFrame, vars: list[str], alpha: list[float]
    ) -> pd.DataFrame:
        """
        Calculate regression-guided variables.

        Parameters
        ----------
        data : pd.DataFrame
            The data to fit the clustering algorithm.
        vars : list[str]
            The variables to use for regression-guided clustering.
        alpha : list[float]
            The alpha values to use for regression-guided clustering.

        Returns
        -------
        pd.DataFrame
            The data with the regression-guided variables.
        """

        # Stack guiding variables into (time, n_vars) array
        X = data.drop(columns=vars)
        Y = np.stack([data[var].values for var in vars], axis=1)

        # Normalize input features
        X_std = X.std().replace(0, 1)
        X_norm = X / X_std

        # Add intercept column to input
        X_design = np.column_stack((np.ones(len(X)), X_norm.values))

        # Normalize guiding targets
        Y_std = np.nanstd(Y, axis=0)
        Y_std[Y_std == 0] = 1.0

        # Fit regression model to predict guiding vars from input
        model = LinearRegression(fit_intercept=False).fit(X_design, Y / Y_std)
        Y_pred = model.predict(X_design) * Y_std  # De-normalize predictions

        # Weight columns by input alpha
        X_weight = 1.0 - np.sum(alpha)
        X_scaled = X_weight * X.values
        Y_scaled = Y_pred * alpha

        df = pd.DataFrame(np.hstack([X_scaled, Y_scaled]), index=data.index)
        df.columns = list(X.columns) + vars

        return df

    @validate_data_kma
    def fit(
        self,
        data: pd.DataFrame,
        directional_variables: list[str] = [],
        custom_scale_factor: dict = {},
        min_number_of_points: int = None,
        max_number_of_iterations: int = 10,
        normalize_data: bool = False,
        regression_guided: dict[str, list] = {},
        init_mda_centroids: pd.DataFrame | None = None,
    ) -> None:
        """
        Fit the clustering algorithm to the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the clustering algorithm.
        directional_variables : list[str], optional
            A list of directional variables that will be transformed to u and v
            components.
            Then, to use custom_scale_factor, you must specify the variables names with
            the u and v suffixes.
            Example: directional_variables=["Dir"],
            custom_scale_factor={"Dir_u": [0, 1], "Dir_v": [0, 1]}.
            Default is [].
        custom_scale_factor : dict, optional
            A dictionary specifying custom scale factors for normalization.
            If normalize_data is True, this will be used to normalize the data.
            Example: {"Hs": [0, 10], "Tp": [0, 10]}.
            Default is {}.
        min_number_of_points : int, optional
            The minimum number of points to consider a cluster.
            Default is None.
        max_number_of_iterations : int, optional
            The maximum number of iterations for the clustering algorithm.
            This is used when min_number_of_points is not None.
            Default is 10.
        normalize_data : bool, optional
            A flag to normalize the data.
            If True, the data will be normalized using the custom_scale_factor.
            Default is False.
        regression_guided: dict, optional
            A dictionary specifying regression-guided clustering variables and
            relative weights.
            Example: {"vars": ["Fe"], "alpha": [0.6]}.
            Default is {}.
        init_mda_centroids : pd.DataFrame, optional
            Normalized centroids from MDA algorithm to use as initialization.
            Should have shape (num_clusters, n_features) and be in normalized space.
            If provided, the clustering algorithm will be initialized with these
            centroids. Default is None.
        """

        if regression_guided:
            data = self.add_regression_guided(
                data=data,
                vars=regression_guided.get("vars", None),
                alpha=regression_guided.get("alpha", None),
            )

        super().fit(
            data=data,
            directional_variables=directional_variables,
            custom_scale_factor=custom_scale_factor,
            normalize_data=normalize_data,
        )

        # Handle MDA initialization if provided
        if init_mda_centroids is not None:
            if not isinstance(init_mda_centroids, pd.DataFrame):
                raise ValueError("init_mda_centroids must be a pd.DataFrame")
            if init_mda_centroids.shape[0] != self.num_clusters:
                raise ValueError(
                    f"init_mda_centroids must have {self.num_clusters} rows, "
                    f"but got {init_mda_centroids.shape[0]}"
                )
            if list(init_mda_centroids.columns) != self.fitting_variables:
                raise ValueError(
                    f"init_mda_centroids columns must match fitting_variables: "
                    f"{self.fitting_variables}"
                )

            # Store MDA centroids for initialization
            self._init_centers = init_mda_centroids
            self.logger.info(f"Initializing {self.algorithm_name} with MDA centroids.")

        # Fit clustering algorithm
        if min_number_of_points is not None:
            stable_cluster = False
            number_of_tries = 0
            while not stable_cluster:
                # Create a new instance
                # If MDA init was provided, use it for first try, then random
                if init_mda_centroids is not None and number_of_tries == 0:
                    self._init_centers = init_mda_centroids
                else:
                    self._init_centers = None
                    self.seed = None  # Use different random state each try

                self._fit_clustering(self.normalized_data)
                predicted_labels = self._labels.copy()

                _unique_labels, counts = np.unique(predicted_labels, return_counts=True)
                if np.all(counts >= min_number_of_points):
                    stable_cluster = True
                number_of_tries += 1
                if number_of_tries > max_number_of_iterations:
                    raise ValueError(
                        f"Failed to find a stable {self.algorithm_name} configuration "
                        f"after {max_number_of_iterations} attempts. "
                        "Change max_number_of_iterations or min_number_of_points."
                    )
            self.logger.info(
                f"Found a stable {self.algorithm_name} configuration after "
                f"{number_of_tries} attempts."
            )
        else:
            self._fit_clustering(self.normalized_data)

        # Calculate the centroids
        self.centroid_real_indices = self._labels.copy()

        # Use the cluster centers from the model
        if self._cluster_centers is not None:
            centers = self._cluster_centers
        else:
            # Fallback: compute centroids from labels
            centers = np.array(
                [
                    self.normalized_data[self._labels == i].mean(axis=0)
                    for i in range(self.num_clusters)
                ]
            )

        self.normalized_centroids = pd.DataFrame(
            centers, columns=self.fitting_variables
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

    def predict(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predict the nearest centroid for the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the prediction.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the nearest centroid index for each data point,
            and the nearest centroids.
        """

        if self.is_fitted is False:
            raise KMAError("KMA model is not fitted.")

        normalized_data = super().predict(data=data)

        y = self._predict_clustering(normalized_data)

        return (
            pd.DataFrame(y, columns=["kma_bmus"], index=data.index),
            self.centroids.iloc[y],
        )

    def fit_predict(
        self,
        data: pd.DataFrame,
        directional_variables: list[str] = [],
        custom_scale_factor: dict = {},
        min_number_of_points: int = None,
        max_number_of_iterations: int = 10,
        normalize_data: bool = False,
        regression_guided: dict[str, list] = {},
        init_mda_centroids: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit the clustering algorithm to the provided data and predict the nearest
        centroid for each data point.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the clustering algorithm.
        directional_variables : list[str], optional
            A list of directional variables that will be transformed to u and v
            components.
            Then, to use custom_scale_factor, you must specify the variables names with
            the u and v suffixes.
            Example: directional_variables=["Dir"],
            custom_scale_factor={"Dir_u": [0, 1], "Dir_v": [0, 1]}.
            Default is [].
        custom_scale_factor : dict, optional
            A dictionary specifying custom scale factors for normalization.
            If normalize_data is True, this will be used to normalize the data.
            Example: {"Hs": [0, 10], "Tp": [0, 10]}.
            Default is {}.
        min_number_of_points : int, optional
            The minimum number of points to consider a cluster.
            Default is None.
        max_number_of_iterations : int, optional
            The maximum number of iterations for the clustering algorithm.
            This is used when min_number_of_points is not None.
            Default is 10.
        normalize_data : bool, optional
            A flag to normalize the data.
            If True, the data will be normalized using the custom_scale_factor.
            Default is False.
        regression_guided: dict, optional
            A dictionary specifying regression-guided clustering variables and
            relative weights.
            Example: {"vars": ["Fe"], "alpha": [0.6]}.
            Default is {}.
        init_mda_centroids : pd.DataFrame, optional
            Normalized centroids from MDA algorithm to use as initialization.
            Should have shape (num_clusters, n_features) and be in normalized space.
            If provided, the clustering algorithm will be initialized with these
            centroids. Default is None.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
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
            regression_guided=regression_guided,
            init_mda_centroids=init_mda_centroids,
        )

        return self.predict(data=data)
