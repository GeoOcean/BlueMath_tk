from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from ..core.decorators import validate_data_mda
from ._base_datamining import BaseClustering


def calculate_normalized_squared_distance(
    data_array: Union[np.ndarray, pd.DataFrame],
    array_to_compare: Union[np.ndarray, pd.DataFrame],
    directional_indices: List[int] = None,
    weights: List[float] = None,
) -> np.ndarray:
    """
    Calculate the normalized squared distance between the data_array and the array_to_compare.
    ALERT: directional_indices will be deprecated in the future.

    Parameters
    ----------
    data_array : Union[np.ndarray, pd.DataFrame]
        The data array to compare. Dimensions: (1, n_features).
    array_to_compare : Union[np.ndarray, pd.DataFrame]
        The array to compare against. Dimensions: (n_samples, n_features).
    directional_indices : List[int], optional
        List of column indices that contain directional data.
        For these columns, the minimum circular distance will be used.
        Default is None.
    weights : List[float], optional
        List of weights to apply to each column's distance.
        Must have the same length as the number of columns.
        Default is None (equal weights).

    Returns
    -------
    np.ndarray
        An array of normalized squared distance between the two arrays.
        Dimensions: (n_samples, 1).

    Raises
    ------
    ValueError
        If the arrays have different numbers of columns.
        If weights are provided but length doesn't match number of columns.

    Examples
    --------
    >>> calculate_normalized_squared_distance(
    ...     data_array=np.array([[1, 2, 3]]),
    ...     array_to_compare=np.array([[1, 2, 3], [4, 5, 6]]),
    ... )
    [0.0, 27.0]

    Notes
    -----
    - IMPORTANT: Data is assumed to be normalized before calling this function.
    - For directional variables, the function calculates the minimum circular distance.
      Assuming data is between 0 and 1 (normalized).
    - The function calculates weighted squared differences for each row.
    - If DataFrames are provided, they will be converted to numpy arrays.
    """

    if isinstance(data_array, pd.DataFrame):
        data_array = data_array.values
    if isinstance(array_to_compare, pd.DataFrame):
        array_to_compare = array_to_compare.values

    if data_array.shape[1] != array_to_compare.shape[1]:
        raise ValueError("Arrays must have the same number of columns")

    if weights is not None and len(weights) != data_array.shape[1]:
        raise ValueError("Length of weights must match number of columns")

    # Calculate initial differences
    diff = data_array - array_to_compare

    # Handle directional variables if specified
    if directional_indices is not None:
        for idx in directional_indices:
            # Calculate absolute angular difference
            abs_diff = np.absolute(diff[:, idx])
            # Use minimum circular distance
            diff[:, idx] = np.minimum(abs_diff, 1 - abs_diff)

    # Apply weights if specified
    if weights is not None:
        for i, weight in enumerate(weights):
            diff[:, i] *= weight

    # Compute the squared sum of differences for each row
    dist = np.sum(diff**2, axis=1)

    return dist


def find_nearest_indices(
    query_points: Union[np.ndarray, pd.DataFrame],
    reference_points: Union[np.ndarray, pd.DataFrame],
    directional_indices: List[int] = None,
    weights: List[float] = None,
) -> np.ndarray:
    """
    Find the indices of nearest points in reference_points for each point in query_points.

    Parameters
    ----------
    query_points : Union[np.ndarray, pd.DataFrame]
        The points to find nearest neighbors for.
    reference_points : Union[np.ndarray, pd.DataFrame]
        The set of points to search in.
    directional_indices : List[int], optional
        List of column indices that contain directional data.
        For these columns, the minimum circular distance will be used.
        Default is None.
    weights : List[float], optional
        List of weights to apply to each column's distance.
        Must have the same length as the number of columns.
        Default is None (equal weights).

    Returns
    -------
    np.ndarray
        An array containing the index of the nearest reference point for each query point.

    Examples
    --------
    >>> # Finding nearest centroids for data points
    >>> data = np.random.rand(100, 3)  # 100 points with 3 features
    >>> centroids = np.random.rand(5, 3)  # 5 centroids
    >>> nearest_centroid_indices = find_nearest_indices(data, centroids)
    """

    if isinstance(query_points, pd.DataFrame):
        query_points = query_points.values
    if isinstance(reference_points, pd.DataFrame):
        reference_points = reference_points.values

    nearest_indices = np.zeros(query_points.shape[0], dtype=int)

    for i in range(query_points.shape[0]):
        rep = np.repeat(
            np.expand_dims(query_points[i, :], axis=0),
            reference_points.shape[0],
            axis=0,
        )
        ndist = calculate_normalized_squared_distance(
            data_array=rep,
            array_to_compare=reference_points,
            directional_indices=directional_indices,
            weights=weights,
        )
        nearest_indices[i] = np.nanargmin(ndist)

    return nearest_indices


class MDAError(Exception):
    """
    Custom exception for MDA class.
    """

    def __init__(self, message: str = "MDA error occurred."):
        self.message = message
        super().__init__(self.message)


class MDA(BaseClustering):
    """
    Maximum Dissimilarity Algorithm (MDA) class.

    This class performs the MDA algorithm on a given dataframe.

    Attributes
    ----------
    num_centers : int
        The number of centers to use in the MDA algorithm.
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
    centroid_iterative_indices : List[int]
        A list of iterative indices of the centroids.
    centroid_real_indices : List[int]
        The real indices of the selected centroids.
    is_fitted : bool
        A flag indicating whether the model is fitted or not.

    Examples
    --------
    .. jupyter-execute::

        import numpy as np
        import pandas as pd
        from bluemath_tk.datamining.mda import MDA

        data = pd.DataFrame(
            {
                "Hs": np.random.rand(1000) * 7,
                "Tp": np.random.rand(1000) * 20,
                "Dir": np.random.rand(1000) * 360
            }
        )
        mda = MDA(num_centers=5)
        nearest_centroids_idxs, nearest_centroids_df = mda.fit_predict(
            data=data,
            directional_variables=["Dir"],
        )

        mda.plot_selected_centroids(plot_text=True)
    """

    def __init__(self, num_centers: int) -> None:
        """
        Initializes the MDA class.

        Parameters
        ----------
        num_centers : int
            The number of centers to use in the MDA algorithm.
            Must be greater than 0.

        Raises
        ------
        ValueError
            If num_centers is not greater than 0.
        """

        super().__init__()
        self.set_logger_name(name=self.__class__.__name__)

        if num_centers > 0:
            self.num_centers = int(num_centers)
        else:
            raise ValueError("Variable num_centers must be > 0")

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
        self.centroid_iterative_indices: List[int] = []
        self.centroid_real_indices: np.ndarray = np.array([])
        self.is_fitted: bool = False

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

    def _nearest_indices(
        self, normalized_data: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Compute nearest centroids to the provided data.

        Parameters
        ----------
        normalized_data : pd.DataFrame
            The input data to be used to compute nearest centroids.

        Returns
        -------
        Tuple[np.ndarray, pd.DataFrame]
            An array containing the index of the nearest centroid to the data,
            and a DataFrame containing the nearest centroids.

        Raises
        ------
        MDAError
            If the data is empty.
        """

        if normalized_data.empty:
            raise MDAError("Data cannot be empty.")

        nearest_indices_array = find_nearest_indices(
            query_points=normalized_data,
            reference_points=self.normalized_centroids,
        )

        return nearest_indices_array, self.centroids.iloc[nearest_indices_array]

    @validate_data_mda
    def fit(
        self,
        data: pd.DataFrame,
        directional_variables: List[str] = [],
        custom_scale_factor: dict = {},
        first_centroid_seed: int = None,
        normalize_data: bool = False,
    ) -> None:
        """
        Fit the Maximum Dissimilarity Algorithm (MDA) to the provided data.

        This method initializes centroids for the MDA algorithm using the provided
        dataframe, directional variables, and custom scale factor. It normalizes the
        data, iteratively selects centroids based on maximum dissimilarity, and
        denormalizes the centroids before returning them.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the MDA algorithm.
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
        first_centroid_seed : int, optional
            The index of the first centroid to use in the MDA algorithm.
            Default is None.
        normalize_data : bool, optional
            A flag to normalize the data.
            If True, the data will be normalized using the custom_scale_factor.
            Default is False.

        Notes
        -----
        - When first_centroid_seed is not provided, max value centroid is used.
        """

        super().fit(
            data=data,
            directional_variables=directional_variables,
            custom_scale_factor=custom_scale_factor,
            normalize_data=normalize_data,
        )

        # Select seed point
        if first_centroid_seed is not None:
            seed = first_centroid_seed
            self.logger.info(f"Using specified seed={seed} as first centroid.")
        else:
            seed = np.argmax(self.normalized_data.sum(axis=1).values)
            self.logger.info(
                f"Using max calculated value seed={seed} as first centroid."
            )

        # Initialize centroids subset
        subset = np.array(
            [self.normalized_data.values[seed]]
        )  # The row that starts as seed
        train = np.delete(self.normalized_data.values, seed, axis=0)

        # Repeat until we have the desired num_centers
        n_c = 1
        while n_c < self.num_centers:
            m2 = subset.shape[0]
            if m2 == 1:
                xx2 = np.repeat(subset, train.shape[0], axis=0)
                d_last = calculate_normalized_squared_distance(
                    data_array=xx2,
                    array_to_compare=train,
                )
            else:
                xx = np.array([subset[-1, :]])
                xx2 = np.repeat(xx, train.shape[0], axis=0)
                d_prev = calculate_normalized_squared_distance(
                    data_array=xx2,
                    array_to_compare=train,
                )
                d_last = np.minimum(d_prev, d_last)

            qerr, bmu = np.nanmax(d_last), np.nanargmax(d_last)

            if not np.isnan(qerr):
                self.centroid_iterative_indices.append(bmu)
                subset = np.append(subset, np.array([train[bmu, :]]), axis=0)
                train = np.delete(train, bmu, axis=0)
                d_last = np.delete(d_last, bmu, axis=0)

                # Log
                fmt = "0{0}d".format(len(str(self.num_centers)))
                self.logger.info(
                    "   MDA centroids: {1:{0}}/{2:{0}}".format(
                        fmt, subset.shape[0], self.num_centers
                    )
                )

            n_c = subset.shape[0]

        # De-normalize scalar and directional data
        self.normalized_centroids = pd.DataFrame(subset, columns=self.fitting_variables)
        self.centroids = self.denormalize(
            normalized_data=self.normalized_centroids, scale_factor=self.scale_factor
        )
        for directional_variable in self.directional_variables:
            self.centroids[directional_variable] = self.get_degrees_from_uv(
                xu=self.centroids[f"{directional_variable}_u"].values,
                xv=self.centroids[f"{directional_variable}_v"].values,
            )

        # Calculate the real indices of the centroids
        self.centroid_real_indices = find_nearest_indices(
            query_points=self.normalized_centroids,
            reference_points=self.normalized_data,
        )

        # Set the fitted flag to True
        self.is_fitted = True

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

        if self.is_fitted is False:
            raise MDAError("MDA model is not fitted.")

        normalized_data = super().predict(data=data)

        return self._nearest_indices(normalized_data=normalized_data)

    def fit_predict(
        self,
        data: pd.DataFrame,
        directional_variables: List[str] = [],
        custom_scale_factor: dict = {},
        first_centroid_seed: int = None,
        normalize_data: bool = False,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Fits the MDA model to the data and predicts the nearest centroids.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the MDA algorithm.
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
        first_centroid_seed : int, optional
            The index of the first centroid to use in the MDA algorithm.
            Default is None.
        normalize_data : bool, optional
            A flag to normalize the data.
            If True, the data will be normalized using the custom_scale_factor.
            Default is False.

        Returns
        -------
        Tuple[np.ndarray, pd.DataFrame]
            A tuple containing the nearest centroid index for each data point and the nearest centroids.
        """

        self.fit(
            data=data,
            directional_variables=directional_variables,
            custom_scale_factor=custom_scale_factor,
            first_centroid_seed=first_centroid_seed,
            normalize_data=normalize_data,
        )

        return self.predict(data=data)
