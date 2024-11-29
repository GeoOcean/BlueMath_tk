import numpy as np
import pandas as pd
from typing import List, Tuple
from ._base_datamining import BaseClustering
from ..core.decorators import validate_data_mda


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
    data : pd.DataFrame
        The input data.
    normalized_data : pd.DataFrame
        The normalized input data.
    data_variables : List[str]
        A list of all data variables.
    directional_variables : List[str]
        A list with directional variables.
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
        A list of real indices of the centroids.

    Notes
    -----
    - This class is designed to perform the MDA algorithm.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from bluemath_tk.datamining.mda import MDA
    >>> data = pd.DataFrame(
    ...     {
    ...         'Hs': np.random.rand(1000) * 7,
    ...         'Tp': np.random.rand(1000) * 20,
    ...         'Dir': np.random.rand(1000) * 360
    ...     }
    ... )
    >>> mda = MDA(num_centers=10)
    >>> mda_centroids_df = mda.fit_predict(
    ...     data=data,
    ...     directional_variables=['Dir'],
    ...     custom_scale_factor={'Dir': [0, 360]},
    ... )
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
        self.data_variables: List[str] = []
        self.directional_variables: List[str] = []
        self.custom_scale_factor: dict = {}
        self.scale_factor: dict = {}
        self.centroids: pd.DataFrame = pd.DataFrame()
        self.normalized_centroids: pd.DataFrame = pd.DataFrame()
        self.centroid_iterative_indices: List[int] = []
        self.centroid_real_indices: List[int] = []
        self.is_fitted: bool = False

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def normalized_data(self) -> pd.DataFrame:
        return self._normalized_data

    def _normalized_distance(
        self, array_to_compare: np.ndarray, all_rest_data: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the normalized distance between the array_to_compare and all_rest_data.

        Parameters
        ----------
        array_to_compare : np.ndarray
            The array to compare against.
        all_rest_data : np.ndarray)
            The rest of the data.

        Returns
        -------
        dist : np.ndarray
            An array of squared Euclidean distances between the two arrays for each row.

        Raises
        ------
        MDAError
            If the function is NOT called before or during fitting.

        Notes
        -----
        - IMPORTANT: Data is assumed to be normalized before calling this function.
        - The function assumes that the data_variables, directional_variables, and scale_factor
        attributes have been set.
        - The function calculates the squared sum of differences for each row.
        - The calculation for directional variables is different, as it considers the minimum
        distance between the absolute difference and 1 minus the absolute difference,
        effectively "wrapping around" the normalized data range.
        """

        if not self.data_variables:
            raise MDAError(
                "_normalized_distance must be called after or during fitting, not before."
            )

        diff = np.zeros(all_rest_data.shape)

        # Calculate differences for columns
        ix = 0
        for data_var in self.data_variables:
            if data_var in self.directional_variables:
                distance = np.absolute(array_to_compare[:, ix] - all_rest_data[:, ix])
                diff[:, ix] = np.minimum(
                    distance, 1 - distance
                )  # * 2  # self.scale_factor.get(data_var)[1]
            else:
                diff[:, ix] = array_to_compare[:, ix] - all_rest_data[:, ix]
            ix = ix + 1

        # Compute the squared sum of differences for each row
        dist = np.sum(diff**2, axis=1)

        return dist

    def _nearest_indices_to_centroids(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Compute nearest data points to calculated centroids.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used to compute nearest data point to centroids.

        Returns
        -------
        nearest_indices_array : np.ndarray
            An array containing the index of the nearest data point to centroids.
        normalized_data.iloc[nearest_indices_array] : pd.DataFrame
            A DataFrame containing the nearest data points to centroids.

        Raises
        ------
        MDAError
            If the fit method has not been called before this method.
            Or if the data is empty.
        """

        if self.is_fitted is False:
            raise MDAError(
                "_nearest_indices must be called after or during fitting, not before."
            )
        if data.empty:
            raise MDAError("Data cannot be empty.")

        normalized_data, _scale_factor = self.normalize(
            data=data, custom_scale_factor=self.scale_factor
        )

        # Compute distances and store nearest distance index
        nearest_indices_array = np.zeros(self.normalized_centroids.shape[0], dtype=int)

        for i in range(self.normalized_centroids.shape[0]):
            rep = np.repeat(
                np.expand_dims(self.normalized_centroids.values[i, :], axis=0),
                normalized_data.values.shape[0],
                axis=0,
            )
            ndist = self._normalized_distance(
                array_to_compare=rep, all_rest_data=normalized_data.values
            )
            nearest_indices_array[i] = np.nanargmin(ndist)

        return nearest_indices_array, normalized_data.iloc[nearest_indices_array]

    def _nearest_indices(self, data: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Compute nearest centroids to the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used to compute nearest centroids.

        Returns
        -------
        nearest_indices_array : np.ndarray
            An array containing the index of the nearest centroid to the data.
        self.centroids.iloc[nearest_indices_array] : pd.DataFrame
            A DataFrame containing the nearest centroids to the data.

        Raises
        ------
        MDAError
            If the fit method has not been called before this method.
            Or if the data is empty.
        """

        if self.is_fitted is False:
            raise MDAError(
                "_nearest_indices must be called after or during fitting, not before."
            )
        if data.empty:
            raise MDAError("Data cannot be empty.")

        normalized_data, _scale_factor = self.normalize(
            data=data, custom_scale_factor=self.scale_factor
        )

        # Compute distances and store nearest distance index
        nearest_indices_array = np.zeros(normalized_data.shape[0], dtype=int)

        for i in range(normalized_data.shape[0]):
            rep = np.repeat(
                np.expand_dims(normalized_data.values[i, :], axis=0),
                self.normalized_centroids.values.shape[0],
                axis=0,
            )
            ndist = self._normalized_distance(
                array_to_compare=rep, all_rest_data=self.normalized_centroids.values
            )
            nearest_indices_array[i] = np.nanargmin(ndist)

        return nearest_indices_array, self.centroids.iloc[nearest_indices_array]

    @validate_data_mda
    def fit(
        self,
        data: pd.DataFrame,
        directional_variables: List[str] = [],
        custom_scale_factor: dict = {},
    ):
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
            A list of names of the directional variables within the data.
        custom_scale_factor : dict, optional
            A dictionary specifying custom scale factors for normalization.

        Notes
        -----
        - The function assumes that the data is validated by the `validate_data_mda`
        decorator before execution.
        - The method logs the progress of centroid initialization.
        """

        self._data = data.copy()
        self.data_variables = list(self.data.columns)
        self.directional_variables = directional_variables
        self.custom_scale_factor = custom_scale_factor

        # TODO: add good explanation for fitting
        self.logger.info(
            f"\nmda parameters: {self.data.shape[0]} --> {self.num_centers}\n"
        )

        # Normalize provided data with instantiated custom_scale_factor
        self._normalized_data, self.scale_factor = self.normalize(
            data=self.data, custom_scale_factor=self.custom_scale_factor
        )

        # [DEPRECATED] Select the point with the maximum value in the first column of pandas dataframe
        # seed = self.normalized_data[self.normalized_data.columns[0]].idxmax()
        # Select the point with the maximum summed value
        seed = self.normalized_data.sum(axis=1).idxmax()

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
                d_last = self._normalized_distance(
                    array_to_compare=xx2, all_rest_data=train
                )
            else:
                xx = np.array([subset[-1, :]])
                xx2 = np.repeat(xx, train.shape[0], axis=0)
                d_prev = self._normalized_distance(
                    array_to_compare=xx2, all_rest_data=train
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

        # Set the fitted flag to True
        self.is_fitted = True

        # De-normalize scalar and directional data
        self.normalized_centroids = pd.DataFrame(subset, columns=self.data_variables)
        self.centroids = self.denormalize(
            normalized_data=self.normalized_centroids, scale_factor=self.scale_factor
        )

        # TODO: use the normalized centroids and the norm_data to avoid rounding errors.
        # Calculate the real indices of the centroids
        self.centroid_real_indices = self._nearest_indices_to_centroids(
            data=self.normalized_data
        )

    def predict(self, data: pd.DataFrame):
        """
        Predict the nearest centroid for the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the prediction.

        Returns
        -------
        self._nearest_indices(data=data) : Tuple[np.ndarray, pd.DataFrame]
            A tuple containing the nearest centroid index for each data point and the nearest centroids.
        """

        return self._nearest_indices(data=data)

    def fit_predict(
        self,
        data: pd.DataFrame,
        directional_variables: List[str] = [],
        custom_scale_factor: dict = {},
    ):
        """
        Fits the MDA model to the data and predicts the nearest centroids.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be used for the MDA algorithm.
        directional_variables : List[str], optional
            A list of names of the directional variables within the data.
        custom_scale_factor : dict, optional
            A dictionary specifying custom scale factors for normalization.

        Returns
        -------
        self.predict(data=data) : Tuple[np.ndarray, pd.DataFrame]
            A tuple containing the nearest centroid index for each data point and the nearest centroids.
        """

        self.fit(
            data=data,
            directional_variables=directional_variables,
            custom_scale_factor=custom_scale_factor,
        )
        return self.predict(data=data)
