"""

- Project: Bluemath{toolkit}.datamining
- File: mda.py
- Description: Maximum Dissimilarity Algorithm
- Author: GeoOcean Research Group, Universidad de Cantabria
- Created Date: 19 January 2024
- License: MIT
- Repository: https://gitlab.com/geoocean/bluemath/toolkit/

"""

import numpy as np
import pandas as pd
from typing import List
from ..core.models import BlueMathModel


class MDAError(Exception):
    """
    Custom exception for MDA class.
    """

    def __init__(self, message="MDA error occurred."):
        self.message = message
        super().__init__(self.message)


class MDA(BlueMathModel):
    """
    This class implements the MDA algorithm (Maximum Dissimilarity Algorithm)

    ...

    Attributes
    ----------
    data : pd.DataFrame
        The data to be clustered. Each column will represent a different variable

    ix_directional : List[str]
        List with the names of the directional variables in the data. If no directional
        variables are present, this list should be empty.

    Methods
    -------
    run()
        Normalize data and calculate centers using maxdiss algorithm

    scatter_data()
        Plot the data and/or the centroids

    Examples
    --------
    df = pd.DataFrame({
        'Hs': np.random.rand(1000)*7,
        'Tp': np.random.rand(1000)*20,
        'Dir': np.random.rand(1000)*360
    })
    mda_ob = MDA(data=df, ix_directional=['Dir'])
    mda_ob.run(10)
    mda_ob.scatter_data()
    """

    def __init__(
        self,
        num_centers: int,
        directional_variables: List[str] = [],
        custom_scale_factor: dict = {},
    ) -> None:
        super().__init__()
        if num_centers > 0:
            self.num_centers = num_centers
        else:
            raise ValueError("Variable num_centers must be > 0")
        for directional_variable in directional_variables:
            if directional_variable not in custom_scale_factor:
                raise KeyError(
                    "All directional_variables must have an associated custom_scale_factor"
                )
        self.directional_variables = directional_variables
        self.custom_scale_factor = custom_scale_factor
        # NOTE: Below, important class variables will be declared to be filled
        self.data_variables: List[str] = []
        self.scale_factor: dict = {}
        # TO DEPRECATE: All class variables below
        self.centroid_iterative_indices: List[int] = []
        self.centroid_real_indices: List[int] = []

    def _normalized_distance(self, train, subset):
        """
        Compute the normalized distance between rows in train and subset.

        Args
        ----
        train : numpy.ndarray
            Train matrix
        subset : numpy.ndarray
            Subset matrix

        Returns
        -------
        dist: numpy.ndarray
            normalized distances
        """

        if not self.data_variables or not self.scale_factor:
            raise MDAError(
                "Normalized distance must be called after or during fitting, not before."
            )

        diff = np.zeros(train.shape)

        # Calculate differences for columns
        ix = 0
        for data_var in self.data_variables:
            if data_var in self.directional_variables:
                ab = np.absolute(subset[:, ix] - train[:, ix])
                diff[:, ix] = (
                    np.minimum(ab, self.scale_factor.get(data_var)[1] - ab) * 2
                )
            else:
                diff[:, ix] = subset[:, ix] - train[:, ix]
            ix = ix + 1

        # Compute the squared sum of differences for each row
        dist = np.sum(diff**2, axis=1)

        return dist

    def _nearest_indices(
        self, normalized_centroids: pd.DataFrame, normalized_data: pd.DataFrame
    ):
        """
        Find the index of the nearest point in self.data for each entry in self.centroids.

        Returns:
        - ix_near: Array of indexes of the nearest point for each entry in self.centroids
        """

        # Compute distances and store nearest distance index
        nearest_indices_array = np.zeros(normalized_centroids.shape[0], dtype=int)

        for i in range(normalized_centroids.shape[0]):
            rep = np.repeat(
                np.expand_dims(normalized_centroids.values[i, :], axis=0),
                normalized_data.values.shape[0],
                axis=0,
            )
            ndist = self._normalized_distance(train=self.data_norm.values, subset=rep)

            nearest_indices_array[i] = np.nanargmin(ndist)

        return nearest_indices_array

    # TODO: Implement @validate_data
    def fit(self, data: pd.DataFrame):
        """
        Normalize data and calculate centers using maxdiss  algorithm

        Args:
        - num_centers: Number of centers to calculate

        Returns:
        - centroids: Calculated centroids
        """

        # Check if data is correctly set
        if data is None:
            raise MDAError("No data was provided.")
        elif not isinstance(data, pd.DataFrame):
            raise MDAError("Data should be a pandas DataFrame.")

        self.logger.info(
            f"\nmda parameters: {self.data.shape[0]} --> {self.num_centers}\n"
        )

        # Normalize provided data with instantiated custom_scale_factor
        normalized_data, self.scale_factor = self.normalize(
            data=data, custom_scale_factor=self.custom_scale_factor
        )

        # [DEPRECATED] Select the point with the maximum value in the first column of pandas dataframe
        # seed = normalized_data[normalized_data.columns[0]].idxmax()
        # Select the point with the maximum summed value
        seed = normalized_data.sum(axis=1).idxmax()

        # Initialize centroids subset
        subset = np.array([normalized_data.values[seed]])  # The row that starts as seed
        train = np.delete(normalized_data.values, seed, axis=0)

        # Repeat until we have the desired num_centers
        n_c = 1
        while n_c < self.num_centers:
            m2 = subset.shape[0]
            self.logger.info(
                f"   MDA centroids: {subset.shape[0]}/{self.num_centers}", end="\r"
            )
            if m2 == 1:
                xx2 = np.repeat(subset, train.shape[0], axis=0)
                d_last = self._normalized_distance(train, xx2)
            else:
                xx = np.array([subset[-1, :]])
                xx2 = np.repeat(xx, train.shape[0], axis=0)
                d_prev = self._normalized_distance(train, xx2)
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
                    ),
                    end="\r",
                )

            n_c = subset.shape[0]

        # De-normalize scalar and directional data
        normalized_centroids = pd.DataFrame(subset, columns=data.columns)
        centroids = self.denormalize(
            normalized_data=normalized_centroids, scale_factor=self.scale_factor
        )

        # TODO: use the normalized centroids and the norm_data to avoid rounding errors.
        # Calculate the real indices of the centroids
        self.centroid_real_indices = self._nearest_indices(
            normalized_centroids=normalized_centroids, normalized_data=normalized_data
        )

        return centroids

    def nearest_centroid_indices(self, data_q):
        """
        Find the index of the nearest centroid for each entry in 'data_q'

        Args
        ----
        - data_q (pandas.DataFrame):
            Query data (example: df[[5]], df[[5,6,10]])

        Returns:
        - ix_near_cent: Array of indices of the nearest centroids for each entry in data_q
        """

        # # Reshape if only one data point was selected
        if len(np.shape(data_q)) == 1:
            data_q = data_q.reshape(1, -1)

        # Normalize data point
        data_q_pd = pd.DataFrame(data_q, columns=self.data.columns)
        data_q_norm, b = self.normalize(
            data=data_q_pd,
            custom_scale_factor=self.scale_factor,
        )

        # Check centroids were calculated beforehand
        if len(self.centroids) == 0:
            raise MDAError(
                "Centroids have not been calculated, first apply .run method"
            )

        # Compute distances to centroids and store nearest distance index
        ix_near_cent = np.zeros(data_q_norm.values.shape[0], dtype=int)
        for i in range(data_q_norm.values.shape[0]):
            norm_dists_centroids = self._normalized_distance(
                self.centroids_norm.values,
                np.repeat(
                    np.expand_dims(data_q_norm.values[i, :], axis=0),
                    self.centroids_norm.values.shape[0],
                    axis=0,
                ),
            )
            ix_near_cent[i] = np.nanargmin(norm_dists_centroids)

        return ix_near_cent

    def nearest_centroid(self, data_q):
        """
        Find the nearest centroid for each entry in 'data_q'

        Args:
        - data_q: Query data (example: df[[5]], df[[5,6,10]])

        Returns:
        - nearest_cents: Nearest MDA centroids
        """

        ix_near_cents = self.nearest_centroid_indices(data_q)
        nearest_cents = self.centroids.values[ix_near_cents]

        return nearest_cents
