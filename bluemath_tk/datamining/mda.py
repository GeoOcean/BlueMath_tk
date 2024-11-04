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
from bluemath_tk.core.data import normalize, denormalize, scatter


class MDA:
    """
    This class implements the MDA algorithm (Maximum Dissimilarity Algorithm)

    ...

    Attributes
    ----------
    data : pandas.core.frame.DataFrame
        The data to be clustered. Each column will represent a different variable

    ix_directional : list of str
        List with the names of the directional variables in the data. If no directional variables are present, this list should be empty.

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

    def __init__(self, data=None, ix_directional=[]):
        self.data = data
        self.ix_directional = ix_directional
        self.scale_factor = {}
        self.data_norm = []
        self.centroids_norm = []
        self.centroids = []
        self.centroid_iterative_indices = []
        self.centroid_real_indices = []

    def run(self, num_centers):
        """
        Normalize data and calculate centers using
        maxdiss  algorithm

        Args:
        - num_centers: Number of centers to calculate

        Returns:
        - centroids: Calculated centroids
        """

        # Check if data is correctly set
        if self.data is None:
            raise MDAError("No data was provided.")
        elif type(self.data) is not pd.DataFrame:
            raise MDAError("Data should be a pandas DataFrame.")

        print("\nmda parameters: {0} --> {1}\n".format(self.data.shape[0], num_centers))

        self.data_norm, self.scale_factor = normalize(self.data, self.ix_directional)

        # TODO: Improve the calculation of the centroids
        # mda seed. Select the point with the maximum value in the first column of pandas dataframe
        seed = self.data_norm[self.data_norm.columns[0]].idxmax()

        # Initialize centroids subset
        subset = np.array([self.data_norm.values[seed]])
        train = np.delete(self.data_norm.values, seed, axis=0)

        # In the future, we could use panda dataframes instead of numpy arrays.
        # seed= self.data_norm[self.data_norm.columns[0]].idxmax()
        # # Create a pandas dataarray containing the seed point
        # subset = pd.DataFrame(self.data_norm.loc[seed]).T
        # # Get the data without the seed
        # train = self.data_norm.drop(seed, axis=0)

        # Repeat until we have the desired num_centers
        n_c = 1
        while n_c < num_centers:

            m2 = subset.shape[0]
            print(
                f"   MDA centroids: {subset.shape[0]:04d}/{num_centers:04d}", end="\r"
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
                fmt = "0{0}d".format(len(str(num_centers)))
                print(
                    "   MDA centroids: {1:{0}}/{2:{0}}".format(
                        fmt, subset.shape[0], num_centers
                    ),
                    end="\r",
                )

            n_c = subset.shape[0]

        # De-normalize scalar and directional data
        self.centroids_norm = pd.DataFrame(subset, columns=self.data.columns)
        self.centroids = denormalize(
            self.centroids_norm, self.ix_directional, self.scale_factor
        )

        # TODO: use the normalized centroids and the norm_data to avoid rounding errors.
        # Calculate the real indices of the centroids
        self.centroid_real_indices = self._nearest_indices()

    def _normalized_distance(self, M, D):
        """
        Compute the normalized distance between rows in M and D.

        Args
        ----
        M : numpy.ndarray
            Train matrix
        D : numpy.ndarray
            Subset matrix

        Returns
        -------
        dist: numpy.ndarray
            normalized distances
        """

        dif = np.zeros(M.shape)

        # Calculate differences for columns
        ix = 0
        for column in self.data.columns:
            # First column
            if column in self.ix_directional:
                ab = np.absolute(D[:, ix] - M[:, ix])
                dif[:, ix] = np.minimum(ab, 2 * np.pi - ab) / np.pi
            else:
                dif[:, ix] = D[:, ix] - M[:, ix]
            ix = ix + 1

        # Compute the squared sum of differences for each row
        dist = np.sum(dif**2, axis=1)

        return dist

    def _nearest_indices(self):
        """
        Find the index of the nearest point in self.data for each entry in self.centroids.

        Returns:
        - ix_near: Array of indexes of the nearest point for each entry in self.centroids
        """

        # Compute distances and store nearest distance index
        nearest_indices_array = np.zeros(self.centroids_norm.shape[0], dtype=int)
        for i in range(self.centroids_norm.shape[0]):
            rep = np.repeat(
                np.expand_dims(self.centroids_norm.values[i, :], axis=0),
                self.data_norm.values.shape[0],
                axis=0,
            )
            ndist = self._normalized_distance(self.data_norm.values, rep)

            nearest_indices_array[i] = np.nanargmin(ndist)

        return nearest_indices_array

    def nearest_centroid_indices(self, data_q):
        """
        Find the index of the nearest centroid for each entry in 'data_q'

        Args
        ----
        - data_q (pandas.core.frame.DataFrame):
            Query data (example: df[[5]], df[[5,6,10]])

        Returns:
        - ix_near_cent: Array of indices of the nearest centroids for each entry in data_q
        """

        # # Reshape if only one data point was selected
        if len(np.shape(data_q)) == 1:
            data_q = data_q.reshape(1, -1)

        # Normalize data point
        data_q_pd = pd.DataFrame(data_q, columns=self.data.columns)
        data_q_norm, b = normalize(
            data_q_pd,
            ix_directional=self.ix_directional,
            scale_factor=self.scale_factor,
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

    def scatter_data(self, norm=False, plot_centroids=False, custom_params=None):
        """
        Plot the data and/or the centroids.

        Parameters
        ----------
        norm : bool
            If True, the normalized data will be plotted. Default is False.

        plot_centroids : bool
            If True, the centroids will be plotted. Default is False.

        custom_params : dict
            Custom parameters for the scatter plot. Default is None.
        """

        if norm == True:
            data = self.data_norm
            centroids = self.centroids_norm
        else:
            data = self.data
            centroids = self.centroids

        if plot_centroids:
            scatter(data, centroids=centroids, custom_params=custom_params)
        else:
            scatter(data, custom_params=custom_params)


class MDAError(Exception):
    """Custom exception for MDA class."""

    def __init__(self, message="MDA error occurred."):
        self.message = message
        super().__init__(self.message)
