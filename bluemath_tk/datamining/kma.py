"""

- Project: Bluemath{toolkit}.datamining
- File: kma.py
- Description: KMeans algorithm
- Author: GeoOcean Research Group, Universidad de Cantabria
- Created Date: 23 January 2024
- License: MIT
- Repository: https://gitlab.com/geoocean/bluemath/toolkit/

"""

# 1. Format Document (black). Documentar
# Preferences: Open User Setting JSON
# "python.formatting.provider": "black"


# Laura
# 3. Laura comprobar que el código funciona y las gráficas están bien


# 4. Mejorar control de errores. Poner alguno más
# 5. Cómo guardar los objetos de la clase y como pasarselos a las funciones auxiliares como scatter_data



# PREDICTIA:
# - Autopep8, longitud de línea 120?
# - Documentar con qué
# - Cómo se guarda el objeto de la clase
# - Logging
# - Mejorar control de errores. Poner alguno más
# - Cómo guardar los objetos de la clase y como pasarselos a las funciones auxiliares como scatter_data


import pandas as pd
from bluemath_tk.core.data import normalize, denormalize, scatter

# Kmeans algorithm
from sklearn.cluster import KMeans


class KMA:
    """
    This class implements the KMeans Algorithm (KMA)
    KMA is a clustering algorithm that divides a dataset into K distinct groups based on data similarity. It iteratively assigns data points to the nearest cluster center and updates centroids until convergence, aiming to minimize the sum of squared distances. While efficient and widely used, KMA requires specifying the number of clusters and is sensitive to initial centroid selection.

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
        Normalize data and calculate centers using kmeans algorithm
    
    scatter_data()
        Plot the data and/or the centroids

    Examples
    --------
    df = pd.DataFrame({
        'Hs': np.random.rand(1000)*7,
        'Tp': np.random.rand(1000)*20,
        'Dir': np.random.rand(1000)*360
    })

    kma_ob = KMA(data=df, ix_directional=['Dir'])
    kma_ob.run(10)
    kma_ob.scatter_data()
    """

    def __init__(self, data=None, ix_directional=[]):
        self.data = data
        self.ix_directional = ix_directional
        self.scale_factor = {}
        self.data_norm = []
        self.centroids_norm = []
        self.centroids = []

    def run(self, n_clusters):
        """
        Normalize data and calculate centers using k-means algorithm.

        Parameters 
        ---------- 
        n_clusters: int
            Number of clusters to be calculated.

        Returns
        -------
        pandas.core.frame.DataFrame
            Calculated centroids.
        """

        # Check if data is correctly set
        if self.data is None:
            raise KMAError("No data was provided.")
        elif type(self.data) is not pd.DataFrame:
            raise KMAError("Data should be a pandas DataFrame.")
            

        print("\nkma parameters: {0} --> {1}\n".format(self.data.shape[0], n_clusters))

        # if not np.shape(self.data)[1] == len(self.ix_scalar) + len(self.ix_directional):
        #     raise KMAError(
        #         "ix_scalar and ix_directional should match with the number of data columns"
        #     )

        self.data_norm, self.scale_factor = normalize(self.data, self.ix_directional)

        kma = KMeans(n_clusters=n_clusters, n_init=100).fit(self.data_norm)

        # De-normalize scalar and directional data

        self.bmus = kma.labels_
        self.centroids_norm = pd.DataFrame(
            kma.cluster_centers_, columns=self.data_norm.keys()
        )

        self.centroids = denormalize(
            self.centroids_norm, self.ix_directional, self.scale_factor
        )
        return self.centroids

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
            data=self.data_norm
            centroids=self.centroids_norm
        else:
            data=self.data
            centroids=self.centroids_norm
        
        if plot_centroids:    
            scatter(
                data, centroids=centroids, custom_params=custom_params
            )
        else:
            scatter(
                data, custom_params=custom_params
            )

    def scatter_bmus(self, norm=False, plot_centroids=False, custom_params=None):
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
            data=self.data_norm
            centroids=self.centroids_norm
        else:
            data=self.data
            centroids=self.centroids
        
        if plot_centroids:    
            scatter(
                data, centroids=centroids, color_data=self.bmus, custom_params=custom_params
            )
        else:
            scatter(
                data, color_data=self.bmus, custom_params=custom_params
            )


class KMAError(Exception):
    """Custom exception for KMA class."""

    def __init__(self, message="KMA error occurred."):
        self.message = message
        super().__init__(self.message)
