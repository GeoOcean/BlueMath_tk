import numpy as np

# scatter_data
import bluemath_tk.colors
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import itertools


def normalize(data, ix_directional, scale_factor={}):
    """
    Normalize a subset of data using min-max scaling.
    Parameters
    ----------
    data : pandas.DataFrame
        The data to be normalized. Each column represents a different variable.
        List of column names corresponding to directional variables in the data.
        Directional variables are normalized by dividing by 180. If no directional
        variables are present, this list should be empty.
    ix_directional : list of str
        List with the names of the directional variables in the data. If no
        directional variables are present, this list should be empty.
    scale_factor : dict, optional
        Dictionary to store the minimum and maximum values for each variable.
        If not provided, it will be computed from the data. The keys should be
        column names and the values should be lists of the form [min, max].
    Returns
    -------
    data_norm : pandas.DataFrame
        The normalized data. Each column is scaled to the range [0, 1] using
        min-max scaling, except for directional variables which are scaled by 180.
    scale_factor : dict
        Dictionary containing the minimum and maximum values used for scaling
        each variable. This can be used for inverse transformation or to apply
        the same scaling to new data.
    Notes
    -----
    - The normalization formula for scalar data is:
      norm = (val - min) / (max - min)
    - Directional variables are normalized by dividing by 180.
    - If a directional variable has a value greater than 360, it should be handled
      appropriately before normalization.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'Hs': np.random.rand(1000)*7,
    ...     'Tp': np.random.rand(1000)*20,
    ...     'Dir': np.random.rand(1000)*360
    ... })
    >>> ix_directional = ['Dir']
    >>> data_norm, scale_factor = normalize(data, ix_directional)
    """

    # TODO
    # ¿qué hacer cuando nos dan un valor direccional > 360 o < 0?.
    # Lo quitamos nosotros las vueltas o devolvemos un error. Lo hacemos aquí o en el código que llama a esta función?
    # Devolvemos 2 valores, uno con los datos normalizados y otro con los factores de escala. ¿Es la mejor manera de hacerlo?


    # Initialize data.
    # data_norm = data (it is a pointer an modify the original data)
    data_norm = data.copy()

    # Normalize scalar data
    for ix in data.columns:
        if ix in ix_directional:
            data_norm[ix] = data[ix] / 180.0
        else:
            v = data[ix]
            if ix not in scale_factor:
                mi = np.amin(v)
                ma = np.amax(v)
                scale_factor[ix] = [mi, ma]

            data_norm[ix] = (v - scale_factor[ix][0]) / (
                scale_factor[ix][1] - scale_factor[ix][0]
            )

    return data_norm, scale_factor


def denormalize(data_norm, ix_directional, scale_factor):

    """
    Denormalizes the normalized data.

    This function takes a DataFrame of normalized data and denormalizes it
    using provided scale factors. Directional data is denormalized by multiplying
    it by 180, while scalar data is denormalized using specific scale factors.

    Parameters
    ----------
    data_norm : pd.DataFrame
        DataFrame containing the normalized data.
    ix_directional : list
        List of column names that contain directional data.
    scale_factor : dict
        Dictionary containing the minimum and maximum values used for scaling
        each variable.
    Returns
    -------
    pd.DataFrame
        DataFrame with the denormalized data.
    """
    
    # Initialize data
    data = data_norm.copy()

    # Scalar data
    for ix in data.columns:
        if ix in ix_directional:
            data[ix] = data_norm[ix] * 180
        else:
            data[ix] = (
                data_norm[ix] * (scale_factor[ix][1] - scale_factor[ix][0])
                + scale_factor[ix][0]
            )

    return data

    # Initialize data
    data = data_norm.copy()

    # Scalar data
    for ix in data.columns:
        if ix in ix_directional:
            data[ix] = data_norm[ix] * 180
        else:
            data[ix] = (
                data_norm[ix] * (scale_factor[ix][1] - scale_factor[ix][0])
                + scale_factor[ix][0]
            )

    return data


def scatter(data, centroids=None, color_data=None, custom_params=None):
    """
    Create scatter plots for all combinations of variables in the data.

    Arguments
    ---------
    data: pandas DataFrame
        Data to be plotted.

    centroids: pandas DataFrame
        Centroids to be plotted.

    color_data: array
        Array of values to color the data points.

    custom_params: dict
        Custom parameters for scatter plots.
    """

    scatter_params = (
        {**bluemath_tk.colors.scatter_defaults, **custom_params}
        if custom_params
        else bluemath_tk.colors.scatter_defaults
    )

    # Create figure and axes
    num_variables = data.shape[1]
    fig, axes = plt.subplots(
        nrows=num_variables - 1,
        ncols=num_variables - 1,
        figsize=scatter_params["figsize"],
    )

    # Create scatter plots
    combinations = list(itertools.combinations(data.columns, 2))

    i = 0
    j = num_variables - 2

    for combination in combinations:

        # If number of variables is greater than 2, create subplots
        if num_variables > 2:
            ax = axes[i, j]
        else:
            ax = axes

        if color_data is not None:
            # Define a continuous colormap using the 'rainbow' colormap from Matplotlib
            cmap_continuous = plt.cm.rainbow
            # Create a discretized colormap by sampling the continuous colormap at evenly spaced intervals
            # The number of intervals is determined by the number of unique values in 'bmus'
            cmap_discretized = ListedColormap(
                cmap_continuous(np.linspace(0, 1, len(np.unique(color_data))))
            )

            # Plot scatter data
            im = ax.scatter(
                data[combination[0]],
                data[combination[1]],
                c=color_data,
                s=scatter_params["size_data"],
                label="data",
                cmap=cmap_discretized,
                alpha=scatter_params["alpha_subset"],
            )
            plt.colorbar(im, ticks=np.arange(0, len(np.unique(color_data))))

        else:
            ax.scatter(
                data[combination[0]],
                data[combination[1]],
                s=scatter_params["size_data"],
                c=scatter_params["color_data"],
                alpha=scatter_params["alpha_data"],
                label="Data",
            )

        if centroids is not None:
            if color_data is not None:
                # Add centroids to the plot
                ax.scatter(
                    centroids[combination[0]],
                    centroids[combination[1]],
                    s=scatter_params["size_centroid"],
                    c=np.array(range(len(np.unique(color_data)))) + 1,
                    cmap=cmap_discretized,
                    ec="k",
                    label="Centroids",
                )
            else:
                ax.scatter(
                    centroids[combination[0]],
                    centroids[combination[1]],
                    s=scatter_params["size_centroid"],
                    c=scatter_params["color_data"],
                    ec="k",
                    label="Centroids",
                )

        ax.set_xlabel(combination[0], fontsize=scatter_params["fontsize"])
        ax.set_ylabel(combination[1], fontsize=scatter_params["fontsize"])
        ax.legend(fontsize=scatter_params["size_data"])
        ax.tick_params(axis="both", labelsize=scatter_params["fontsize"])

        # Update i and j for subplots
        if j > i:
            j = j - 1
        else:
            # Remove axis for empty subplots
            if j > 0:
                for empty in range(0, j):
                    ax = axes[i, empty]
                    ax.axis("off")
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

            i += 1
            j = num_variables - 2

    plt.tight_layout()
    plt.show()


# def scatter_subset(self, norm=False, custom_params=None):

#     self.scatter_data(
#         norm=norm,
#         plot_centroids=True,
#         custom_params=custom_params,
#     )
#     plt.show()
