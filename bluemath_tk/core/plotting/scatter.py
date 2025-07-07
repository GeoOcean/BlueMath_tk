from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde, probplot
from sklearn.metrics import mean_squared_error

from .base_plotting import DefaultStaticPlotting
from .colors import default_colors


def rmse(pred: np.ndarray, tar: np.ndarray) -> float:
    """
    Calculate the Root Mean Square Error between predicted and target values.

    Parameters
    ----------
    pred : np.ndarray
        Array of predicted values.
    tar : np.ndarray
        Array of target/actual values.

    Returns
    -------
    float
        The Root Mean Square Error value.
    """

    if len(pred) != len(tar):
        raise ValueError("pred and tar must have the same length")

    return np.sqrt(((pred - tar) ** 2).mean())


def bias(pred: np.ndarray, tar: np.ndarray) -> float:
    """
    Calculate the bias between predicted and target values.

    Parameters
    ----------
    pred : np.ndarray
        Array of predicted values.
    tar : np.ndarray
        Array of target/actual values.

    Returns
    -------
    float
        The bias value (mean difference between predictions and targets).
    """

    if len(pred) != len(tar):
        raise ValueError("pred and tar must have the same length")

    return sum(pred - tar) / len(pred)


def si(pred: np.ndarray, tar: np.ndarray) -> float:
    """
    Calculate the Scatter Index between predicted and target values.

    Parameters
    ----------
    pred : np.ndarray
        Array of predicted values.
    tar : np.ndarray
        Array of target/actual values.

    Returns
    -------
    float
        The Scatter Index value.
    """

    if len(pred) != len(tar):
        raise ValueError("pred and tar must have the same length")

    pred_mean = pred.mean()
    tar_mean = tar.mean()

    return np.sqrt(sum(((pred - pred_mean) - (tar - tar_mean)) ** 2) / (sum(tar**2)))


def density_scatter(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a density scatter for two arrays using gaussian KDE.

    Parameters
    ----------
    x : np.ndarray
        X values for the scatter plot.
    y : np.ndarray
        Y values for the scatter plot.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - Sorted x values
        - Sorted y values
        - Density values corresponding to each point
    """

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x1, y1, z = x[idx], y[idx], z[idx]

    return x1, y1, z


def validation_scatter(
    axs: Axes,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    cmap: str = "rainbow",
) -> None:
    """
    Plot a density scatter and Q-Q plot for validation.

    Parameters
    ----------
    axs : Axes
        Matplotlib axes to plot on.
    x : np.ndarray
        X values for the scatter plot.
    y : np.ndarray
        Y values for the scatter plot.
    xlabel : str
        Label for the X-axis.
    ylabel : str
        Label for the Y-axis.
    title : str
        Title for the plot.
    cmap : str, optional
        Colormap to use for the scatter plot. Default is "rainbow".
    """

    x2, y2, z = density_scatter(x, y)

    # plot
    axs.scatter(x2, y2, c=z, s=5, cmap=cmap)

    # labels
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_title(title)

    # axis limits
    maxt = np.ceil(max(max(x) + 0.1, max(y) + 0.1))
    axs.set_xlim(0, maxt)
    axs.set_ylim(0, maxt)
    axs.plot([0, maxt], [0, maxt], "-r")
    axs.set_xticks(np.linspace(0, maxt, 5))
    axs.set_yticks(np.linspace(0, maxt, 5))
    axs.set_aspect("equal")

    # qq-plot
    xq = probplot(x, dist="norm")
    yq = probplot(y, dist="norm")
    axs.plot(xq[0][1], yq[0][1], "o", markersize=0.5, color="k", label="Q-Q plot")

    # diagnostic errors
    props = dict(
        boxstyle="round", facecolor="w", edgecolor="grey", linewidth=0.8, alpha=0.5
    )
    mse = mean_squared_error(x2, y2)
    rmse_e = rmse(x2, y2)
    BIAS = bias(x2, y2)
    SI = si(x2, y2)
    label = "\n".join(
        (
            r"RMSE = %.2f" % (rmse_e,),
            r"MSE =  %.2f" % (mse,),
            r"BIAS = %.2f" % (BIAS,),
            R"SI = %.2f" % (SI,),
        )
    )
    axs.text(
        0.05,
        0.95,
        label,
        transform=axs.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )


def plot_scatters_in_triangle(
    dataframes: List[pd.DataFrame],
    data_colors: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[Figure, np.ndarray]:
    """
    Plot scatter plots of the dataframes with axes in a triangle arrangement.

    Parameters
    ----------
    dataframes : List[pd.DataFrame]
        List of dataframes to plot. Each dataframe should contain the same columns.
    data_colors : Optional[List[str]], optional
        List of colors for the dataframes. If None, uses default_colors.
    **kwargs : dict, optional
        Additional keyword arguments for the scatter plot. These will be passed to
        matplotlib.pyplot.scatter. Common parameters include:
        - s : float, marker size
        - alpha : float, transparency
        - marker : str, marker style

    Returns
    -------
    Tuple[Figure, np.ndarray]
        A tuple containing:
        - Figure object
        - 2D array of Axes objects

    Raises
    ------
    ValueError
        If the variables in the first dataframe are not present in all other dataframes.
    """

    if data_colors is None:
        data_colors = default_colors

    # Get the number and names of variables from the first dataframe
    variables_names = list(dataframes[0].columns)
    num_variables = len(variables_names)

    # Check variables names are in all dataframes
    for df in dataframes:
        if not all(v in df.columns for v in variables_names):
            raise ValueError(
                f"Variables {variables_names} are not in dataframe {df.columns}."
            )

    # Create figure and axes
    default_static_plot = DefaultStaticPlotting()
    fig, axes = default_static_plot.get_subplots(
        nrows=num_variables - 1,
        ncols=num_variables - 1,
        sharex=False,
        sharey=False,
    )
    if isinstance(axes, Axes):
        axes = np.array([[axes]])

    for c1, v1 in enumerate(variables_names[1:]):
        for c2, v2 in enumerate(variables_names[:-1]):
            for idf, df in enumerate(dataframes):
                default_static_plot.plot_scatter(
                    ax=axes[c2, c1],
                    x=df[v1],
                    y=df[v2],
                    c=data_colors[idf],
                    alpha=0.6,
                    **kwargs,
                )
            if c1 == c2:
                axes[c2, c1].set_xlabel(variables_names[c1 + 1])
                axes[c2, c1].set_ylabel(variables_names[c2])
            elif c1 > c2:
                axes[c2, c1].xaxis.set_ticklabels([])
                axes[c2, c1].yaxis.set_ticklabels([])
            else:
                fig.delaxes(axes[c2, c1])

    return fig, axes
