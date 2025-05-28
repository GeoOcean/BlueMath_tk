from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_plotting import DefaultStaticPlotting
from .colors import default_colors


def plot_scatters_in_triangle(
    dataframes: List[pd.DataFrame],
    data_colors: List[str] = default_colors,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """
    Plot a scatter plot of the dataframes with axes in a triangle.

    Parameters
    ----------
    dataframes : List[pd.DataFrame]
        List of dataframes to plot.
    data_colors : List[str], optional
        List of colors for the dataframes.
    **kwargs : dict, optional
        Keyword arguments for the scatter plot. Will be passed to the
        DefaultStaticPlotting.plot_scatter method, which is the same
        as the one in matplotlib.pyplot.scatter.
        For example, to change the marker size, you can use:
        ``plot_scatters_in_triangle(dataframes, s=10)``

    Returns
    -------
    fig : Figure
        Figure object.
    axes : Axes
        Axes object.
    """

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
