import pandas as pd
from ..core.models import BlueMathModel
from ..plotting.base_plotting import DefaultStaticPlotting


class BaseDataMining(BlueMathModel):
    def __init__(self):
        super().__init__()

    def plot_selected_data(
        self,
        data: pd.DataFrame,
        centroids: pd.DataFrame = None,
        data_color: str = "blue",
        centroids_color: str = "red",
        **kwargs,
    ):
        """
        Create scatter plots for all combinations of variables in the data.
        """

        if list(data.columns) == list(centroids.columns):
            variables_names = list(data.columns)
            num_variables = len(variables_names)
        else:
            raise ValueError("Data and centroids must have the same number of columns.")

        # Create figure and axes
        default_static_plot = DefaultStaticPlotting()
        fig, axes = default_static_plot.get_subplots(
            nrows=num_variables - 1,
            ncols=num_variables - 1,
            sharex=False,
            sharey=False,
        )

        for c1, v1 in enumerate(variables_names[1:]):
            for c2, v2 in enumerate(variables_names[:-1]):
                default_static_plot.plot_scatter(
                    ax=axes[c2, c1],
                    x=data[v1],
                    y=data[v2],
                    c=data_color,
                    s=kwargs.get("s", default_static_plot.default_scatter_size),
                    alpha=kwargs.get("alpha", 0.7),
                )
                # Plot centroids in selected ax if passed
                if centroids is not None:
                    default_static_plot.plot_scatter(
                        ax=axes[c2, c1],
                        x=centroids[v1],
                        y=centroids[v2],
                        c=centroids_color,
                        s=kwargs.get("s", default_static_plot.default_scatter_size),
                        alpha=kwargs.get("alpha", 0.9),
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
