import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.collections import Collection

from ..core.dask import setup_dask_client
from ..core.decorators import validate_data_xwt
from ..core.models import BlueMathModel
from ..core.pipeline import BlueMathPipeline
from ..core.plotting.colors import get_cluster_colors, get_config_variables
from ..datamining.kma import KMA
from ..datamining.pca import PCA

warnings.filterwarnings("ignore")
config_variables = get_config_variables()


def get_dynamic_estela_predictor(
    data: xr.Dataset,
    estela: xr.Dataset,
    check_interpolation: bool = True,
    verbose: bool = False,
) -> xr.Dataset:
    """
    Transform an xarray dataset of longitude, latitude, and time into one where
    each longitude, latitude value at each time is replaced by the corresponding
    time - t, where t is specified in the estela dataset.

    Parameters
    ----------
    data : xr.Dataset
        The input dataset with dimensions longitude, latitude, and time.
    estela : xr.Dataset
        The dataset containing the F values with dimensions longitude and latitude.
    check_interpolation : bool, optional
        Whether to check if the data is interpolated. Default is True.
    verbose : bool, optional
        Whether to print verbose output. Default is False.
        If False, Dask logs are suppressed.
        If True, Dask logs are shown.

    Returns
    -------
    xr.Dataset
        The transformed dataset.
    """

    if not verbose:
        # Suppress Dask logs
        logging.getLogger("distributed").setLevel(logging.ERROR)
        logging.getLogger("distributed.client").setLevel(logging.ERROR)
        logging.getLogger("distributed.scheduler").setLevel(logging.ERROR)
        logging.getLogger("distributed.worker").setLevel(logging.ERROR)
        logging.getLogger("distributed.nanny").setLevel(logging.ERROR)
        # Also suppress bokeh and tornado logs that Dask uses
        logging.getLogger("bokeh").setLevel(logging.ERROR)
        logging.getLogger("tornado").setLevel(logging.ERROR)

    # TODO: Add customization for dask client
    _dask_client = setup_dask_client(n_workers=4, memory_limit=0.25)

    if check_interpolation:
        if (
            "longitude" not in data.dims
            or "latitude" not in data.dims
            or "time" not in data.dims
        ):
            raise ValueError("Data must have longitude, latitude, and time dimensions.")
        if "longitude" not in estela.dims or "latitude" not in estela.dims:
            raise ValueError("Estela must have longitude and latitude dimensions.")
        estela = estela.interp_like(data)  # TODO: Check NaNs interpolation

    data = data.chunk({"time": 365}).where(estela.F >= 0.0, np.nan)
    estela_traveltimes = estela.where(estela.F >= 0, np.nan).traveltime.astype(int)
    estela_max_traveltime = estela_traveltimes.max().values

    for traveltime in range(estela_max_traveltime):
        data = data.where(estela_traveltimes != traveltime, data.shift(time=traveltime))

    return data.compute()


def check_model_is_fitted(func):
    """
    Decorator to check if the model is fitted.
    """

    def wrapper(self, *args, **kwargs):
        if self.kma_bmus is None:
            raise XWTError("Fit the model before calling this property.")
        return func(self, *args, **kwargs)

    return wrapper


class XWTError(Exception):
    """Custom exception for XWT class."""

    def __init__(self, message="XWT error occurred."):
        self.message = message
        super().__init__(self.message)


class XWT(BlueMathModel, BlueMathPipeline):
    """
    Xly Weather Types (XWT) class.

    This class implements the XWT method to identify and classify weather patterns
    in a dataset. The XWT method is a combination of Principal Component Analysis (PCA)
    and K-means clustering (KMA).

    Attributes
    ----------
    steps : Dict[str, BlueMathModel]
        The steps of the XWT method.
    num_clusters : int
        The number of clusters.
    kma_bmus : pd.DataFrame
        The KMA best matching units (BMUs).
    """

    def __init__(self, steps: Dict[str, BlueMathModel]) -> None:
        """
        Initialize the XWT.

        Parameters
        ----------
        steps : Dict[str, BlueMathModel]
            The steps of the XWT method. The steps must include a PCA and a KMA model.
        """

        super().__init__()
        self.set_logger_name(name=self.__class__.__name__, level="INFO")

        # Save XWT attributes
        if steps:
            if (
                not all(isinstance(step, BlueMathModel) for step in steps.values())
                or "pca" not in steps
                or "kma" not in steps
            ):
                raise XWTError("The steps must include a PCA and a KMA model.")
        self.steps = steps
        self._data: xr.Dataset = None
        self.num_clusters: int = None
        self.kma_bmus: pd.DataFrame = None

        # Exclude attributes from being saved
        self._exclude_attributes = ["_data"]

    @property
    def data(self) -> xr.Dataset:
        return self._data

    @property
    @check_model_is_fitted
    def clusters_probs_df(self) -> pd.DataFrame:
        """
        Calculate the probabilities for each XWT.
        """

        # Calculate probabilities for each cluster
        clusters_probs = (
            self.kma_bmus["kma_bmus"].value_counts(normalize=True).sort_index()
        )

        return clusters_probs

    @property
    @check_model_is_fitted
    def clusters_monthly_probs_df(self) -> pd.DataFrame:
        """
        Calculate the monthly probabilities for each XWT.
        """

        # Calculate probabilities for each month
        monthly_probs = (
            self.kma_bmus.groupby(self.kma_bmus.index.month)["kma_bmus"]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
        )

        return monthly_probs

    @property
    @check_model_is_fitted
    def clusters_seasonal_probs_df(self) -> pd.DataFrame:
        """
        Calculate the seasonal probabilities for each XWT.
        """

        # Calculate probabilities for each season
        # Define seasons: DJF (Dec, Jan, Feb), MAM (Mar, Apr, May),
        # JJA (Jun, Jul, Aug), SON (Sep, Oct, Nov)
        seasons = {
            "DJF": [12, 1, 2],
            "MAM": [3, 4, 5],
            "JJA": [6, 7, 8],
            "SON": [9, 10, 11],
        }
        # Add a 'season' column to the DataFrame
        kma_bmus_season = self.kma_bmus.copy()
        kma_bmus_season["season"] = kma_bmus_season.index.month.map(
            lambda x: next(season for season, months in seasons.items() if x in months)
        )

        # Calculate probabilities for each season
        seasonal_probs = (
            kma_bmus_season.groupby("season")["kma_bmus"]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
        )

        return seasonal_probs

    @property
    @check_model_is_fitted
    def clusters_annual_probs_df(self) -> pd.DataFrame:
        """
        Calculate the annual probabilities for each XWT.
        """

        # Calculate probabilities for each year
        annual_probs = (
            self.kma_bmus.groupby(self.kma_bmus.index.year)["kma_bmus"]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
        )

        return annual_probs

    @property
    @check_model_is_fitted
    def clusters_perpetual_year_probs_df(self) -> pd.DataFrame:
        """
        Calculate the perpetual year probabilities for each XWT.
        """

        # Calculate probabilities for each natural day in the year
        natural_day_probs = (
            self.kma_bmus.groupby(self.kma_bmus.index.dayofyear)["kma_bmus"]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
        )
        # Set index to be the datetime first day of month
        natural_day_probs.index = [
            datetime(2000, 1, 1) + timedelta(days=i - 1)
            for i in natural_day_probs.index
        ]

        return natural_day_probs

    @property
    @check_model_is_fitted
    def get_conditioned_probabilities(self) -> pd.DataFrame:
        """
        Calculate conditional probabilities P(X_t = j | X_{t-lag} = i)
        """

        # Convert to numpy array if not already
        data = self.kma_bmus.values.flatten()

        # Find unique values in the data
        unique_values = np.unique(data)

        # Create empty matrix for conditional probabilities
        cond_probs = np.zeros((self.num_clusters, self.num_clusters))

        # Count transitions
        for i in range(len(data) - 1):
            prev_idx = np.where(unique_values == data[i])[0][0]
            next_idx = np.where(unique_values == data[i + 1])[0][0]
            cond_probs[prev_idx, next_idx] += 1

        # Normalize to get probabilities
        row_sums = cond_probs.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        cond_probs = cond_probs / row_sums

        # Create DataFrame with labels
        df_cond_probs = pd.DataFrame(
            cond_probs,
            index=[f"Cluster {v}" for v in unique_values],
            columns=[f"Cluster {v}" for v in unique_values],
        )

        return df_cond_probs

    @validate_data_xwt
    def fit(
        self,
        data: xr.Dataset,
        fit_params: Dict[str, Dict[str, Any]] = {},
        variable_to_sort_bmus: str = None,
    ) -> None:
        """
        Fit the XWT model to the data.

        Parameters
        ----------
        data : xr.Dataset
            The data to fit the model to. Must be PCA formatted.
        fit_params : Dict[str, Dict[str, Any]], optional
            The fitting parameters for the PCA and KMA models. Default is {}.
        variable_to_sort_bmus : str, optional
            The variable to sort the BMUs. Default is None.

        Raises
        ------
        XWTError
            If the data is not PCA formatted.

        TODO: Standarize PCs by first PC variance.
              pca.pcs_df / pca.pcs.stds.isel(n_component=0).values ??
        """

        # Make a copy of the data to avoid modifying the original dataset
        self._data = data.copy()

        pca: PCA = self.steps.get("pca")
        if pca.pcs is None:
            try:
                _pcs_ds = pca.fit_transform(
                    data=data,
                    **fit_params.get("pca", {}),
                )
            except Exception as e:
                raise XWTError(f"Error during PCA fitting: {e}")
        else:
            self.logger.info("PCA already fitted, skipping PCA fitting.")

        kma: KMA = self.steps.get("kma")
        self.num_clusters = kma.num_clusters

        data_to_kma = pca.pcs_df.copy()

        if "regression_guided" in fit_params.get("kma", {}):
            guiding_vars = fit_params["kma"]["regression_guided"].get("vars", [])

            if guiding_vars:
                guiding_data = pd.DataFrame(
                    {var: data[var].values for var in guiding_vars},
                    index=data.time.values,
                )
                data_to_kma = pd.concat([data_to_kma, guiding_data], axis=1)

        kma_bmus, _kma_bmus_df = kma.fit_predict(
            data=data_to_kma,
            **fit_params.get("kma", {}),
        )
        self.kma_bmus = kma_bmus + 1  # TODO: Check if this is necessary!!!

        # Re-sort kma clusters based on variable if specified
        if variable_to_sort_bmus:
            pca.pcs["kma_bmus"] = (("time"), self.kma_bmus["kma_bmus"].values)
            sorted_bmus = (
                pca.inverse_transform(
                    PCs=pca.pcs.groupby("kma_bmus")
                    .mean()
                    .rename({"kma_bmus": pca.pca_dim_for_rows})
                )
                .mean(dim=pca.coords_to_stack)
                .sortby(variable_to_sort_bmus)[f"{pca.pca_dim_for_rows}"]
                .values
            )
            sorted_bmus_mapping = dict(
                zip(sorted_bmus, range(1, self.num_clusters + 1))
            )
            self.kma_bmus.replace(sorted_bmus_mapping, inplace=True)

        # Add the KMA bmus to the PCs and data
        pca.pcs["kma_bmus"] = (("time"), self.kma_bmus["kma_bmus"].values)
        self.data["kma_bmus"] = (("time"), self.kma_bmus["kma_bmus"].values)

    def plot_map_features(
        self, ax: Axes, land_color: str = cfeature.COLORS["land"]
    ) -> None:
        """
        Plot map features on an axis.

        Parameters
        ----------
        ax : Axes
            The axis to plot the map features on.
        land_color : str, optional
            The color of the land. Default is cfeature.COLORS["land"].
        """

        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAND, edgecolor="black", color=land_color)
        ax.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.5)

    def plot_xwts(
        self, var_to_plot: str, anomaly: bool = False, map_center: tuple = None
    ) -> Collection:
        """
        Plot the XWTs for a variable.

        Parameters
        ----------
        var_to_plot : str
            The variable to plot.
        anomaly : bool, optional
            Whether to plot the anomaly of the variable. Default is False.
        map_center : tuple, optional
            The center of the map. Default is None.

        Returns
        -------
        GridSpec
            The grid specification with the XWTs plot.
        """

        if anomaly:
            data_to_plot = self.data.groupby("kma_bmus").mean()[
                var_to_plot
            ] - self.data[var_to_plot].mean("time")
        else:
            data_to_plot = self.data.groupby("kma_bmus").mean()[var_to_plot]

        if self.num_clusters > 3:
            col_wrap = int(np.ceil(np.sqrt(self.num_clusters)))
        else:
            col_wrap = self.num_clusters

        # Get the configuration for the variable to plot if it exists
        var_to_plot_config = config_variables.get(var_to_plot, {})
        # Get the cluster colors for each XWT
        xwts_colors = get_cluster_colors(num_clusters=self.num_clusters)

        # Create figure with enough space at bottom for colorbar
        fig = plt.figure(figsize=(15, 16))
        gs = gridspec.GridSpec(
            col_wrap,
            col_wrap,
            wspace=0.05,
            hspace=0.05,
        )

        # Plot the XWTs for the variable
        vmin = var_to_plot_config.get("vmin", data_to_plot.min().values)
        vmax = var_to_plot_config.get("vmax", data_to_plot.max().values)
        for i, (bmus, xwt_color) in enumerate(
            zip(data_to_plot.kma_bmus.values, xwts_colors)
        ):
            row = i // col_wrap
            col = i % col_wrap
            if map_center:
                ax = fig.add_subplot(
                    gs[row, col], projection=ccrs.Orthographic(*map_center)
                )
                p = data_to_plot.sel(kma_bmus=bmus).plot(
                    ax=ax,
                    cmap=var_to_plot_config.get("cmap", "RdBu_r"),
                    add_colorbar=False,
                    transform=ccrs.PlateCarree(),
                    vmin=vmin,
                    vmax=vmax,
                )
                self.plot_map_features(ax=ax, land_color=xwt_color)
            else:
                ax = fig.add_subplot(gs[row, col])
                p = data_to_plot.sel(kma_bmus=bmus).plot(
                    ax=ax,
                    cmap=var_to_plot_config.get("cmap", "RdBu_r"),
                    add_colorbar=False,
                    vmin=vmin,
                    vmax=vmax,
                )
                for border in ["top", "bottom", "left", "right"]:
                    ax.spines[border].set_color(xwt_color)
            ax.set_title("")
            ax.text(
                0.05,
                0.05,
                int(bmus),
                ha="left",
                va="bottom",
                fontsize=15,
                fontweight="bold",
                color="navy",
                transform=ax.transAxes,
            )

        # Add colorbar in little custom axes at the bottom
        cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02])
        _cb = fig.colorbar(
            p,
            cax=cbar_ax,
            orientation="horizontal",
            label=var_to_plot_config.get("label", var_to_plot),
        )

        return p

    def _axplot_wt_probs(
        self,
        ax: Axes,
        wt_probs: np.ndarray,
        ttl: str = "",
        vmin: float = 0.0,
        vmax: float = 0.1,
        cmap: str = "Blues",
        caxis: str = "black",
        plot_text: bool = False,
    ) -> Collection:
        """
        Axes plot WT cluster probabilities.

        Parameters
        ----------
        ax : Axes
            The axis to plot the WT cluster probabilities on.
        wt_probs : np.ndarray
            The WT cluster probabilities.
        ttl : str, optional
            The title of the plot. Default is "".
        vmin : float, optional
            The minimum value of the colorbar. Default is 0.0.
        vmax : float, optional
            The maximum value of the colorbar. Default is 0.1.
        cmap : str, optional
            The colormap to use. Default is "Blues".
        caxis : str, optional
            The color of the axis. Default is "black".
        plot_text : bool, optional
            Whether to plot the text in each cell. Default is False.
        """

        # cluster transition plot
        pc = ax.pcolor(
            np.flipud(wt_probs),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors="k",
        )
        # plot text in each cell
        if plot_text:
            for i in range(wt_probs.shape[0]):
                for j in range(wt_probs.shape[1]):
                    ax.text(
                        j + 0.5,
                        wt_probs.shape[0] - 0.5 - i,
                        f"{wt_probs[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        fontweight="bold",
                        color="black",
                    )

        # customize axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(ttl, {"fontsize": 10, "fontweight": "bold"})

        # axis color
        plt.setp(ax.spines.values(), color=caxis)
        plt.setp(
            [ax.get_xticklines(), ax.get_yticklines()],
            color=caxis,
        )

        # axis linewidth
        if caxis != "black":
            plt.setp(ax.spines.values(), linewidth=3)

        return pc

    def _axplot_wt_hist(self, ax: Axes, ttl: str = "") -> Axes:
        """
        Axes plot WT cluster count histogram.

        Parameters
        ----------
        ax : Axes
            The axis to plot the WT cluster count histogram on.
        ttl : str, optional
            The title of the plot. Default is "".

        Returns
        -------
        Axes
            The axis with the WT cluster count histogram.
        """

        # cluster transition plot
        ax.hist(
            self.kma_bmus.values.reshape(-1),
            bins=np.arange(1, self.num_clusters + 2),
            edgecolor="k",
        )

        # customize axes
        # ax.grid('y')

        ax.set_xticks(np.arange(1, self.num_clusters + 1) + 0.5)
        ax.set_xticklabels(np.arange(1, self.num_clusters + 1))
        ax.set_xlim([1, self.num_clusters + 1])
        ax.tick_params(axis="both", which="major", labelsize=6)
        ax.set_title(ttl, {"fontsize": 10, "fontweight": "bold"})

        return ax

    def plot_dwts_probs(
        self,
        vmax: float = 0.15,
        vmax_seasonality: float = 0.15,
        plot_text: bool = False,
    ) -> None:
        """
        Plot Daily Weather Types bmus probabilities.

        Parameters
        ----------
        vmax : float, optional
            The maximum value of the colorbar. Default is 0.15.
        vmax_seasonality : float, optional
            The maximum value of the colorbar for seasonality. Default is 0.15.
        plot_text : bool, optional
            Whether to plot the text in each cell. Default is False.

        Raises
        ------
        ValueError
            If the kma_bmus time sampling is not daily.
        """

        if (self.kma_bmus.index[-1] - self.kma_bmus.index[-2]) != timedelta(days=1):
            raise ValueError("The kma_bmus time sampling must be daily.")

        # Best rows cols combination
        if self.num_clusters > 3:
            n_rows = n_cols = int(np.ceil(np.sqrt(self.num_clusters)))
        else:
            n_cols = self.num_clusters
            n_rows = 1

        # figure
        fig = plt.figure(figsize=(15, 9))
        gs = gridspec.GridSpec(4, 7, wspace=0.10, hspace=0.25)

        # list all plots params
        l_months = [
            (1, "January", gs[1, 3]),
            (2, "February", gs[2, 3]),
            (3, "March", gs[0, 4]),
            (4, "April", gs[1, 4]),
            (5, "May", gs[2, 4]),
            (6, "June", gs[0, 5]),
            (7, "July", gs[1, 5]),
            (8, "August", gs[2, 5]),
            (9, "September", gs[0, 6]),
            (10, "October", gs[1, 6]),
            (11, "November", gs[2, 6]),
            (12, "December", gs[0, 3]),
        ]
        l_3months = [
            ([12, 1, 2], "DJF", gs[3, 3]),
            ([3, 4, 5], "MAM", gs[3, 4]),
            ([6, 7, 8], "JJA", gs[3, 5]),
            ([9, 10, 11], "SON", gs[3, 6]),
        ]

        # plot total probabilities
        c_T = self.clusters_probs_df.values
        C_T = c_T.reshape(n_rows, n_cols)
        ax_probs_T = plt.subplot(gs[:2, :2])
        pc = self._axplot_wt_probs(
            ax_probs_T, C_T, ttl="DWT Probabilities", plot_text=plot_text
        )

        # plot counts histogram
        ax_hist = plt.subplot(gs[2:, :3])
        _ax_hist = self._axplot_wt_hist(ax_hist, ttl="DWT Counts")

        # plot probabilities by month
        for m_ix, m_name, m_gs in l_months:
            try:
                c_M = self.clusters_monthly_probs_df.loc[m_ix, :].values
                C_M = c_M.reshape(n_rows, n_cols)
                ax_M = plt.subplot(m_gs)
                self._axplot_wt_probs(
                    ax_M, C_M, ttl=m_name, vmax=vmax, plot_text=plot_text
                )
            except Exception as e:
                self.logger.error(e)

        # plot probabilities by 3 month sets
        for m_ix, m_name, m_gs in l_3months:
            try:
                c_M = self.clusters_seasonal_probs_df.loc[m_name, :].values
                C_M = c_M.reshape(n_rows, n_cols)
                ax_M = plt.subplot(m_gs)
                self._axplot_wt_probs(
                    ax_M,
                    C_M,
                    ttl=m_name,
                    vmax=vmax_seasonality,
                    cmap="Greens",
                    plot_text=plot_text,
                )
            except Exception as e:
                self.logger.error(e)

        # add custom colorbar
        pp = ax_probs_T.get_position()
        cbar_ax = fig.add_axes([pp.x1 + 0.02, pp.y0, 0.02, pp.y1 - pp.y0])
        cb = fig.colorbar(pc, cax=cbar_ax, cmap="Blues")
        cb.ax.tick_params(labelsize=8)

    def plot_perpetual_year(self) -> Axes:
        """
        Plot perpetual year bmus probabilities.

        Returns
        -------
        Axes
            The plot with the perpetual year bmus probabilities.
        """

        # Get cluster colors for stacked bar plot
        cluster_colors = get_cluster_colors(self.num_clusters)
        cluster_colors_list = [
            tuple(cluster_colors[cluster, :]) for cluster in range(self.num_clusters)
        ]

        # Plot perpetual year bmus
        fig, ax = plt.subplots(1, figsize=(15, 5))
        clusters_perpetual_year_probs_df = self.clusters_perpetual_year_probs_df
        clusters_perpetual_year_probs_df.plot.area(
            ax=ax,
            stacked=True,
            color=cluster_colors_list,
            legend=False,
        )
        ax.set_ylim(0, 1)

        return ax
