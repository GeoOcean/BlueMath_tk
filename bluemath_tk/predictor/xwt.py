import os
from typing import Any, Dict

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from dask.diagnostics import ProgressBar

from ..core.dask import setup_dask_client
from ..core.decorators import validate_data_xwt
from ..core.models import BlueMathModel
from ..core.pipeline import BlueMathPipeline
from ..core.plotting.colors import get_cluster_colors, get_config_variables
from ..datamining.kma import KMA
from ..datamining.pca import PCA

config_variables = get_config_variables()


def get_dynamic_estela_predictor(
    data: xr.Dataset,
    estela: xr.Dataset,
    check_interpolation: bool = False,
    num_workers: int = None,
    memory_limit: float = 0.5,
    chunk_sizes: dict = {"time": 365},
) -> xr.Dataset:
    """
    Transform an xarray dataset of longitude, latitude, and time into one where
    each longitude, latitude value at each time is replaced by the corresponding
    time - t, where t is specified in estela.

    Parameters
    ----------
    data : xr.Dataset
        The input dataset with dimensions longitude, latitude, and time.
    estela : xr.Dataset
        The dataset containing the t values with dimensions longitude and latitude.
    check_interpolation : bool, optional
        Whether to check if the data is interpolated. Default is False.
    num_workers : int, optional
        The number of workers to use. Default is None.
    memory_limit : float, optional
        The memory limit to use. Default is 0.5.
    chunk_sizes : dict, optional
        The chunk sizes to use. Default is {"time": 24}.

    Returns
    -------
    xr.Dataset
        The transformed dataset.
    """

    # Setup Dask client
    if num_workers is None:
        num_workers = os.environ.get("BLUEMATH_NUM_WORKERS", 2)
    client = setup_dask_client(n_workers=num_workers, memory_limit=memory_limit)

    try:
        # Process with controlled chunks
        if check_interpolation:
            # Check both data and estela have the same lon, lat, and time dimensions
            if (
                "longitude" not in data.dims
                or "latitude" not in data.dims
                or "time" not in data.dims
            ):
                raise ValueError(
                    "Data must have longitude, latitude, and time dimensions."
                )
            if "longitude" not in estela.dims or "latitude" not in estela.dims:
                raise ValueError("Estela must have longitude and latitude dimensions.")
            with ProgressBar():
                data = (
                    data.chunk({"time": chunk_sizes.get("time", 365)})
                    .interp_like(estela)
                    .compute()
                )
        data = data.where(estela.estela_mask == 1.0, np.nan).copy()
        estela_traveltimes = estela.where(estela.estela_mask, np.nan).traveltime.astype(
            int
        )
        estela_max_traveltime = estela_traveltimes.max().values
        for traveltime in range(estela_max_traveltime):
            data = data.where(
                estela_traveltimes != traveltime, data.shift(time=traveltime)
            )
        return data

    finally:
        client.close()


class XWT(BlueMathModel, BlueMathPipeline):
    """
    Xly Weather Types (XWT) class.
    """

    def __init__(self, steps: Dict[str, BlueMathModel]):
        """
        Initialize the XWT.
        """

        super().__init__()
        self.set_logger_name(name=self.__class__.__name__, level="INFO")

        # Save XWT attributes
        self.steps = steps
        self._data: xr.Dataset = None
        self.num_clusters: int = None
        self.kma_bmus: xr.Dataset = None

    @property
    def data(self) -> xr.Dataset:
        return self._data

    @property
    def clusters_probs_df(self) -> pd.DataFrame:
        """
        Calculate the probabilities for each XWT.
        """

        # Calculate probabilities for each cluster
        clusters_probs = self.kma_bmus["kma_bmus"].value_counts(normalize=True)

        return clusters_probs

    @property
    def monthly_clusters_probs_df(self) -> pd.DataFrame:
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
    def seasonal_clusters_probs_df(self) -> pd.DataFrame:
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

    @validate_data_xwt
    def fit(
        self,
        data: xr.Dataset,
        fit_params: Dict[str, Dict[str, Any]] = {},
    ):
        """
        Fit the XWT model.
        """

        # Make a copy of the data to avoid modifying the original dataset
        self._data = data.copy()

        pca: PCA = self.steps.get("pca")
        _pcs_ds = pca.fit_transform(
            data=data,
            **fit_params.get("pca", {}),
        )

        kma: KMA = self.steps.get("kma")
        self.num_clusters = kma.num_clusters
        kma_bmus, _kma_bmus_df = kma.fit_predict(
            data=pca.pcs_df,
            **fit_params.get("kma", {}),
        )
        self.kma_bmus = kma_bmus

        # Add the KMA bmus to the data
        self.data["kma_bmus"] = (("time"), kma_bmus["kma_bmus"].values)

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
    ):
        """
        Plot the XWTs.
        """

        if anomaly:
            data_to_plot = self.data.groupby("kma_bmus").mean()[
                var_to_plot
            ] - self.data[var_to_plot].mean("time")
        else:
            data_to_plot = self.data.groupby("kma_bmus").mean()[var_to_plot]

        # Get the configuration for the variable to plot if it exists
        var_to_plot_config = config_variables.get(var_to_plot, {})
        # Get the cluster colors for each XWT
        xwts_colors = get_cluster_colors(num_clusters=self.num_clusters)

        if map_center:
            p = data_to_plot.plot(
                col="kma_bmus",
                col_wrap=var_to_plot_config.get("col_wrap", 6),
                cmap=var_to_plot_config.get("cmap", "RdBu"),
                add_colorbar=False,
                # cbar_kwargs={
                #     "orientation": "horizontal",
                #     "label": var_to_plot_config.get("label", var_to_plot),
                #     "shrink": var_to_plot_config.get("shrink", 0.8),
                # },
                subplot_kws={"projection": ccrs.Orthographic(*map_center)},
                transform=ccrs.PlateCarree(),
            )
            for ax, xwt_color in zip(p.axes.flat, xwts_colors):
                self.plot_map_features(ax=ax, land_color=xwt_color)
        else:
            p = data_to_plot.plot(
                col="kma_bmus",
                col_wrap=var_to_plot_config.get("col_wrap", 6),
                cmap=var_to_plot_config.get("cmap", "RdBu"),
                add_colorbar=False,
                # cbar_kwargs={
                #     "orientation": "horizontal",
                #     "label": var_to_plot_config.get("label", var_to_plot),
                #     "shrink": var_to_plot_config.get("shrink", 0.8),
                # },
            )
            for ax, xwt_color in zip(p.axes.flat, xwts_colors):
                for border in ["top", "bottom", "left", "right"]:
                    ax.spines[border].set_color(xwt_color)

        for i, ax in enumerate(p.axes.flat):
            ax.set_title("")
            ax.text(
                0.05,
                0.05,
                i + 1,
                ha="left",
                va="bottom",
                fontsize=15,
                fontweight="bold",
                color="navy",
                transform=ax.transAxes,
            )

        plt.subplots_adjust(
            # left=0.02, right=0.98, top=0.92, bottom=0.01,
            wspace=0.05,
            hspace=0.05,
        )

    def axplot_wt_probs(
        self, ax, wt_probs, ttl="", vmin=0, vmax=0.1, cmap="Blues", caxis="black"
    ):
        "axes plot WT cluster probabilities"

        # clsuter transition plot
        pc = ax.pcolor(
            np.flipud(wt_probs),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors="k",
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

    def axplot_wt_hist(self, ax, bmus, n_clusters, ttl=""):
        "axes plot WT cluster count histogram"

        # cluster transition plot
        ax.hist(bmus, bins=np.arange(1, n_clusters + 2), edgecolor="k")

        # customize axes
        # ax.grid('y')

        ax.set_xticks(np.arange(1, n_clusters + 1) + 0.5)
        ax.set_xticklabels(np.arange(1, n_clusters + 1))
        ax.set_xlim([1, n_clusters + 1])
        ax.tick_params(axis="both", which="major", labelsize=6)

        ax.set_title(ttl, {"fontsize": 10, "fontweight": "bold"})

    def plot_dwts_probs(self, vmax=0.15, vmax_seasonality=0.15):
        """
        Plot Daily Weather Types bmus probabilities
        """

        wt_set = np.arange(self.num_clusters) + 1

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
        C_T = np.reshape(c_T, (n_rows, n_cols))

        ax_probs_T = plt.subplot(gs[:2, :2])
        pc = self.axplot_wt_probs(ax_probs_T, C_T, ttl="DWT Probabilities")

        # plot counts histogram
        ax_hist = plt.subplot(gs[2:, :3])
        self.axplot_wt_hist(
            ax_hist, self.kma_bmus.values, self.num_clusters, ttl="DWT Counts"
        )

        # plot probabilities by month
        for m_ix, m_name, m_gs in l_months:
            try:
                # get probs matrix
                # c_M = self.ClusterProbs_Month(bmus, bmus_time, wt_set, m_ix)
                c_M = self.monthly_clusters_probs_df.loc[m_ix].values
                C_M = np.reshape(c_M, (n_rows, n_cols))
                # plot axes
                ax_M = plt.subplot(m_gs)
                self.axplot_wt_probs(ax_M, C_M, ttl=m_name, vmax=vmax)
            except Exception as e:
                self.logger.error(e)

        # plot probabilities by 3 month sets
        for m_ix, m_name, m_gs in l_3months:
            try:
                # get probs matrix
                # c_M = self.ClusterProbs_Month(bmus, bmus_time, wt_set, m_ix)
                c_M = self.seasonal_clusters_probs_df.loc[m_name].values
                C_M = np.reshape(c_M, (n_rows, n_cols))

                # plot axes
                ax_M = plt.subplot(m_gs)
                self.axplot_wt_probs(
                    ax_M, C_M, ttl=m_name, vmax=vmax_seasonality, cmap="Greens"
                )
            except Exception as e:
                self.logger.error(e)

        # add custom colorbar
        pp = ax_probs_T.get_position()
        cbar_ax = fig.add_axes([pp.x1 + 0.02, pp.y0, 0.02, pp.y1 - pp.y0])
        cb = fig.colorbar(pc, cax=cbar_ax, cmap="Blues")
        cb.ax.tick_params(labelsize=8)

    def Generate_PerpYear_Matrix(
        self, num_clusters, bmus_values, bmus_dates, num_sim=1, month_ini=1
    ):
        """
        Calculates and returns matrix for stacked bar plotting

        bmus_dates - datetime.datetime (only works if daily resolution)
        bmus_values has to be 2D (time, nsim)
        """

        # generate perpetual year list
        list_pyear = self.GenOneYearDaily(month_ini=month_ini)

        # generate aux arrays
        m_plot = np.zeros((num_clusters, len(list_pyear))) * np.nan
        bmus_dates_months = np.array([pd.to_datetime(d).month for d in bmus_dates])
        bmus_dates_days = np.array([pd.to_datetime(d).day for d in bmus_dates])

        # sort data
        for i, dpy in enumerate(list_pyear):
            _, s = np.where(
                [(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)]
            )

            b = bmus_values[s]
            b = b.flatten()

            for j in range(num_clusters):
                _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!

                m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)

        return m_plot

    def GenOneYearDaily(self, yy=1981, month_ini=1):
        "returns one generic year in a list of datetimes. Daily resolution"

        dp1 = datetime(yy, month_ini, 1)
        dp2 = dp1 + timedelta(days=365)

        return [dp1 + timedelta(days=i) for i in range((dp2 - dp1).days)]

    def plot_perpetual_year(self):
        num_clusters = len(self.dwt_centroids.bmus)
        bmus_values = self.data.bmus.values
        bmus_dates = self.data.time.values
        month_ini = 1
        p_site = self.p_site

        "axes plot bmus perpetual year"

        # get cluster colors for stacked bar plot
        np_colors_int = get_cluster_colors(num_clusters)

        # generate dateticks
        x_val = self.GenOneYearDaily(month_ini=month_ini)

        # generate plot matrix
        m_plot = self.Generate_PerpYear_Matrix(
            num_clusters, bmus_values + 1, bmus_dates, month_ini=month_ini
        )

        fig, ax = plt.subplots(1, figsize=(15, 5))

        # plot stacked bars
        bottom_val = np.zeros(m_plot[1, :].shape)
        for r in range(num_clusters):
            row_val = m_plot[r, :]
            ax.bar(
                x_val,
                row_val,
                bottom=bottom_val,
                width=1,
                color=np.array([np_colors_int[r]]),
                alpha=0.7,
            )

            # store bottom
            bottom_val += row_val

        # customize  axis
        months = mdates.MonthLocator()
        monthsFmt = mdates.DateFormatter("%b")

        ax.set_xlim(x_val[0], x_val[-1])
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthsFmt)
        ax.set_ylim(0, 1)
        ax.set_ylabel("")

        fig.savefig(op.join(p_site, "plot_perpetual_year.png"), bbox_inches="tight")
        with open(op.join(p_site, "plot_perpetual_year.pkl"), "wb") as f:
            pickle.dump(fig, f)


class DWTError(Exception):
    """Custom exception for DWT class."""

    def __init__(self, message="DWT error occurred."):
        self.message = message
        super().__init__(self.message)
