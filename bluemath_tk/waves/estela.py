import os
import re

import glob
import numpy as np
import xarray as xr

from scipy.ndimage import gaussian_filter
import requests
from bs4 import BeautifulSoup

from olas.estela import calc

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class ESTELA:
    """Wave influence area using ESTELA (a method for Evaluating the Source and Travel-time of the wave Energy reaching a Local Area).

    Based on Perez et al., 2014 (https://doi.org/10.1007/s10236-014-0740-7). Codes can be found in https://github.com/jorgeperezg/olas/

    Attributes
    ----------
    p_waves : str
        Path to wave data files (local path or HTTP/HTTPS URL).
    source : str
        Data source type, default is "cawcr".
    site : str
        Site name for the analysis.
    lon0 : float
        Longitude of the analysis point (0-360).
    lat0 : float
        Latitude of the analysis point.
    PATHS : list
        List of file paths selected for processing.
    estela : xr.Dataset
        Computed ESTELA results as xarray Dataset.

    Methods
    -------
    set_site(site, location)
        Set the site name and geographic coordinates.
    set_years(years)
        Select data files within a year range.
    run(groupers, nblocks, only_sea, overwrite)
        Execute ESTELA computation.
    save(filename)
        Save results to NetCDF file.
    load(filename)
        Load results from NetCDF file.
    plot_mean_estela(custom_params)
        Plot mean wave energy distribution.
    plot_season_estela(anomaly, custom_params)
        Plot seasonal wave energy distributions.
    plot_monthly_estela(anomaly, custom_params)
        Plot monthly wave energy distributions.
    plot_grouped_estela(anomaly, custom_params, times, ncols, max_panels)
        Plot multiple time steps in a grid layout.
    """

    def __init__(self, p_waves, source="cawcr"):

        self.p_waves = p_waves.rstrip("/")
        self.source = source.lower()

        self.site = None
        self.lon0 = None
        self.lat0 = None

        self.PATHS = None
        self.estela = None

    def set_site(self, site, location):
        """Set the site name and geographic coordinates.

        Parameters
        ----------
        site : str
            Site name for the analysis.
        location : tuple
            Tuple of (longitude, latitude) coordinates.
        """
        self.site = site
        self.lon0 = location[0] % 360
        self.lat0 = location[1]

    def set_years(self, years):
        """Select data files within a year range.

        Parameters
        ----------
        years : tuple
            Tuple of (start_year, end_year) to filter input files.
        """

        if self.p_waves.startswith(("http://", "https://")):
            url = self.p_waves + "/catalog.html"
            soup = BeautifulSoup(requests.get(url).text, "html.parser")

            files = [
                self.p_waves.replace("/catalog", "/dodsC") + "/" + a.text
                for a in soup.find_all("a")
                if a.text.endswith(".nc")
            ]

        else:
            files = glob.glob(os.path.join(self.p_waves, "*.nc"))

        selected = []

        for f in sorted(files):
            filename = os.path.basename(f)
            year = int(re.search(r"\d{4}", filename).group())

            if years[0] <= year <= years[1]:
                selected.append(f)

        self.PATHS = selected

    def _prepare_era5_for_estela(self):
        """
        Transform ERA5 variables into CAWCR-like partition structure
        expected by olas.estela
        """
        pestela = os.path.join(f"outputs/estela_{self.site}")
        os.makedirs(pestela, exist_ok=True)

        new_paths = []

        for path in self.PATHS:
            filename = os.path.basename(path).replace(".nc", "_for_estela.nc")
            temp_path = os.path.join(pestela, filename)

            if os.path.exists(temp_path):
                new_paths.append(temp_path)
                continue

            ds = xr.open_dataset(path)

            if "valid_time" in ds.coords:
                ds = ds.rename({"valid_time": "time"})

            ds["hs.0"] = ds["shww"]
            ds["hs.1"] = ds["swh1"]
            ds["hs.2"] = ds["swh2"]
            ds["hs.3"] = ds["swh3"]

            # Empirical relationship from Cagigal et al. (2021), considering JONSWAP spectrum with gamma=3.3
            ds["tp.0"] = ds["mpww"] * 1.2
            ds["tp.1"] = ds["mwp1"] * 1.2
            ds["tp.2"] = ds["mwp2"] * 1.2
            ds["tp.3"] = ds["mwp3"] * 1.2

            ds["dir.0"] = ds["mdww"]
            ds["dir.1"] = ds["mwd1"]
            ds["dir.2"] = ds["mwd2"]
            ds["dir.3"] = ds["mwd3"]

            # TODO: wind and swell waves spreading from ERA5 dataset can be included separately (dwww and dwps variables)
            for i in range(4):
                ds[f"si.{i}"] = ds["wdw"]

            # Delete old variables
            old_vars = [
                "shww",
                "swh1",
                "swh2",
                "swh3",
                "mpww",
                "mwp1",
                "mwp2",
                "mwp3",
                "mdww",
                "mwd1",
                "mwd2",
                "mwd3",
                "wdw",
            ]
            ds = ds.drop_vars(old_vars, errors="ignore")

            if ds.longitude.min() < 0:
                ds = ds.assign_coords(longitude=(ds.longitude % 360))
            ds = ds.sortby("longitude")

            if ds.latitude[0] < ds.latitude[-1]:
                ds = ds.sortby("latitude", ascending=False)

            ds.to_netcdf(temp_path)

            new_paths.append(temp_path)

        self.PATHS = new_paths

    def run(self, groupers=None, nblocks=1, only_sea=False, overwrite=False):
        """Execute ESTELA computation.

        Parameters
        ----------
        groupers : list, optional
            Grouping variables for temporal aggregation ("H", "D", "M", "Y", "ALL", "time.month" and "time.season").
            Default are "ALL", "time.month" and "time.season".
        nblocks : int, optional
            Number of blocks for computation, default is 1.
        only_sea : bool, optional
            If True, use only sea waves, default is False.
        overwrite : bool, optional
            If True, overwrite existing results, default is False.
        """

        if self.PATHS is None:
            raise ValueError("Call set_years first.")

        estela_dir = os.path.join(f"outputs/estela_{self.site}")
        os.makedirs(estela_dir, exist_ok=True)

        years = (
            f"{os.path.basename(self.PATHS[0])[:4]}-"
            f"{os.path.basename(self.PATHS[-1])[:4]}"
        )

        if only_sea:
            estela_file = os.path.join(f"outputs/estela_{self.site}_{years}_onlysea.nc")
        else:
            estela_file = os.path.join(f"outputs/estela_{self.site}_{years}.nc")

        if os.path.exists(estela_file) and not overwrite:
            print("Loading existing ESTELA file...")
            self.estela = xr.open_dataset(estela_file)
            return

        if self.source == "era5":
            self._prepare_era5_for_estela()

        if self.p_waves.startswith(("http://", "https://")):
            hs_name = r"hs0" if only_sea else r"hs\d"
            tp_name = r"tp0" if only_sea else r"tp\d"
            dp_name = r"th0" if only_sea else r"th\d"
            si_name = r"si0" if only_sea else r"si\d"
        else:
            hs_name = r"hs\.0" if only_sea else r"hs\.\d"
            tp_name = r"tp\.0" if only_sea else r"tp\.\d"
            dp_name = r"dir\.0" if only_sea else r"dir\.\d"
            si_name = r"si\.0" if only_sea else r"si\.\d"

        self.estela = calc(
            self.PATHS,
            self.lat0,
            self.lon0,
            hs=hs_name,
            tp=tp_name,
            dp=dp_name,
            si=si_name,
            groupers=groupers,
            nblocks=nblocks,
        )

        self.save(estela_file)

    def save(self, filename):

        if self.estela is None:
            raise ValueError("Run estela first.")

        self.estela.to_netcdf(filename)

    def load(self, filename):

        self.estela = xr.open_dataset(filename)


    # ----------------------------------------------------------------------------------
    # ------------------------------------ PLOTTING ------------------------------------
    # ----------------------------------------------------------------------------------


    plot_defaults = dict(
        extent=None,
        figsize=[20, 12],
        vmin=None,
        vmax=None,
        vmin_anomaly=None,
        vmax_anomaly=None,
        central_longitude=0,
        cbar=True,
        fontsize=15,
        cmap="rainbow",
        cmap_anomaly="RdBu_r",
        c_color="grey",
        alpha=1,
    )

    def get_figsize(self, plot_params, N=20, ratio=1):

        if plot_params.get("figsize") is None:
            return [N, N / ratio]

        return plot_params["figsize"]

    def plot_map_features(self, ax, land_color=cfeature.COLORS["land"]):

        ax.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.5)

        land_10m = cfeature.NaturalEarthFeature(
            "physical",
            "land",
            "10m",
            edgecolor="black",
            linewidth=0.5,
            facecolor=land_color,
            zorder=1,
        )

        ax.add_feature(land_10m)
        ax.gridlines(draw_labels=False)

    def plot_estela(self, est, plot_params, anomaly=False, fig=None, ax=None):

        extent = plot_params.get("extent")

        if extent is None:
            lonmin = float(est.longitude.min())
            lonmax = float(est.longitude.max())
            latmin = float(est.latitude.min())
            latmax = float(est.latitude.max())
            extent = (lonmin, lonmax, latmin, latmax)
            plot_params["extent"] = extent

        ratio = (extent[1] - extent[0]) / (extent[3] - extent[2])

        if fig is None or ax is None:
            figsize = plot_params.get("figsize", [20, 20 / ratio])
            fig = plt.figure(figsize=figsize)
            ax = plt.axes(
                projection=ccrs.PlateCarree(
                    central_longitude=plot_params["central_longitude"]
                )
            )

        self.plot_map_features(ax)

        energy = est.F.copy()

        if anomaly:
            energy = energy - self.estela.F.mean("time")

            vmin = plot_params["vmin_anomaly"]
            vmax = plot_params["vmax_anomaly"]
            cmap_energy = plot_params["cmap_anomaly"]

            levels = np.linspace(vmin, vmax, 101)

        else:
            vmin = plot_params["vmin"]
            vmax = plot_params["vmax"]
            cmap_energy = plot_params["cmap"]

            levels = 100

        p1 = ax.contourf(
            est.longitude,
            est.latitude,
            energy,
            levels=levels,
            transform=ccrs.PlateCarree(),
            zorder=2,
            alpha=plot_params["alpha"],
            cmap=cmap_energy,
            extend="both" if anomaly else "neither",
        )

        if not anomaly:
            levels = np.arange(1, 26, 1)

            ax.contour(
                est.longitude,
                est.latitude,
                gaussian_filter(est.traveltime, sigma=1.4),
                levels=levels[levels % 3 != 0],
                transform=ccrs.PlateCarree(),
                colors="lightgrey",
                linewidths=0.8,
                zorder=3,
            )

            cim = ax.contour(
                est.longitude,
                est.latitude,
                gaussian_filter(est.traveltime, sigma=1.4),
                levels=levels[levels % 3 == 0],
                transform=ccrs.PlateCarree(),
                colors="black",
                linewidths=1.2,
                zorder=4,
            )

            lab = ax.clabel(cim, inline=True, fontsize=plot_params["fontsize"] * 0.7)
            for txt in lab:
                txt.set_rotation(0)

        ax.plot(
            self.lon0,
            self.lat0,
            marker="*",
            color="black",
            markersize=12,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

        if plot_params["cbar"]:
            cbar = plt.colorbar(p1, shrink=0.7)
            cbar.set_label("kW/m/º", fontsize=plot_params["fontsize"])

        ax.set_extent(extent, crs=ccrs.PlateCarree())

        return p1

    def plot_mean_estela(self, custom_params=None):

        est = self.estela.copy()

        if "time" in est.dims:
            est = est.mean("time")

        plot_params = {**self.plot_defaults, **(custom_params or {})}

        self.plot_estela(est, plot_params)

    def plot_season_estela(self, anomaly=False, custom_params=None):

        seasons = ["DJF", "MAM", "JJA", "SON"]
        plot_params = {**self.plot_defaults, **(custom_params or {})}

        if not anomaly:
            vmin = 0
            vmax = float(self.estela.F.quantile(0.995))
            plot_params["vmin"] = vmin
            plot_params["vmax"] = vmax
        else:
            aux = self.estela.F - self.estela.F.mean("time")
            lim = float(abs(aux).quantile(0.995))
            vmin, vmax = -lim, lim
            plot_params["vmin_anomaly"] = vmin
            plot_params["vmax_anomaly"] = vmax

        plot_params["cbar"] = False

        ratio = 2
        figsize = plot_params.get("figsize", [18, 18 / ratio])

        fig, axs = plt.subplots(
            2,
            2,
            figsize=figsize,
            subplot_kw={
                "projection": ccrs.PlateCarree(
                    central_longitude=plot_params["central_longitude"]
                )
            },
        )

        fig.subplots_adjust(
            left=0.07, right=0.93, bottom=0.2, top=0.92, wspace=0, hspace=0.2
        )

        fig_estela = None

        for i, ax in enumerate(axs.flat):
            ax.set_title(seasons[i], fontsize=plot_params["fontsize"])

            fig_estela = self.plot_estela(
                self.estela.sel(time=seasons[i]),
                plot_params,
                fig=fig,
                ax=ax,
                anomaly=anomaly,
            )

        cbar = fig.colorbar(
            fig_estela, ax=axs, orientation="horizontal", fraction=0.04, pad=0.07
        )

        cbar.set_label("kW/m/°", fontsize=plot_params["fontsize"])

    def plot_monthly_estela(self, anomaly=False, custom_params=None):

        plot_params = {**self.plot_defaults, **(custom_params or {})}

        if not anomaly:
            vmin = 0
            vmax = float(self.estela.F.quantile(0.995))
            plot_params["vmin"] = vmin
            plot_params["vmax"] = vmax
        else:
            aux = self.estela.F - self.estela.F.mean("time")
            lim = float(abs(aux).quantile(0.995))
            vmin, vmax = -lim, lim
            plot_params["vmin_anomaly"] = vmin
            plot_params["vmax_anomaly"] = vmax

        plot_params["cbar"] = False

        figsize = plot_params.get("figsize", [22, 16])

        fig, axs = plt.subplots(
            4,
            3,
            figsize=figsize,
            subplot_kw={
                "projection": ccrs.PlateCarree(
                    central_longitude=plot_params["central_longitude"]
                )
            },
        )

        fig.subplots_adjust(
            left=0.07, right=0.88, bottom=0.08, top=0.92, wspace=0, hspace=0.15
        )

        last_p1 = None
        month_labels = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        for i, ax in enumerate(axs.flat):
            ax.set_title(month_labels[i], fontsize=plot_params["fontsize"])

            last_p1 = self.plot_estela(
                self.estela.sel(time=f"m{i + 1:02d}"),
                plot_params,
                fig=fig,
                ax=ax,
                anomaly=anomaly,
            )

        cbar = fig.colorbar(
            last_p1, ax=axs, orientation="vertical", fraction=0.02, pad=0.02
        )

        cbar.set_label("kW/m/°", fontsize=plot_params["fontsize"])

    def plot_grouped_estela(
        self, anomaly=False, custom_params=None, times=None, ncols=3, max_panels=None
    ):

        plot_params = {**self.plot_defaults, **(custom_params or {})}

        available_times = [str(t) for t in self.estela.time.values]

        if times is None:
            times = available_times
        else:
            times = [str(t) for t in np.atleast_1d(times)]

        missing = [t for t in times if t not in available_times]
        if missing:
            raise ValueError(
                f"Some requested times are not available: {missing}\n"
                f"Available times are: {available_times}"
            )

        if max_panels is not None:
            times = times[:max_panels]

        nplots = len(times)
        nrows = int(np.ceil(nplots / ncols))

        if not anomaly:
            plot_params["vmin"] = (
                0 if plot_params["vmin"] is None else plot_params["vmin"]
            )
            plot_params["vmax"] = (
                float(self.estela.F.quantile(0.995))
                if plot_params["vmax"] is None
                else plot_params["vmax"]
            )
        else:
            aux = self.estela.F - self.estela.F.mean("time")
            lim = float(abs(aux).quantile(0.995))

            plot_params["vmin_anomaly"] = (
                -lim
                if plot_params["vmin_anomaly"] is None
                else plot_params["vmin_anomaly"]
            )
            plot_params["vmax_anomaly"] = (
                lim
                if plot_params["vmax_anomaly"] is None
                else plot_params["vmax_anomaly"]
            )

        plot_params["cbar"] = False

        if plot_params.get("figsize") is None:
            figsize = [5.5 * ncols, 4.5 * nrows]
        else:
            figsize = plot_params["figsize"]

        fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            subplot_kw={
                "projection": ccrs.PlateCarree(
                    central_longitude=plot_params["central_longitude"]
                )
            },
        )

        axs = np.atleast_1d(axs).ravel()

        last_p1 = None

        for i, time_label in enumerate(times):
            ax = axs[i]

            ax.set_title(time_label, fontsize=plot_params["fontsize"])

            est_i = self.estela.sel(time=time_label)

            last_p1 = self.plot_estela(
                est_i,
                plot_params,
                anomaly=anomaly,
                fig=fig,
                ax=ax,
            )

        for j in range(nplots, len(axs)):
            axs[j].set_visible(False)

        fig.subplots_adjust(
            left=0.06,
            right=0.90,
            bottom=0.08,
            top=0.92,
            wspace=0.05,
            hspace=0.20,
        )

        cbar = fig.colorbar(
            last_p1,
            ax=axs[:nplots],
            orientation="vertical",
            fraction=0.025,
            pad=0.02,
        )

        cbar.set_label("kW/m/°", fontsize=plot_params["fontsize"])
