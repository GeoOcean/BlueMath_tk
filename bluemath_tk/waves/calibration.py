from datetime import timedelta as td

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.stats import gaussian_kde
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


# -----------------------------------------------------------------------------#
# Metric and accuracy functions
def rmse(pred, tar):
    return np.sqrt(((pred - tar) ** 2).mean())


def bias(pred, tar):
    return sum(pred - tar) / len(pred)


def si(pred, tar):
    pred_mean = pred.mean()
    tar_mean = tar.mean()
    return np.sqrt(sum(((pred - pred_mean) - (tar - tar_mean)) ** 2) / (sum(tar**2)))


# -----------------------------------------------------------------------------#
# Direc selection function
def create_vec_direc(waves, direcs):
    data = np.zeros((len(waves), 16))
    for i in range(len(waves)):
        if ((i / len(waves)) * 100) % 5 == 0:
            print(str((i / len(waves)) * 100) + "% completed...")
        if direcs[i] < 0:
            direcs[i] = direcs[i] + 360
        if direcs[i] > 0 and waves[i] > 0:
            if direcs[i] >= 0 and direcs[i] < 22.5:
                data[i, 0] = waves[i]
            elif direcs[i] >= 22.5 and direcs[i] < 45:
                data[i, 1] = waves[i]
            elif direcs[i] >= 45 and direcs[i] < 67.5:
                data[i, 2] = waves[i]
            elif direcs[i] >= 67.5 and direcs[i] < 90:
                data[i, 3] = waves[i]
            elif direcs[i] >= 90 and direcs[i] < 112.5:
                data[i, 4] = waves[i]
            elif direcs[i] >= 112.5 and direcs[i] < 135:
                data[i, 5] = waves[i]
            elif direcs[i] >= 135 and direcs[i] < 157.5:
                data[i, 6] = waves[i]
            elif direcs[i] >= 157.5 and direcs[i] < 180:
                data[i, 7] = waves[i]
            elif direcs[i] >= 180 and direcs[i] < 202.5:
                data[i, 8] = waves[i]
            elif direcs[i] >= 202.5 and direcs[i] < 225:
                data[i, 9] = waves[i]
            elif direcs[i] >= 225 and direcs[i] < 247.5:
                data[i, 10] = waves[i]
            elif direcs[i] >= 247.5 and direcs[i] < 270:
                data[i, 11] = waves[i]
            elif direcs[i] >= 270 and direcs[i] < 292.5:
                data[i, 12] = waves[i]
            elif direcs[i] >= 292.5 and direcs[i] < 315:
                data[i, 13] = waves[i]
            elif direcs[i] >= 315 and direcs[i] < 335.5:
                data[i, 14] = waves[i]
            elif direcs[i] >= 335.5 and direcs[i] < 360:
                data[i, 15] = waves[i]
    return data


# -----------------------------------------------------------------------------#
# Time calibration
def calibration_time(sat, hind, sh=True, min_time=2):
    # create the empty arrays
    times_sat = np.array([], dtype="datetime64")
    times_hind = np.array([], dtype="datetime64")
    # perform the calibration
    if sh:
        for i in range(len(sat)):
            diff = np.min(abs(hind - sat[i]))
            if diff < np.timedelta64(min_time, "h"):
                min_index = np.argmin(abs(hind - sat[i]))
                times_hind = np.append(times_hind, hind[min_index])
                times_sat = np.append(times_sat, sat[i])
    else:
        for i in range(len(hind)):
            diff = np.min(abs(sat - hind[i]))
            if diff < np.timedelta64(min_time, "h"):
                min_index = np.argmin(abs(sat - hind[i]))
                times_sat = np.append(times_sat, sat[min_index])
                times_hind = np.append(times_hind, hind[i])
    # return
    return times_sat, times_hind


# Calibration-Validation class
class CalVal(object):
    """
    This class CalVal calibrates the wave reanalysis information with
    buoy and satellite data, although we prefer to calibrate it first
    with satellites, so then the data can be compared and validated
    with the buoys. Following this procedure, the calibration is always
    performed and after this, if buoy data is available, then this new
    wave reanalysis calibrated hindcast can be validated
    """

    def __init__(
        self,
        hindcast,
        n_parts,
        satellite,
        buoy=(False, None),
        buoy_corrections=False,
        hindcast_longitude=-4.5,
        hindcast_latitude=44.0,
        buoy_longitude=None,
        buoy_latitude=None,
    ):
        """Initializes the class with all the necessary attributes that
        will be used in the different methods
        ------------
        Parameters
        hindcast: Hindcast data as a dataframe
        n_parts: Number of partitions in the wave hindcast
        satellite: Satellite data as a netCDF (see extract_satellite.py)
        buoy: Buoy data as a dataframe, and boolean to indicate if it
            is available or not
        buoy_corrections: Boolean to indicate if the buoy corrections
            will be applied or not
        ------------
        Returns
        Attributes initialized and plots for both calibrations, the one
        done with the satellite and the other one performed with buoy
        data, if done. The parameters for the calibrations are also stored
        """

        print("\n Plotting region to be working with!! \n")

        # plot data domains for hindcast, satellite and buoy
        fig, ax = plt.subplots(
            figsize=(10, 10),
            subplot_kw={
                "projection": ccrs.PlateCarree(central_longitude=hindcast_longitude)
            },
        )
        land_10m = cfeature.NaturalEarthFeature(
            "physical",
            "land",
            "10m",
            edgecolor="face",
            facecolor=cfeature.COLORS["land"],
        )  # add land to image
        ax.scatter(
            satellite.LONGITUDE,
            satellite.LATITUDE,
            s=0.01,
            c="k",
            transform=ccrs.PlateCarree(),
        )
        ax.scatter(
            hindcast_longitude,
            hindcast_latitude,
            s=50,
            c="red",
            zorder=10,
            transform=ccrs.PlateCarree(),
        )
        ax.scatter(
            buoy_longitude,
            buoy_latitude,
            s=50,
            c="orange",
            zorder=10,
            transform=ccrs.PlateCarree(),
        ) if buoy[0] else None
        ax.set_extent(
            [
                hindcast_longitude - 4,
                hindcast_longitude + 4,
                hindcast_latitude - 4,
                hindcast_latitude + 2,
            ]
        )
        ax.stock_img()
        ax.add_feature(land_10m)
        plt.show()  # TODO: add better plotting!!
        # TODO: add labels
        # gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,
        #                   linewidth=2,color='gray',linestyle='--')
        # xlabels = np.arange(plot_region[1][0],plot_region[1][1],plot_labels[1])
        # xlabels = np.where(xlabels<180,xlabels,xlabels-360)
        # ylabels = np.arange(plot_region[1][3],plot_region[1][2],plot_labels[2])
        # gl.xlocator = mticker.FixedLocator(list(xlabels))
        # gl.ylocator = mticker.FixedLocator(list(ylabels))
        # gl.xlabels_top = False
        # gl.ylabels_right = False

        # save all datasets and variables in class
        self.hindcast = hindcast.copy()
        self.n_parts = n_parts
        self.hindcast_longitude = hindcast_longitude
        self.hindcast_latitude = hindcast_latitude
        self.possible_to_correct = np.where(hindcast["Hs_cal"] > 0.01)[0]
        self.hind_to_corr = hindcast.iloc[self.possible_to_correct].copy()
        self.satellite = satellite.copy()
        self.hindcast_sat_corr = hindcast.copy()
        self.params_sat_corr = self.calibration("satellite")
        if buoy[0]:
            self.buoy = buoy[1].copy()
            self.buoy_longitude = buoy_longitude
            self.buoy_latitude = buoy_latitude
        if buoy_corrections:
            self.hindcast_buoy_corr = hindcast.copy()
            self.params_buoy_corr = self.calibration("buoy")
        else:
            print("\n No buoy corrections will be done!! \n")

    def calibration(self, calibration_type):
        """Calibrates hindcast with satellite or buoy data. This calibration
        is performed using a linear regression and selecting only those
        parameters that are significantly representative
        ------------
        Parameters
        calibration_type: (str) Type of calibration to be done
            it can be : (satellite, buoy)
        ------------
        Returns
        Corrected data and calculated params
        """

        print(" -------------------------------------------------------- ")
        print(calibration_type.upper() + " CALIBRATION will be performed")
        print(" -------------------------------------------------------- \n ")

        # Initializes satellite data to calibrate
        if calibration_type == "satellite":
            print("Satellite box values: ")
            ini_lat = float(input("South latitude: "))
            end_lat = float(input("North latitude: "))
            ini_lon = float(input("West longitude: "))
            end_lon = float(input("East longitude: "))
            print(" \n ")
            hs_calibrate, calibration = self.satellite_values(
                ini_lat=ini_lat, end_lat=end_lat, ini_lon=ini_lon, end_lon=end_lon
            )
            title = "Satellite"
        elif calibration_type == "buoy":
            calibration_tot = pd.concat([self.buoy, self.hind_to_corr], axis=1)
            notna_values = calibration_tot["Hs_Buoy"].notna().values
            calibration = calibration_tot.iloc[notna_values]
            hs_calibrate = calibration["Hs_Buoy"].values
            title = "Buoy"
        else:
            return "Not a valid value for calibration_type"

        print(" \n ")

        # Construct matrices to calibrate
        print("Constructing matrices and calibrating... \n ")
        print("This might take a few minutes... \n")

        print("Sea... \n")
        Hsea = create_vec_direc(calibration["Hsea"], calibration["Dirsea"])
        print("\n")

        print("Swells 1, 2, 3... \n")
        Hs_swells = np.zeros(Hsea.shape)
        for part in np.arange(1, self.n_parts):
            Hs_swells += (
                create_vec_direc(
                    calibration["Hswell" + str(part)],
                    calibration["Dirswell" + str(part)],
                )
            ) ** 2

        # concatenate seas and swells
        Hs_ncorr_mat = np.concatenate([Hsea**2, Hs_swells], axis=1)

        Hs_ncorr = np.sqrt(np.sum(Hs_ncorr_mat, axis=1))
        # ---------------------------------------------------------------------#
        print("\n")
        print("Threshold of minimum Hs to calibrate")
        print(
            "Directional families with a mean Hs under this threshold will not be calibrated: "
        )
        th_ne = float(input("----- Threshold ----- : "))
        print(" \n ")
        # ---------------------------------------------------------------------#
        nedata = np.where(np.mean(Hs_ncorr_mat, axis=0) < th_ne)[0]
        reg = linear_model.LinearRegression()
        hs_calibrate_2 = hs_calibrate**2
        reg.fit(Hs_ncorr_mat, hs_calibrate_2)
        X = sm.add_constant(Hs_ncorr_mat)
        est = sm.OLS(hs_calibrate_2, X)
        est2 = est.fit()
        params = np.array([], dtype=float)
        for p in range(1, len(est2.params)):
            if est2.pvalues[p] < 0.05 and reg.coef_[p - 1] > 0:
                params = np.append(params, reg.coef_[p - 1])
            else:
                params = np.append(params, 1.0)
        params[nedata] = 1.0
        paramss = np.array([params])
        Hs_corr_mat = paramss * Hs_ncorr_mat
        Hs_corr = np.sqrt(np.sum(Hs_corr_mat, axis=1))
        params = np.sqrt(params)

        print(" \n ")
        print("Params used for the " + title.upper() + " calibration are: \n ")
        print(params)
        print(" \n ")

        # Plotting corrected results
        print("Plotting just the data used to calibrate... \n ")

        self.calibration_plots(
            Hs_ncorr, Hs_corr, hs_calibrate, calibration, params, title
        )

        # Now, we will save all the data corrected
        print("Saving corrected results... \n ")
        print("This might take more than a few minutes... \n")

        for part in np.arange(0, self.n_parts):
            if part == 0:
                print("Sea... \n")
                Hsea = create_vec_direc(
                    self.hind_to_corr["Hsea"], self.hind_to_corr["Dirsea"]
                )
                Hsea_corr_mat = paramss[:, 0:16] * Hsea**2
                Hsea_corr = np.sqrt(np.sum(Hsea_corr_mat, axis=1))
                Hs_ncorr_sea = Hsea**2
                index_hsea = np.where(self.hindcast.columns.values == "Hsea")[0][0]
                print("\n")
            else:
                print("Swell:" + str(part) + "\n")
                globals()["Hswell%s" % part] = create_vec_direc(
                    self.hind_to_corr["Hswell" + str(part)],
                    self.hind_to_corr["Dirswell" + str(part)],
                )
                globals()["Hswell%s_corr_mat" % part] = (
                    paramss[:, 16:32] * globals()["Hswell" + str(part)] ** 2
                )
                globals()["Hswell%s_corr" % part] = np.sqrt(
                    np.sum(globals()["Hswell%s_corr_mat" % part], axis=1)
                )
                # globals()['index_hswell%s' % part] = np.where(self.hindcast.columns.values==globals()['Hswell%s' % part])[0][0]
                globals()["index_hswell%s" % part] = np.where(
                    self.hindcast.columns.values == ["Hswell" + str(part)]
                )[0][0]
                if part == 1:
                    Hs_ncorr_swell = globals()["Hswell%s" % part] ** 2
                else:
                    Hs_ncorr_swell = Hs_ncorr_swell + globals()["Hswell%s" % part] ** 2
        # print(np.shape(Hs_ncorr_sea))
        # print(np.shape(Hs_ncorr_swell))

        Hs_ncorr_mat = np.concatenate((Hs_ncorr_sea, Hs_ncorr_swell), axis=1)
        Hs_ncorr = np.sqrt(np.sum(Hs_ncorr_mat, axis=1))
        Hs_corr_mat = paramss * Hs_ncorr_mat
        Hs_corr = np.sqrt(np.sum(Hs_corr_mat, axis=1))
        index_hs = np.where(self.hindcast.columns.values == "Hs")[0][0]
        index_hs_cal = np.where(self.hindcast.columns.values == "Hs_cal")[0][0]

        if calibration_type == "satellite":
            self.hindcast_sat_corr.iloc[self.possible_to_correct, index_hs] = Hs_corr
            self.hindcast_sat_corr.iloc[self.possible_to_correct, index_hsea] = (
                Hsea_corr
            )
            for part in np.arange(1, self.n_parts):
                self.hindcast_sat_corr.iloc[
                    self.possible_to_correct, globals()["index_hswell%s" % part]
                ] = globals()["Hswell%s_corr" % part]
            self.hindcast_sat_corr.iloc[self.possible_to_correct, index_hs_cal] = (
                Hs_corr
            )

        else:
            self.hindcast_buoy_corr.iloc[self.possible_to_correct, index_hs] = Hs_corr
            self.hindcast_buoy_corr.iloc[self.possible_to_correct, index_hsea] = (
                Hsea_corr
            )
            for part in np.arange(1, self.n_parts):
                self.hindcast_buoy_corr.iloc[
                    self.possible_to_correct, globals()["index_hswell%s" % part]
                ] = globals()["Hswell%s_corr" % part]
            self.hindcast_buoy_corr.iloc[self.possible_to_correct, index_hs_cal] = (
                Hs_corr
            )
        print(" \n  \n ")

        # return
        return params

    def satellite_values(self, ini_lat, end_lat, ini_lon, end_lon):
        """Performs the time calibration step that allows us to perform the
        calibration between the hindcast and the satellite data
        ------------
        Parameters
        Lats and lons to generate the box with the satellite data that
        will be used, previously selected by input
        ------------
        Returns
        Significant wave height for the satellite and a reduced dataframe
        for the hindcast data
        """

        # SATELLITE
        print("Selecting the satellite data choosed... \n ")

        self.satellite = self.satellite.isel(
            TIME=np.where(self.satellite.LATITUDE.values > ini_lat)[0]
        )
        self.satellite = self.satellite.isel(
            TIME=np.where(self.satellite.LATITUDE.values < end_lat)[0]
        )
        self.satellite = self.satellite.isel(
            TIME=np.where(self.satellite.LONGITUDE.values > ini_lon)[0]
        )
        self.satellite = self.satellite.isel(
            TIME=np.where(self.satellite.LONGITUDE.values < end_lon)[0]
        )

        print("Satellite length: " + str(len(self.satellite.TIME.values)))

        # HINDCAST
        print(
            "Hindcast information able to calibrate: "
            + str(len(self.hind_to_corr))
            + " \n "
        )

        # We perform the calibration
        print("Choose the way to calibrate the data: ")
        type_calib_way = bool(
            input(
                "True (not recomended): hindcast for each satellite \n"
                + "False (empty box): satellite for each hindcast \n"
                + "----- Select ----- : "
            )
        )
        print(" \n ")

        print("Performing the time calibration... \n ")
        times_sat, times_hind = calibration_time(
            self.satellite.TIME.values,
            self.hind_to_corr.index.values,
            sh=type_calib_way,
        )
        sat_times = self.satellite.sel(TIME=times_sat)

        # All the necessary Satellite data (Quality)
        wave_height_qlt = np.nansum(
            np.concatenate(
                (
                    sat_times["SWH_KU_quality_control"].values[:, np.newaxis],
                    sat_times["SWH_KA_quality_control"].values[:, np.newaxis],
                ),
                axis=1,
            ),
            axis=1,
        )
        good_qlt = np.where(wave_height_qlt < 1.5)
        # All necessary Satellite data (Heights)
        wave_height_cal = np.nansum(
            np.concatenate(
                (
                    sat_times["SWH_KU_CAL"].values[:, np.newaxis],
                    sat_times["SWH_KA_CAL"].values[:, np.newaxis],
                ),
                axis=1,
            ),
            axis=1,
        )
        wave_height_cal = wave_height_cal[good_qlt]

        calibration = self.hind_to_corr.loc[times_hind].iloc[good_qlt]

        print("Length of data to calibrate: " + str(len(calibration)) + " \n ")

        return wave_height_cal, calibration

    def density_scatter(self, x, y):
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x1, y1, z = x[idx], y[idx], z[idx]

        return (x1, y1, z)

    def validation_scatter(self, axs, x, y, xlabel, ylabel, title):
        x2, y2, z = self.density_scatter(x, y)

        # plot
        axs.scatter(x2, y2, c=z, s=5, cmap="rainbow")

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
        xq = stats.probplot(x, dist="norm")
        yq = stats.probplot(y, dist="norm")
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
                r"mse =  %.2f" % (mse,),
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

    def calibration_plots(self, xx1, xx2, hs, data, coefs, big_title):
        """Plots differnet graphs for the calibration of hindcast data with
        buoy and satellite information. Plots are explained by their own
        ------------
        Parameters
        xx1: No corrected data
        xx2: Corrected data
        hs: Satellite / Buoy data that has been used to calibrate
        data: Dataframe with more hindcast information
        coefs: Parameters calculated in the calibration
        big_title: Can be 'Buoy' or 'Satellite'
        ------------
        Returns
        Different auto explicative plots
        """

        num = "1"

        fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        fig.subplots_adjust(hspace=0.4, wspace=0.1)
        fig.suptitle(
            "CSIRO hindcast calibration with " + big_title + " data",
            y=0.99,
            fontsize=12,
            fontweight="bold",
        )

        for i in range(2):
            for j in range(3):
                if i == j == 0 or i == 1 and j == 0:
                    if i == 0:
                        x, y = hs, xx1
                        title = "Not corrected, $H_{s}$ (m)"
                    else:
                        x, y = hs, xx2
                        title = "Corrected, $H_{s}$ (m)"

                    self.validation_scatter(
                        axs[i, j], x, y, big_title, "Hindcast", title
                    )

                elif i == 0 and j == 1 or i == 0 and j == 2:
                    if j == 1:
                        dataj1 = data[["Dirsea", "Hsea"]].dropna(axis=0, how="any")
                        x, y = dataj1["Dirsea"], dataj1["Hsea"]
                        index = 2
                        title = "SEA $Wave$ $Climate$"
                    else:
                        dataj2 = data[["Dirswell" + num, "Hswell" + num]].dropna(
                            axis=0, how="any"
                        )
                        x, y = dataj2["Dirswell" + num], dataj2["Hswell" + num]
                        index = 3
                        title = "SWELL {0} $Wave$ $Climate$".format(num)

                    x = (x * np.pi) / 180
                    axs[i, j].axis("off")
                    axs[i, j] = fig.add_subplot(2, 3, index, projection="polar")
                    x2, y2, z = self.density_scatter(x, y)
                    axs[i, j].scatter(x2, y2, c=z, s=3, cmap="jet")
                    axs[i, j].set_theta_zero_location("N", offset=0)
                    axs[i, j].set_xticklabels(
                        ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                    )
                    axs[i, j].xaxis.grid(True, color="lavender", linestyle="-")
                    axs[i, j].yaxis.grid(True, color="lavender", linestyle="-")
                    # axs[i,j].xaxis.set_tick_params(labelsize=20)
                    axs[i, j].set_theta_direction(-1)
                    axs[i, j].set_xlabel("$\u03b8_{m}$ ($\degree$)")
                    axs[i, j].set_ylabel("$H_{s}$ (m)", labelpad=20)
                    axs[i, j].set_title(title, pad=15, fontweight="bold")

                else:
                    if j == 1:
                        color_vals = coefs[0:16]
                        title = "SEA $Correction$"
                    else:
                        color_vals = coefs[16:32]
                        title = "SWELL 1 $Correction$"

                    norm = 0.3
                    fracs = np.repeat(10, 16)
                    my_norm = mpl.colors.Normalize(1 - norm, 1 + norm)
                    my_cmap = mpl.cm.get_cmap("bwr", len(color_vals))
                    axs[i, j].pie(
                        fracs,
                        labels=None,
                        colors=my_cmap(my_norm(color_vals)),
                        startangle=90,
                        counterclock=False,
                        radius=1.2,
                    )
                    axs[i, j].set_title(title, fontweight="bold")  # fontsize=12, ,

                    if j == 2:
                        ax1_divider = make_axes_locatable(axs[i, j])
                        cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
                        cb = mpl.colorbar.ColorbarBase(cax1, cmap=my_cmap, norm=my_norm)
                        cb.set_label("Correction Coefficients")
                        cb.outline.set_color("white")

        # show results
        plt.show()

    def buoy_comparison(self, comparison_type, buoy_name="Buoy NAME"):
        """Compares data with buoy, even if it's been corrected or not
        ------------
        Parameters
        comparison_type: Type of comparison to be done
                         (raw, sat_corr, buoy_corr)
        ------------
        Returns
        Different auto explicative plots
        """

        # Initialize the data to compare
        if comparison_type == "raw":
            comparison = pd.concat([self.buoy, self.hindcast], axis=1)
        elif comparison_type == "sat_corr":
            comparison = pd.concat([self.buoy, self.hindcast_sat_corr], axis=1)
        elif comparison_type == "buoy_corr":
            comparison = pd.concat([self.buoy, self.hindcast_buoy_corr], axis=1)
        else:
            return "Not a valid value for comparison_type"

        print("--------------------------------------------------------")
        print(comparison_type.upper() + " comparison will be performed")
        print("-------------------------------------------------------- \n ")

        comparison = comparison[["Hs_Buoy", "Tp_Buoy", "Dir_Buoy", "Hs", "Tp", "DirM"]]

        # Perform the comparison
        n = int(input("Number of years: "))
        years = list(map(int, input("Years separated by one space: ").strip().split()))[
            :n
        ]
        # years = [2006, 2007, 2008]

        print(" \n ")
        print("Comparing data... \n ")

        for year in years:
            year_plot = comparison.copy()
            ini = str(year) + "-01-01 00:00:00"
            end = str(year) + "-12-31 23:00:00"
            year_plot = year_plot.loc[ini:end]
            fig, axs = plt.subplots(3, 1, figsize=(20, 15), sharex=True)
            fig.subplots_adjust(hspace=0.05, wspace=0.1)
            fig.suptitle(
                "Year: "
                + str(year)
                + ", "
                + buoy_name
                + " buoy comparison with "
                + comparison_type.upper()
                + " CSIRO",
                fontsize=24,
                y=0.94,
                fontweight="bold",
            )
            months = [
                "                        Jan",
                "                        Feb",
                "                        Mar",
                "                        Apr",
                "                        May",
                "                        Jun",
                "                        Jul",
                "                        Aug",
                "                        Sep",
                "                        Oct",
                "                        Nov",
                "                        Dec",
            ]
            labels = ["$H_S$ [m]", "$T_P$ [s]", "$\u03b8_{m}$ [$\degree$]"]

            i = 0
            while i < 3:
                if i == 2:
                    axs[i].plot(
                        year_plot[year_plot.columns.values[i]],
                        ".",
                        markersize=1,
                        color="darkblue",
                    )
                    axs[i].plot(
                        year_plot[year_plot.columns.values[i + 3]],
                        ".",
                        markersize=1,
                        color="red",
                    )
                    axs[i].set_ylabel(labels[i], fontsize=12, fontweight="bold")
                    axs[i].grid()
                    axs[i].set_xlim(ini, end)
                    axs[i].set_xticks(
                        np.arange(
                            pd.to_datetime(ini), pd.to_datetime(end), td(days=30.5)
                        )
                    )
                    axs[i].tick_params(direction="in")
                    axs[i].set_xticklabels(months, fontsize=12, fontweight="bold")
                else:
                    axs[i].plot(
                        year_plot[year_plot.columns.values[i]],
                        color="darkblue",
                        linewidth=1,
                    )
                    axs[i].plot(
                        year_plot[year_plot.columns.values[i + 3]],
                        color="red",
                        linewidth=1,
                    )
                    axs[i].set_ylabel(labels[i], fontsize=12, fontweight="bold")
                    axs[i].grid()
                    axs[i].tick_params(direction="in")
                fig.legend(["Buoy", "Modelo"], loc=(0.66, 0.04), ncol=3, fontsize=14)
                i += 1

        # show results
        plt.show()

    def buoy_validation(self, validation_type, buoy_name="Buoy NAME"):
        """Validate data with buoy, even if it's been corrected or not
        ------------
        Parameters
        validation_type: Type of comparison to be done
                         (raw, sat_corr, buoy_corr)
        ------------
        Returns
        Different auto explicative plots
        """

        # Initialize the data to validate
        if validation_type == "raw":
            validation = pd.concat([self.buoy, self.hindcast], axis=1)
            title = "No previous correction"
        elif validation_type == "sat_corr":
            validation = pd.concat([self.buoy, self.hindcast_sat_corr], axis=1)
            title = "Previosly corrected with satellite data"
        elif validation_type == "buoy_corr":
            validation = pd.concat([self.buoy, self.hindcast_buoy_corr], axis=1)
            title = "Previosly corrected with buoy data"
        else:
            return "Not a valid value for validation_type"

        print("--------------------------------------------------------")
        print(validation_type.upper() + " VALIDATION will be performed")
        print("-------------------------------------------------------- \n ")

        validation = validation[["Hs_Buoy", "Tp_Buoy", "Dir_Buoy", "Hs", "Tp", "DirM"]]
        validation = validation.dropna(axis=0, how="any")

        print("Validating and plotting validated data... \n ")
        print("Length of data to validate: " + str(len(validation)) + " \n ")

        fig, axs = plt.subplots(2, 3, figsize=(20, 20))
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        fig.suptitle(
            "Hindcast: CSIRO" + ", " + buoy_name + " buoy validation \n " + title,
            fontsize=24,
            y=0.98,
            fontweight="bold",
        )

        for i in range(2):
            for j in range(3):
                if i == j == 0 or i == 1 and j == 0:
                    if i == 0:
                        x, y = validation["Hs_Buoy"], validation["Hs"]
                        title = "$H_S$ [m]"
                    else:
                        x, y = validation["Tp_Buoy"], validation["Tp"]
                        title = "$T_P$ [s]"

                    xy = np.vstack([x, y])
                    z = gaussian_kde(xy)(xy)
                    idx = z.argsort()
                    x2, y2, z = x[idx], y[idx], z[idx]
                    axs[i, j].scatter(x2, y2, c=z, s=1, cmap=cmocean.cm.haline)
                    axs[i, j].set_xlabel("Buoy", fontsize=12, fontweight="bold")
                    axs[i, j].set_ylabel("Model", fontsize=12, fontweight="bold")
                    axs[i, j].set_title(title, fontsize=12, fontweight="bold")
                    maxt = np.ceil(max(max(x) + 0.1, max(y) + 0.1))
                    axs[i, j].set_xlim(0, maxt)
                    axs[i, j].set_ylim(0, maxt)
                    axs[i, j].plot([0, maxt], [0, maxt], "-k", linewidth=0.6)
                    axs[i, j].set_xticks(np.linspace(0, maxt, 5))
                    axs[i, j].set_yticks(np.linspace(0, maxt, 5))
                    axs[i, j].set_aspect("equal")
                    xq = stats.probplot(x2, dist="norm")
                    yq = stats.probplot(y2, dist="norm")
                    axs[i, j].plot(
                        xq[0][1],
                        yq[0][1],
                        "o",
                        markersize=1,
                        color="k",
                        label="Q-Q plot",
                    )
                    mse = mean_squared_error(x2, y2)
                    rmse_e = rmse(x2, y2)
                    BIAS = bias(x2, y2)
                    SI = si(x2, y2)
                    label = "\n".join(
                        (
                            r"RMSE = %.2f" % (rmse_e,),
                            r"mse =  %.2f" % (mse,),
                            r"BIAS = %.2f" % (BIAS,),
                            R"SI = %.2f" % (SI,),
                        )
                    )
                    axs[i, j].text(0.7, 0.05, label, transform=axs[i, j].transAxes)

                elif i == 0 and j == 1 or i == 0 and j == 2:
                    idx_buoy = validation["Tp_Buoy"].argsort()
                    idx_hind = validation["Tp"].argsort()
                    if j == 1:
                        x, y = (
                            validation["Dir_Buoy"][idx_buoy],
                            validation["Hs_Buoy"][idx_buoy],
                        )
                        index = 2
                        c = validation["Tp_Buoy"][idx_buoy]
                        title = "Boya"
                    else:
                        x, y = validation["DirM"][idx_hind], validation["Hs"][idx_hind]
                        index = 3
                        c = validation["Tp"][idx_hind]
                        title = "Modelo"
                    x = (x * np.pi) / 180
                    axs[i, j].axis("off")
                    axs[i, j] = fig.add_subplot(2, 3, index, projection="polar")
                    c = axs[i, j].scatter(x, y, c=c, s=5, cmap="magma_r", alpha=0.75)
                    cbar = plt.colorbar(c, pad=0.1)
                    cbar.ax.set_ylabel("$T_P$ [s]", fontsize=12, fontweight="bold")
                    axs[i, j].set_theta_zero_location("N", offset=0)
                    axs[i, j].set_xticklabels(
                        ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                    )
                    axs[i, j].set_theta_direction(-1)
                    axs[i, j].set_xlabel(
                        "$\u03b8_{m}$ [$\degree$]", fontsize=12, fontweight="bold"
                    )
                    axs[i, j].set_ylabel(
                        "$H_S$ [m]", labelpad=20, fontsize=12, fontweight="bold"
                    )
                    axs[i, j].set_title(title, pad=15, fontsize=12, fontweight="bold")

                else:
                    if j == 1:
                        x, y = validation["Tp_Buoy"], validation["Hs_Buoy"]
                        c = "darkblue"
                        title = "Boya"
                    else:
                        x, y = validation["Tp"], validation["Hs"]
                        c = "red"
                        title = "Modelo"
                    xy = np.vstack([x, y])
                    z = gaussian_kde(xy)(xy)
                    idx = z.argsort()
                    x2, y2, z = x[idx], y[idx], z[idx]
                    axs[i, j].scatter(x2, y2, c=z, s=3, cmap="Blues_r")
                    axs[i, j].set_xlabel("$T_P$ [s]", fontsize=12, fontweight="bold")
                    axs[i, j].set_ylabel("$H_S$ [m]", fontsize=12, fontweight="bold")
                    axs[i, j].set_title(title, fontsize=12, fontweight="bold")
                    axs[i, j].set_xlim(0, 20)
                    axs[i, j].set_ylim(0, 7.5)

        # show results
        plt.show()
