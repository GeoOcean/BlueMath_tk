from typing import Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from sklearn import linear_model

from ..core.models import BlueMathModel
from ..core.plotting.scatter import validation_scatter


def create_vec_direc(waves: np.ndarray, direcs: np.ndarray) -> np.ndarray:
    """
    Creates a vector of wave heights for each directional bin.
    TODO: check if this is correct!

    Parameters
    ----------
    waves : np.ndarray
        Wave heights.
    direcs : np.ndarray
        Wave directions in degrees.

    Returns
    -------
    np.ndarray
        Matrix of wave heights for each directional bin.
    """

    data = np.zeros((len(waves), 16))
    for i in range(len(waves)):
        if ((i / len(waves)) * 100) % 5 == 0:
            print(f"{str((i / len(waves)) * 100)}% completed...")
        if direcs[i] < 0:
            direcs[i] = direcs[i] + 360
        if direcs[i] > 0 and waves[i] > 0:
            bin_idx = int(direcs[i] / 22.5)
            data[i, bin_idx] = waves[i]

    return data


def get_matching_times(
    times1: np.ndarray,
    times2: np.ndarray,
    min_time: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds matching time indices between two arrays of timestamps.
    For each time in times1, finds the closest time in times2 that is within min_time hours.
    Returns the indices of matching times in both arrays.

    Parameters
    ----------
    times1 : np.ndarray
        First array of timestamps (reference times).
    times2 : np.ndarray
        Second array of timestamps (times to match against).
    min_time : int, optional
        Maximum time difference in hours for considering times as matching.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two arrays containing the indices of matching times:
        - First array: indices in times1 that have matches
        - Second array: corresponding indices in times2 that match
    """

    indices1 = np.array([], dtype=int)
    indices2 = np.array([], dtype=int)

    for i in range(len(times1)):
        # Find minimum time difference for current time1
        time_diffs = np.abs(times2 - times1[i])
        min_diff = np.min(time_diffs)

        # If minimum difference is within threshold, record the indices
        if min_diff < np.timedelta64(min_time, "h"):
            min_index = np.argmin(time_diffs)
            indices1 = np.append(indices1, i)
            indices2 = np.append(indices2, min_index)

    return indices1, indices2


def process_imos_satellite_data(
    satellite_data: xr.Dataset,
    ini_lat: float,
    end_lat: float,
    ini_lon: float,
    end_lon: float,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Processes IMOS satellite data for calibration.

    Parameters
    ----------
    satellite_data : xr.Dataset
        IMOS satellite dataset. This is the output of XXX.
    ini_lat : float
        South latitude of the satellite box.
    end_lat : float
        North latitude of the satellite box.
    ini_lon : float
        West longitude of the satellite box.
    end_lon : float
        East longitude of the satellite box.

    Returns
    -------
    pd.DataFrame
        The processed IMOS satellite data into pd.DataFrame.
    """

    # Filter satellite data by coordinates
    satellite_data = satellite_data[
        (satellite_data.LATITUDE > ini_lat)
        & (satellite_data.LATITUDE < end_lat)
        & (satellite_data.LONGITUDE > ini_lon)
        & (satellite_data.LONGITUDE < end_lon)
    ]

    # Process quality control
    wave_height_qlt = np.nansum(
        np.concatenate(
            (
                satellite_data["SWH_KU_quality_control"].values[:, np.newaxis],
                satellite_data["SWH_KA_quality_control"].values[:, np.newaxis],
            ),
            axis=1,
        ),
        axis=1,
    )
    good_qlt = np.where(wave_height_qlt < 1.5)

    # Process wave heights
    wave_height_cal = np.nansum(
        np.concatenate(
            (
                satellite_data["SWH_KU_CAL"].values[:, np.newaxis],
                satellite_data["SWH_KA_CAL"].values[:, np.newaxis],
            ),
            axis=1,
        ),
        axis=1,
    )
    wave_height_cal = wave_height_cal[good_qlt]

    return (
        satellite_data.isel(TIME=good_qlt)[["SWH_KU_CAL", "SWH_KA_CAL"]]
        .to_dataframe()
        .reset_index()
    )


class CalVal(BlueMathModel):
    """
    Calibrates wave data using reference data.
    The calibration can be validated with additional data if available.

    Attributes
    ----------
    data : pd.DataFrame
        Original data to be calibrated.
    data_to_calibrate : pd.DataFrame
        Data used for calibration.
    data_to_validate : pd.DataFrame
        Data used for validation (optional).
    n_parts : int
        Number of partitions in the wave data.
    longitude : float
        Longitude of the data point.
    latitude : float
        Latitude of the data point.
    validation_longitude : float
        Longitude of the validation point (if available).
    validation_latitude : float
        Latitude of the validation point (if available).
    calibrated_data : pd.DataFrame
        Data after calibration.
    calibration_params : np.ndarray
        Calibration parameters.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        data_to_calibrate: pd.DataFrame,
        data_to_validate: pd.DataFrame = None,
        n_parts: int = 1,
        longitude: float = None,
        latitude: float = None,
        validation_longitude: float = None,
        validation_latitude: float = None,
    ) -> None:
        """
        Initialize the CalVal class.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be calibrated.
        data_to_calibrate : pd.DataFrame
            Data used for calibration.
        data_to_validate : pd.DataFrame, optional
            Data used for validation (if available).
        n_parts : int, optional
            Number of partitions in the wave data.
        longitude : float, optional
            Longitude of the data point.
        latitude : float, optional
            Latitude of the data point.
        validation_longitude : float, optional
            Longitude of the validation point (if available).
        validation_latitude : float, optional
            Latitude of the validation point (if available).
        """

        super().__init__()
        self.set_logger_name(name="CalVal", level="INFO", console=True)

        # Save input data
        self._data = data.copy()
        self._data_to_calibrate = data_to_calibrate.copy()
        self._data_to_validate = (
            data_to_validate.copy() if data_to_validate is not None else None
        )

        # Save parameters
        self.n_parts = n_parts
        self.longitude = longitude
        self.latitude = latitude
        self.validation_longitude = validation_longitude
        self.validation_latitude = validation_latitude

        # Initialize calibration results
        self._calibrated_data = None
        self._calibration_params = None

        # Plot data domains
        self._plot_data_domains()

        # Exclude large attributes from model saving
        self._exclude_attributes += [
            "_data",
            "_data_to_calibrate",
            "_data_to_validate",
        ]

    @property
    def calibrated_data(self) -> pd.DataFrame:
        """Returns the calibrated data."""
        return self._calibrated_data

    @property
    def calibration_params(self) -> np.ndarray:
        """Returns the calibration parameters."""
        return self._calibration_params

    def _plot_data_domains(self) -> None:
        """
        Plots the domains of the data points.
        """

        fig, ax = plt.subplots(
            figsize=(10, 10),
            subplot_kw={
                "projection": ccrs.PlateCarree(central_longitude=self.longitude)
            },
        )

        land_10m = cfeature.NaturalEarthFeature(
            "physical",
            "land",
            "10m",
            edgecolor="face",
            facecolor=cfeature.COLORS["land"],
        )

        # Plot calibration data
        ax.scatter(
            self._data_to_calibrate.LONGITUDE,
            self._data_to_calibrate.LATITUDE,
            s=0.01,
            c="k",
            transform=ccrs.PlateCarree(),
        )

        # Plot main data point
        ax.scatter(
            self.longitude,
            self.latitude,
            s=50,
            c="red",
            zorder=10,
            transform=ccrs.PlateCarree(),
        )

        # Plot validation point if available
        if (
            self.validation_longitude is not None
            and self.validation_latitude is not None
        ):
            ax.scatter(
                self.validation_longitude,
                self.validation_latitude,
                s=50,
                c="orange",
                zorder=10,
                transform=ccrs.PlateCarree(),
            )

        # Set plot extent
        ax.set_extent(
            [
                self.longitude - 4,
                self.longitude + 4,
                self.latitude - 4,
                self.latitude + 2,
            ]
        )

        ax.add_feature(land_10m)
        plt.show()

    def calibrate(
        self,
        calibration_type: str = "primary",
        min_time: int = 2,
        type_calib_way: bool = False,
        th_ne: float = 0.1,
    ) -> np.ndarray:
        """
        Calibrates the data using reference data.

        Parameters
        ----------
        calibration_type : str, optional
            Type of calibration ('primary' or 'validation').
        min_time : int, optional
            Minimum time difference in hours for matching.
        type_calib_way : bool, optional
            Whether to calibrate data for each reference point (True) or vice versa (False).
        th_ne : float, optional
            Threshold for minimum wave height to calibrate.

        Returns
        -------
        np.ndarray
            Calibration parameters.
        """

        # Construct matrices for calibration
        self.logger.info("Constructing matrices and calibrating...")

        # Process sea waves
        Hsea = create_vec_direc(self._data["Hsea"], self._data["Dirsea"])

        # Process swells
        Hs_swells = np.zeros(Hsea.shape)
        for part in range(1, self.n_parts):
            Hs_swells += (
                create_vec_direc(
                    self._data[f"Hswell{part}"],
                    self._data[f"Dirswell{part}"],
                )
            ) ** 2

        # Combine sea and swell matrices
        Hs_ncorr_mat = np.concatenate([Hsea**2, Hs_swells], axis=1)
        Hs_ncorr = np.sqrt(np.sum(Hs_ncorr_mat, axis=1))

        # Perform calibration
        nedata = np.where(np.mean(Hs_ncorr_mat, axis=0) < th_ne)[0]
        reg = linear_model.LinearRegression()
        hs_calibrate_2 = self._data_to_calibrate["Hs"] ** 2
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

        # Save calibration results
        self._calibration_params = params
        self._calibrated_data = self._data.copy()

        # Apply calibration to all data
        self._apply_calibration(params)

        # Plot calibration results
        self._plot_calibration_results(
            Hs_ncorr,
            Hs_corr,
            self._data_to_calibrate["Hs"],
            self._data,
            params,
            calibration_type,
        )

        return params

    def _plot_calibration_results(
        self,
        xx1: np.ndarray,
        xx2: np.ndarray,
        hs: np.ndarray,
        data: pd.DataFrame,
        coefs: np.ndarray,
        big_title: str,
    ) -> None:
        """
        Plots calibration results.

        Parameters
        ----------
        xx1 : np.ndarray
            Not corrected data.
        xx2 : np.ndarray
            Corrected data.
        hs : np.ndarray
            Reference data used to calibrate.
        data : pd.DataFrame
            Dataframe with more wave information.
        coefs : np.ndarray
            Parameters calculated in the calibration.
        big_title : str
            Title for the plot.
        """
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        fig.subplots_adjust(hspace=0.4, wspace=0.1)
        fig.suptitle(
            f"Wave data calibration with {big_title} data",
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

                    validation_scatter(axs[i, j], x, y, big_title, "Data", title)

                elif i == 0 and j == 1 or i == 0 and j == 2:
                    if j == 1:
                        dataj1 = data[["Dirsea", "Hsea"]].dropna(axis=0, how="any")
                        x, y = dataj1["Dirsea"], dataj1["Hsea"]
                        index = 2
                        title = "SEA $Wave$ $Climate$"
                    else:
                        dataj2 = data[["Dirswell1", "Hswell1"]].dropna(
                            axis=0, how="any"
                        )
                        x, y = dataj2["Dirswell1"], dataj2["Hswell1"]
                        index = 3
                        title = "SWELL 1 $Wave$ $Climate$"

                    x = (x * np.pi) / 180
                    axs[i, j].axis("off")
                    axs[i, j] = fig.add_subplot(2, 3, index, projection="polar")
                    x2, y2, z = self._density_scatter(x, y)
                    axs[i, j].scatter(x2, y2, c=z, s=3, cmap="jet")
                    axs[i, j].set_theta_zero_location("N", offset=0)
                    axs[i, j].set_xticklabels(
                        ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                    )
                    axs[i, j].xaxis.grid(True, color="lavender", linestyle="-")
                    axs[i, j].yaxis.grid(True, color="lavender", linestyle="-")
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
                    axs[i, j].set_title(title, fontweight="bold")

                    if j == 2:
                        ax1_divider = make_axes_locatable(axs[i, j])
                        cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
                        cb = mpl.colorbar.ColorbarBase(cax1, cmap=my_cmap, norm=my_norm)
                        cb.set_label("Correction Coefficients")
                        cb.outline.set_color("white")

        plt.show()
