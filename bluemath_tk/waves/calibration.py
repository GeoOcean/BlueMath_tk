from typing import Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.decorators import validate_data_calval
from ..core.models import BlueMathModel


def get_matching_times_between_arrays(
    times1: np.ndarray,
    times2: np.ndarray,
    min_time_diff: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds matching time indices between two arrays of timestamps.
    For each time in times1, finds the closest time in times2 that is within min_time_diff hours.
    Returns the indices of matching times in both arrays.

    Parameters
    ----------
    times1 : np.ndarray
        First array of timestamps (reference times).
    times2 : np.ndarray
        Second array of timestamps (times to match against).
    min_time_diff : int, optional
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
        if min_diff < np.timedelta64(min_time_diff, "h"):
            min_index = np.argmin(time_diffs)
            indices1 = np.append(indices1, i)
            indices2 = np.append(indices2, min_index)

    return indices1, indices2


def process_imos_satellite_data(
    satellite_df: pd.DataFrame,
    ini_lat: float,
    end_lat: float,
    ini_lon: float,
    end_lon: float,
    depth_threshold: float = -200,
) -> pd.DataFrame:
    """
    Processes IMOS satellite data for calibration.

    Parameters
    ----------
    satellite_df : pd.DataFrame
        IMOS satellite data as a pd.DataFrame. Must contain the following columns:
        - LATITUDE
        - LONGITUDE
        - SWH_KU_quality_control
        - SWH_KA_quality_control
        - SWH_KU_CAL
        - SWH_KA_CAL
        - BOT_DEPTH
    ini_lat : float
        South latitude to filter the satellite data.
    end_lat : float
        North latitude to filter the satellite data.
    ini_lon : float
        West longitude to filter the satellite data.
    end_lon : float
        East longitude to filter the satellite data.
    depth_threshold : float, optional
        Depth threshold to filter the satellite data.

    Returns
    -------
    pd.DataFrame
        The processed IMOS satellite data into pd.DataFrame.
    """

    # Filter satellite data by coordinates
    satellite_df = satellite_df[
        (satellite_df.LATITUDE > ini_lat)
        & (satellite_df.LATITUDE < end_lat)
        & (satellite_df.LONGITUDE > ini_lon)
        & (satellite_df.LONGITUDE < end_lon)
        & (satellite_df.BOT_DEPTH < depth_threshold)
    ]

    # Process quality control
    wave_height_qlt = np.nansum(
        np.concatenate(
            (
                satellite_df["SWH_KU_quality_control"].values[:, np.newaxis],
                satellite_df["SWH_KA_quality_control"].values[:, np.newaxis],
            ),
            axis=1,
        ),
        axis=1,
    )
    good_qlt = np.where(wave_height_qlt < 1.5)

    # Process wave heights
    satellite_df["Hs_CAL"] = np.nansum(
        np.concatenate(
            (
                satellite_df["SWH_KU_CAL"].values[:, np.newaxis],
                satellite_df["SWH_KA_CAL"].values[:, np.newaxis],
            ),
            axis=1,
        ),
        axis=1,
    )

    return satellite_df.iloc[good_qlt]


class CalVal(BlueMathModel):
    """
    Calibrates wave data using reference data.
    """

    direction_bin_size: int = 22.5
    direction_bins: np.ndarray = np.arange(
        direction_bin_size, 360.5, direction_bin_size
    )

    def __init__(self) -> None:
        """
        Initialize the CalVal class.
        """

        super().__init__()
        self.set_logger_name(name="CalVal", level="INFO", console=True)

        # Save input data
        self._data: pd.DataFrame = None
        self._data_longitude: float = None
        self._data_latitude: float = None
        self._data_to_calibrate: pd.DataFrame = None
        self._min_time_diff: int = None

        # Initialize calibration results
        self._data_to_fit: Tuple[pd.DataFrame, pd.DataFrame] = (None, None)
        self._calibration_model: sm.OLS = None
        self._calibrated_data: pd.DataFrame = None
        self._calibration_params: pd.Series = None

        # Exclude large attributes from model saving
        self._exclude_attributes += [
            "_data",
            "_data_to_calibrate",
            "_data_to_fit",
        ]

    @property
    def calibration_model(self) -> sm.OLS:
        """Returns the calibration model."""

        if self._calibration_model is None:
            raise ValueError(
                "Calibration model is not available. Please run the fit method first."
            )

        return self._calibration_model

    @property
    def calibrated_data(self) -> pd.DataFrame:
        """Returns the calibrated data."""

        if self._calibrated_data is None:
            raise ValueError(
                "Calibrated data is not available. Please run the fit method first."
            )

        return self._calibrated_data

    @property
    def calibration_params(self) -> pd.Series:
        """Returns the calibration parameters."""

        if self._calibration_params is None:
            raise ValueError(
                "Calibration parameters are not available. Please run the fit method first."
            )

        return self._calibration_params

    def plot_calibration_results(self) -> Tuple[Figure, Axes]:
        """
        Plots the calibration results.
        """

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        sea_correction = self.calibration_params["sea_correction"].values()
        swell_correction = self.calibration_params["swell_correction"].values()

        for ax, coeffs, title in zip(
            axs[0, :],
            [sea_correction, swell_correction],
            ["SEA $Correction$", "SWELL 1 $Correction$"],
        ):
            norm = 0.3
            fracs = np.repeat(10, len(coeffs))
            my_norm = mpl.colors.Normalize(1 - norm, 1 + norm)
            my_cmap = mpl.cm.get_cmap("bwr", len(coeffs))
            ax.pie(
                fracs,
                labels=None,
                colors=my_cmap(my_norm(coeffs)),
                startangle=90,
                counterclock=False,
                radius=1.2,
            )
            ax.set_title(title, fontweight="bold")

        return fig, axs

    def _plot_data_domains(self) -> Tuple[Figure, Axes]:
        """
        Plots the domains of the data points.
        """

        fig, ax = plt.subplots(
            figsize=(10, 10),
            subplot_kw={
                "projection": ccrs.PlateCarree(central_longitude=self._data_longitude)
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
            self._data_longitude,
            self._data_latitude,
            s=50,
            c="red",
            zorder=10,
            transform=ccrs.PlateCarree(),
        )

        # Set plot extent
        ax.set_extent(
            [
                self._data_longitude - 2,
                self._data_longitude + 2,
                self._data_latitude - 2,
                self._data_latitude + 2,
            ]
        )
        ax.add_feature(land_10m)

        return fig, ax

    def _create_vec_direc(self, waves: np.ndarray, direcs: np.ndarray) -> np.ndarray:
        """
        Creates a vector of wave heights for each directional bin.

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

        data = np.zeros((len(waves), len(self.direction_bins)))
        for i in range(len(waves)):
            if direcs[i] < 0:
                direcs[i] = direcs[i] + 360
            if direcs[i] > 0 and waves[i] > 0:
                bin_idx = int(direcs[i] / self.direction_bin_size)
                data[i, bin_idx] = waves[i]

        return data

    @staticmethod
    def _get_nparts(data: pd.DataFrame) -> int:
        """
        Gets the number of parts in the wave data.

        Parameters
        ----------
        data : pd.DataFrame
            Wave data.

        Returns
        -------
        int
            The number of parts in the wave data.
        """

        return len([col for col in data.columns if col.startswith("Hswell")])

    def _get_joined_sea_swell_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Joins the sea and swell data.

        Parameters
        ----------
        data : pd.DataFrame
            Wave data.

        Returns
        -------
        np.ndarray
            The joined sea and swell matrix.
        """

        # Process sea waves
        Hsea = self._create_vec_direc(data["Hsea"], data["Dirsea"])

        # Process swells
        Hs_swells = np.zeros(Hsea.shape)
        for part in range(1, self._get_nparts(data) + 1):
            Hs_swells += (
                self._create_vec_direc(data[f"Hswell{part}"], data[f"Dirswell{part}"])
            ) ** 2

        # Combine sea and swell matrices
        sea_swell_matrix = np.concatenate([Hsea**2, Hs_swells], axis=1)

        return sea_swell_matrix

    @validate_data_calval
    def fit(
        self,
        data: pd.DataFrame,
        data_longitude: float,
        data_latitude: float,
        data_to_calibrate: pd.DataFrame,
        min_time_diff: int = 2,
    ) -> None:
        """
        Calibrates the data using reference data.
        """

        # Save input data
        self._data = data.copy()
        self._data_longitude = data_longitude
        self._data_latitude = data_latitude
        self._data_to_calibrate = data_to_calibrate.copy()
        self._min_time_diff = min_time_diff

        # Plot data domains
        self._plot_data_domains()

        # Construct matrices for calibration
        self.logger.info("Constructing matrices and calibrating...")

        # Get matching times
        times_data_to_fit, times_data_to_calibrate = get_matching_times_between_arrays(
            self._data.index.values,
            self._data_to_calibrate.index.values,
            min_time_diff=self._min_time_diff,
        )
        self._data_to_fit = (
            self._data.iloc[times_data_to_fit],
            self._data_to_calibrate.iloc[times_data_to_calibrate],
        )

        # Get joined sea and swell data
        sea_swell_matrix = self._get_joined_sea_swell_data(self._data_to_fit[0])

        # Perform calibration
        X = sm.add_constant(sea_swell_matrix)
        self._calibration_model = sm.OLS(self._data_to_fit[1]["Hs_CAL"] ** 2, X)
        calibrated_model_results = self._calibration_model.fit()

        # Get significant correction coefficients
        significant_model_params = [
            model_param
            if calibrated_model_results.pvalues[imp] < 0.05 and model_param > 0
            else 1.0
            for imp, model_param in enumerate(calibrated_model_results.params)
        ]

        # Save sea and swell correction coefficients
        self._calibration_params = {
            "sea_correction": {
                ip: param
                for ip, param in enumerate(
                    np.sqrt(significant_model_params[: len(self.direction_bins)])
                )
            },
            "swell_correction": {
                ip: param
                for ip, param in enumerate(
                    np.sqrt(significant_model_params[len(self.direction_bins) :])
                )
            },
        }

    def predict(
        self, data: Union[pd.DataFrame, xr.Dataset]
    ) -> Union[pd.DataFrame, xr.Dataset]:
        """
        Predicts the wave heights using the calibration model.
        """

        if isinstance(data, xr.Dataset):
            self.logger.warning(
                "Spectra data detected. Correcting data by peak directions."
            )
            corrected_data = data.copy()  # Copy data to avoid modifying original data
            peak_directions = corrected_data.spec.stats(["dp"]).load()
            correction_coeffs = np.ones(peak_directions.dp.shape)
            for n_part in peak_directions.part:
                if n_part == 0:
                    correction_coeffs[n_part, :] = np.array(
                        [
                            self.calibration_params["sea_correction"][
                                int(peak_direction / self.direction_bin_size)
                            ]
                            for peak_direction in peak_directions.isel(
                                part=n_part
                            ).dp.values
                        ]
                    )
                else:
                    correction_coeffs[n_part, :] = np.array(
                        [
                            self.calibration_params["swell_correction"][
                                int(peak_direction / self.direction_bin_size)
                            ]
                            for peak_direction in peak_directions.isel(
                                part=n_part
                            ).dp.values
                        ]
                    )
            corrected_data["corr_coeffs"] = (("part", "time"), correction_coeffs)
            corrected_data["corr_efth"] = (
                corrected_data.efth * corrected_data.corr_coeffs
            )

            return corrected_data

        elif isinstance(data, pd.DataFrame):
            self.logger.warning("Wave data detected. Correcting data.")
            corrected_data = data.copy()
            corrected_data["Hsea"] = (
                corrected_data["Hsea"] ** 2
                * np.array(
                    [
                        self.calibration_params["sea_correction"][
                            int(peak_direction / self.direction_bin_size)
                        ]
                        for peak_direction in corrected_data["Dirsea"]
                    ]
                )
                ** 2
            )
            corrected_data["Hs_CORR"] = corrected_data["Hsea"]
            for n_part in range(1, self._get_nparts(corrected_data) + 1):
                corrected_data[f"Hswell{n_part}"] = (
                    corrected_data[f"Hswell{n_part}"] ** 2
                    * np.array(
                        [
                            self.calibration_params["swell_correction"][
                                int(peak_direction / self.direction_bin_size)
                            ]
                            for peak_direction in corrected_data[f"Dirswell{n_part}"]
                        ]
                    )
                    ** 2
                )
                corrected_data["Hs_CORR"] += corrected_data[f"Hswell{n_part}"]

            corrected_data["Hs_CORR"] = np.sqrt(corrected_data["Hs_CORR"])

            return corrected_data[["Hs", "Hs_CORR"]]
