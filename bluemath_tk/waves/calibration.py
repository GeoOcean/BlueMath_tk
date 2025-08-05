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
from ..core.plotting.scatter import density_scatter, validation_scatter


def get_matching_times_between_arrays(
    times1: np.ndarray,
    times2: np.ndarray,
    max_time_diff: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds matching time indices between two arrays of timestamps.

    For each time in `times1`, finds the closest time in `times2` that is within `max_time_diff` hours.
    Returns the indices of matching times in both arrays.

    Parameters
    ----------
    times1 : np.ndarray
        First array of timestamps (reference times, e.g., from model data).
    times2 : np.ndarray
        Second array of timestamps (e.g., from satellite or validation data).
    max_time_diff : int
        Maximum time difference in hours for considering times as matching.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two arrays containing the indices of matching times:
        - First array: indices in times1 that have matches
        - Second array: corresponding indices in times2 that match

    Example
    -------
    >>> idx1, idx2 = get_matching_times_between_arrays(
    ...     model_df.index.values,
    ...     sat_df.index.values,
    ...     max_time_diff=2,
    ... )
    """

    indices1 = np.array([], dtype=int)
    indices2 = np.array([], dtype=int)

    for i in range(len(times1)):
        # Find minimum time difference for current time1
        time_diffs = np.abs(times2 - times1[i])
        min_diff = np.min(time_diffs)

        # If minimum difference is within threshold, record the indices
        if min_diff < np.timedelta64(max_time_diff, "h"):
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

    This function filters and processes IMOS satellite altimeter data to be used as
    reference data for calibration (e.g., as `data_to_calibrate` in CalVal.fit).

    Parameters
    ----------
    satellite_df : pd.DataFrame
        IMOS satellite data. Must contain columns:
            - 'LATITUDE' (float): Latitude in decimal degrees
            - 'LONGITUDE' (float): Longitude in decimal degrees
            - 'SWH_KU_quality_control' (float): Quality control flag for Ku-band
            - 'SWH_KA_quality_control' (float): Quality control flag for Ka-band
            - 'SWH_KU_CAL' (float): Calibrated significant wave height (Ku-band)
            - 'SWH_KA_CAL' (float): Calibrated significant wave height (Ka-band)
            - 'BOT_DEPTH' (float): Bathymetry (negative values for ocean)
    ini_lat : float
        Minimum latitude (southern boundary) for filtering.
    end_lat : float
        Maximum latitude (northern boundary) for filtering.
    ini_lon : float
        Minimum longitude (western boundary) for filtering.
    end_lon : float
        Maximum longitude (eastern boundary) for filtering.
    depth_threshold : float, optional
        Only include points with BOT_DEPTH < depth_threshold. Default is -200.

    Returns
    -------
    pd.DataFrame
        Filtered and processed satellite data, suitable for use as `data_to_calibrate` in CalVal.fit.
        Includes a new column 'Hs_CAL' (combination of Ku-band and Ka-band calibrated significant wave heights).

    Notes
    -----
    - The returned DataFrame can be used directly as the `data_to_calibrate` argument in CalVal.fit.
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

    This class provides a framework for calibrating wave model outputs (e.g., hindcast or reanalysis)
    using reference data (e.g., satellite or buoy observations).
    It supports directionally-dependent calibration for both sea and swell components.

    Attributes
    ----------
    direction_bin_size : int
        Size of directional bins in degrees.
    direction_bins : np.ndarray
        Array of bin edges for directions.
    calibration_model : sm.OLS
        The calibration model, more details in `statsmodels.api.OLS`.
    calibrated_data : pd.DataFrame
        DataFrame with columns ['Hs', 'Hs_CORR', 'Hs_CAL'] after calibration.
        The time domain is the same as the model data.
    calibration_params : dict
        Dictionary with 'sea_correction' and 'swell_correction' correction coefficients.

    Examples
    --------
    .. jupyter-execute::

        import pandas as pd
        from bluemath_tk.waves.calibration import CalVal, process_imos_satellite_data

        # Load your model data (must have columns: 'Hs', 'Hsea', 'Dirsea', 'Hswell1', 'Dirswell1', ...)
        model_df = pd.read_csv('model_data.csv', index_col=0, parse_dates=True)

        # Load IMOS satellite data and process it for calibration
        sat_df = pd.read_csv('imos_satellite.csv', index_col=0, parse_dates=True)
        data_to_calibrate = process_imos_satellite_data(
            sat_df, ini_lat=-40, end_lat=-30, ini_lon=140, end_lon=150
        )

        # Initialize and fit the calibration
        cal = CalVal()
        cal.fit(
            data=model_df,
            data_longitude=145.0,
            data_latitude=-35.0,
            data_to_calibrate=data_to_calibrate,
            max_time_diff=2,
        )

        # Plot results
        cal.plot_calibration_results()

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
        self._max_time_diff: int = None

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

    def _plot_data_domains(self) -> Tuple[Figure, Axes]:
        """
        Plots the domains of the data points.

        Returns
        -------
        Tuple[Figure, Axes]
            A tuple containing the figure and axes objects.
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
        ax.set_facecolor("lightblue")
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
                # Handle direction = 360° case by mapping to the first bin (0-22.5°)
                if direcs[i] >= 360:
                    bin_idx = 0
                else:
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
        Hsea = self._create_vec_direc(data["Hsea"], data["Dirsea"]) ** 2

        # Process swells
        Hs_swells = np.zeros(Hsea.shape)
        for part in range(1, self._get_nparts(data) + 1):
            Hs_swells += (
                self._create_vec_direc(data[f"Hswell{part}"], data[f"Dirswell{part}"])
            ) ** 2

        # Combine sea and swell matrices
        sea_swell_matrix = np.concatenate([Hsea, Hs_swells], axis=1)

        return sea_swell_matrix

    @validate_data_calval
    def fit(
        self,
        data: pd.DataFrame,
        data_longitude: float,
        data_latitude: float,
        data_to_calibrate: pd.DataFrame,
        max_time_diff: int = 2,
    ) -> None:
        """
        Calibrate the model data using reference (calibration) data.

        This method matches the model data and calibration data in time,
        constructs directionally-binned sea and swell matrices,
        and fits a linear regression to obtain correction coefficients
        for each direction bin.

        Parameters
        ----------
        data : pd.DataFrame
            Model data to calibrate. Must contain columns:
                - 'Hs' (float): Significant wave height
                - 'Hsea' (float): Sea component significant wave height
                - 'Dirsea' (float): Sea component mean direction (degrees)
                - 'Hswell1', 'Dirswell1', ... (float): Swell components (at least one required)
            The index must be datetime-like.
        data_longitude : float
            Longitude of the model location (used for plotting and filtering).
        data_latitude : float
            Latitude of the model location (used for plotting and filtering).
        data_to_calibrate : pd.DataFrame
            Reference data for calibration. Must contain column:
                - 'Hs_CAL' (float): Calibrated significant wave height (e.g., from satellite)
            The index must be datetime-like.
        max_time_diff : int, optional
            Maximum time difference (in hours) allowed when matching model and calibration data.
            Default is 2.

        Notes
        -----
        - After calling this method, the calibration parameters are stored in `self.calibration_params`
        and the calibrated data is available in `self.calibrated_data`.
        - The calibration is directionally dependent, meaning it uses different correction coefficients
        for different wave directions.
        - The coefficients with p-values greater than 0.05 or negative values are set to 1.0,
        indicating no correction is applied for those directions.
        """

        self.logger.info("Starting calibration fit procedure.")

        # Save input data
        self._data = data.copy()
        self._data_longitude = data_longitude
        self._data_latitude = data_latitude
        self._data_to_calibrate = data_to_calibrate.copy()
        self._max_time_diff = max_time_diff

        # Plot data domains
        self.logger.info("Plotting data domains.")
        self._plot_data_domains()

        # Construct matrices for calibration
        self.logger.info("Matching times and constructing matrices for calibration.")

        # Get matching times
        times_data_to_fit, times_data_to_calibrate = get_matching_times_between_arrays(
            self._data.index.values,
            self._data_to_calibrate.index.values,
            max_time_diff=self._max_time_diff,
        )
        self._data_to_fit = (
            self._data.iloc[times_data_to_fit],
            self._data_to_calibrate.iloc[times_data_to_calibrate],
        )

        # Get joined sea and swell data
        sea_swell_matrix = self._get_joined_sea_swell_data(self._data_to_fit[0])

        # Perform calibration
        self.logger.info("Fitting OLS regression for calibration.")
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

        # Save calibrated data to be used in plot_calibration_results()
        self._calibrated_data = self.correct(self._data_to_fit[0])
        self._calibrated_data["Hs_CAL"] = self._data_to_fit[1]["Hs_CAL"].values

        self.logger.info("Calibration fit procedure completed.")

    def correct(
        self, data: Union[pd.DataFrame, xr.Dataset]
    ) -> Union[pd.DataFrame, xr.Dataset]:
        """
        Apply the calibration correction to new data.

        Parameters
        ----------
        data : pd.DataFrame or xr.Dataset
            Data to correct. If DataFrame, must contain columns:
                - 'Hs', 'Hsea', 'Dirsea', 'Hswell1', 'Dirswell1', ...
            If xarray.Dataset, must have variable 'efth' and dimension 'part'.

        Returns
        -------
        pd.DataFrame or xr.Dataset
            Corrected data. For DataFrame, returns columns ['Hs', 'Hs_CORR'] (original and corrected SWH).
            For Dataset, adds variables 'corr_coeffs' and 'corr_efth'.

        Notes
        -----
        - The correction is directionally dependent and uses the coefficients obtained from `fit`.
        """

        if self._calibration_params is None:
            raise ValueError(
                "Calibration parameters are not available. Run fit() first."
            )

        if isinstance(data, xr.Dataset):
            self.logger.info(
                "Input is xarray.Dataset. Applying correction to spectra data."
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
                                if peak_direction < 360
                                else 0  # TODO: Check if this with Javi
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
                                if peak_direction < 360
                                else 0  # TODO: Check if this with Javi
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
            self.logger.info("Spectra correction complete.")

            return corrected_data

        elif isinstance(data, pd.DataFrame):
            self.logger.info(
                "Input is pandas.DataFrame. Applying correction to wave data."
            )

            corrected_data = data.copy()
            corrected_data["Hsea"] = (
                corrected_data["Hsea"] ** 2
                * np.array(
                    [
                        self.calibration_params["sea_correction"][
                            int(peak_direction / self.direction_bin_size)
                            if peak_direction < 360
                            else 0
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
                                if peak_direction < 360
                                else 0
                            ]
                            for peak_direction in corrected_data[f"Dirswell{n_part}"]
                        ]
                    )
                    ** 2
                )
                corrected_data["Hs_CORR"] += corrected_data[f"Hswell{n_part}"]

            corrected_data["Hs_CORR"] = np.sqrt(corrected_data["Hs_CORR"])
            self.logger.info("Wave data correction complete.")

            return corrected_data[["Hs", "Hs_CORR"]]

    def plot_calibration_results(self) -> Tuple[Figure, list]:
        """
        Plot the calibration results, including:
        - Pie charts of correction coefficients for sea and swell
        - Scatter plots of model vs. reference (before and after correction)
        - Polar density plots of sea and swell wave climate

        Returns
        -------
        Tuple[Figure, list]
            The matplotlib Figure and a list of Axes objects for all subplots.
        """

        self.logger.info("Plotting calibration results.")

        fig = plt.figure(figsize=(10, 15))
        gs = fig.add_gridspec(8, 2, wspace=0.4, hspace=0.7)

        # Create subplots with proper projections
        ax1 = fig.add_subplot(gs[:2, 0])  # Sea correction pie
        ax2 = fig.add_subplot(gs[:2, 1])  # Swell correction pie
        ax1_cbar = fig.add_subplot(gs[2, 0])  # Sea correction colorbar
        ax2_cbar = fig.add_subplot(gs[2, 1])  # Swell correction colorbar
        ax3 = fig.add_subplot(gs[3:5, 0])  # No correction scatter
        ax4 = fig.add_subplot(gs[3:5, 1])  # With correction scatter
        ax5 = fig.add_subplot(gs[6:, 0], projection="polar")  # Sea climate
        ax6 = fig.add_subplot(gs[6:, 1], projection="polar")  # Swell climate

        # Plot sea correction pie chart
        sea_norm = 0.35  # Normalization factor for sea correction
        sea_fracs = np.repeat(10, len(self.calibration_params["sea_correction"]))
        sea_norm = mpl.colors.Normalize(1 - sea_norm, 1 + sea_norm)
        sea_cmap = mpl.cm.get_cmap(
            "bwr", len(self.calibration_params["sea_correction"])
        )
        sea_colors = sea_cmap(
            sea_norm(list(self.calibration_params["sea_correction"].values()))
        )
        ax1.pie(
            sea_fracs,
            labels=None,
            colors=sea_colors,
            startangle=90,
            counterclock=False,
            radius=1.2,
        )
        ax1.set_title("SEA $Correction$", fontweight="bold")
        # Add colorbar for sea correction below the pie chart, shrink it
        _sea_cbar = mpl.colorbar.ColorbarBase(
            ax1_cbar,
            cmap=sea_cmap,
            norm=sea_norm,
            orientation="horizontal",
            label="Correction Factor",
        )
        box = ax1_cbar.get_position()
        ax1_cbar.set_position(
            [
                box.x0 + 0.15 * box.width,
                box.y0 + 0.3 * box.height,
                0.7 * box.width,
                0.4 * box.height,
            ]
        )
        ax1_cbar.set_frame_on(False)
        ax1_cbar.tick_params(
            left=False, right=False, labelleft=False, labelbottom=True, bottom=True
        )

        # Plot swell correction pie chart
        swell_norm = 0.35  # Normalization factor for swell correction
        swell_fracs = np.repeat(10, len(self.calibration_params["swell_correction"]))
        swell_norm = mpl.colors.Normalize(1 - swell_norm, 1 + swell_norm)
        swell_cmap = mpl.cm.get_cmap(
            "bwr", len(self.calibration_params["swell_correction"])
        )
        swell_colors = swell_cmap(
            swell_norm(list(self.calibration_params["swell_correction"].values()))
        )
        ax2.pie(
            swell_fracs,
            labels=None,
            colors=swell_colors,
            startangle=90,
            counterclock=False,
            radius=1.2,
        )
        ax2.set_title("SWELL $Correction$", fontweight="bold")
        # Add colorbar for swell correction below the pie chart, shrink it
        _swell_cbar = mpl.colorbar.ColorbarBase(
            ax2_cbar,
            cmap=swell_cmap,
            norm=swell_norm,
            orientation="horizontal",
            label="Correction Factor",
        )
        box = ax2_cbar.get_position()
        ax2_cbar.set_position(
            [
                box.x0 + 0.15 * box.width,
                box.y0 + 0.3 * box.height,
                0.7 * box.width,
                0.4 * box.height,
            ]
        )
        ax2_cbar.set_frame_on(False)
        ax2_cbar.tick_params(
            left=False, right=False, labelleft=False, labelbottom=True, bottom=True
        )

        # Plot no correction scatter
        validation_scatter(
            axs=ax3,
            x=self.calibrated_data["Hs"].values,
            y=self.calibrated_data["Hs_CAL"].values,
            xlabel="Hindcast",
            ylabel="Satellite",
            title="No Correction",
        )

        # Plot with correction scatter
        validation_scatter(
            axs=ax4,
            x=self.calibrated_data["Hs_CORR"].values,
            y=self.calibrated_data["Hs_CAL"].values,
            xlabel="Hindcast",
            ylabel="Satellite",
            title="With Correction",
        )

        # Plot sea wave climate
        sea_dirs = self._data["Dirsea"].iloc[::10] * np.pi / 180
        sea_heights = self._data["Hsea"].iloc[::10]
        # Filter out NaN and infinite values
        valid_mask = np.isfinite(sea_dirs) & np.isfinite(sea_heights)
        sea_dirs_valid = sea_dirs[valid_mask]
        sea_heights_valid = sea_heights[valid_mask]

        if len(sea_dirs_valid) > 0:
            x, y, z = density_scatter(sea_dirs_valid, sea_heights_valid)
            ax5.scatter(x, y, c=z, s=3, cmap="jet")
        ax5.set_theta_zero_location("N", offset=0)
        ax5.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        ax5.xaxis.grid(True, color="lavender", linestyle="-")
        ax5.yaxis.grid(True, color="lavender", linestyle="-")
        ax5.set_theta_direction(-1)
        ax5.set_xlabel("$\u03b8_{m}$ ($\degree$)")
        ax5.set_ylabel("$H_{s}$ (m)", labelpad=20)
        ax5.set_title("SEA $Wave$ $Climate$", pad=35, fontweight="bold")

        # Plot swell wave climate
        swell_dirs = self._data["Dirswell1"].iloc[::10] * np.pi / 180
        swell_heights = self._data["Hswell1"].iloc[::10]
        # Filter out NaN and infinite values
        valid_mask = np.isfinite(swell_dirs) & np.isfinite(swell_heights)
        swell_dirs_valid = swell_dirs[valid_mask]
        swell_heights_valid = swell_heights[valid_mask]

        if len(swell_dirs_valid) > 0:
            x, y, z = density_scatter(swell_dirs_valid, swell_heights_valid)
            ax6.scatter(x, y, c=z, s=3, cmap="jet")
        ax6.set_theta_zero_location("N", offset=0)
        ax6.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        ax6.xaxis.grid(True, color="lavender", linestyle="-")
        ax6.yaxis.grid(True, color="lavender", linestyle="-")
        ax6.set_theta_direction(-1)
        ax6.set_xlabel("$\u03b8_{m}$ ($\degree$)")
        ax6.set_ylabel("$H_{s}$ (m)", labelpad=20)
        ax6.set_title("SWELL 1 $Wave$ $Climate$", pad=35, fontweight="bold")

        return fig, [ax1, ax2, ax1_cbar, ax2_cbar, ax3, ax4, ax5, ax6]

    def validate_calibration(
        self, data_to_validate: pd.DataFrame
    ) -> Tuple[Figure, list]:
        """
        Validate the calibration using independent validation data.

        This method compares the original and corrected model data to the validation data,
        both as time series and with scatter plots.

        Parameters
        ----------
        data_to_validate : pd.DataFrame
            Validation data. Must contain column:
                - 'Hs_VAL' (float): Validation significant wave height (e.g., from buoy)
            The index must be datetime-like.

        Returns
        -------
        Tuple[Figure, list]
            The matplotlib Figure and a list of Axes objects:
            [time series axis, scatter (no correction), scatter (corrected)].
        """

        if "Hs_VAL" not in data_to_validate.columns:
            raise ValueError("Validation data is missing required column: 'Hs_VAL'")

        data_corr = self.correct(data=self._data)
        data_times, data_to_validate_times = get_matching_times_between_arrays(
            times1=data_corr.index,
            times2=data_to_validate.index,
            max_time_diff=1,
        )

        # Create figure with a 2-row, 2-column grid, top row spans both columns
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 3], hspace=0.4, wspace=0.3)

        # Top row: time series plot (spans both columns)
        ax_ts = fig.add_subplot(gs[0, :])
        t = data_corr.index[data_times]
        ax_ts.plot(
            t,
            data_to_validate["Hs_VAL"].iloc[data_to_validate_times],
            label="Validation",
            color="k",
            lw=1.5,
        )
        ax_ts.plot(
            t,
            data_corr["Hs"].iloc[data_times],
            label="Model (No Correction)",
            color="tab:blue",
            alpha=0.7,
        )
        ax_ts.plot(
            t,
            data_corr["Hs_CORR"].iloc[data_times],
            label="Model (Corrected)",
            color="tab:orange",
            alpha=0.7,
        )
        ax_ts.set_ylabel("$H_s$ (m)")
        ax_ts.set_xlabel("Time")
        ax_ts.set_title("Time Series Comparison")
        ax_ts.legend(loc="upper right")
        ax_ts.grid(True, linestyle=":", alpha=0.5)

        # Bottom row: scatter plots
        ax_sc1 = fig.add_subplot(gs[1, 0])
        ax_sc2 = fig.add_subplot(gs[1, 1])
        validation_scatter(
            axs=ax_sc1,
            x=data_corr["Hs"].iloc[data_times].values,
            y=data_to_validate["Hs_VAL"].iloc[data_to_validate_times].values,
            xlabel="Model (No Correction)",
            ylabel="Validation",
            title="No Correction",
        )
        validation_scatter(
            axs=ax_sc2,
            x=data_corr["Hs_CORR"].iloc[data_times].values,
            y=data_to_validate["Hs_VAL"].iloc[data_to_validate_times].values,
            xlabel="Model (Corrected)",
            ylabel="Validation",
            title="With Correction",
        )

        return fig, [ax_ts, ax_sc1, ax_sc2]
