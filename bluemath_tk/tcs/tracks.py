from datetime import timedelta
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from numpy import polyfit

from ..config.paths import get_paths
from ..core.constants import EARTH_RADIUS
from ..core.geo import geodesic_distance, geodesic_distance_azimuth, shoot

PATHS = get_paths(verbose=True)

# Configuration dictionaries and constants for IBTrACS data
centers_config_params: Dict[str, Dict[str, Union[str, List[str], float]]] = {
    "USA": {
        "id": "usa",
        "basins": ["NA", "SA", "EP", "WP", "SP", "NI", "SI"],
        "windfac": 1,
        "color": "r",
    },
    "TOKYO": {
        "id": "tokyo",
        "basins": ["EP", "WP"],
        "windfac": 0.93,
        "color": "lightcoral",
    },
    "CMA": {
        "id": "cma",
        "basins": ["EP", "WP"],
        "windfac": 0.99,
        "color": "lime",
    },
    "HKO": {
        "id": "hko",
        "basins": ["EP", "WP"],
        "windfac": 0.93,
        "color": "gold",
    },
    "NEWDELHI": {
        "id": "newdelhi",
        "basins": ["WP", "NI"],
        "windfac": 0.99,
        "color": "m",
    },
    "REUNION": {
        "id": "reunion",
        "basins": ["SI"],
        "windfac": 0.93,
        "color": "magenta",
    },
    "BOM": {
        "id": "bom",
        "basins": ["SP", "SI"],
        "windfac": 0.93,
        "color": "b",
    },
    "WELLINGTON": {
        "id": "wellington",
        "basins": ["SP", "SI"],
        "windfac": 0.93,
        "color": "slategrey",
    },
    "NADI": {
        "id": "nadi",
        "basins": ["SP"],
        "windfac": 0.93,
        "color": "c",
    },
    "WMO": {
        "id": "wmo",
        "basins": ["NA", "SA", "EP", "WP", "SP", "NI", "SI"],
        "windfac": 1,
        "color": "k",
    },
    "FORECAST": {
        "id": "forecast",
        "basins": ["AL", "LS", "EP", "CP", "WP", "IO", "SH"],
        "windfac": 1,
        "color": "k",
    },
}

all_centers: List[str] = [
    "USA",
    "TOKYO",
    "CMA",
    "HKO",
    "NEWDELHI",
    "REUNION",
    "BOM",
    "WELLINGTON",
    "NADI",
    "WMO",
]

all_basins: List[str] = ["NA", "SA", "EP", "WP", "SP", "NI", "SI"]


def get_center_information(center: str = "WMO") -> Dict[str, Union[str, None]]:
    """
    Get the center information from the configuration parameters for IBTrACS data.

    Parameters
    ----------
    center : str, optional
        The name of the center to get information for. Default is "WMO".

    Returns
    -------
    Dict[str, Union[str, None]]
        A dictionary containing the center information with the following keys:
        - source: Center ID
        - basin: Basin identifier
        - time: Time variable name
        - longitude: Longitude variable name
        - latitude: Latitude variable name
        - pressure: Pressure variable name
        - maxwinds: Maximum winds variable name
        - rmw: Radius of maximum winds variable name (if available)
        - dist2land: Distance to land variable name

    Raises
    ------
    ValueError
        If the specified center is not found in the configuration parameters.

    Examples
    --------
    >>> info = get_center_information("USA")
    >>> info["source"]
    'usa'
    >>> info["rmw"]
    'usa_rmw'
    """

    if center not in centers_config_params:
        raise ValueError(
            f"Center {center} not found. Available centers: {list(centers_config_params.keys())}"
        )

    return {
        "source": centers_config_params[center]["id"],
        "basin": "basin",
        "time": "time",
        "longitude": "lon"
        if center == "WMO"
        else centers_config_params[center]["id"] + "_lon",
        "latitude": "lat"
        if center == "WMO"
        else centers_config_params[center]["id"] + "_lat",
        "pressure": centers_config_params[center]["id"] + "_pres",
        "maxwinds": centers_config_params[center]["id"] + "_wind",
        "rmw": centers_config_params[center]["id"] + "_rmw"
        if center in ["USA", "REUNION", "BOM"]
        else None,
        "dist2land": "dist2land",
    }


def check_and_plot_track_data(track_data: xr.Dataset) -> plt.Figure:
    """
    Check the track data for missing values and plot the track.

    Parameters
    ----------
    track_data : xr.Dataset
        The track data to check in IBTrACS format. Must contain variables
        for time, longitude, latitude, pressure, and maximum winds for
        each center.

    Returns
    -------
    plt.Figure
        Figure object containing four subplots:
        1. Storm coordinates (lon vs lat)
        2. Minimum central pressure time series
        3. Maximum sustained winds time series
        4. Radii of maximum winds time series (if available)

    Notes
    -----
    - Longitude values are converted to [0-360º] convention
    - Wind speeds are converted to 1-minute average using center-specific factors
    - Variables that are entirely NaN are omitted from plots
    """

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    for center in all_centers:
        # dictionary for IBTrACS center
        center_info = get_center_information(center=center)
        # get var time
        ytime = track_data[center_info["time"]].values
        # remove NaTs (time)
        st_lon = track_data[center_info["longitude"]].values[~np.isnat(ytime)]
        st_lat = track_data[center_info["latitude"]].values[~np.isnat(ytime)]
        st_prs = track_data[center_info["pressure"]].values[~np.isnat(ytime)]
        st_win = track_data[center_info["maxwinds"]].values[~np.isnat(ytime)]
        if center_info["rmw"]:
            st_rmw = track_data[center_info["rmw"]].values[~np.isnat(ytime)]
        st_tim = ytime[~np.isnat(ytime)]
        # longitude convention [0-360º]
        st_lon[st_lon < 0] += 360
        # winds are converted to 1-min avg
        st_win = st_win / centers_config_params[center]["windfac"]
        # plot center data (full NaN variables ommitted)
        color = centers_config_params[center]["color"]

        # Plot storm data
        axes[0].plot(st_lon, st_lat, "-", c=color, label=center)
        axes[0].set_title("Storm coordinates", fontweight="bold")
        axes[0].set_xlabel("Longitude (º)")
        axes[0].set_ylabel("Latitude (º)")
        if not np.isnan(st_prs).all():
            axes[1].plot(st_tim, st_prs, ".-", c=color, label=center)
            axes[1].set_title("Minimum central pressure [mbar]", fontweight="bold")
        if not np.isnan(st_win).all():
            axes[2].plot(st_tim, st_win, ".-", c=color, label=center)
            axes[2].set_title("Maximum sustained winds [kt]", fontweight="bold")
        if center_info["rmw"] and not np.isnan(st_rmw).all():
            axes[3].plot(st_tim, st_rmw, ".-", c=color, label=center)
            axes[3].set_title("Radii of max winds [nmile]", fontweight="bold")

    # plot attributes
    for i, ax in enumerate(axes):
        ax.legend(loc="upper left")
        if i > 0:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    return fig


def filter_track_by_basin(tracks_data: xr.Dataset, id_basin: str) -> xr.Dataset:
    """
    Filter the tracks data by basin.

    Parameters
    ----------
    tracks_data : xr.Dataset
        The tracks data to filter in IBTrACS format.
        Must contain a 'basin' variable.
    id_basin : str
        The basin ID to filter by (e.g., 'NA', 'SP', etc.)

    Returns
    -------
    xr.Dataset
        The filtered tracks data containing only storms from the specified basin

    Examples
    --------
    >>> sp_tracks = filter_track_by_basin(tracks_data, "SP")
    >>> print(sp_tracks.storm.size)
    """

    # TODO: check whether just [0, :] is needed, as it is the only one used
    return tracks_data.isel(
        storm=np.where(tracks_data.basin.values[0, :].astype(str) == id_basin)
    )


def ibtracs_fit_pmin_wmax(ibtracs_data: xr.Dataset = None, N: int = 3) -> xr.Dataset:
    """
    Generate polynomial fit coefficients for pressure-wind relationship.

    Parameters
    ----------
    ibtracs_data : xr.Dataset
        The IBTrACS dataset containing storm track data. Must include pressure
        and wind variables for each center.
    N : int, optional
        The order of the polynomial fit. Default is 3.

    Returns
    -------
    xr.Dataset
        A dataset containing:
        - coef_fit : Polynomial coefficients (center, basin, N+1)
        - pres_data : Pressure data used for fitting
        - wind_data : Wind data used for fitting

    Notes
    -----
    - Maximum wind speeds are converted to 1-min average for all RSMC centers
    - The polynomial fit is of the form: p = a₀ + a₁w + a₂w² + ... + aₙwⁿ
      where p is pressure and w is wind speed

    Examples
    --------
    >>> fit_data = ibtracs_fit_pmin_wmax(ibtracs_data, N=3)
    >>> print(fit_data.coef_fit.shape)  # (n_centers, n_basins, N+1)
    """

    try:
        return xr.open_dataset(PATHS["SHYTCWAVES_COEFS"])

    except Exception as e:
        print(f"File could not be opened: {PATHS['SHYTCWAVES_COEFS']}. Error: {e}")

        coef_fit = np.nan * np.zeros((len(all_centers), len(all_basins), N + 1))
        pres_data = np.nan * np.zeros((len(all_centers), len(all_basins), 200000))
        wind_data = np.nan * np.zeros((len(all_centers), len(all_basins), 200000))

        for ic, center in enumerate(all_centers):
            center_info = get_center_information(center=center)

            for basin in center_info["basin"]:
                # filter tracks data by basin
                filtered_tracks = filter_track_by_basin(
                    tracks_data=ibtracs_data, id_basin=basin
                )
                # extract and reshape basin data: pressure, wind, landbasin
                newshape = filtered_tracks.storm.size * filtered_tracks.date_time.size
                landbasin = filtered_tracks[center_info["dist2land"]].values.reshape(
                    newshape
                )
                Pbasin = filtered_tracks[center_info["pressure"]].values.reshape(
                    newshape
                )
                Wbasin = filtered_tracks[center_info["maxwinds"]].values.reshape(
                    newshape
                )
                # winds are converted to 1-min avg [kt]
                Wbasin /= center_info["windfac"]
                PWbasin_s = np.column_stack((Pbasin, Wbasin, landbasin))
                # index for removing NaNs (including landmask)
                ix_nonan = ~np.isnan(PWbasin_s).any(axis=1)
                PWbasin_s = PWbasin_s[ix_nonan]
                # Fitting Polynomial Regression to the dataset
                X, y = PWbasin_s[:, 0], PWbasin_s[:, 1]
                u = polyfit(X, y, deg=N)
                # store coefficients
                ibasin = np.where(basin == np.array(all_basins))[0][0]
                coef_fit[ic, ibasin, :] = u
                pres_data[ic, ibasin, : PWbasin_s.shape[0]] = PWbasin_s[:, 0].T
                wind_data[ic, ibasin, : PWbasin_s.shape[0]] = PWbasin_s[:, 1].T

        # dataset
        xds = xr.Dataset(
            {
                "coef": (("center", "basin", "polynomial"), coef_fit),
                "pres": (("center", "basin", "data"), pres_data),
                "wind": (("center", "basin", "data"), wind_data),
            },
            {
                "center": np.asarray(all_centers),
                "basin": np.asarray(all_basins),
            },
        )
        xds.coef.attrs["name"] = "Fitting polynomial coefficients (Pmin, Wmax)"
        xds.coef.attrs["units"] = "Pressure (mbar), Wind speed (kt, 1-min avg)"

        return xds


def get_category(ycpres: np.ndarray) -> np.ndarray:
    """
    Calculate storm category based on central pressure.

    Parameters
    ----------
    ycpres : np.ndarray
        Array of central pressures in millibars (mbar).

    Returns
    -------
    np.ndarray
        Array of storm categories:
        - 6: Missing data (pressure = 0 or NaN)
        - 5: Category 5 (pressure < 920 mbar)
        - 4: Category 4 (920 ≤ pressure < 944 mbar)
        - 3: Category 3 (944 ≤ pressure < 964 mbar)
        - 2: Category 2 (964 ≤ pressure < 979 mbar)
        - 1: Category 1 (979 ≤ pressure < 1000 mbar)
        - 0: Tropical Storm/Depression (pressure ≥ 1000 mbar)

    Notes
    -----
    Uses np.select for vectorized operations. Categories are based on the
    Saffir-Simpson Hurricane Wind Scale pressure thresholds.

    Examples
    --------
    >>> pressures = np.array([915, 950, 980, 1005, np.nan])
    >>> get_category(pressures)
    array([5, 3, 1, 0, 6])
    """

    conditions = [
        (ycpres == 0) | (np.isnan(ycpres)),
        (ycpres < 920),
        (ycpres < 944),
        (ycpres < 964),
        (ycpres < 979),
        (ycpres < 1000),
        (ycpres >= 1000),
    ]
    choices = [6, 5, 4, 3, 2, 1, 0]

    return np.select(conditions, choices)


def wind2rmw(wmax: np.ndarray, vmean: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """
    Calculate radius of maximum winds using Knaff et al. (2016) formula.

    Parameters
    ----------
    wmax : np.ndarray
        Maximum sustained winds in knots
    vmean : np.ndarray
        Mean translational speed in knots
    latitude : np.ndarray
        Latitude of the storm center in degrees

    Returns
    -------
    np.ndarray
        Radius of maximum winds (RMW) in nautical miles

    Notes
    -----
    Implements the Knaff et al. (2016) formula for estimating RMW:
    1. Subtracts translational speed from observed maximum wind speed
    2. Converts 10m wind speed to gradient level wind speed using beta factor
    3. Applies empirical formula accounting for wind speed and latitude

    References
    ----------
    Knaff, J. A., et al. (2016). "Estimation of Tropical Cyclone Wind Structure
    Parameters from Satellite Imagery", Weather and Forecasting.

    Examples
    --------
    >>> rmw = wind2rmw(np.array([100]), np.array([10]), np.array([25]))
    array([26.51])
    """

    pifac = np.arccos(-1) / 180  # pi/180
    beta = 0.9  # conversion factor of wind speed

    # Subtract translational speed from observed maximum wind speed
    vkt = wmax - vmean  # [kt]

    # Convert 10m wind speed to gradient level wind speed
    vgrad = vkt / beta  # [kt]

    # Knaff et al. (2016) formula
    rm = (
        218.3784
        - 1.2014 * vgrad
        + np.power(vgrad / 10.9844, 2)
        - np.power(vgrad / 35.3052, 3)
        - 145.509 * np.cos(latitude * pifac)
    )  # [nmile]

    return rm


def get_vmean(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
    deltat: Union[float, np.ndarray],
) -> Tuple[
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
]:
    """
    Calculate storm translation speed and direction between two points.

    Parameters
    ----------
    lat1 : Union[float, np.ndarray]
        Latitude of starting point(s) in degrees
    lon1 : Union[float, np.ndarray]
        Longitude of starting point(s) in degrees
    lat2 : Union[float, np.ndarray]
        Latitude of ending point(s) in degrees
    lon2 : Union[float, np.ndarray]
        Longitude of ending point(s) in degrees
    deltat : Union[float, np.ndarray]
        Time step(s) in hours

    Returns
    -------
    Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]
        Tuple containing:
        - gamma_h : Forward direction in degrees from North
        - vmean : Translation speed in km/h
        - vu : x-component of translation velocity in km/h
        - vy : y-component of translation velocity in km/h

    Notes
    -----
    Uses great circle distance calculation to determine distance between points.
    Translation speed is calculated by dividing distance by time step.
    Direction components are calculated using trigonometry.

    Examples
    --------
    >>> gamma, v, vx, vy = get_vmean(0, 0, 0, 1, 6)
    (270.0, 18.55, 18.55, 5.68e-15)
    """

    arcl_h, gamma_h = geodesic_distance_azimuth(lat2, lon2, lat1, lon1)  # great circle

    r = arcl_h * np.pi / 180.0 * EARTH_RADIUS  # distance between coordinates [km]
    vmean = r / deltat  # translation speed [km/h]

    vx = vmean * np.sin((gamma_h + 180) * np.pi / 180)
    vy = vmean * np.cos((gamma_h + 180) * np.pi / 180)

    return gamma_h, vmean, vx, vy  # [º], [km/h]


def wind2pres(
    xds_coef: xr.Dataset, st_wind: np.ndarray, st_center: str, st_basin: str
) -> np.ndarray:
    """
    Convert maximum wind speeds to minimum central pressure using fitted coefficients.
    As many other functions in this module, this works for IBTrACS data for the moment.

    Parameters
    ----------
    xds_coef : xr.Dataset
        Dataset containing polynomial fitting coefficients for Pmin-Wmax relationship.
    st_wind : np.ndarray
        Storm maximum winds in knots (1-min average).
    st_center : str
        Storm center/agency identifier (e.g., 'WMO', 'TOKYO').
    st_basin : str
        Storm basin identifier (e.g., 'NA', 'WP').

    Returns
    -------
    np.ndarray
        Predicted minimum central pressure values in millibars

    Notes
    -----
    Uses polynomial regression coefficients to estimate minimum central pressure
    from maximum wind speeds. The polynomial is of the form:
    p = a₀ + a₁w + a₂w² + ... + aₙwⁿ
    where p is pressure and w is wind speed.

    Examples
    --------
    >>> pres = wind2pres(coef_dataset, np.array([100]), 'WMO', 'NA')
    array([955.2])
    """

    pmin_pred = []

    for iwind in st_wind:
        # select Pmin-Wmax coefficients
        coeff = xds_coef.sel(center=st_center, basin=st_basin).coef.values[:]

        # aisle equation
        coeff[-1] -= iwind

        # pressure root
        pmin_pred.append(
            np.real(np.roots(coeff))[np.where(np.imag(np.roots(coeff)) == 0)[0][0]]
        )

    return np.array(pmin_pred)  # [mbar]


def resample_storm_6h(storm: xr.Dataset) -> xr.Dataset:
    """
    Resample storm data to 6-hour intervals.

    Parameters
    ----------
    storm : xr.Dataset
        Storm dataset containing time series data with arbitrary time intervals

    Returns
    -------
    xr.Dataset
        Resampled storm dataset with 6-hour intervals

    Notes
    -----
    This function:
    1. Removes NaT (Not a Time) values from time coordinate
    2. Removes NaN values from WMO pressure data
    3. Resets lon/lat coordinates
    4. Resamples all variables to 6-hour intervals using linear interpolation
    5. Preserves the original basin identifier

    Examples
    --------
    >>> resampled = resample_storm_6h(storm_data)
    >>> np.all(np.diff(resampled.time) == np.timedelta64(6, 'h'))
    True
    """

    storm = storm.isel(date_time=~np.isnat(storm.time.values))
    storm = storm.isel(date_time=~np.isnan(storm.wmo_pres.values))
    storm = storm.reset_coords(["lon", "lat"])
    ibasin = storm.basin.values[0]
    storm = (
        storm.swap_dims({"date_time": "time"}).resample(time="6H").interpolate("linear")
    )
    storm["basin"] = (("time"), np.array([ibasin] * storm.time.size))

    return storm


def historic_track_preprocessing(
    xds: xr.Dataset,
    center: str = "WMO",
    forecast_on: bool = False,
    database_on: bool = False,
    st_param: bool = False,
) -> pd.DataFrame:
    """
    Preprocess historical storm track data from IBTrACS or forecast sources.

    Parameters
    ----------
    xds : xr.Dataset
        Historical storm track dataset with storm dimension.
    center : str, optional
        IBTrACS center code (e.g., 'WMO', 'TOKYO'). Default is "WMO".
    forecast_on : bool, optional
        Whether track is forecasted (not IBTrACS). Default is False.
    database_on : bool, optional
        Whether to keep data only at 0,6,12,18 hours. Default is False.
    st_param : bool, optional
        Whether to keep data as original. Default is False.

    Returns
    -------
    pd.DataFrame
        Preprocessed storm track data with columns:
        - center : Storm center identifier
        - basin : Storm basin identifier
        - dist2land : Distance to nearest land (km)
        - longitude : Storm longitude (0-360°)
        - latitude : Storm latitude
        - move : Forward direction (degrees)
        - mean_velocity : Translation speed (kt)
        - pressure : Central pressure (mbar)
        - maxwinds : Maximum winds (kt, 1-min avg)
        - rmw : Radius of maximum winds (nmile)
        - category : Storm category (0-6)
        - timestep : Time step (hours)
        - storm_vmean : Mean translation speed (kt)

    Notes
    -----
    Processing steps include:
    1. Removing NaT/NaN values from time and pressure data
    2. Converting longitudes to [0°-360°] convention
    3. Rounding dates to nearest hour
    4. Calculating translation speed and direction
    5. Converting wind speeds to 1-min average
    6. Determining storm category based on pressure

    For forecast tracks:
    - Basin IDs are converted to IBTrACS equivalents
    - Missing pressures are estimated from wind speeds using fitted coefficients
    - Southern hemisphere basins are determined by latitude and longitude

    Examples
    --------
    >>> df = historic_track_preprocessing(track_data, center='WMO')
    >>> print(f"Track duration: {len(df)} time steps")
    Track duration: 48
    >>> print(f"Storm category: {df['category'].iloc[0]}")
    Storm category: 3
    """

    # dictionary for IBTrACS center
    d_vns = get_center_information(center=center)

    # get names of variables
    nm_tim = d_vns["time"]
    nm_bas = d_vns["basin"]
    nm_lon = d_vns["longitude"]
    nm_lat = d_vns["latitude"]
    nm_prs = d_vns["pressure"]
    nm_win = d_vns["maxwinds"]
    nm_rmw = d_vns["rmw"]

    # get var time
    ytime = xds[nm_tim].values  # dates format: datetime64

    # remove NaTs (time)
    ydist = xds["dist2land"].values[~np.isnat(ytime)]  # distance to land [km]
    ybasin = xds[nm_bas].values[~np.isnat(ytime)]  # basin
    ylat_tc = xds[nm_lat].values[~np.isnat(ytime)]  # latitude
    ylon_tc = xds[nm_lon].values[~np.isnat(ytime)]  # longitude
    ycpres = xds[nm_prs].values[~np.isnat(ytime)]  # pressure [mbar]
    ywind = xds[nm_win].values[~np.isnat(ytime)]  # wind speed [kt]
    if nm_rmw:
        yradii = xds[nm_rmw].values[~np.isnat(ytime)]  # rmw [nmile]
    ytime = ytime[~np.isnat(ytime)]

    # remove common NaNs (pressure & wind)
    posnonan_p_w = np.unique(
        np.concatenate((np.argwhere(~np.isnan(ycpres)), np.argwhere(~np.isnan(ywind))))
    )
    ytime = ytime[posnonan_p_w]
    ydist = ydist[posnonan_p_w]
    ybasin = ybasin[posnonan_p_w]
    ylat_tc = ylat_tc[posnonan_p_w]
    ylon_tc = ylon_tc[posnonan_p_w]
    ycpres = ycpres[posnonan_p_w]
    ywind = ywind[posnonan_p_w]
    if nm_rmw:
        yradii = yradii[posnonan_p_w]
    if not nm_rmw:
        yradii = np.full(ycpres.size, np.nan)  # [np.nan] * ycpres.size

    ###########################################################################
    # forecast input may lack pressure
    if forecast_on:
        # convert basin ids (IBTrACS equivalent)
        dict_basin_forecast = {
            "AL": "NA",
            "LS": "SA",
            "EP": "EP",
            "CP": "WP",
            "WP": "WP",
            "SH": "SP",
            "IO": "NI",
        }
        ybasin = np.array([dict_basin_forecast[ybas] for ybas in ybasin])
        ybasin[(ylat_tc < 0) & (ylon_tc < 135)] = "SI"

        if np.isnan(ycpres).any():
            # ibtracs fitting coefficients path
            xds_coef = ibtracs_fit_pmin_wmax()

            # fill pressure gaps
            ycpres[np.isnan(ycpres)] = wind2pres(
                xds_coef, ywind[np.isnan(ycpres)], center, ybasin[0]
            )

        ybasin = np.array([np.bytes_(ybas) for ybas in ybasin])
    ###########################################################################

    # remove NaNs (pressure)
    ytime = ytime[~np.isnan(ycpres)]
    st_dist2land = ydist[~np.isnan(ycpres)]
    ybasin = ybasin[~np.isnan(ycpres)]
    st_lat = ylat_tc[~np.isnan(ycpres)]
    st_lon = ylon_tc[~np.isnan(ycpres)]
    st_pres = ycpres[~np.isnan(ycpres)]
    st_wind = ywind[~np.isnan(ycpres)]
    if nm_rmw:
        st_rmw = yradii[~np.isnan(ycpres)]
    if not nm_rmw:
        st_rmw = np.full(st_pres.size, np.nan)  # [np.nan] * st_pres.size

    # conflicting times
    dft = pd.DataFrame({"time": ytime})
    dft["time"] = pd.to_datetime(dft["time"], format="%Y-%m-%d %H:%M:%S").dt.round("1h")
    hr = dft["time"].dt.hour.values

    # duplicate times
    pos_out = np.where(np.diff(hr) == 0)[0]

    # keep 0,3,6,9,12,15,18 hours (remove in between data)
    # TODO: option to interpolate to fixed target hours
    if database_on:
        pos_in = np.where((hr == 0) | (hr == 6) | (hr == 12) | (hr == 18))[0]
    else:
        pos_in = np.where(
            (hr == 0)
            | (hr == 6)
            | (hr == 12)
            | (hr == 18)
            | (hr == 3)
            | (hr == 9)
            | (hr == 15)
            | (hr == 21)
        )[0]
    # for parameterized tracks hours can be random
    if st_param:
        pos_in = np.where(hr > 0)

    # remove duplicates
    pos = np.setdiff1d(pos_in, pos_out)  # pos_in[pos_in != pos_out].ravel()
    ytime = ytime[pos]
    st_dist2land = st_dist2land[pos]
    ybasin = ybasin[pos]
    st_lat = st_lat[pos]
    st_lon = st_lon[pos]
    st_pres = st_pres[pos]
    st_wind = st_wind[pos]
    st_rmw = st_rmw[pos]

    # only when storm data available
    if st_pres.size > 0:
        # longitude convention: [0º,360º]
        st_lon[st_lon < 0] = st_lon[st_lon < 0] + 360

        # round dates to hour ---> half hour
        round_to = 3600 / 2
        st_time = []
        for i in range(len(ytime)):
            dt = ytime[i].astype("datetime64[s]").tolist()
            seconds = (dt - dt.min).seconds
            rounding = (seconds + round_to / 2) // round_to * round_to
            out = dt + timedelta(0, rounding - seconds, -dt.microsecond)
            st_time.append(out)
        st_time = np.asarray(st_time)

        # storm coordinates timestep [hours]
        ts = st_time[1:] - st_time[:-1]
        ts = [ts[i].total_seconds() / 3600 for i in range(ts.size)]
        ts.append(np.nan)

        # calculate Vmean
        st_vmean, st_move = [], []
        for i in range(0, len(st_time) - 1):
            # consecutive storm coordinates
            lon1, lon2 = st_lon[i], st_lon[i + 1]
            lat1, lat2 = st_lat[i], st_lat[i + 1]

            # translation speed
            gamma_h, vel_mean, _, _ = get_vmean(lat1, lon1, lat2, lon2, ts[i])
            st_vmean.append(vel_mean / 1.852)  # translation speed [km/h to kt]
            st_move.append(gamma_h)  # forward direction [º]
        st_vmean.append(np.nan)
        st_move.append(np.nan)

        # mean value
        vmean = np.nanmean(st_vmean)  # [kt]

        # storm category
        categ = get_category(st_pres)

        # convert (X)-min to 1-min avg winds (depends on center)
        st_wind = st_wind / centers_config_params[center]["windfac"]

        # get basin string
        st_basin = [str(c.decode("UTF-8")) for c in ybasin]

        # store storm variables
        df = pd.DataFrame(
            index=st_time,
            columns=[
                "center",
                "basin",
                "dist2land",
                "longitude",
                "latitude",
                "move",
                "mean_velocity",
                "pressure",
                "maxwinds",
                "rmw",
                "category",
                "timestep",
                "storm_vmean",
            ],
        )

        df["center"] = center
        df["basin"] = st_basin
        if "dist2land" in xds.keys():
            df["dist2land"] = st_dist2land
        else:
            df["dist2land"] = np.nan
        df["longitude"] = st_lon
        df["latitude"] = st_lat
        df["move"] = st_move
        df["mean_velocity"] = st_vmean
        df["pressure"] = st_pres
        df["maxwinds"] = st_wind
        df["rmw"] = st_rmw
        df["category"] = categ
        df["timestep"] = ts
        df["storm_vmean"] = vmean

        df.attrs = {
            "dist2land": "km",
            "velocity": "kt",
            "pressure": "mbar",
            "maxwinds": "kt, 1-min average sustained winds (converted from X-min)",
            "rmw": "nmile",
            "timestep": "hour",
        }

    else:
        df = pd.DataFrame(
            index=[],
            columns=[
                "center",
                "basin",
                "dist2land",
                "longitude",
                "latitude",
                "move",
                "mean_velocity",
                "pressure",
                "maxwinds",
                "rmw",
                "category",
                "timestep",
                "storm_vmean",
            ],
        )

    return df


def historic_track_interpolation(
    df: pd.DataFrame,
    dt_comp: float,
    y0: float = None,
    x0: float = None,
    great_circle: bool = True,
    wind_estimate_on: bool = False,
    fit: bool = False,
    interpolation: bool = True,
    mode: str = "first",
    radi_estimate_on: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Interpolate storm track variables to computational time steps.

    Parameters
    ----------
    df : pd.DataFrame
        Storm track DataFrame with historical data.
    dt_comp : float
        Computation time step in minutes.
    y0 : float, optional
        Target latitude coordinate. Default is None.
    x0 : float, optional
        Target longitude coordinate. Default is None.
    great_circle : bool, optional
        Whether to use great circle distances. Default is True.
    wind_estimate_on : bool, optional
        Whether to use empirical estimates instead of historical winds. Default is False.
    fit : bool, optional
        Whether to estimate winds when wind=0. Default is False.
    interpolation : bool, optional
        Whether to interpolate storm variables. Default is True.
    mode : str, optional
        Value selection for constant segments ('first' or 'mean'). Default is "first".
    radi_estimate_on : bool, optional
        Whether to estimate missing RMW values. Default is True.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        - DataFrame with interpolated storm track variables.
        - Array of interpolated time coordinates.

    Notes
    -----
    The function:
    1. Interpolates track points to match computational time step
    2. Estimates missing values using empirical relationships:
        - Maximum winds from pressure (Pmin-Wmax relationship)
        - Radius of maximum winds (Knaff et al. 2016)
    3. Can use either linear interpolation or constant segments
    4. Preserves metadata and adds computational parameters

    Examples
    --------
    >>> st, times = historic_track_interpolation(track_data, dt_comp=30)
    >>> print(f"Interpolated points: {len(st)}")
    Interpolated points: 144
    """

    # historic storm variables
    st_time = df.index.values[:]  # datetime format
    st_center = df["center"].values[0]
    st_basin = df["basin"].values[:]
    st_dist2land = df["dist2land"].values[:]  # [km]
    st_lon = df["longitude"].values[:]
    st_lat = df["latitude"].values[:]
    st_vmean = df["mean_velocity"].values[:]  # [kt]
    st_pres = df["pressure"].values[:]  # [mbar]
    st_wind = df["maxwinds"].values[:]  # [kt, 1-min avg]
    st_rmw = df["rmw"].values[:]  # [nmile]
    st_step = df["timestep"].values[:]  # [hour]
    wind_fill = np.full(st_pres.size, False)
    rmw_fill = np.full(st_pres.size, False)

    # select Pmin-Wmax polynomial fitting coefficients (IBTrACS center,basin)
    xds_coef = ibtracs_fit_pmin_wmax()
    try:
        coefs = xds_coef.sel(
            center=st_center,
            basin=st_basin,
        ).coef.values[:]
    except Exception:
        coefs = xds_coef.sel(
            center=st_center.encode("utf-8"),
            basin=st_basin.astype("bytes"),
        ).coef.values[:]

    p1, p2, p3, p4 = coefs[:, 0], coefs[:, 1], coefs[:, 2], coefs[:, 3]
    wind_estimate = (
        p1 * np.power(st_pres, 3)
        + p2 * np.power(st_pres, 2)
        + p3 * np.power(st_pres, 1)
        + p4
    )

    # maxwinds gaps filled with Pmin-Wmax coefficients
    posnan = np.argwhere(np.isnan(st_wind))  # np.where(st_wind==np.nan)[0]
    poszero = np.where(st_wind == 0)[0]
    if np.isnan(st_wind).all() or wind_estimate_on:  # wind NOT provided
        wind_fill = np.full(st_wind.size, True)
        st_wind = wind_estimate
    ###########################################################################
    elif np.isnan(st_wind).any():  # wind with some NaNs
        if fit and posnan.size > 0:  # data filled (for wind=0)
            wind_fill[posnan] = True
            st_wind[posnan] = wind_estimate[posnan]
    elif fit and poszero.size > 0:  # wind provided
        wind_fill[poszero] = True
        st_wind[poszero] = wind_estimate[poszero]
    ###########################################################################
    #    else:                                             # wind provided
    #        pos = np.where(st_wind==0)[0]
    #        if fit and pos.size>0:    # data filled (for wind=0)
    #            wind_fill[pos] = True
    #            st_wind[pos]   = wind_estimate[pos]

    # radii of maximum winds gaps filled with Knaff et al. (2016) estimate
    rmw_estimate = wind2rmw(st_wind, st_vmean, st_lat)
    if radi_estimate_on:
        if np.isnan(st_rmw).all():  # all
            rmw_fill = np.full(rmw_estimate.size, True)
            st_rmw = rmw_estimate
        elif np.isnan(st_rmw).any():  # some
            rmw_fill[np.isnan(st_rmw)] = True
            st_rmw[np.isnan(st_rmw)] = rmw_estimate[np.isnan(st_rmw)]
    else:
        if np.isnan(st_rmw).any():
            st_rmw = df.rmw.interpolate(method="time").values

    # storm variables (original values or mean values)
    if mode == "mean":
        st_pres = np.mean((st_pres, np.append(st_pres[1:], np.nan)), axis=0)
        st_wind = np.mean((st_wind, np.append(st_wind[1:], np.nan)), axis=0)
        st_rmw = np.mean((st_rmw, np.append(st_rmw[1:], np.nan)), axis=0)

    # number of time steps between consecutive interpolated storm coordinates
    # to match SWAN computational timestep
    ts_h = st_step  # hours
    nts = np.asarray(st_step) * 60 / dt_comp  # number of intervals

    # initialize interpolated variables
    bas, dist, lon, lat, move, vmean, vu, vy = [], [], [], [], [], [], [], []
    p0, vmax, rmw, vmaxfill, rmwfill = [], [], [], [], []
    time_input = np.empty((0,), dtype="datetime64[ns]")

    for i in range(0, len(st_time) - 1):
        # time array for SWAN input
        date_ini = st_time[i]
        time_input0 = pd.date_range(
            date_ini, periods=int(nts[i]), freq="{0}min".format(dt_comp)
        )
        time_input = np.append(np.array(time_input), np.array(time_input0))

        # consecutive storm coordinates
        lon1, lon2 = st_lon[i], st_lon[i + 1]
        lat1, lat2 = st_lat[i], st_lat[i + 1]

        # translation speed
        arcl_h, gamma_h = geodesic_distance_azimuth(lat2, lon2, lat1, lon1)
        r = arcl_h * np.pi / 180.0 * EARTH_RADIUS  # distance between coordinates [km]
        dx = r / nts[i]  # distance during time step [km]
        tx = ts_h[i] / nts[i]  # time period during time step [h]
        vx = float(dx) / tx / 3.6  # translation speed [km/h to m/s]
        vx = vx / 0.5144  # translation speed [m/s to kt]

        # interpolate at timesteps
        for j in range(int(nts[i])):
            bas.append(st_basin[i])
            dist.append(st_dist2land[i])

            # append interpolated lon, lat
            if not great_circle:
                glon = (
                    lon1
                    - (dx * 180 / (EARTH_RADIUS * np.pi))
                    * np.sin(gamma_h * np.pi / 180)
                    * j
                )
                glat = (
                    lat1
                    - (dx * 180 / (EARTH_RADIUS * np.pi))
                    * np.cos(gamma_h * np.pi / 180)
                    * j
                )
            else:
                glon, glat, baz = shoot(lon1, lat1, gamma_h + 180, float(dx) * j)
            lon.append(glon)
            lat.append(glat)

            # append storm track parameters
            move.append(gamma_h)
            vmean.append(vx)
            vu.append(vx * np.sin((gamma_h + 180) * np.pi / 180))
            vy.append(vx * np.cos((gamma_h + 180) * np.pi / 180))

            # append pmin, wind, rmw (interpolated/constant)
            if interpolation:
                p0.append(st_pres[i] + j * (st_pres[i + 1] - st_pres[i]) / nts[i])
                vmax.append(st_wind[i] + j * (st_wind[i + 1] - st_wind[i]) / nts[i])
                rmw.append(st_rmw[i] + j * (st_rmw[i + 1] - st_rmw[i]) / nts[i])
                vmaxfill.append(wind_fill[i] or wind_fill[i + 1])
                rmwfill.append(rmw_fill[i] or rmw_fill[i + 1])
            else:
                p0.append(st_pres[i])
                vmax.append(st_wind[i])
                rmw.append(st_rmw[i])
                vmaxfill.append(wind_fill[i])
                rmwfill.append(rmw_fill[i])

    # longitude convention [0º,360º]
    lon = np.array(lon)
    lon[lon < 0] = lon[lon < 0] + 360

    # method attribute
    n_hour = dt_comp / 60
    if interpolation:
        method = "interpolation ({0}h)".format(n_hour)
    else:
        if mode == "mean":
            method = "segment ({0}h, mean value)".format(n_hour)
        elif mode == "first":
            method = "segment ({0}h, node value)".format(n_hour)

    # storm track (pd.DataFrame)
    st = pd.DataFrame(
        index=time_input,
        columns=[
            "center",
            "basin",
            "dist2land",
            "lon",
            "lat",
            "move",
            "vf",
            "vfx",
            "vfy",
            "pn",
            "p0",
            "vmax",
            "rmw",
            "vmaxfill",
            "rmwfill",
        ],
    )

    st["center"] = st_center
    st["basin"] = bas
    st["dist2land"] = dist  # distance to nearest land [km]
    st["lon"] = lon  # longitude coordinate
    st["lat"] = lat  # latitude coordinate
    st["move"] = move  # gamma, forward direction
    st["vf"] = vmean  # translational speed [kt]
    st["vfx"] = vu  # x-component
    st["vfy"] = vy  # y-component
    st["pn"] = 1013  # average pressure at the surface [mbar]
    st["p0"] = p0  # minimum central pressure [mbar]
    st["vmax"] = vmax  # maximum winds [kt, 1-min avg]
    st["rmw"] = rmw  # radii of maximum winds [nmile]
    st["vmaxfill"] = vmaxfill  # True if maxwinds filled with Pmin-Vmax
    st["rmwfill"] = rmwfill  # True if radii filled with Knaff 2016

    # add metadata
    st.attrs = {
        "method": method,
        "center": st_center,
        "dist": "km",
        "vf": "kt",
        "p0": "mbar",
        "vmax": "kt, 1-min avg",
        "rmw": "nmile",
        "x0": x0,
        "y0": y0,
        "R": 4,
    }

    return st, time_input


def track_triming(
    st: pd.DataFrame, lat00: float, lon00: float, lat01: float, lon01: float
) -> pd.DataFrame:
    """
    Trim storm track to specified geographical bounds.

    Parameters
    ----------
    st : pd.DataFrame
        Storm track DataFrame containing time series data.
    lat00 : float
        Southern latitude bound in degrees.
    lon00 : float
        Western longitude bound in degrees.
    lat01 : float
        Northern latitude bound in degrees.
    lon01 : float
        Eastern longitude bound in degrees.

    Returns
    -------
    pd.DataFrame
        Trimmed storm track containing only points within the specified bounds
        and preserving continuous time segments.

    Notes
    -----
    The function:
    1. Identifies all track points within the specified bounds
    2. Finds the earliest and latest times for points within bounds
    3. Returns all track points between these times to maintain continuity
    4. Preserves original metadata in the returned DataFrame

    Examples
    --------
    >>> trimmed = track_triming(track_data, 20, -80, 30, -60)
    >>> print(f"Points in bounds: {len(trimmed)}")
    Points in bounds: 24
    """

    lo, la = st.lon.values[:], st.lat.values[:]

    # select track coordinates within the target domain area
    st_trim = st.iloc[(lo <= lon01) & (lo >= lon00) & (la <= lat01) & (la >= lat00)]

    # get min/max times within the area
    tmin = np.min(st_trim.index.values)
    tmax = np.max(st_trim.index.values)

    # select all the time window
    st_trim = st.iloc[(st.index >= tmin) & (st.index <= tmax)]

    # add metadata
    st_trim.attrs = st.attrs

    return st_trim


def track_triming_circle(
    st: pd.DataFrame, plon: float, plat: float, radii: float
) -> pd.DataFrame:
    """
    Trim storm track to points within a circular domain.

    Parameters
    ----------
    st : pd.DataFrame
        Storm track DataFrame containing time series data.
    plon : float
        Longitude of circle center in degrees.
    plat : float
        Latitude of circle center in degrees.
    radii : float
        Radius of circular domain in degrees.

    Returns
    -------
    pd.DataFrame
        Trimmed storm track containing only points within the circular domain
        and preserving continuous time segments.

    Notes
    -----
    The function:
    1. Calculates geodesic distance from each track point to circle center
    2. Identifies points within the specified radius
    3. Returns continuous time segment containing all points within radius
    4. Preserves original metadata in the returned DataFrame

    Examples
    --------
    >>> trimmed = track_triming_circle(track_data, -75, 25, 5)
    >>> print(f"Points in circle: {len(trimmed)}")
    Points in circle: 18
    """

    lo, la = st.lon.values[:], st.lat.values[:]

    # stack storm longitude, latitude
    lonlat_s = np.column_stack((lo, la))

    # calculate geodesic distance (degree)
    geo_dist = []
    for lon_ps, lat_ps in lonlat_s:
        geo_dist.append(geodesic_distance(lat_ps, lon_ps, plat, plon))
    geo_dist = np.asarray(geo_dist)

    # find storm inside circle and calculate parameters
    if (geo_dist < radii).any() & (geo_dist.size > 1):
        # storm inside circle
        ix_in = np.where(geo_dist < radii)[0][:]
        # select track coordinates within the target domain area
        st_trim = st.iloc[ix_in]

    else:
        st_trim = st.iloc[st.lat.values[:] > 90]

    # get min/max times within the area
    tmin = np.min(st_trim.index.values)
    tmax = np.max(st_trim.index.values)

    # select all the time window
    st_trim = st.iloc[(st.index >= tmin) & (st.index <= tmax)]

    # add metadata
    st_trim.attrs = st.attrs

    return st_trim


def stopmotion_trim_circle(
    df_seg: pd.DataFrame, plon: float, plat: float, radii: float
) -> pd.DataFrame:
    """
    Trim storm track segments to those intersecting a circular domain.

    Parameters
    ----------
    df_seg : pd.DataFrame
        DataFrame containing storm track segments with start/end coordinates.
    plon : float
        Longitude of circle center in degrees.
    plat : float
        Latitude of circle center in degrees.
    radii : float
        Radius of circular domain in degrees.

    Returns
    -------
    pd.DataFrame
        Trimmed DataFrame containing only segments that intersect the circular domain.

    Notes
    -----
    The function:
    1. Calculates geodesic distances from both endpoints of each segment to circle center
    2. Takes minimum distance to determine if segment intersects circle
    3. Returns continuous time segment containing all intersecting segments
    4. Preserves original metadata in the returned DataFrame

    Examples
    --------
    >>> trimmed = stopmotion_trim_circle(segments, -75, 25, 5)
    >>> print(f"Segments intersecting circle: {len(trimmed)}")
    Segments intersecting circle: 8
    """

    lo0, la0 = df_seg.lon_i.values[:], df_seg.lat_i.values[:]
    lo1, la1 = df_seg.lon_t.values[:], df_seg.lat_t.values[:]

    # stack storm longitude, latitude
    lonlat_s0 = np.column_stack((lo0, la0))
    lonlat_s1 = np.column_stack((lo1, la1))

    # calculate geodesic distance (degree)
    geo_dist0, geo_dist1 = [], []
    for lon_ps, lat_ps in lonlat_s0:
        geo_dist0.append(geodesic_distance(lat_ps, lon_ps, plat, plon))
    geo_dist0 = np.asarray(geo_dist0)
    for lon_ps, lat_ps in lonlat_s1:
        geo_dist1.append(geodesic_distance(lat_ps, lon_ps, plat, plon))
    geo_dist1 = np.asarray(geo_dist1)

    # get minimum distance of the two target endpoints
    geo_dist = np.min((geo_dist0, geo_dist1), axis=0)

    # find storm inside circle and calculate parameters
    if (geo_dist < radii).any() & (geo_dist.size > 1):
        # storm inside circle
        ix_in = np.where(geo_dist < radii)[0][:]
        # select track coordinates within the target domain area
        st_trim = df_seg.iloc[ix_in]

        # get min/max times within the area
        tmin = np.min(st_trim.index.values)
        tmax = np.max(st_trim.index.values)

        # select all the time window
        st_trim = df_seg.iloc[(df_seg.index >= tmin) & (df_seg.index <= tmax)]

        # add metadata
        st_trim.attrs = df_seg.attrs

    else:  # empty dataframe
        st_trim = pd.DataFrame(columns=df_seg.columns)

    return st_trim


def track_extent(
    st: pd.DataFrame, time_input: np.ndarray, dt_comp: float, time_extent: float = 48.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extend storm track data for wave propagation analysis.

    Parameters
    ----------
    st : pd.DataFrame
        Storm track DataFrame containing time series data.
    time_input : np.ndarray
        Array of time coordinates.
    dt_comp : float
        Computational time step in minutes.
    time_extent : float, optional
        Additional time to extend simulation in hours. Default is 48.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing:
        - st_new : Extended storm track DataFrame.
        - we : Wave event DataFrame with empty fields for wave parameters.

    Notes
    -----
    The function:
    1. Calculates total simulation period (storm + propagation)
    2. Creates extended time array with specified time step
    3. Fills storm data in initial period
    4. Creates empty wave event DataFrame for entire period
    5. Preserves metadata and adds computational time step override

    Examples
    --------
    >>> st_ext, wave_evt = track_extent(track_data, times, 30, time_extent=24)
    >>> print(f"Extended duration: {len(st_ext)} time steps")
    Extended duration: 96 time steps
    """

    # total simulation period (hours): storm + propagation
    time_storm = (time_input[-1] - time_input[0]).astype(
        "timedelta64[h]"
    ) / np.timedelta64(1, "h")
    T = time_storm + time_extent

    # original track period
    size_ini = st.index.size
    date_ini = st.index[0]

    # total simulation period
    size_new = int(T * 60 / dt_comp)
    time_input_new = pd.date_range(
        date_ini, periods=size_new, freq="{0}MIN".format(dt_comp)
    )
    st_new = pd.DataFrame(index=time_input_new, columns=list(st.keys()))

    # combine
    st_new.values[:size_ini, :] = st.values
    if st.attrs["x0"]:
        st_new.x0 = st.attrs["x0"]
    if st.attrs["y0"]:
        st_new.y0 = st.attrs["y0"]

    # [OPTIONAL] override SWAN storm case computational delta time
    st_new.attrs["override_dtcomp"] = "{0}MIN".format(dt_comp)

    # generate wave event (empty)
    we = pd.DataFrame(
        index=time_input_new, columns=["hs", "t02", "dir", "spr", "U10", "V10"]
    )
    we["level"] = 0
    we["tide"] = 0

    return st_new, we


###############################################################################
# HYTCWAVES
# PARAMETERIZED storm tracks
# Given the storm parameters (HyTCWaves methodology) the track coordinates are
# calculated at each swan computational timestep
###############################################################################


def entrance_coords(
    delta: float,
    gamma: float,
    x0: float,
    y0: float,
    R: float,
    lon0: float,
    lon1: float,
    lat0: float,
    lat1: float,
) -> Tuple[float, float]:
    """
    Calculate storm entrance coordinates at computational domain boundary.

    Parameters
    ----------
    delta : float
        Storm track angle parameter in degrees.
    gamma : float
        Storm forward direction in degrees.
    x0 : float
        Site longitude coordinate in degrees.
    y0 : float
        Site latitude coordinate in degrees.
    R : float
        Radius of influence in degrees.
    lon0 : float
        Western longitude bound of computational domain.
    lon1 : float
        Eastern longitude bound of computational domain.
    lat0 : float
        Southern latitude bound of computational domain.
    lat1 : float
        Northern latitude bound of computational domain.

    Returns
    -------
    Tuple[float, float]
        Tuple containing:
        - x1 : Entrance longitude coordinate in degrees.
        - y1 : Entrance latitude coordinate in degrees.

    Notes
    -----
    The function:
    1. Calculates entrance point on the radius of influence
    2. Determines angles to domain corners from entrance point
    3. Uses storm direction to determine intersection with domain boundary
    4. Returns coordinates where storm track enters computational domain

    Examples
    --------
    >>> x1, y1 = entrance_coords(45, 270, -75, 25, 5, -80, -70, 20, 30)
    >>> print(f"Entrance point: ({x1:.1f}°, {y1:.1f}°)")
    Entrance point: (-78.5°, 27.5°)
    """

    # enter point in the radius
    xc = x0 + R * np.sin(delta * np.pi / 180)
    yc = y0 + R * np.cos(delta * np.pi / 180)

    # calculate angles that determine the storm boundary entrance  [degrees]
    ang_1 = np.arctan((lon1 - xc) / (lat1 - yc)) * 180 / np.pi  # upper right corner
    ang_2 = np.arctan((lon1 - xc) / (lat0 - yc)) * 180 / np.pi + 180  # lower right
    ang_3 = np.arctan((lon0 - xc) / (lat0 - yc)) * 180 / np.pi + 180  # lower left
    ang_4 = np.arctan((lon0 - xc) / (lat1 - yc)) * 180 / np.pi + 360  # upper left

    if (gamma > ang_1) & (gamma < ang_2):
        x1 = lon1
        d = (x1 - xc) / np.sin(gamma * np.pi / 180)
        y1 = yc + d * np.cos(gamma * np.pi / 180)

    elif (gamma > ang_2) & (gamma < ang_3):
        y1 = lat0
        d = (y1 - yc) / np.cos(gamma * np.pi / 180)
        x1 = xc + d * np.sin(gamma * np.pi / 180)

    elif (gamma > ang_3) & (gamma < ang_4):
        x1 = lon0
        d = (x1 - xc) / np.sin(gamma * np.pi / 180)
        y1 = yc + d * np.cos(gamma * np.pi / 180)

    elif (gamma > ang_4) | (gamma < ang_1):
        y1 = lat1
        d = (y1 - yc) / np.cos(gamma * np.pi / 180)
        x1 = xc + d * np.sin(gamma * np.pi / 180)

    return x1, y1


def track_site_parameters(
    step: float,
    pmin: float,
    vmean: float,
    delta: float,
    gamma: float,
    x0: float,
    y0: float,
    lon0: float,
    lon1: float,
    lat0: float,
    lat1: float,
    R: float,
    date_ini: str,
    center: str = "WMO",
    basin: str = "SP",
) -> pd.DataFrame:
    """
    Generate parameterized storm track within study area.

    Parameters
    ----------
    step : float
        Computational time step in minutes.
    pmin : float
        Minimum central pressure in millibars.
    vmean : float
        Mean translation speed in knots.
    delta : float
        Storm track angle parameter in degrees.
    gamma : float
        Storm forward direction in degrees.
    x0 : float
        Site longitude coordinate in degrees.
    y0 : float
        Site latitude coordinate in degrees.
    lon0 : float
        Western longitude bound of computational domain.
    lon1 : float
        Eastern longitude bound of computational domain.
    lat0 : float
        Southern latitude bound of computational domain.
    lat1 : float
        Northern latitude bound of computational domain.
    R : float
        Radius of influence in degrees.
    date_ini : str
        Initial date in format 'YYYY-MM-DD HH:MM'.
    center : str, optional
        IBTrACS center code. Default is "WMO".
    basin : str, optional
        Storm basin identifier. Default is "SP".

    Returns
    -------
    pd.DataFrame
        Storm track DataFrame with columns:
        - lon : Storm longitude coordinates
        - lat : Storm latitude coordinates
        - move : Forward direction (gamma)
        - vf : Translation speed (kt)
        - vfx : x-component of velocity
        - vfy : y-component of velocity
        - pn : Surface pressure (1013 mbar)
        - p0 : Minimum central pressure
        - vmax : Maximum winds (kt, 1-min avg)
        - rmw : Radius of maximum winds (nmile)

    Notes
    -----
    The function:
    1. Calculates entrance coordinates at domain boundary
    2. Generates track points at computational time steps
    3. Estimates maximum winds from pressure using fitted coefficients
    4. Estimates radius of maximum winds using Knaff et al. (2016)
    5. Preserves metadata including site coordinates and radius

    Examples
    --------
    >>> track = track_site_parameters(
    ...     step=30, pmin=950, vmean=10, delta=45, gamma=270,
    ...     x0=-75, y0=25, lon0=-80, lon1=-70, lat0=20, lat1=30,
    ...     R=5, date_ini='2020-09-01 00:00'
    ... )
    >>> print(f"Track points: {len(track)}")
    Track points: 96
    """

    # storm entrance coordinates at the domain boundary
    x1, y1 = entrance_coords(delta, gamma, x0, y0, R, lon0, lon1, lat0, lat1)

    # calculate computational timestep storm coordinates
    # Note: velocity input in shoot function must be km/h
    lon, lat = [x1], [y1]
    i = 1
    glon, glat, baz = shoot(x1, y1, gamma + 180, vmean * 1.852 * i * step / 60)
    if glon < 0:
        glon += 360
    while (glon < lon1) & (glon > lon0) & (glat < lat1) & (glat > lat0):
        lon.append(glon)
        lat.append(glat)
        i += 1
        glon, glat, baz = shoot(x1, y1, gamma + 180, vmean * 1.852 * i * step / 60)
        if glon < 0:
            glon += 360
    frec = len(lon)

    # select Pmin-Vmax polynomial fitting coefficients (IBTrACS center,basin)
    xds_coef = ibtracs_fit_pmin_wmax()
    p1, p2, p3, p4 = xds_coef.sel(center=center, basin=basin).coef.values[:]
    wind_estimate = (
        p1 * np.power(pmin, 3) + p2 * np.power(pmin, 2) + p3 * np.power(pmin, 1) + p4
    )

    # radii of maximum winds is filled with Knaff (2016) estimate
    rmw_estimate = wind2rmw(
        np.full(lat.size, wind_estimate), np.full(lat.size, vmean), lat
    )

    # velocity components
    vfx = vmean * np.sin((gamma + 180) * np.pi / 180)  # [kt]
    vfy = vmean * np.cos((gamma + 180) * np.pi / 180)  # [kt]

    # time array for SWAN input
    time_input = pd.date_range(date_ini, periods=frec, freq="{0}MIN".format(step))

    # storm track (pd.DataFrame)
    st = pd.DataFrame(
        index=time_input,
        columns=["lon", "lat", "move", "vf", "vfx", "vfy", "pn", "p0", "vmax", "rmw"],
    )

    st["lon"] = lon
    st["lat"] = lat
    st["move"] = gamma  # gamma, forward direction
    st["vf"] = vmean  # translation speed [kt]
    st["vfx"] = vfx  # x-component
    st["vfy"] = vfy  # y-component
    st["pn"] = 1013  # average pressure at the surface [mbar]
    st["p0"] = pmin  # minimum central pressure [mbar]
    st["vmax"] = wind_estimate  # maximum winds [kt, 1-min avg]
    st["rmw"] = rmw_estimate  # radii of maximum winds [nmile]

    # add metadata
    st.attrs = {
        "vf": "kt",
        "p0": "mbar",
        "vmax": "kt, 1-min avg",
        "rmw": "nmile",
        "x0": x0,
        "y0": y0,
        "R": 4,
    }

    return st


###############################################################################
# synthetic tracks
###############################################################################


def nakajo_track_preprocessing(xds: xr.Dataset, center: str = "WMO") -> pd.DataFrame:
    """
    Preprocess synthetic storm track data from Nakajo format.

    Parameters
    ----------
    xds : xr.Dataset
        Synthetic storm track dataset with storm dimension in Nakajo format.
        Must contain variables for time, longitude (ylon_TC), latitude (ylat_TC),
        and pressure (yCPRES).
    center : str, optional
        IBTrACS center code. Default is "WMO".

    Returns
    -------
    pd.DataFrame
        Preprocessed storm track data with columns:
        - center : Storm center identifier
        - basin : Storm basin identifier (determined from coordinates)
        - dist2land : Distance to nearest land (fixed at 100 km)
        - longitude : Storm longitude (0-360°)
        - latitude : Storm latitude
        - move : Forward direction (degrees)
        - mean_velocity : Translation speed (kt)
        - pressure : Central pressure (mbar)
        - maxwinds : Maximum winds (kt, not provided)
        - rmw : Radius of maximum winds (not provided)
        - category : Storm category (0-6)
        - timestep : Time step (hours)
        - storm_vmean : Mean translation speed (kt)

    Notes
    -----
    The function:
    1. Removes NaT (Not a Time) values from time coordinate
    2. Removes NaN values from pressure data
    3. Converts longitudes to [0°-360°] convention
    4. Determines basin based on storm coordinates
    5. Rounds dates to nearest hour
    6. Calculates translation speed and direction
    7. Determines storm category based on pressure

    Examples
    --------
    >>> processed = nakajo_track_preprocessing(nakajo_data, center='WMO')
    >>> print(f"Track duration: {len(processed)} time steps")
    Track duration: 48
    >>> print(f"Storm category: {processed['category'].iloc[0]}")
    Storm category: 3
    """

    # get names of variables
    nm_tim = "time"
    nm_lon = "ylon_TC"
    nm_lat = "ylat_TC"
    nm_prs = "yCPRES"

    # get var time
    ytime = xds[nm_tim].values  # dates format: datetime64

    # remove NaTs (time)
    ylat_tc = xds[nm_lat].values[~np.isnat(ytime)]  # latitude
    ylon_tc = xds[nm_lon].values[~np.isnat(ytime)]  # longitude
    ycpres = xds[nm_prs].values[~np.isnat(ytime)]  # pressure [mbar]
    ytime = ytime[~np.isnat(ytime)]

    # remove common NaNs (pressure)
    posnonan_p_w = np.unique(np.argwhere(~np.isnan(ycpres)))
    ytime = ytime[posnonan_p_w]
    st_lat = ylat_tc[posnonan_p_w]
    st_lon = ylon_tc[posnonan_p_w]
    st_pres = ycpres[posnonan_p_w]

    ###########################################################################
    # longitude convention: [0º,360º]
    st_lon[st_lon < 0] = st_lon[st_lon < 0] + 360

    # get basin
    ybasin = np.empty(ytime.size, dtype="S10")
    for i in range(ytime.size):
        ilo, ila = st_lon[i], st_lat[i]
        if (ila < 0) & (ilo < 135) & (ilo > 20):
            ybasin[i] = "SI"
        elif (ila < 0) & (ilo > 135) & (ilo < 290):
            ybasin[i] = "SP"
        elif (ila < 0) & (ilo < 20) & (ilo > 290):
            ybasin[i] = "SA"
        elif (ila > 0) & (ilo > 20) & (ilo < 100):
            ybasin[i] = "NI"
        elif (ila > 0) & (ilo > 100) & (ilo < 180):
            ybasin[i] = "WP"
        elif (ila > 0) & (ilo > 180) & (ilo < 260):
            ybasin[i] = "EP"
        elif (ila > 0) & (ila < 15) & (ilo > 260) & (ilo < 275):
            ybasin[i] = "EP"
        else:
            ybasin[i] = "NA"

    ###########################################################################

    # only when storm data available
    if st_pres.size > 0:
        # round dates to hour
        round_to = 3600
        st_time = []
        for i in range(len(ytime)):
            dt = ytime[i].astype("datetime64[s]").tolist()
            seconds = (dt - dt.min).seconds
            rounding = (seconds + round_to / 2) // round_to * round_to
            out = dt + timedelta(0, rounding - seconds, -dt.microsecond)
            st_time.append(out)
        st_time = np.asarray(st_time)

        # storm coordinates timestep [hours]
        ts = st_time[1:] - st_time[:-1]
        ts = [ts[i].total_seconds() / 3600 for i in range(ts.size)]
        ts.append(np.nan)

        # calculate Vmean
        st_vmean, st_move = [], []
        for i in range(0, len(st_time) - 1):
            # consecutive storm coordinates
            lon1, lon2 = st_lon[i], st_lon[i + 1]
            lat1, lat2 = st_lat[i], st_lat[i + 1]

            # translation speed
            gamma_h, vel_mean, _, _ = get_vmean(lat1, lon1, lat2, lon2, ts[i])
            st_vmean.append(vel_mean / 1.852)  # translation speed [km/h to kt]
            st_move.append(gamma_h)  # forward direction [º]
        st_vmean.append(np.nan)
        st_move.append(np.nan)

        # mean value
        vmean = np.nanmean(st_vmean)  # [kt]

        # storm category
        categ = get_category(st_pres)

        # get basin string
        st_basin = [str(c.decode("UTF-8")) for c in ybasin]

        # store storm variables
        df = pd.DataFrame(
            index=st_time,
            columns=[
                "center",
                "basin",
                "dist2land",
                "longitude",
                "latitude",
                "move",
                "mean_velocity",
                "pressure",
                "maxwinds",
                "rmw",
                "category",
                "timestep",
                "storm_vmean",
            ],
        )

        df["center"] = center  # hypothesis winds 1-min
        df["basin"] = st_basin
        df["dist2land"] = 100
        df["longitude"] = st_lon
        df["latitude"] = st_lat
        df["move"] = st_move
        df["mean_velocity"] = st_vmean
        df["pressure"] = st_pres
        df["maxwinds"] = np.nan
        df["rmw"] = np.nan
        df["category"] = categ
        df["timestep"] = ts
        df["storm_vmean"] = vmean

        df.attrs = {
            "dist2land": "km",
            "velocity": "kt",
            "pressure": "mbar",
            "maxwinds": "kt, 1-min average sustained winds (converted from X-min)",
            "rmw": "nmile",
            "timestep": "hour",
        }

    else:
        df = pd.DataFrame(
            index=[],
            columns=[
                "center",
                "basin",
                "dist2land",
                "longitude",
                "latitude",
                "move",
                "mean_velocity",
                "pressure",
                "maxwinds",
                "rmw",
                "category",
                "timestep",
                "storm_vmean",
            ],
        )

    return df
