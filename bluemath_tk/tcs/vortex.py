from math import radians

import numpy as np
import pandas as pd
import xarray as xr

"""
Dynamic Holland Model for Wind Vortex Fields
This module implements the Dynamic Holland Model to generate wind vortex fields
from storm track parameters. It computes wind speed and direction based on
the storm's position, pressure, and wind parameters, using either spherical or
cartesian coordinates.
The model is optimized for vectorized operations to enhance performance.
It supports both spherical coordinates (latitude, longitude) and cartesian
coordinates (x, y) for storm track data.
The output is an xarray Dataset containing wind speed and direction.
The model is based on the Dynamic Holland Model, which uses storm parameters
to compute wind fields, considering factors like the Coriolis effect,
central pressure deficit, and the radius of maximum winds.
This implementation is designed to be efficient and scalable, suitable for
large datasets and real-time applications.
"""


def geo_distance_azimuth(
    lat_matrix: np.ndarray,
    lon_matrix: np.ndarray,
    lat_point: np.ndarray,
    lon_point: np.ndarray,
) -> tuple:
    """
    Returns geodesic distance and azimuth between lat,lon matrix and lat,lon
    point in degrees.
    Parameters:
    ----------
    lat_matrix : np.ndarray
        2D array of latitudes.
    lon_matrix : np.ndarray
        2D array of longitudes.
    lat_point : float
        Latitude of the point.
    lon_point : float
        Longitude of the point.
    Returns:
    -------
    tuple
        Tuple containing:
        - arcl : np.ndarray
            Array of geodesic distances in degrees.
        - azi : np.ndarray
            Array of azimuths in degrees from north.
    """
    # Vectorized computation of distances and azimuths
    lat_point_rad, lon_point_rad = map(radians, [lat_point, lon_point])
    lat_matrix_rad, lon_matrix_rad = np.radians(lat_matrix), np.radians(lon_matrix)

    dlat = lat_matrix_rad - lat_point_rad
    dlon = lon_matrix_rad - lon_point_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_point_rad) * np.cos(lat_matrix_rad) * np.sin(dlon / 2) ** 2
    )
    a = np.clip(a, 0, 1)  # Clamp values to avoid numerical errors

    arcl = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    arcl = np.degrees(arcl)

    azi = np.arctan2(
        np.cos(lat_matrix_rad) * np.sin(dlon),
        np.cos(lat_point_rad) * np.sin(lat_matrix_rad)
        - np.sin(lat_point_rad) * np.cos(lat_matrix_rad) * np.cos(dlon),
    )
    azi = np.degrees(azi % (2 * np.pi))

    return arcl, azi


def geo_distance_cartesian(
    y_matrix: np.ndarray, x_matrix, y_point: np.ndarray, x_point: np.ndarray
) -> np.ndarray:
    """
    Returns cartesian distance between y,x matrix and y,x point.
    Optimized using vectorized operations.
    Parameters:
    ----------
    y_matrix : np.ndarray
        2D array of y-coordinates (latitude or y in Cartesian).
    x_matrix : np.ndarray
        2D array of x-coordinates (longitude or x in Cartesian).
    y_point : np.ndarray
        y-coordinate of the point (latitude or y in Cartesian).
    x_point : np.ndarray
        x-coordinate of the point (longitude or x in Cartesian).
    Returns:
    -------
    np.ndarray
        Array of distances in the same units as x_matrix and y_matrix.
    """
    dist = np.sqrt((y_point - y_matrix) ** 2 + (x_point - x_matrix) ** 2)
    return dist


def geo_distance_meters(
    y_matrix: np.ndarray,
    x_matrix: np.ndarray,
    y_point: np.ndarray,
    x_point: np.ndarray,
    coords_mode: str = "SPHERICAL",
) -> np.ndarray:
    """
    Returns geodesic distance in meters between y,x matrix and y,x point.
    Parameters:
    ----------
    y_matrix : np.ndarray
        2D array of y-coordinates (latitude or y in Cartesian).
    x_matrix : np.ndarray
        2D array of x-coordinates (longitude or x in Cartesian).
    y_point : np.ndarray
        y-coordinate of the point (latitude or y in Cartesian).
    x_point : np.ndarray
        x-coordinate of the point (longitude or x in Cartesian).
    coords_mode : str
        'SPHERICAL' for spherical coordinates (latitude, longitude),
        'CARTESIAN' for Cartesian coordinates (x, y).
    Returns:
    -------
    np.ndarray
        Array of distances in meters.
    """
    RE = 6378.135 * 1000  # Earth radius [m]

    if coords_mode == "SPHERICAL":
        arcl, _ = geo_distance_azimuth(y_matrix, x_matrix, y_point, x_point)
        r = arcl * np.pi / 180.0 * RE
    elif coords_mode == "CARTESIAN":
        r = geo_distance_cartesian(y_matrix, x_matrix, y_point, x_point)

    return r


def vortex_model_grid(
    storm_track: pd.DataFrame,
    cg_lon: np.ndarray,
    cg_lat: np.ndarray,
    coords_mode: str = "SPHERICAL",
) -> xr.Dataset:
    """
    storm_track - (pandas.DataFrame)
        - obligatory fields:            vfx, vfy, p0, pn, vmax, rmw
        - for SPHERICAL coordinates:    lon, lat
        - for CARTESIAN coordinates:    x, y, latitude

    cg_lon, cg_lat - computational grid in longitudes and latitudes

    coords_mode - 'SPHERICAL' / 'CARTESIAN' swan project coordinates mode

    The Dynamic Holland Model is used to generate wind vortex fields from
    storm track coordinate parameters.

    Returns: xarray.Dataset with wind speed W and direction Dir (º from north)
    """
    # Convert negative longitudes to 0-360 range
    converted_coords = False
    if coords_mode == "SPHERICAL":
        cg_lon = np.where(cg_lon < 0, cg_lon + 360, cg_lon)
        converted_coords = True

    # Define model constants
    beta = 0.9  # Conversion factor of wind speed
    rho_air = 1.15  # Air density [kg/m³]
    omega = 2 * np.pi / 86184.2  # Earth's rotation rate [rad/s]
    deg2rad = np.pi / 180  # Degrees to radians conversion
    conv_1min_to_10min = 0.93  # Convert 1-min avg winds to 10-min avg

    # Extract storm parameters from the DataFrame
    vfx, vfy = storm_track.vfx.values, storm_track.vfy.values  # [kt] translation
    p0, pn = storm_track.p0.values, storm_track.pn.values  # [mbar] pressure
    vmax, rmw = storm_track.vmax.values, storm_track.rmw.values  # [kt], [nmile]
    times = storm_track.index  # time values

    # Select coordinates depending on mode
    if coords_mode == "SPHERICAL":
        x, y, lat = (
            storm_track.lon.values,
            storm_track.lat.values,
            storm_track.lat.values,
        )
    else:
        x, y, lat = (
            storm_track.x.values,
            storm_track.y.values,
            storm_track.latitude.values,
        )

    # Check if the storm is in the southern hemisphere
    is_southern = np.any(lat < 0)

    # Create 2D meshgrid of computational grid
    lon2d, lat2d = np.meshgrid(cg_lon, cg_lat)
    shape = (len(cg_lat), len(cg_lon), len(p0))

    # Initialize output arrays for wind magnitude and direction
    W = np.zeros(shape)
    D = np.zeros(shape)
    p = np.zeros(shape)

    # Loop over time steps to compute vortex fields
    for i in range(len(p0)):
        lo, la, la0 = x[i], y[i], lat[i]

        # Skip time steps with NaN values
        if np.isnan([lo, la, la0, p0[i], pn[i], vfx[i], vfy[i], vmax[i], rmw[i]]).any():
            continue

        # Compute distance between grid points and storm center
        r = geo_distance_meters(lat2d, lon2d, la, lo, coords_mode)

        # Compute direction from storm center to each grid point
        dlat = (lat2d - la) * deg2rad
        dlon = (lon2d - lo) * deg2rad
        theta = np.arctan2(dlat, -dlon if is_southern else dlon)

        # Compute central pressure deficit [Pa]
        CPD = max((pn[i] - p0[i]) * 100, 100)

        # Compute Coriolis parameter
        coriolis = 2 * omega * np.sin(abs(la0) * deg2rad)

        # Compute adjusted gradient wind speed
        v_trans = np.hypot(vfx[i], vfy[i])  # [kt] translation magnitude
        vkt = vmax[i] - v_trans  # corrected max wind [kt]
        vgrad = vkt / beta  # gradient wind [kt]
        vm = vgrad * 0.52  # convert to [m/s]

        # Compute radius and nondimensional radius
        rm = rmw[i] * 1852  # [m]
        rn = rm / r  # nondimensional

        # Compute Holland B parameter, with bounds
        B = np.clip(rho_air * np.exp(1) * vm**2 / CPD, 1, 2.5)

        # Gradient wind velocity at each grid point
        vg = (
            np.sqrt((rn**B) * np.exp(1 - rn**B) * vm**2 + (r**2 * coriolis**2) / 4)
            - r * coriolis / 2
        )

        # Convert to wind components at 10m height
        sign = 1 if is_southern else -1
        ve = sign * vg * beta * np.sin(theta) * conv_1min_to_10min
        vn = vg * beta * np.cos(theta) * conv_1min_to_10min

        # Add translation velocity components
        vtae = (np.abs(vg) / vgrad) * vfx[i]
        vtan = (np.abs(vg) / vgrad) * vfy[i]
        vfe = ve + vtae
        vfn = vn + vtan

        # Total wind magnitude [m/s]
        W[:, :, i] = np.hypot(vfe, vfn)

        # Pressure gradient to estimate wind direction
        pr = p0[i] + (pn[i] - p0[i]) * np.exp(-(rn**B))  # surface pressure
        p[:, :, i] = pr
        py, px = np.gradient(pr)
        angle = np.arctan2(py, px) + np.sign(la0) * np.pi / 2

        # Wind direction in degrees from north (clockwise)
        D[:, :, i] = (270 - np.rad2deg(angle)) % 360

    # Define coordinate labels based on coordinate mode
    ylab, xlab = ("lat", "lon") if coords_mode == "SPHERICAL" else ("y", "x")

    if converted_coords:
        # Convert longitudes back to -180 to 180 range if they were converted
        cg_lon = np.where(cg_lon > 180, cg_lon - 360, cg_lon)

    # Return results as xarray Dataset
    return xr.Dataset(
        {
            "W": ((ylab, xlab, "time"), W, {"units": "m/s"}),
            "Dir": ((ylab, xlab, "time"), D, {"units": "º"}),
            "p": ((ylab, xlab, "time"), p, {"units": "Pa"}),
        },
        coords={ylab: cg_lat, xlab: cg_lon, "time": times},
    )