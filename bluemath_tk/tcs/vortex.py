import numpy as np
import pandas as pd
import xarray as xr

from ..core.geo import geodesic_distance, geo_distance_cartesian

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


def vortex_model_grid(
    storm_track: pd.DataFrame,
    cg_lon: np.ndarray,
    cg_lat: np.ndarray,
    coords_mode: str = "SPHERICAL",
) -> xr.Dataset:
    """

    The Dynamic Holland Model is used to generate wind vortex fields from
    storm track coordinate parameters.

    Parameters
    ----------
    storm_track : pd.DataFrame
        Obligatory fields: vfx, vfy, p0, pn, vmax, rmw.
        For SPHERICAL coordinates: lon, lat.
        For CARTESIAN coordinates: x, y, latitude.
    cg_lon : np.ndarray
        Computational grid in longitudes.
    cg_lat : np.ndarray
        Computational grid in latitudes.
    coords_mode : str
        'SPHERICAL' / 'CARTESIAN' swan project coordinates mode. Default is "SPHERICAL".

    Returns
    -------
    xr.Dataset
        xarray.Dataset with wind speed W and direction Dir (º from north).
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
        if coords_mode == "SPHERICAL":
            r = geodesic_distance(lat2d, lon2d, la, lo)
        else:
            r = geo_distance_cartesian(lat2d, lon2d, la, lo)

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
