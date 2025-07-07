from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from ..core.geo import geo_distance_cartesian, geodesic_distance

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
    Generate wind vortex fields from storm track parameters using the Dynamic Holland Model.

    Parameters
    ----------
    storm_track : pd.DataFrame
        DataFrame containing storm track parameters.
        - obligatory fields: vfx, vfy, p0, pn, vmax, rmw
        - for SPHERICAL coordinates: lon, lat
        - for CARTESIAN coordinates: x, y, latitude
    cg_lon : np.ndarray
        Computational grid longitudes.
    cg_lat : np.ndarray
        Computational grid latitudes.
    cg_lon, cg_lat : np.ndarray
        Computational grid in longitudes and latitudes.
    coords_mode : str
        'SPHERICAL' for spherical coordinates (latitude, longitude),
        'CARTESIAN' for Cartesian coordinates (x, y).

    Returns
    -------
    xarray.Dataset
        Dataset containing wind speed W, direction Dir (º from north),
        and pressure p at each grid point.

    Examples
    --------
    >>> storm_track = pd.DataFrame({
    ...     'vfx': [10, 12], 'vfy': [5, 6],
    ...     'p0': [1000, 990], 'pn': [980, 970],
    ...     'vmax': [50, 55], 'rmw': [30, 35],
    ...     'lon': [10, 12], 'lat': [20, 22]
    ... })
    >>> cg_lon = np.array([10, 11, 12])
    >>> cg_lat = np.array([20, 21, 22])
    >>> coords_mode = 'SPHERICAL'
    >>> result = vortex_model_grid(storm_track, cg_lon, cg_lat, coords_mode)
    >>> print(result)
    <xarray.Dataset>
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
    * lat      (lat) float64 20.0 21.0 22.0
    * lon      (lon) float64 10.0 11.0 12.0
    * time     (time) datetime64[ns] 2023-10-01 2023-10-02
    Data variables:
    W        (lat, lon, time) float64 0.0 0.0 0.0 ... 0.0 0.0 0.0
    Dir      (lat, lon, time) float64 0.0 0.0 0.0 ... 0.0 0.0 0.0
    p        (lat, lon, time) float64 0.0 0.0 0.0 ... 0.0 0.0 0.0
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
            geo_dis = geodesic_distance(lat2d, lon2d, la, lo)
            RE = 6378.135 * 1000  # Earth radius [m]
            r = geo_dis * np.pi / 180.0 * RE
        elif coords_mode == "CARTESIAN":
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
        rm = rmw[i] * 1.852 * 1000  # [m]
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
            "p": ((ylab, xlab, "time"), p * 100, {"units": "Pa"}),
        },
        coords={ylab: cg_lat, xlab: cg_lon, "time": times},
    )


def vortex2delft_3D_FM_nc(
    mesh: xr.Dataset,
    ds_vortex: xr.Dataset,
) -> xr.Dataset:
    """
    Convert the vortex dataset to a Delft3D FM compatible netCDF forcing file.

    Parameters
    ----------
    mesh : xarray.Dataset
        The mesh dataset containing the node coordinates.
    ds_vortex : xarray.Dataset
        The vortex dataset containing wind speed and pressure data.
    path_output : str
        The output path where the netCDF file will be saved.
    ds_name : str
        The name of the output netCDF file, default is "forcing_Tonga_vortex.nc".
    forcing_ext : str
        The extension for the forcing file, default is "GreenSurge_GFDcase_wind.ext".
    Returns
    -------
    xarray.Dataset
        A dataset containing the interpolated wind speed and pressure data,
        ready for use in Delft3D FM.
    """
    longitude = mesh.mesh2d_node_x.values
    latitude = mesh.mesh2d_node_y.values
    n_time = ds_vortex.time.size

    lat_interp = xr.DataArray(latitude, dims="node")
    lon_interp = xr.DataArray(longitude, dims="node")

    angle = np.deg2rad((270 - ds_vortex.Dir.values) % 360)
    W = ds_vortex.W.values

    ds_vortex_interp = xr.Dataset(
        {
            "windx": (
                ("latitude", "longitude", "time"),
                (W * np.cos(angle)).astype(np.float32),
            ),
            "windy": (
                ("latitude", "longitude", "time"),
                (W * np.sin(angle)).astype(np.float32),
            ),
            "airpressure": (
                ("latitude", "longitude", "time"),
                ds_vortex.p.values.astype(np.float32),
            ),
        },
        coords={
            "latitude": ds_vortex.lat.values,
            "longitude": ds_vortex.lon.values,
            "time": np.arange(n_time),
        },
    )

    forcing_dataset = ds_vortex_interp.interp(latitude=lat_interp, longitude=lon_interp)

    reference_date_str = (
        ds_vortex.time.values[0]
        .astype("M8[ms]")
        .astype(datetime)
        .strftime("%Y-%m-%d %H:%M:%S")
    )

    forcing_dataset["windx"].attrs = {
        "coordinates": "time node",
        "long_name": "Wind speed in x direction",
        "standard_name": "windx",
        "units": "m s-1",
    }
    forcing_dataset["windy"].attrs = {
        "coordinates": "time node",
        "long_name": "Wind speed in y direction",
        "standard_name": "windy",
        "units": "m s-1",
    }

    forcing_dataset["airpressure"].attrs = {
        "coordinates": "time node",
        "long_name": "Atmospheric Pressure",
        "standard_name": "air_pressure",
        "units": "Pa",
    }

    forcing_dataset["time"].attrs = {
        "standard_name": "time",
        "long_name": f"Time - hours since {reference_date_str} +00:00",
        "time_origin": f"{reference_date_str}",
        "units": f"hours since {reference_date_str} +00:00",
        "calendar": "gregorian",
        "description": "Time definition for the forcing data",
    }

    forcing_dataset["longitude"].attrs = {
        "description": "Longitude of each mesh node of the computational grid",
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
    }
    forcing_dataset["latitude"].attrs = {
        "description": "Latitude of each mesh node of the computational grid",
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
    }

    return forcing_dataset
