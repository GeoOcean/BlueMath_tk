import numpy as np
import xarray as xr


def generate_grid_parameters(bathy_data: xr.DataArray) -> dict:
    """
    Generate the grid parameters for the SWAN model.

    Parameters
    ----------
    bathy_data : xr.DataArray
        Bathymetry data.
        Must have the following dimensions:
        - lon: longitude
        - lat: latitude

    Returns
    -------
    dict
        Grid parameters for the SWAN model.

    Contact @bellidog on GitHub for more information.
    """

    return {
        "xpc": int(np.nanmin(bathy_data.lon)),  # x origin
        "ypc": int(np.nanmin(bathy_data.lat)),  # y origin
        "alpc": 0,  # x-axis direction
        "xlenc": int(
            np.nanmax(bathy_data.lon) - np.nanmin(bathy_data.lon)
        ),  # grid length in x
        "ylenc": int(
            np.nanmax(bathy_data.lat) - np.nanmin(bathy_data.lat)
        ),  # grid length in y
        "mxc": len(bathy_data.lon) - 1,  # number mesh x, una menos pq si no SWAN peta
        "myc": len(bathy_data.lat) - 1,  # number mesh y, una menos pq si no SWAN peta
        "xpinp": np.nanmin(bathy_data.lon),  # x origin
        "ypinp": np.nanmin(bathy_data.lat),  # y origin
        "alpinp": 0,  # x-axis direction
        "mxinp": len(bathy_data.lon) - 1,  # number mesh x
        "myinp": len(bathy_data.lat) - 1,  # number mesh y
        "dxinp": abs(
            bathy_data.lon[1].values - bathy_data.lon[0].values
        ),  # size mesh x (resolution in x)
        "dyinp": abs(
            bathy_data.lat[1].values - bathy_data.lat[0].values
        ),  # size mesh y (resolution in y)
    }
