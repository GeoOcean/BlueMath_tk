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
        "xpc": np.nanmin(bathy_data.lon),  # x origin
        "ypc": np.nanmin(bathy_data.lat),  # y origin
        "alpc": 0,  # x-axis direction
        "xlenc": np.nanmax(bathy_data.lon)
        - np.nanmin(bathy_data.lon),  # grid length in x
        "ylenc": np.nanmax(bathy_data.lat)
        - np.nanmin(bathy_data.lat),  # grid length in y
        "mxc": len(bathy_data.lon) - 1,  # number mesh x, una menos pq si no SWAN peta
        "myc": len(bathy_data.lat) - 1,  # number mesh y, una menos pq si no SWAN peta
        "dxinp": bathy_data.lon[1].values
        - bathy_data.lon[0].values,  # size mesh x (resolution in x)
        "dyinp": bathy_data.lat[1].values
        - bathy_data.lat[0].values,  # size mesh y (resolution in y)
    }
