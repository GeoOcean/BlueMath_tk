from typing import Dict, Tuple

import xarray as xr


def superpoint_calculation(
    stations_data: xr.DataArray,
    stations_dimension_name: str,
    sectors_for_each_station: Dict[str, Tuple[float, float]],
) -> xr.DataArray:
    """
    Join multiple station spectral data for each directional sector.

    Parameters
    ----------
    stations_data : xr.DataArray
        DataArray containing spectral data for multiple stations.
    stations_dimension_name : str
        Name of the dimension representing different stations in the DataArray.
    sectors_for_each_station : Dict[str, Tuple[float, float]]
        Dictionary mapping each station ID to a tuple of (min_direction, max_direction)
        representing the directional sector for that station.

    Returns
    -------
    xr.DataArray
        A new DataArray where each point is the sum of spectral data from all stations
        for the specified directional sector.

    Notes
    -----
    If your stations_data is saved in different files, you can load them all and then
    concatenate them using xr.open_mfdataset function. Example below:

    ```python
    files = [
        "path/to/station1.nc",
        "path/to/station2.nc",
        "path/to/station3.nc"
    ]

    def load_station_data(ds: xr.Dataset) -> xr.DataArray:
        return ds.efth.expand_dims("station")

    stations_data = xr.open_mfdataset(
        files,
        concat_dim="station",
        preprocess=load_station_data,
    )
    ```
    """

    superpoint_dataarray = xr.zeros_like(
        stations_data.isel({stations_dimension_name: 0})
    )

    for station_id, (dir_min, dir_max) in sectors_for_each_station.items():
        # Use where to select specific directions for each point
        superpoint_dataarray += stations_data.sel(
            {stations_dimension_name: station_id}
        ).where(
            (stations_data["dir"] >= dir_min) & (stations_data["dir"] < dir_max),
            0.0,
        )

    return superpoint_dataarray
