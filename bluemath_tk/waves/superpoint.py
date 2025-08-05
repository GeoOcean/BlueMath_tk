from typing import Dict, Tuple

import xarray as xr


def superpoint_calculation(
    stations_data: xr.DataArray,
    stations_dimension_name: str,
    sectors_for_each_station: Dict[str, Tuple[float, float]],
    overlap_angle: float = 0.0,
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
    overlap_angle : float
        Angle in degrees that defines the overlap between two sectors.

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

    if overlap_angle == 0:
        for station_id, (dir_min, dir_max) in sectors_for_each_station.items():
            if dir_min < dir_max:
                mask = (stations_data["dir"] >= dir_min) & (
                    stations_data["dir"] < dir_max
                )
            else:
                # Handle wrap-around (e.g., 350° to 10°)
                mask = (stations_data["dir"] >= dir_min) | (
                    stations_data["dir"] < dir_max
                )
            superpoint_dataarray += stations_data.sel(
                {stations_dimension_name: station_id}
            ).where(mask, 0.0)

    else:
        # With overlap - expand sectors and average overlaps
        directions = stations_data["dir"]
        count_array = xr.zeros_like(superpoint_dataarray)  # Counter for overlaps

        for station_id, (dir_min, dir_max) in sectors_for_each_station.items():
            station_data = stations_data.sel({stations_dimension_name: station_id})

            # Expand sector boundaries by overlap_angle
            if (dir_max - dir_min) < 0:
                mask = (directions >= dir_min - overlap_angle) | (
                    directions <= dir_max + overlap_angle
                )
            else:
                mask = (directions >= dir_min - overlap_angle) & (
                    directions <= dir_max + overlap_angle
                )

            # Add contribution where mask is true
            superpoint_dataarray += station_data.where(mask, 0.0)
            count_array += xr.where(mask, 1, 0)

        # Average where there are overlaps (count > 1)
        overlap_mask = count_array > 1
        superpoint_dataarray = xr.where(
            overlap_mask, superpoint_dataarray / count_array, superpoint_dataarray
        )

    return superpoint_dataarray
