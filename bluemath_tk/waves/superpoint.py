from typing import List, Tuple

import numpy as np
import xarray as xr


def stations_superposition(
    stations_data: xr.Dataset,
    sectors: List[Tuple[float, float]],
    degrees_superposition: float,
    station_wind_id: str,
    efth_to_rad: bool = True,
    freq_name: str = "frequency",
    dir_name: str = "direction",
    efth_name: str = "Efth",
    wspeed_name: str = "u10m",
    wdir_name: str = "udir",
) -> xr.Dataset:
    """
    Join multiple station spectral data for each directional sector with superposition.

    This function combines wave spectra from multiple stations, each responsible for a specific
    directional sector. It handles overlapping regions through superposition and includes
    wind data from a specified reference station.

    Parameters
    ----------
    stations_data : xr.Dataset
        Xarray Dataset containing spectral data for each station.
    sectors : List[Tuple[float, float]]
        List of directional sectors for each station in degrees.
        Each tuple contains (start_angle, end_angle) where:
        - start_angle: beginning of sector in degrees [0, 360]
        - end_angle: end of sector in degrees [0, 360]
    degrees_superposition : float
        Degrees of superposition at sector boundaries.
    station_wind_id : str
        Station ID from which to extract wind data.
    efth_to_rad : bool, optional
        Convert energy frequency-direction spectrum to radians. Default is True.
    freq_name : str, optional
        Name of frequency dimension in datasets. Default is "frequency".
    dir_name : str, optional
        Name of direction dimension in datasets. Default is "direction".
    efth_name : str, optional
        Name of energy spectrum variable in datasets. Default is "Efth".
    wspeed_name : str, optional
        Name of wind speed variable in datasets. Default is "u10m".
    wdir_name : str, optional
        Name of wind direction variable in datasets. Default is "udir".

    Returns
    -------
    xr.Dataset
        Combined dataset containing:
        - efth: Energy frequency-direction spectrum
        - wspeed: Wind speed from reference station
        - wdir: Wind direction from reference station
        - depth: Water depth values

    Notes
    -----
    - The function handles circular sectors (e.g., 350° to 10°) correctly
    - Superposition is handled by averaging in overlapping regions
    - Time dimension is rounded to hourly values
    """

    # Get dimensions from first station
    efth_all = np.full(
        [
            len(stations_data.time),
            len(stations_data[freq_name]),
            len(stations_data[dir_name]),
            len(stations_data.point),
        ],
        0.0,
    )
    cont = np.full([len(stations_data[dir_name])], 0)

    wsp = None
    wdir = None
    depth = None

    # Process each station
    for s_ix in stations_data.point:
        station_data = stations_data.sel(point=s_ix)
        sector = sectors[s_ix]

        # Find station data indexes inside sector (and superposition degrees)
        if (sector[1] - sector[0]) < 0:  # Handles circular sectors (e.g., 350° to 10°)
            d = np.where(
                (station_data[dir_name].values > sector[0] - degrees_superposition)
                | (station_data[dir_name].values <= sector[1] + degrees_superposition)
            )[0]
        else:
            d = np.where(
                (station_data[dir_name].values > sector[0] - degrees_superposition)
                & (station_data[dir_name].values <= sector[1] + degrees_superposition)
            )[0]

        cont[d] += 1
        efth_all[:, :, d, s_ix] = station_data[efth_name][:, :, d]

        # Get wind data from chosen wind station
        if s_ix == station_wind_id:
            wsp = station_data[wspeed_name].values
            wdir = station_data[wdir_name].values
            depth = np.full([len(station_data.time.values)], station_data.depth)

    if wsp is None or wdir is None or depth is None:
        raise ValueError(
            f"Wind station {station_wind_id} not found in provided stations"
        )

    # Average superimposed station data
    efth_all = np.sum(efth_all, axis=3) / np.maximum(cont, 1)  # Avoid division by zero
    if efth_to_rad:
        efth_all = efth_all * (np.pi / 180)

    # Create output dataset
    super_point = xr.Dataset(
        {
            "efth": (["time", "freq", "dir"], efth_all),
            "wspeed": (["time"], wsp),
            "wdir": (["time"], wdir),
            "depth": (["time"], depth),
        },
        coords={
            "time": stations_data.time.values,
            "dir": stations_data[dir_name].values,
            "freq": stations_data[freq_name].values,
        },
    )

    # Round time to hour
    super_point["time"] = super_point["time"].dt.round("H").values

    return super_point
