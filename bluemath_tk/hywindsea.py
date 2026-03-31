import xarray as xr
import numpy as np
from bluemath_tk.core.operations import get_uv_components



def transform_rescale_wind_data(
    wind_file: xr.Dataset,
    bathymetry: xr.Dataset,
    percentile: float,
    umbral: float,
    day_hour: int,
    
    
    
    self, case_dir: str, case_context: dict) -> None:
    """
    Transform wind data for the HyWindSea model.
    """

    # Open wind data and slice for bathymetry adaptation
    wind = xr.open_dataset(wind_file).sel(lev=10).squeeze()
    wind_edit = wind.sel(
        lon=slice(
            bathymetry.lon.values.min(),
            case_context.get("bathy").lon.values.max(),
        ),
        lat=slice(
            case_context.get("bathy").lat.values.min(),
            case_context.get("bathy").lat.values.max(),
        ),
    )
    bathy_interp = case_context.get("bathy").interp(lon=wind_edit.lon, lat=wind_edit.lat)
    wind_edit["bathy"] = bathy_interp.depth

    # Calculate percentiles and modify wind speeds
    perc_dataset = wind_edit.where(wind_edit.bathy > 0).quantile(
        case_context.get("percentile") / 100, dim=["lon", "lat"]
    )
    alpha = case_context.get("umbral") / perc_dataset
    alpha["M"] = (
        "time",
        np.where(perc_dataset["M"] > case_context.get("umbral"), 1, alpha["M"]),
    )
    wind["M"] = wind["M"] * alpha["M"]

    # Calculate u10 and v10 components
    u10, v10 = get_uv_components(wind.Dir)
    wind["u10"] = -u10 * wind.M
    wind["v10"] = -v10 * wind.M	
    w = wind.isel(time=case_context.get("day_hour")).interp(
        lat=case_context.get("bathy").lat.values,
        lon=case_context.get("bathy").lon.values,
        method="linear",
    )

    # extract and save
    u10 = w.u10.values
    v10 = w.v10.values
    arr = np.vstack((u10, v10))
    
    
    return arr


from wavespectra.input.swan import read_swan
from typing import List

def process_kp_coefficients(
    list_of_input_spectra: List[str],
    list_of_output_spectra: List[str],
) -> xr.Dataset:
    """
    Process the kp coefficients from the output and input spectra.

    Parameters
    ----------
    list_of_input_spectra : List[str]
        The list of input spectra files.
    list_of_output_spectra : List[str]
        The list of output spectra files.

    Returns
    -------
    xr.Dataset
        The kp coefficients Dataset in frequency and direction.
    """

    output_kp_list = []

    for i, (input_spec_file, output_spec_file) in enumerate(
        zip(list_of_input_spectra, list_of_output_spectra)
    ):
        try:
            input_spec = read_swan(input_spec_file).squeeze().efth
            output_spec = (
                read_swan(output_spec_file)
                .efth.squeeze()
                .drop_vars("time")
                .expand_dims({"case_num": [i]})
            )
            kp = output_spec / input_spec.sum(dim=["freq", "dir"])
            output_kp_list.append(kp)
        except Exception as e:
            print(f"Error processing {input_spec_file} and {output_spec_file}")
            print(e)

    # Concat files one by one
    concatened_kp = output_kp_list[0]
    for file in output_kp_list[1:]:
        concatened_kp = xr.concat([concatened_kp, file], dim="case_num")

    return concatened_kp.fillna(0.0).sortby("freq").sortby("dir")
