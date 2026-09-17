"""
BinWaves utilities for processing SWAN input and output files.
"""

import numpy as np
import xarray as xr
from wavespectra.input.swan import read_swan


def generate_swan_cases_and_fixed_parameters(
    frequencies_array: list | np.ndarray,
    directions_array: list | np.ndarray,
) -> tuple[dict, dict]:
    """
    Generate the SWAN cases dictionary and fixed parameters.

    Parameters
    ----------
    frequencies_array : list | np.ndarray
        The frequencies array.
    directions_array : list | np.ndarray
        The directions array.

    Returns
    -------
    tuple[dict, dict]
        A tuple containing the SWAN monocromatic cases dictionary with keys as case IDs
        and values as dictionaries containing frequency and direction,
        and the fixed parameters.
    """

    if len(frequencies_array) != len(np.unique(frequencies_array)):
        raise ValueError("The frequencies_array contains duplicate values.")
    if len(directions_array) != len(np.unique(directions_array)):
        raise ValueError("The directions_array contains duplicate values.")

    return {
        "dm": directions_array,
        "fp": frequencies_array,
    }, {
        "mdc": len(directions_array),
        "flow": min(frequencies_array),
        "fhigh": max(frequencies_array),
        "freq_discretization": len(frequencies_array),
        "dir_discretization": len(directions_array),
        "frequencies_array": frequencies_array,
        "directions_array": directions_array,
    }


def process_kp_coefficients(
    list_of_input_spectra: list[str],
    list_of_output_spectra: list[str],
) -> xr.Dataset:
    """
    Process the kp coefficients from the output and input spectra.

    Parameters
    ----------
    list_of_input_spectra : list[str]
        The list of input spectra files.
    list_of_output_spectra : list[str]
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


def transform_spectra_to_binwaves(
    spectra_dataset: xr.Dataset,
    kps_dataset: xr.Dataset,
) -> xr.Dataset:
    """
    Transform the wave spectra to binwaves format.

    Parameters
    ----------
    spectra_dataset : xr.Dataset
        The wave spectra dataset.
    kps_dataset : xr.Dataset
        The kp coefficients dataset.

    Returns
    -------
    spectra_binwaves_format : xr.Dataset
        The wave spectra dataset in binwaves format with case_num dimension.
    """

    case_num_spectra = []
    for case_num, (case_dir, case_freq) in enumerate(
        zip(
            kps_dataset["dm"].values,
            kps_dataset["fp"].values,
        )
    ):
        try:
            closest_case = (
                spectra_dataset.efth.sel(
                    freq=case_freq, method="nearest", tolerance=0.001
                )
                .sel(dir=case_dir, method="nearest", tolerance=1.0)
                .expand_dims({"case_num": [case_num]})
            )
            case_num_spectra.append(closest_case)
        except Exception as _e:
            # Add a zeros array if the case number is not available
            case_num_spectra.append(
                xr.zeros_like(spectra_dataset.efth.isel(freq=0, dir=0)).expand_dims(
                    {"case_num": [case_num]}
                )
            )

    return (
        xr.concat(case_num_spectra, dim="case_num").drop_vars("dir").drop_vars("freq")
    )


def reconstruct_spectra(
    offshore_spectra: xr.DataArray,
    kp_coeffs: xr.Dataset,
) -> xr.Dataset:
    """
    Reconstruct onshore spectra from offshore spectra and kp coefficients.

    Parameters
    ----------
    offshore_spectra : xr.DataArray
        Offshore spectral energy binned by SWAN case, with dims
        `(time, case_num)`.
    kp_coeffs : xr.Dataset
        Propagation coefficients with data variable `"kps"` and dims
        `(case_num, site, freq, dir)`.

    Returns
    -------
    xr.Dataset
        Reconstructed onshore spectra: data variable `"kps"`, dims
        `(time, site, freq, dir)`.
    """

    kp = kp_coeffs["kps"].transpose("case_num", "site", "freq", "dir")
    kp_matrix = kp.values.reshape(kp.sizes["case_num"], -1).astype(np.float32)

    offshore = offshore_spectra.transpose("time", "case_num")
    offshore_matrix = offshore.values.astype(np.float32)

    result = offshore_matrix @ kp_matrix  # (time, site * freq * dir)

    reconstructed = xr.DataArray(
        result.reshape(
            offshore.sizes["time"], kp.sizes["site"], kp.sizes["freq"], kp.sizes["dir"]
        ),
        dims=("time", "site", "freq", "dir"),
        coords={
            "time": offshore["time"],
            "site": kp["site"],
            "freq": kp["freq"],
            "dir": kp["dir"],
        },
        name="kps",
    )

    # Carry over auxiliary site/global coordinates (coord_x, coord_y, lat, lon, ...)
    extra_coords = {
        name: coord
        for name, coord in kp_coeffs.coords.items()
        if name not in reconstructed.coords
        and set(coord.dims) <= set(reconstructed.dims)
    }

    return reconstructed.assign_coords(extra_coords).to_dataset()
