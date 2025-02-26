import os.path as op

import numpy as np
import wavespectra
import xarray as xr
from wavespectra.construct import construct_partition

from bluemath_tk.waves.binwaves import (
    generate_swan_cases,
    process_kp_coefficients,
    reconstruc_spectra,
)
from bluemath_tk.wrappers.swan.swan_wrapper import SwanModelWrapper

example_directions = [
    1.5,
    4.5,
    7.5,
    10.5,
    13.5,
    16.5,
    19.5,
    22.5,
    25.5,
    28.5,
    31.5,
    34.5,
    37.5,
    40.5,
    43.5,
    46.5,
    49.5,
    52.5,
    55.5,
    58.5,
    61.5,
    64.5,
    67.5,
    70.5,
    73.5,
    76.5,
    79.5,
    82.5,
    85.5,
    88.5,
    91.5,
    94.5,
    97.5,
    100.5,
    103.5,
    106.5,
    109.5,
    112.5,
    115.5,
    118.5,
    121.5,
    124.5,
    127.5,
    130.5,
    133.5,
    136.5,
    139.5,
    142.5,
    145.5,
    148.5,
    151.5,
    154.5,
    157.5,
    160.5,
    163.5,
    166.5,
    169.5,
    172.5,
    175.5,
    178.5,
    181.5,
    184.5,
    187.5,
    190.5,
    193.5,
    196.5,
    199.5,
    202.5,
    205.5,
    208.5,
    211.5,
    214.5,
    217.5,
    220.5,
    223.5,
    226.5,
    229.5,
    232.5,
    235.5,
    238.5,
    241.5,
    244.5,
    247.5,
    250.5,
    253.5,
    256.5,
    259.5,
    262.5,
    265.5,
    268.5,
    271.5,
    274.5,
    277.5,
    280.5,
    283.5,
    286.5,
    289.5,
    292.5,
    295.5,
    298.5,
    301.5,
    304.5,
    307.5,
    310.5,
    313.5,
    316.5,
    319.5,
    322.5,
    325.5,
    328.5,
    331.5,
    334.5,
    337.5,
    340.5,
    343.5,
    346.5,
    349.5,
    352.5,
    355.5,
    358.5,
]
example_frequencies = [
    0.03,
    0.033,
    0.0363,
    0.0399,
    0.0438,
    0.0482,
    0.053,
    0.0582,
    0.064,
    0.0704,
    0.0774,
    0.0851,
    0.0935,
    0.1028,
    0.1131,
    0.1243,
    0.1367,
    0.1503,
    0.1652,
    0.1816,
    0.1997,
    0.2195,
    0.2413,
    0.2653,
    0.2917,
    0.3207,
    0.3526,
    0.3876,
    0.4262,
    0.4685,
    0.5151,
    0.5663,
    0.6226,
    0.6845,
    0.7525,
    0.8273,
    0.9096,
    1.0,
]

laura_directions = [
    262.5,
    247.5,
    232.5,
    217.5,
    202.5,
    187.5,
    172.5,
    157.5,
    142.5,
    127.5,
    112.5,
    97.5,
    82.5,
    67.5,
    52.5,
    37.5,
    22.5,
    7.5,
    352.5,
    337.5,
    322.5,
    307.5,
    292.5,
    277.5,
]
laura_frequencies = [
    0.03500004,
    0.03850004,
    0.04234991,
    0.04658508,
    0.05124342,
    0.05636788,
    0.06200474,
    0.06820491,
    0.07502551,
    0.082528,
    0.09078117,
    0.0998592,
    0.10984545,
    0.12082986,
    0.13291333,
    0.14620311,
    0.16082342,
    0.17690661,
    0.19459796,
    0.21405484,
    0.23546032,
    0.25900697,
    0.2849084,
    0.31340103,
    0.34474437,
    0.37921881,
    0.41713594,
    0.45884188,
    0.5047446,
]


def transform_CAWCR_WS(
    cawcr_dataset: xr.Dataset,
    subset_parameters: dict,
    available_case_num: np.ndarray,
) -> xr.Dataset:
    """
    Transform the wave spectra from CAWCR format to binwaves format.

    Parameters
    ----------
    cawcr_dataset : xr.Dataset
        The wave spectra dataset in CAWCR format.
    subset_parameters : dict
        A dictionary containing parameters for the subset processing.
    available_case_num : np.ndarray
        The available case numbers.

    Returns
    -------
    xr.Dataset
        The wave spectra dataset in binwaves format.
    """

    # First, reproject the wave spectra to the binwaves format
    ds = cawcr_dataset.rename({"frequency": "freq", "direction": "dir"})
    ds["efth"] = ds["efth"] * np.pi / 180.0
    ds["dir"] = ds["dir"] - 180.0
    ds["dir"] = np.where(ds["dir"] < 0, ds["dir"] + 360, ds["dir"])
    ds = ds.sortby("dir").sortby("freq")

    # Second, reproject into the available case numbers dimension
    case_num_spectra = []
    for case_num, case_dir, case_freq in zip(
        available_case_num,
        np.array(subset_parameters.get("dir"))[available_case_num],
        np.array(subset_parameters.get("freq"))[available_case_num],
    ):
        case_num_spectra.append(
            ds.efth.sel(freq=case_freq, dir=case_dir, method="nearest").expand_dims(
                {"case_num": [case_num]}
            )
        )
    ds_case_num = (
        xr.concat(case_num_spectra, dim="case_num").drop_vars("dir").drop_vars("freq")
    )

    return ds, ds_case_num


class BinWavesWrapper(SwanModelWrapper):
    """
    Wrapper example for the BinWaves model.
    """

    def build_case(self, case_dir: str, case_context: dict):
        input_spectrum = construct_partition(
            freq_name="jonswap",
            freq_kwargs={
                "freq": example_frequencies,
                "fp": 1.0 / case_context.get("tp"),
                "hs": case_context.get("hs"),
            },
            dir_name="cartwright",
            dir_kwargs={
                "dir": example_directions,
                "dm": case_context.get("dir"),
                "dspr": case_context.get("spr"),
            },
        )
        argmax_bin = np.argmax(input_spectrum.values)
        mono_spec_array = np.zeros(input_spectrum.freq.size * input_spectrum.dir.size)
        mono_spec_array[argmax_bin] = input_spectrum.sum(dim=["freq", "dir"])
        mono_spec_array = mono_spec_array.reshape(
            input_spectrum.freq.size, input_spectrum.dir.size
        )
        mono_input_spectrum = xr.Dataset(
            {
                "efth": (["freq", "dir"], mono_spec_array),
            },
            coords={
                "freq": input_spectrum.freq,
                "dir": input_spectrum.dir,
            },
        )
        wavespectra.SpecDataset(mono_input_spectrum).to_swan(
            op.join(case_dir, "input_spectra.bnd")
        )

    def build_cases(self, mode="one_by_one"):
        super().build_cases(mode)
        for case_dir, case_context in zip(self.cases_dirs, self.cases_context):
            self.build_case(case_dir, case_context)


# Usage example
if __name__ == "__main__":
    # Define the input templates and output directory
    templates_dir = (
        "/home/tausiaj/GitHub-GeoOcean/BlueMath/bluemath_tk/wrappers/swan/templates/"
    )
    templates_name = ["input.swn", "depth_main_cantabria.dat", "buoys.loc"]
    output_dir = "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_cases/swan/CAN_mono/"
    # Generate swan model parameters
    model_parameters = (
        generate_swan_cases(
            directions_array=laura_directions,
            frequencies_array=laura_frequencies,
        )
        .astype(float)
        .to_dataframe()
        .reset_index()
        .to_dict(orient="list")
    )
    # Create an instance of the SWAN model wrapper
    swan_wrapper = BinWavesWrapper(
        templates_dir=templates_dir,
        templates_name=templates_name,
        model_parameters=model_parameters,
        output_dir=output_dir,
    )
    # Build the input files
    # swan_wrapper.build_cases(mode="one_by_one")
    # Set the cases directories from the output directory
    swan_wrapper.set_cases_dirs_from_output_dir()
    # List available launchers
    # print(swan_wrapper.list_available_launchers())
    # Run the model
    # swan_wrapper.run_cases(launcher="docker", parallel=True)
    # Post-process the output files
    # postprocessed_ds = swan_wrapper.postprocess_cases()
    # postprocessed_ds.to_netcdf(op.join(swan_wrapper.output_dir, "waves_part.nc"))
    # print(postprocessed_ds)
    # Get input and ouput spectra files from self.cases_dirs
    input_files = [op.join(d, "input_spectra.bnd") for d in swan_wrapper.cases_dirs]
    output_files = [op.join(d, "output.spec") for d in swan_wrapper.cases_dirs]
    # Extract binwaves kp coeffs
    kp_coeffs = process_kp_coefficients(
        list_of_input_spectra=input_files,
        list_of_output_spectra=output_files,
    )
    # Load interest spectra
    _, offshore_spectra = transform_CAWCR_WS(
        cawcr_dataset=xr.open_dataset(
            "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_data/ERA5_full.nc"
        ),
        subset_parameters=model_parameters,
        available_case_num=kp_coeffs.case_num.values,
    )
    # Reconstruct spectra
    onshore_spectra = reconstruc_spectra(
        offshore_spectra=offshore_spectra,
        kp_coeffs=kp_coeffs,
    )
    print(onshore_spectra)
