import os.path as op

import numpy as np
import wavespectra
import xarray as xr
from wavespectra.construct import construct_partition

from bluemath_tk.waves.binwaves import generate_swan_cases
from bluemath_tk.wrappers.swan.swan_wrapper import SwanModelWrapper

example_directions = [
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
example_frequencies = [
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
    output_dir = "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_cases/swan/CAN_part/"
    # Generate swan model parameters
    model_parameters = (
        generate_swan_cases(
            directions_array=example_directions,
            frequencies_array=example_frequencies,
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
