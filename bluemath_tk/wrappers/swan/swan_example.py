import os.path as op

import numpy as np
import wavespectra
import xarray as xr
from wavespectra.construct import construct_partition

from bluemath_tk.waves.binwaves import generate_swan_cases
from bluemath_tk.wrappers.swan.swan_wrapper import SwanModelWrapper

example_directions = np.linspace(0, 360, 24)
example_frequencies = np.linspace(0.03, 0.5, 29)


class BinWavesWrapper(SwanModelWrapper):
    """
    Wrapper example for the BinWaves model.
    """

    def build_case(self, case_dir: str, case_context: dict) -> None:
        """
        Build the input files for the BinWaves model.
        """

        input_spectrum = construct_partition(
            freq_name="jonswap",
            freq_kwargs={
                "freq": sorted(example_frequencies),
                "fp": 1.0 / case_context.get("tp"),
                "hs": case_context.get("hs"),
            },
            dir_name="cartwright",
            dir_kwargs={
                "dir": sorted(example_directions),
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
        for side in ["N", "S", "E", "W"]:
            wavespectra.SpecDataset(mono_input_spectrum).to_swan(
                op.join(case_dir, f"input_spectra_{side}.bnd")
            )


# Usage example
if __name__ == "__main__":
    # Define the input templates and output directory
    templates_dir = (
        "/home/tausiaj/GitHub-GeoOcean/BlueMath_tk/bluemath_tk/wrappers/swan/templates"
    )
    output_dir = "/home/tausiaj/GitHub-GeoOcean/BlueMath_tk/test_cases/swan/CAN_good/"
    # Generate swan model parameters
    metamodel_parameters = (
        generate_swan_cases(
            directions_array=example_directions,
            frequencies_array=example_frequencies,
        )
        .astype(float)
        .to_dataframe()
        .iloc[::33]
        .reset_index()
        .to_dict(orient="list")
    )
    # Create an instance of the SWAN model wrapper
    swan_wrapper = BinWavesWrapper(
        templates_dir=templates_dir,
        metamodel_parameters=metamodel_parameters,
        fixed_parameters={},
        output_dir=output_dir,
    )
    # Build the input files
    swan_wrapper.build_cases(mode="one_by_one")
    # List available launchers
    print(swan_wrapper.list_available_launchers())
    # Run the model
    # swan_wrapper.run_cases(launcher="docker_serial", num_workers=10)
    # Post-process the output files
    # postprocessed_ds = swan_wrapper.postprocess_cases()
    # print(postprocessed_ds)
