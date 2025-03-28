import os.path as op

import numpy as np
import wavespectra
import xarray as xr
from wavespectra.construct import construct_partition

from bluemath_tk.waves.binwaves import generate_swan_cases
from bluemath_tk.wrappers._utils_wrappers import write_array_in_file
from bluemath_tk.wrappers.swan.swan_wrapper import SwanModelWrapper

example_directions = np.linspace(0, 360, 24)
example_frequencies = np.linspace(0.03, 0.5, 29)


class BinWavesWrapper(SwanModelWrapper):
    """
    Wrapper example for the BinWaves model.
    """

    def __init__(
        self,
        templates_dir: str,
        metamodel_parameters: dict,
        fixed_parameters: dict,
        output_dir: str,
        depth_dataarray: xr.DataArray = None,
        templates_name: dict = "all",
        debug: bool = True,
    ) -> None:
        """
        Initialize the SWAN model wrapper.
        """

        depth_array = depth_dataarray.values
        locations_x, locations_y = np.meshgrid(
            depth_dataarray.lon.values, depth_dataarray.lat.values
        )
        self.locations = np.column_stack((locations_x.ravel(), locations_y.ravel()))
        # Add Virgen del Mar exact buoy location
        self.locations = np.vstack((self.locations, [428845.10, 4815606.89]))
        self.locations = np.array([[428845.10, 4815606.89]])

        super().__init__(
            templates_dir=templates_dir,
            metamodel_parameters=metamodel_parameters,
            fixed_parameters=fixed_parameters,
            output_dir=output_dir,
            templates_name=templates_name,
            depth_array=depth_array,
            debug=debug,
        )

    def build_case(self, case_dir: str, case_context: dict) -> None:
        super().build_case(case_context=case_context, case_dir=case_dir)
        write_array_in_file(
            array=self.locations, filename=op.join(case_dir, "locations.loc")
        )

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
