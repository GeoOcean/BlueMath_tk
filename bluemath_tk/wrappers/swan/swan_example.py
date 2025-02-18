import os.path as op

import wavespectra
import xarray as xr
from wavespectra.construct import construct_partition

from bluemath_tk.topo_bathy.swan_grid import generate_grid_parameters
from bluemath_tk.waves.binwaves import (
    process_kp_coefficients,
    transform_CAWCR_WS,
)
from bluemath_tk.wrappers.swan.swan_wrapper import SwanModelWrapper

example_directions = [
    7.5,
    22.5,
    37.5,
    52.5,
    67.5,
    82.5,
    97.5,
    112.5,
    127.5,
    142.5,
    157.5,
    172.5,
    187.5,
    202.5,
    217.5,
    232.5,
    247.5,
    262.5,
    277.5,
    292.5,
    307.5,
    322.5,
    337.5,
    352.5,
]
example_frequencies = [
    0.035,
    0.0385,
    0.042349998,
    0.046585,
    0.051243503,
    0.05636785,
    0.062004633,
    0.068205096,
    0.07502561,
    0.082528174,
    0.090780996,
    0.099859096,
    0.10984501,
    0.120829515,
    0.13291247,
    0.14620373,
    0.1608241,
    0.17690653,
    0.19459718,
    0.21405691,
    0.2354626,
    0.25900885,
    0.28490975,
    0.31340075,
    0.3447408,
    0.37921488,
    0.4171364,
    0.45885003,
    0.50473505,
]


class BinWavesWrapper(SwanModelWrapper):
    """ """

    def build_case(self, case_dir: str, case_context: dict):
        self.logger.info(f"Saving spectrum for {case_dir}")
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
        wavespectra.SpecDataset(input_spectrum.to_dataset(name="efth")).to_swan(
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
    templates_name = ["input.swn", "depth_main.dat", "buoys.loc"]
    output_dir = "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_cases/swan/javi/"
    # Load swan model parameters
    model_parameters = (
        xr.open_dataset("/home/tausiaj/GitHub-GeoOcean/BlueMath/test_data/subset.nc")
        .to_dataframe()
        .iloc[::60]
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
    swan_wrapper.build_cases(mode="one_by_one")
    # List available launchers
    print(swan_wrapper.list_available_launchers())
    # Run the model
    swan_wrapper.run_cases(launcher="docker", parallel=True)
    # Post-process the output files
    postprocessed_ds = swan_wrapper.postprocess_cases()
    postprocessed_ds.to_netcdf(op.join(swan_wrapper.output_dir, "waves_part.nc"))
    print(postprocessed_ds)
    # # Load spectra example
    # spectra = xr.open_dataset(
    #     "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_data/Waves_Cantabria_356.08_43.82.nc"
    # )
    # spectra_transformed = transform_CAWCR_WS(spectra)
    # # Extract binwaves kp coeffs
    # kp_coeffs = process_kp_coefficients(
    #     swan_ds=postprocessed_ds,
    #     spectrum_freq=spectra_transformed.freq.values,
    #     spectrum_dir=spectra_transformed.dir.values,
    #     latitude=43.3,
    #     longitude=173.0,
    # )
    # print(kp_coeffs)
