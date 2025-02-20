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


class BinWavesWrapper(SwanModelWrapper):
    """ """

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
    templates_name = ["input.swn", "depth_main_cantabria.dat", "buoys.loc"]
    output_dir = "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_cases/swan/CAN/"
    # Load swan model parameters
    model_parameters = (
        xr.open_dataset("/home/tausiaj/GitHub-GeoOcean/BlueMath/test_data/subset.nc")
        .to_dataframe()
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
    # swan_wrapper.run_cases(launcher="docker", parallel=True)
    # Post-process the output files
    # postprocessed_ds = swan_wrapper.postprocess_cases()
    # postprocessed_ds.to_netcdf(op.join(swan_wrapper.output_dir, "waves_part.nc"))
    # print(postprocessed_ds)
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
