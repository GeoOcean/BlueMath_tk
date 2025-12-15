import os
import re
from typing import List, Union

import numpy as np
import pandas as pd
import scipy.io as sio
import wavespectra
import xarray as xr
from wavespectra.construct import construct_partition

from .._base_wrappers import BaseModelWrapper
from .._utils_wrappers import write_array_in_file


class SwanModelWrapper(BaseModelWrapper):
    """
    Wrapper for the SWAN model.
    https://swanmodel.sourceforge.io/online_doc/swanuse/swanuse.html

    Attributes
    ----------
    default_parameters : dict
        The default parameters type for the wrapper.
    available_launchers : dict
        The available launchers for the wrapper.
    output_variables : dict
        The output variables for the wrapper.
    """

    default_parameters = {
        "Hs": {
            "type": float,
            "value": None,
            "description": "Significant wave height.",
        },
        "Tp": {
            "type": float,
            "value": None,
            "description": "Wave peak period.",
        },
        "Dir": {
            "type": float,
            "value": None,
            "description": "Wave direction.",
        },
        "Spr": {
            "type": float,
            "value": None,
            "description": "Directional spread.",
        },
        "dir_dist": {
            "type": str,
            "choices": ["CIRCLE", "SECTOR"],
            "value": "CIRCLE",
            "description": "CIRCLE indicates that the spectral directions cover the full circle. SECTOR indicates that the spectral directions cover a limited sector of the circle.",
        },
        "dir1": {
            "type": float,
            "value": None,
            "description": "Only with SECTOR option. The direction of the right-hand boundary of the sector when looking outward from the sector (in degrees).",
        },
        "dir2": {
            "type": float,
            "value": None,
            "description": "Only with SECTOR option. The direction of the left-hand boundary of the sector when looking outward from the sector (in degrees).",
        },
        "mdc": {
            "type": int,
            "value": 24,
            "description": "Spectral directional discretization.",
        },
        "flow": {
            "type": float,
            "value": 0.03,
            "description": "Low values for frequency.",
        },
        "fhigh": {
            "type": float,
            "value": 0.5,
            "description": "High value for frequency.",
        },
        "Freq_array": {
            "type": np.ndarray,
            "value": None,
            "description": "Array of frequencies for the model.",
        },
        "Dir_array": {
            "type": np.ndarray,
            "value": None,
            "description": "Array of directions for the model.",
        },
    }

    available_launchers = {
        "serial": "swan_serial.exe",
        "docker_serial": "docker run --rm -v .:/case_dir -w /case_dir geoocean/rocky8 swan_serial.exe",
        "geoocean-cluster": "launchSwan.sh",
    }

    output_variables = {
        "Depth": {
            "long_name": "Water depth at the point",
            "units": "m",
        },
        "Hsig": {
            "long_name": "Significant wave height",
            "units": "m",
        },
        "Tm02": {
            "long_name": "Mean wave period",
            "units": "s",
        },
        "Dir": {
            "long_name": "Wave direction",
            "units": "degrees",
        },
        "PkDir": {
            "long_name": "Peak wave direction",
            "units": "degrees",
        },
        "TPsmoo": {
            "long_name": "Peak wave period",
            "units": "s",
        },
        "Dspr": {
            "long_name": "Directional spread",
            "units": "degrees",
        },
    }

    def __init__(
        self,
        templates_dir: str,
        metamodel_parameters: dict,
        fixed_parameters: dict,
        output_dir: str,
        templates_name: dict = "all",
        depth_array: np.ndarray = None,
        locations: np.ndarray = None,
        debug: bool = True,
    ) -> None:
        """
        Initialize the SWAN model wrapper.
        """

        super().__init__(
            templates_dir=templates_dir,
            metamodel_parameters=metamodel_parameters,
            fixed_parameters=fixed_parameters,
            output_dir=output_dir,
            templates_name=templates_name,
            default_parameters=self.default_parameters,
        )
        self.set_logger_name(
            name=self.__class__.__name__, level="DEBUG" if debug else "INFO"
        )

        if depth_array is not None:
            self.depth_array = np.round(depth_array, 5)
        else:
            self.depth_array = None

        if locations is not None:
            self.locations = np.column_stack(locations)
        else:
            self.locations = None

    def list_available_output_variables(self) -> List[str]:
        """
        List available output variables.

        Returns
        -------
        List[str]
            The available output variables.
        """

        return list(self.output_variables.keys())

    def _convert_case_output_files_to_nc(
        self, case_num: int, output_path: str, output_vars: List[str]
    ) -> xr.Dataset:
        """
        Convert mat file to netCDF file.

        Parameters
        ----------
        case_num : int
            The case number.
        output_path : str
            The output path.
        output_vars : List[str]
            The output variables to use.

        Returns
        -------
        xr.Dataset
            The xarray Dataset.
        """

        # Read mat file
        output_dict = sio.loadmat(output_path)

        # Create Dataset
        ds_output_dict = {var: (("Yp", "Xp"), output_dict[var]) for var in output_vars}
        ds = xr.Dataset(
            ds_output_dict,
            coords={"Xp": output_dict["Xp"][0, :], "Yp": output_dict["Yp"][:, 0]},
        )

        # assign correct coordinate case_num
        ds.coords["case_num"] = case_num

        return ds

    def get_case_percentage_from_file(self, output_log_file: str) -> str:
        """
        Get the case percentage from the output log file.

        Parameters
        ----------
        output_log_file : str
            The output log file.

        Returns
        -------
        str
            The case percentage.
        """

        if not os.path.exists(output_log_file):
            return "0 %"

        progress_pattern = r"OK in\s+(\d+\.\d+)\s*%"
        with open(output_log_file, "r") as f:
            for line in reversed(f.readlines()):
                match = re.search(progress_pattern, line)
                if match:
                    if float(match.group(1)) > 99.5:
                        return "100 %"
                    return f"{match.group(1)} %"

        return "0 %"  # if no progress is found

    def monitor_cases(self, value_counts: str = None) -> Union[pd.DataFrame, dict]:
        """
        Monitor the cases based on the wrapper_out.log file.
        """

        cases_status = {}

        for case_dir in self.cases_dirs:
            output_log_file = os.path.join(case_dir, "wrapper_out.log")
            progress = self.get_case_percentage_from_file(
                output_log_file=output_log_file
            )
            cases_status[os.path.basename(case_dir)] = progress

        return super().monitor_cases(
            cases_status=cases_status, value_counts=value_counts
        )

    def postprocess_case(
        self,
        case_num: int,
        case_dir: str,
        output_vars: List[str] = ["Hsig", "Tm02", "Dir"],
    ) -> xr.Dataset:
        """
        Convert mat ouput files to netCDF file.

        Parameters
        ----------
        case_num : int
            The case number.
        case_dir : str
            The case directory.
        output_vars : list, optional
            The output variables to postprocess. Default is None.

        Returns
        -------
        xr.Dataset
            The postprocessed Dataset.
        """

        if output_vars is None:
            self.logger.info("Postprocessing all available variables.")
            output_vars = list(self.output_variables.keys())

        output_nc_path = os.path.join(case_dir, "output.nc")
        if not os.path.exists(output_nc_path):
            # Convert tab files to netCDF file
            output_path = os.path.join(case_dir, "output.mat")
            output_nc = self._convert_case_output_files_to_nc(
                case_num=case_num,
                output_path=output_path,
                output_vars=output_vars,
            )
            output_nc.to_netcdf(os.path.join(case_dir, "output.nc"))
        else:
            self.logger.info("Reading existing output.nc file.")
            output_nc = xr.open_dataset(output_nc_path)

        return output_nc

    def join_postprocessed_files(
        self, postprocessed_files: List[xr.Dataset]
    ) -> xr.Dataset:
        """
        Join postprocessed files in a single Dataset.

        Parameters
        ----------
        postprocessed_files : list
            The postprocessed files.

        Returns
        -------
        xr.Dataset
            The joined Dataset.
        """

        return xr.concat(postprocessed_files, dim="case_num")


def generate_fixed_parameters(
    grid_parameters: dict,
    freq_array: np.array,
    dir_array: np.array,
) -> dict:
    """
    Generate fixed parameters for the SWAN model based on grid parameters and frequency/direction arrays.
    Parameters
    ----------
    grid_parameters : dict
        Dictionary with grid configuration for SWAN input.
    freq_array : np.ndarray
        Array of frequencies for the SWAN model.
    dir_array : np.ndarray
        Array of directions for the SWAN model.
    Returns
    -------
    dict
        Dictionary with fixed parameters for the SWAN model.
    """

    dirs = np.sort(np.unique(dir_array)) % 360
    step = np.round(np.median(np.diff(np.sort(dirs))), 4)

    # Compute angular gaps between sorted directions (including wrap-around)
    diffs = np.diff(np.concatenate([dirs, [dirs[0] + 360]]))
    max_gap_idx = np.argmax(diffs)

    if np.isclose(diffs[max_gap_idx], step, atol=1e-2):
        dir_dist = "CIRCLE"
        dir1, dir2 = None, None
    else:
        dir_dist = "SECTOR"
        dir1 = float((dirs[(max_gap_idx + 1) % len(dirs)]) % 360)  # right-hand boundary
        dir2 = float((dirs[max_gap_idx]) % 360)  # left-hand boundary
    print("Distribución direccional:", dir_dist)
    if dir_dist == "SECTOR":
        print(f"Direcciones de {dir1}° a {dir2}°")

    return {
        "xpc": grid_parameters["xpc"],  # origin x
        "ypc": grid_parameters["ypc"],  # origin y
        "alpc": grid_parameters["alpc"],  # x-axis direction
        "xlenc": grid_parameters["xlenc"],  # grid length x
        "ylenc": grid_parameters["ylenc"],  # grid length y
        "mxc": grid_parameters["mxc"],  # num mesh x
        "myc": grid_parameters["myc"],  # num mesh y
        "xpinp": grid_parameters["xpinp"],  # origin x for input grid
        "ypinp": grid_parameters["ypinp"],  # origin y for input grid
        "alpinp": grid_parameters["alpinp"],  # x-axis direction
        "mxinp": grid_parameters["mxinp"],  # num mesh x for input grid
        "myinp": grid_parameters["myinp"],  # num mesh y for input grid
        "dxinp": grid_parameters["dxinp"],  # resolution x for input grid
        "dyinp": grid_parameters["dyinp"],  # resolution y for input grid
        "dir_dist": dir_dist,  # direction distribution type
        "dir1": dir1,  # min direction
        "dir2": dir2,  # max direction
        "freq_discretization": len(np.unique(freq_array)),  # frequency discretization
        "dir_discretization": int(
            360 / (np.unique(dir_array)[1] - np.unique(dir_array)[0])
        ),  # direction discretization
        "mdc": int(
            360 / (np.unique(dir_array)[1] - np.unique(dir_array)[0])
        ),  # number of depth cases
        "flow": float(np.min(np.unique(freq_array))),  # low frequency limit
        "fhigh": float(np.max(np.unique(freq_array))),  # high frequency limit
    }


class BinWavesWrapper(SwanModelWrapper):
    """
    Wrapper example for the BinWaves model.
    """

    def build_case(self, case_dir: str, case_context: dict) -> None:
        if self.depth_array is not None:
            write_array_in_file(self.depth_array, f"{case_dir}/depth.dat")
        if self.locations is not None:
            write_array_in_file(self.locations, f"{case_dir}/locations.loc")

        # Construct the input spectrum
        input_spectrum = construct_partition(
            freq_name="jonswap",
            freq_kwargs={
                "freq": np.geomspace(
                    case_context.get("flow", 0.035),
                    case_context.get("fhigh", 0.5),
                    case_context.get("freq_discretization", 29),
                ),
                # "freq": np.linspace(case_context.get("flow", 0.035), case_context.get("fhigh", 0.5), case_context.get("freq_discretization", 29)),
                "fp": 1.0 / case_context.get("tp"),
                "hs": case_context.get("hs"),
            },
            dir_name="cartwright",
            dir_kwargs={
                "dir": np.linspace(0, 360, case_context.get("dir_discretization", 24)),
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
                os.path.join(case_dir, f"input_spectra_{side}.bnd")
            )
