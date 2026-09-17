"""
Wrapper for the SWAN model.
https://swanmodel.sourceforge.io/online_doc/swanuse/swanuse.html
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import wavespectra
import xarray as xr
from wavespectra.construct import construct_partition

from .._base_wrappers import BaseModelWrapper
from .._utils_wrappers import write_array_in_file
from .swan_utils import generate_forcing_file_GreenWaves, sbatch_file_greenwaves


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

    default_parameters = {}

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

    def list_available_output_variables(self) -> list[str]:
        """
        List available output variables.

        Returns
        -------
        list[str]
            The available output variables.
        """

        return list(self.output_variables.keys())

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
                    if float(match.group(1)) > 98.0:
                        return "100 %"
                    return f"{match.group(1)} %"

        return "0 %"  # if no progress is found

    def monitor_cases(self, value_counts: str = None) -> tuple[pd.DataFrame, dict]:
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

    def join_postprocessed_files(
        self, postprocessed_files: list[xr.Dataset]
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


class SwanStructuredModelWrapper(SwanModelWrapper):
    """
    Wrapper for the SWAN structured model.
    """

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
        Initialize the SWAN structured model wrapper.
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

    def _convert_case_output_files_to_nc(
        self, case_num: int, output_path: str, output_vars: list[str]
    ) -> xr.Dataset:
        """
        Convert mat file to netCDF file.

        Parameters
        ----------
        case_num : int
            The case number.
        output_path : str
            The output path.
        output_vars : list[str]
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

    def postprocess_case(
        self,
        case_num: int,
        case_dir: str,
        case_context: dict,
        output_vars: list[str] = ["Hsig", "Tm02", "Dir"],
    ) -> xr.Dataset:
        """
        Convert mat ouput files to netCDF file.

        Parameters
        ----------
        case_num : int
            The case number.
        case_dir : str
            The case directory.
        case_context : dict
            The case context.
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


class SwanUnstructuredModelWrapper(SwanModelWrapper):
    """
    Wrapper for the SWAN unstructured model.
    """

    def __init__(
        self,
        templates_dir: str,
        metamodel_parameters: dict,
        fixed_parameters: dict,
        output_dir: str,
        templates_name: dict = "all",
        debug: bool = True,
    ) -> None:
        """
        Initialize the SWAN unstructured model wrapper.
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

    def _convert_case_output_files_to_nc(
        self, case_num: int, output_path: str, output_vars: list[str]
    ) -> xr.Dataset:
        """
        Convert mat file to netCDF file.

        Parameters
        ----------
        case_num : int
            The case number.
        output_path : str
            The output path.
        output_vars : list[str]
            The output variables to use.

        Returns
        -------
        xr.Dataset
            The xarray Dataset.
        """

        # Read mat file
        output_dict = sio.loadmat(output_path)

        # Create Dataset
        ds_output_dict = {
            var: (("case_num", "node"), output_dict[var]) for var in output_vars
        }
        ds = xr.Dataset(
            ds_output_dict,
            coords={
                "case_num": [case_num],
                "Xp": (("node"), output_dict["Xp"][0]),
                "Yp": (("node"), output_dict["Yp"][0]),
                "Depth": (("node"), output_dict["Depth"][0]),
            },
        )

        return ds


class BinWavesModelWrapper:
    def plot_cases_to_run(
        self,
        cmap: str = "turbo",
        figsize: tuple[float, float] = (8, 8),
    ) -> plt.Figure:
        """
        Plot the SWAN case library as a polar frequency/direction grid,
        colored by case ID.

        Parameters
        ----------
        cmap : str, optional
            Colormap used to color cases by ID, by default "turbo".
        figsize : tuple[float, float], optional
            Figure size, by default (8, 8).

        Returns
        -------
        Figure
            Matplotlib figure with the polar case grid.
        """

        dirs = np.array([case.get("dm") for case in self.cases_context])
        freqs = np.array([case.get("fp") for case in self.cases_context])
        case_ids = np.arange(len(self.cases_context))

        directions = np.sort(np.unique(dirs))
        frequencies = np.sort(np.unique(freqs))
        dir_to_col = {d: i for i, d in enumerate(directions)}
        freq_to_row = {f: i for i, f in enumerate(frequencies)}

        # grid of case IDs (rows=freq, cols=dir); left as NaN where a
        # dir/freq combination is missing (e.g. a direction-sector subset)
        grid = np.full((len(frequencies), len(directions)), np.nan)
        for case_id, dir_val, freq_val in zip(case_ids, dirs, freqs):
            grid[freq_to_row[freq_val], dir_to_col[dir_val]] = case_id

        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})

        dtheta = np.diff(directions).mean()
        theta = np.deg2rad(np.append(directions, directions[0] + 360) - dtheta / 2)
        radius = np.append(0, frequencies)

        pcm = ax.pcolormesh(
            theta,
            radius,
            grid,
            cmap=cmap,
            edgecolors="grey",
            linewidth=0.1,
            shading="flat",
        )

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(
            f"SWAN case library ({len(self.cases_context)} cases)", pad=20, fontsize=14
        )
        ax.set_ylabel("Frequency [Hz]", labelpad=30)
        ax.tick_params(labelsize=9)

        fig.colorbar(pcm, ax=ax, pad=0.1, shrink=0.7, label="Case ID")
        fig.tight_layout()

        return fig


class BinWavesStructuredWrapper(SwanStructuredModelWrapper, BinWavesModelWrapper):
    """
    Wrapper example for the BinWaves structured model.
    """

    def build_case(self, case_dir: str, case_context: dict) -> None:
        """
        Build the input spectra files for a case.
        """

        if self.depth_array is not None:
            write_array_in_file(self.depth_array, f"{case_dir}/depth.dat")
        if self.locations is not None:
            write_array_in_file(self.locations, f"{case_dir}/locations.loc")

        # Construct the input spectrum
        input_spectrum = construct_partition(
            freq_name="jonswap",
            freq_kwargs={
                "freq": case_context.get("frequencies_array"),
                "fp": case_context.get("fp"),
                "hs": 1.0,
            },
            dir_name="cartwright",
            dir_kwargs={
                "dir": case_context.get("directions_array"),
                "dm": case_context.get("dm"),
                "dspr": 1.0,
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


class BinWavesUnstructuredWrapper(SwanUnstructuredModelWrapper, BinWavesModelWrapper):
    """
    Wrapper example for the BinWaves unstructured model.
    """

    def build_case(self, case_dir: str, case_context: dict) -> None:
        """
        Build the input spectra file for a case.
        """

        # Construct the input spectrum
        input_spectrum = construct_partition(
            freq_name="jonswap",
            freq_kwargs={
                "freq": case_context.get("frequencies_array"),
                "fp": case_context.get("fp"),
                "hs": 1.0 if case_context.get("fp") < 0.2 else 0.1,
            },
            dir_name="cartwright",
            dir_kwargs={
                "dir": case_context.get("directions_array"),
                "dm": case_context.get("dm"),
                "dspr": 1.0,
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
            os.path.join(case_dir, "input_spectra.bnd")
        )

    def postprocess_case(
        self,
        case_num: int,
        case_dir: str,
        case_context: dict,
        output_vars: list[str] = ["Hsig", "Tm02", "Dir"],
    ) -> xr.Dataset:
        """
        Convert mat ouput files to netCDF file.

        Parameters
        ----------
        case_num : int
            The case number.
        case_dir : str
            The case directory.
        case_context : dict
            The case context.
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
            output_nc = output_nc.assign_coords(
                {
                    "dm": (("case_num"), [case_context.get("dm")]),
                    "fp": (("case_num"), [case_context.get("fp")]),
                    "tp": (("case_num"), [1.0 / case_context.get("fp")]),
                }
            )
            output_nc.to_netcdf(os.path.join(case_dir, "output.nc"))
        else:
            self.logger.info("Reading existing output.nc file.")
            output_nc = xr.open_dataset(output_nc_path)

        return output_nc


class GreenWavesWrapper(SwanModelWrapper):
    """
    Wrapper example for the GreenWaves model.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the GreenWaves wrapper.
        """

        super().__init__(*args, **kwargs)
        self.sbatch_file_example = sbatch_file_greenwaves

    def build_case(
        self,
        case_context: dict,
        case_dir: str,
    ) -> None:
        """
        Build the input files for a case.

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        """

        generate_forcing_file_GreenWaves(
            case_context=case_context,
            case_dir=case_dir,
            ds_GFD_info=case_context.get("ds_GFD_info"),
        )
