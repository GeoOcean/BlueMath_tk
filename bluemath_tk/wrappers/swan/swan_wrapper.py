"""
Wrapper for the SWAN model.
https://swanmodel.sourceforge.io/online_doc/swanuse/swanuse.html
"""

import os
import re
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import wavespectra
import xarray as xr
from wavespectra.construct import construct_partition

from ...core.operations import get_uv_components
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
                    if float(match.group(1)) >= 98.0:
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


class HyWindSeaWrapper(SwanModelWrapper):
    """
    Wrapper for the HyWindSea metamodel.

    Cases are driven by a single NetCDF holding the high-resolution wind fields
    already selected for simulation: one case per (time, tide level) pair.

    Expected parameters
    -------------------
    metamodel_parameters : dict
        - tide_level : list of float
            Water levels to simulate.
        - time_index : list of int
            Positional indices into the ``time`` dimension of the wind file.
    fixed_parameters : dict
        - wind_file : str
            Path to the NetCDF with the high-resolution winds. Must contain
            ``M`` (wind speed) and ``Dir`` (wind direction) on a (time, lat, lon)
            grid. A ``lev`` dimension, if still present, is reduced using
            ``wind_level``.
        - wind_level : float, optional
            Level to select when the wind file still has a ``lev`` dimension.
            Default is 10.
        - bathy : xr.Dataset
            Bathymetry with a ``depth`` variable, positive downwards.
        - percentile : float
            Percentile of the wind speed over water used for the rescaling.
        - umbral : float
            Wind speed threshold (m/s) that percentile is rescaled to.

    Notes
    -----
    The previous version took a list of daily wind files plus a ``day_hour``
    index and built every combination, so it simulated 24 hours of each file
    whether or not those hours had been selected. Here the wind file already
    holds exactly the times to run, so the case count is
    ``len(tide_level) * len(time_index)`` and nothing is simulated that will not
    be used.

    Examples
    --------
    >>> wind_file = "inputs/wind_hr_10times.nc"
    >>> n_times = xr.open_dataset(wind_file).sizes["time"]
    >>> wrapper = HyWindSeaWrapper(
    ...     templates_dir="templates",
    ...     metamodel_parameters={
    ...         "tide_level": [0.0],
    ...         "time_index": list(range(n_times)),
    ...     },
    ...     fixed_parameters={
    ...         "wind_file": wind_file,
    ...         "bathy": xr.open_dataset("inputs/bati_santander_50m_LONLAT.nc"),
    ...         "percentile": 50,
    ...         "umbral": 9,
    ...     },
    ...     output_dir="outputs/SANTANDER/swan",
    ... )
    """

    def open_case_wind(self, case_context: dict) -> xr.Dataset:
        """
        Open the wind field corresponding to a single case.

        Parameters
        ----------
        case_context : dict
            The case context. Must contain ``wind_file`` and ``time_index``.

        Returns
        -------
        xr.Dataset
            The wind field at the requested time, squeezed to (lat, lon).

        Raises
        ------
        KeyError
            If ``wind_file`` or ``time_index`` is missing from the context.
        """

        for key in ("wind_file", "time_index"):
            if case_context.get(key) is None:
                raise KeyError(f"'{key}' is required in the case context")

        wind = xr.open_dataset(case_context["wind_file"]).isel(
            time=case_context["time_index"]
        )

        # The wind file may already have been reduced to a single level when it
        # was built, in which case 'lev' survives as a scalar coordinate and
        # must not be selected again.
        if "lev" in wind.dims:
            wind = wind.sel(lev=case_context.get("wind_level", 10))

        return wind.squeeze()

    def calculate_alpha_matrix_for_case(
        self, case_dir: str, case_context: dict
    ) -> None:
        """
        Reescale output data for the HyWindSea model.
        """

        # Open wind data and slice for bathymetry adaptation
        wind = self.open_case_wind(case_context)
        wind_edit = wind.sel(
            lon=slice(
                case_context.get("bathy").lon.values.min(),
                case_context.get("bathy").lon.values.max(),
            ),
            lat=slice(
                case_context.get("bathy").lat.values.min(),
                case_context.get("bathy").lat.values.max(),
            ),
        )
        bathy_interp = case_context.get("bathy").interp(
            lon=wind_edit.lon, lat=wind_edit.lat
        )
        wind_edit["bathy"] = bathy_interp.depth

        # Calculate percentiles and modify wind speeds
        perc_dataset = wind_edit.where(wind_edit.bathy > 0).quantile(
            case_context.get("percentile") / 100, dim=["lon", "lat"]
        )
        alpha = case_context.get("umbral") / perc_dataset
        # alpha["M"] = (
        #     "time",
        #     np.where(perc_dataset["M"] > case_context.get("umbral"), 1, alpha["M"]),
        # )

        # Save wind and alpha data in case_context dict
        case_context["alpha"] = np.where(
            perc_dataset["M"] > case_context.get("umbral"), 1, alpha["M"]
        )
        wind_done = wind.copy()
        wind_done["M"] = wind["M"] * case_context["alpha"]
        case_context["wind"] = wind_done

    def transform_write_wind_data(self, case_dir: str, case_context: dict) -> None:
        """
        Transform wind data for the HyWindSea model.
        """

        # Calculate u10 and v10 components
        u10, v10 = get_uv_components(case_context["wind"].Dir)
        case_context["wind"]["u10"] = -u10 * case_context["wind"].M
        case_context["wind"]["v10"] = -v10 * case_context["wind"].M
        w = case_context["wind"].interp(
            lat=case_context.get("bathy").lat.values,
            lon=case_context.get("bathy").lon.values,
            method="linear",
        )

        # extract and save
        u10 = w.u10.values
        v10 = w.v10.values
        arr = np.vstack((u10, v10))

        # Save wind file
        write_array_in_file(arr, f"{case_dir}/wind_file.dat")

    def transform_postprocess_ouput_data(
        self, wave_data: dict, case_context: dict
    ) -> np.ndarray:
        """
        Transform wind data for the HyWindSea model.
        """

        # Use alpha to rescale wave output data

        wave_data["Hsig"] = wave_data["Hsig"] / case_context.get("alpha")
        return wave_data

    def build_case(self, case_dir: str, case_context: dict) -> None:
        if self.depth_array is not None:
            write_array_in_file(self.depth_array, f"{case_dir}/depth_main.dat")
        if self.locations is not None:
            write_array_in_file(self.locations, f"{case_dir}/locations.loc")
        self.calculate_alpha_matrix_for_case(
            case_dir=case_dir, case_context=case_context
        )
        self.transform_write_wind_data(case_dir=case_dir, case_context=case_context)

    def postprocess_case(
        self, case_num, case_dir, case_context, output_vars=["Hsig", "Tm02", "Dir"]
    ):
        """
        Postprocess a single case, rescaling Hsig back with the case alpha.

        Notes
        -----
        ``alpha`` and ``wind`` are normally written into the context by
        ``build_case``. They are absent whenever the cases were not built in
        this session — after ``load_cases()``, or when the runs were submitted
        to a queue and postprocessed later — so they are recomputed here when
        missing. Both come deterministically from the wind file, so recomputing
        gives the same values the build used.
        """

        if case_context.get("alpha") is None or case_context.get("wind") is None:
            self.calculate_alpha_matrix_for_case(
                case_dir=case_dir, case_context=case_context
            )

        wave_data = super().postprocess_case(
            case_num, case_dir, case_context, output_vars
        )
        reescaled_wind = self.transform_postprocess_ouput_data(wave_data, case_context)
        wave_output = reescaled_wind.expand_dims(
            {
                "time": [case_context.get("wind").time.values],
                "tide": [case_context.get("tide_level")],
            }
        )
        return wave_output

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

        # combined_time = xr.concat(postprocessed_files, dim="time")
        # expand_dims leaves 'tide' as a length-1 dimension, so its .values is a
        # 1-D array. float() on that raises TypeError from NumPy 2.0 onwards
        # (it was only a DeprecationWarning before), hence the ravel()[0].
        def _tide_value(ds: xr.Dataset) -> float:
            return float(np.ravel(ds.tide.values)[0])

        postprocessed_files_sorted = sorted(postprocessed_files, key=_tide_value)

        # Agrupar por valor de marea
        grouped_by_tide = {
            tide: list(group)
            for tide, group in groupby(postprocessed_files_sorted, key=_tide_value)
        }

        combined_by_tide = []

        # Combinar los datasets de cada marea por tiempo
        for tide_val, ds_list in grouped_by_tide.items():
            ds_tide = xr.concat(
                ds_list, dim="time", combine_attrs="override", join="outer"
            )
            combined_by_tide.append(ds_tide)

        # Combinar todas las mareas
        combined_all = xr.concat(
            combined_by_tide, dim="tide", combine_attrs="override", join="outer"
        )

        return combined_all  # time and tide dimensions
