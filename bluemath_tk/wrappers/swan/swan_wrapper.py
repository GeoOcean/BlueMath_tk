"""
Wrapper for the SWAN model.
https://swanmodel.sourceforge.io/online_doc/swanuse/swanuse.html
"""

import os
import re
import shutil
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
        # Total energy (m0) of the partition, properly integrated over the
        # (non-uniform) frequency bin widths and direction step, so collapsing
        # it to a single bin below preserves the intended hs regardless of
        # where fp falls on the log-spaced frequency grid.
        m0 = float(input_spectrum.spec.to_energy().sum(dim=["freq", "dir"]))
        df = input_spectrum.spec.df.values
        dd = input_spectrum.spec.dd

        argmax_bin = np.unravel_index(
            np.argmax(input_spectrum.values), input_spectrum.shape
        )
        mono_spec_array = np.zeros_like(input_spectrum.values)
        mono_spec_array[argmax_bin] = m0 / (df[argmax_bin[0]] * dd)
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

        # Save hs value depending on 5 second peak period (fp) value
        case_context["hs"] = 1.0 if case_context.get("fp") < 0.2 else 0.1

        # Construct the input spectrum
        input_spectrum = construct_partition(
            freq_name="jonswap",
            freq_kwargs={
                "freq": case_context.get("frequencies_array"),
                "fp": case_context.get("fp"),
                "hs": case_context.get("hs"),
            },
            dir_name="cartwright",
            dir_kwargs={
                "dir": case_context.get("directions_array"),
                "dm": case_context.get("dm"),
                "dspr": 1.0,
            },
        )
        # Total energy (m0) of the partition, properly integrated over the
        # (non-uniform) frequency bin widths and direction step, so collapsing
        # it to a single bin below preserves the intended hs regardless of
        # where fp falls on the log-spaced frequency grid.
        m0 = float(input_spectrum.spec.to_energy().sum(dim=["freq", "dir"]))
        df = input_spectrum.spec.df.values
        dd = input_spectrum.spec.dd

        argmax_bin = np.unravel_index(
            np.argmax(input_spectrum.values), input_spectrum.shape
        )
        mono_spec_array = np.zeros_like(input_spectrum.values)
        mono_spec_array[argmax_bin] = m0 / (df[argmax_bin[0]] * dd)
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
                    "hs": (("case_num"), [case_context.get("hs")]),
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


# --------------------------------------------------------------------------------------
# Unstructured SWAN forced along the open boundary at a set of boundary nodes, with
# output at points — the SWAN counterpart of bluemath_tk.wrappers.snapwave
# (same constructor options and template context names).
# --------------------------------------------------------------------------------------

#: Shell lines that put SWAN (``swan.exe``) on the PATH on the GeoOcean cluster.
GEOOCEAN_SWAN_MODULE = (
    "module use /nfs/software/geocean/modulefiles\nmodule load swan/4151"
)

SWAN_SLURM_ARRAY_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={mem}
#SBATCH --output={logs_dir}/%A_%a.out
#SBATCH --error={logs_dir}/%A_%a.err
{extra_sbatch}
{setup}

case_dir=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {case_dirs_file})
cd "$case_dir"
swan.exe > wrapper_out.log 2> wrapper_error.log
"""

_SWAN_NUMBER = (int, float, np.integer, np.floating)


class SwanBoundaryModelWrapper(BaseModelWrapper):
    """
    Unstructured SWAN cases forced along the open boundary, output at points.

    The grid is an ADCIRC file (linked as ``fort.14`` into every case, as SWAN
    requires) with counterclockwise triangles and boundary node strings
    covering the whole outline (open + land). Boundary conditions are given
    at ``boundary_nodes`` (e.g. HyWaves goals), each snapped to the nearest
    vertex of the grid's open boundary, and SWAN interpolates linearly
    between them along it (see
    :func:`~bluemath_tk.wrappers.swan.swan_utils.swan_boundary_side` and
    :func:`~bluemath_tk.wrappers.swan.swan_utils.boundary_par_table`).

    Template context added by :meth:`build_case`:

    - ``boundary_side``: e.g. ``"1 CCW"``, for ``BOUNDSPEC SIDE``;
    - ``boundary_par``: list of ``"len hs per dir dd"`` rows, for
      ``VARIABLE PAR``;
    - ``n_nodes``, ``points_file``, ``table_file``, ``table_quantities``.

    Attributes
    ----------
    table_quantities : tuple of str
        SWAN quantities written by the templates' ``TABLE`` command, in order.
    """

    default_parameters: dict = {}

    available_launchers = {
        "default": "swan.exe",
        "geoocean-cluster": f"{GEOOCEAN_SWAN_MODULE}\nswan.exe",
    }

    table_quantities: tuple[str, ...] = ("HSIGN", "TPS", "DIR", "DSPR")

    def __init__(
        self,
        templates_dir: str,
        metamodel_parameters: dict,
        fixed_parameters: dict,
        output_dir: str,
        grid_file: str,
        boundary_nodes: np.ndarray,
        output_points: np.ndarray | pd.DataFrame,
        templates_name: list[str] | str = "all",
        debug: bool = True,
        open_boundary: int = 0,
        points_filename: str = "output_sites.txt",
        table_filename: str = "output_sites.tab",
        point_dim: str = "points",
        point_coords: dict[str, str] | None = None,
        store_parameters: tuple[str, ...] = (),
    ) -> None:
        """
        Initialise the SWAN boundary-forced wrapper.

        Parameters
        ----------
        templates_dir : str
            Folder with the ``INPUT`` template (SWAN command file).
        metamodel_parameters : dict
            Per-case parameters (lists of equal length).
        fixed_parameters : dict
            Parameters shared by every case.
        output_dir : str
            Folder where case folders are created.
        grid_file : str
            ADCIRC grid (``fort.14`` format, depth positive down) with its
            open-boundary node string.
        boundary_nodes : np.ndarray
            ``(n_nodes, 2)`` boundary node coordinates, in order along the
            open boundary (the per-node forcing order).
        output_points : np.ndarray or pd.DataFrame
            Output points: ``(n, 2)`` array, or a DataFrame with ``lon`` and
            ``lat`` columns.
        templates_name : list of str or "all", optional
            Templates to render. Default is "all".
        debug : bool, optional
            DEBUG-level logging. Default is True.
        open_boundary : int, optional
            Which open boundary of *grid_file* is forced. Default is 0.
        points_filename, table_filename : str, optional
            Output-points file and ``TABLE`` output names.
        point_dim : str, optional
            Name of the output-points dimension. Default is "points".
        point_coords : dict, optional
            ``{coord_name: column}`` of *output_points* (a DataFrame) attached
            as coordinates on *point_dim*, e.g. ``{"sites": "source_id"}``.
        store_parameters : tuple of str, optional
            Scalar case parameters stored as ``f"{name}_forcing"`` along the
            case dimension.
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

        from .swan_utils import read_adcirc_grid, swan_boundary_side

        self.grid_file = os.path.abspath(grid_file)
        self.side_xy, self.boundary_side = swan_boundary_side(
            read_adcirc_grid(self.grid_file), open_boundary
        )
        self.boundary_nodes = np.asarray(boundary_nodes, dtype=float).reshape(-1, 2)
        self.point_metadata = (
            output_points.reset_index(drop=True)
            if isinstance(output_points, pd.DataFrame)
            else None
        )
        if point_coords and self.point_metadata is None:
            raise ValueError("point_coords needs output_points as a DataFrame")
        if isinstance(output_points, pd.DataFrame):
            output_points = output_points[["lon", "lat"]].to_numpy(float)
        self.output_points = np.asarray(output_points, dtype=float).reshape(-1, 2)
        self.points_filename = points_filename
        self.table_filename = table_filename
        self.point_dim = point_dim
        self.point_coords = dict(point_coords or {})
        self.store_parameters = tuple(store_parameters)

    @property
    def n_nodes(self) -> int:
        """Number of boundary nodes."""

        return len(self.boundary_nodes)

    def boundary_values(self, case_context: dict) -> dict[str, list]:
        """
        ``hs``, ``tp``, ``dir``, ``spr`` at each boundary node for one case.

        Parameters
        ----------
        case_context : dict
            Case parameters.

        Returns
        -------
        dict[str, list]
            One list of ``n_nodes`` values per variable.
        """

        raise NotImplementedError

    def build_case(self, case_context: dict, case_dir: str) -> None:
        """
        Link the grid, write the output points and the boundary context.

        Parameters
        ----------
        case_context : dict
            Case parameters; the template context listed in the class
            docstring is added.
        case_dir : str
            Case folder.
        """

        from .swan_utils import boundary_par_table, write_swan_points

        fort14 = os.path.join(case_dir, "fort.14")
        if os.path.lexists(fort14):
            os.remove(fort14)
        try:
            os.symlink(self.grid_file, fort14)
        except OSError:  # no symlinks (e.g. Windows without privileges): copy
            shutil.copyfile(self.grid_file, fort14)
        write_swan_points(
            self.output_points, os.path.join(case_dir, self.points_filename)
        )

        rows = boundary_par_table(
            self.side_xy, self.boundary_nodes, self.boundary_values(case_context)
        )
        case_context["n_nodes"] = self.n_nodes
        case_context["boundary_side"] = self.boundary_side
        case_context["boundary_par"] = [
            f"{length:.8f} {hs:.4f} {tp:.4f} {d:.3f} {spr:.3f}"
            for length, hs, tp, d, spr in zip(
                rows["len"], rows["hs"], rows["tp"], rows["dir"], rows["spr"]
            )
        ]
        case_context["points_file"] = self.points_filename
        case_context["table_file"] = self.table_filename
        case_context["table_quantities"] = " ".join(self.table_quantities)

    def monitor_cases(self, value_counts: str | None = None):
        """
        Case status: FINISHED when SWAN wrote ``norm_end`` and the table.

        Parameters
        ----------
        value_counts : str, optional
            Passed to :meth:`BaseModelWrapper.monitor_cases`.

        Returns
        -------
        pd.DataFrame or dict
            See :meth:`BaseModelWrapper.monitor_cases`.
        """

        def status(case_dir: str) -> str:
            if os.path.exists(os.path.join(case_dir, "norm_end")) and os.path.exists(
                os.path.join(case_dir, self.table_filename)
            ):
                return "FINISHED"
            if os.path.exists(os.path.join(case_dir, "PRINT")) or os.path.exists(
                os.path.join(case_dir, "PRINT-001")
            ):
                return "RUNNING"
            return "NOT STARTED"

        cases_status = {
            os.path.basename(case_dir): status(case_dir) for case_dir in self.cases_dirs
        }

        return super().monitor_cases(
            cases_status=cases_status, value_counts=value_counts
        )

    def write_slurm_array(
        self,
        filename: str = "swan_array.sh",
        partition: str = "geocean",
        mem: str = "4gb",
        job_name: str = "swan",
        setup: str = GEOOCEAN_SWAN_MODULE,
        logs_dir: str = "slurm_logs",
        extra_sbatch: list[str] | None = None,
    ) -> str:
        """
        Write a SLURM job-array script (one serial SWAN run per case) to ``output_dir``.

        Also writes ``case_dirs.txt``, read by ``SLURM_ARRAY_TASK_ID``. Each
        case runs ``swan.exe`` in its folder on one core; the array runs the
        cases in parallel.

        Parameters
        ----------
        filename : str, optional
            Script name inside ``output_dir``. Default is "swan_array.sh".
        partition : str, optional
            SLURM partition. Default is "geocean".
        mem : str, optional
            Memory per case. Default is "4gb" (a 153k-node grid needs ~3 GB).
        job_name : str, optional
            SLURM job name. Default is "swan".
        setup : str, optional
            Shell lines run before SWAN. Default loads the GeoOcean module.
        logs_dir : str, optional
            SLURM log folder, relative to ``output_dir``.
        extra_sbatch : list of str, optional
            Additional ``#SBATCH`` option lines, e.g. ``["--time=02:00:00"]``.

        Returns
        -------
        str
            Path of the written script.
        """

        case_dirs_file = self.cases_dir_to_txt()
        os.makedirs(os.path.join(self.output_dir, logs_dir), exist_ok=True)
        extra = "\n".join(f"#SBATCH {line}" for line in (extra_sbatch or []))
        script = SWAN_SLURM_ARRAY_TEMPLATE.format(
            job_name=job_name,
            partition=partition,
            mem=mem,
            logs_dir=logs_dir,
            extra_sbatch=extra,
            setup=setup,
            case_dirs_file=case_dirs_file,
        )
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write(script)
        self.logger.info(f"SLURM array script for {len(self.cases_dirs)} cases: {path}")

        return path

    def run_cases_slurm(
        self,
        cases_to_run: list[int] | None = None,
        max_parallel: int | None = None,
        **script_kwargs,
    ) -> None:
        """
        Write the SLURM array script and submit it with ``sbatch``.

        Parameters
        ----------
        cases_to_run : list of int, optional
            0-based case indices to submit. Default is every case.
        max_parallel : int, optional
            Maximum simultaneously running tasks (``%`` array throttle).
        **script_kwargs
            Passed to :meth:`write_slurm_array`.
        """

        script = self.write_slurm_array(**script_kwargs)
        if cases_to_run is None:
            array = f"1-{len(self.cases_dirs)}"
        else:
            array = ",".join(str(int(i) + 1) for i in cases_to_run)
        if max_parallel is not None:
            array += f"%{int(max_parallel)}"
        self.run_cases_bulk(launcher=f"sbatch --array={array} {script}")

    def read_case_table(
        self, case_dir: str, case_context: dict, row_dim: str, row
    ) -> xr.Dataset:
        """
        One case's ``TABLE`` output as a dataset on ``(row_dim, point_dim)``.

        Parameters
        ----------
        case_dir : str
            Case folder.
        case_context : dict
            Case parameters (for :attr:`store_parameters`).
        row_dim : str
            Case dimension (``case_num`` or ``time``).
        row
            Coordinate value of this case on *row_dim*.

        Returns
        -------
        xr.Dataset
            ``hs``, ``tp``, ``dir``, ``spr`` (and other table quantities).
        """

        from .swan_utils import read_swan_table

        table = read_swan_table(
            os.path.join(case_dir, self.table_filename), list(self.table_quantities)
        )
        if len(table) != len(self.output_points):
            raise ValueError(
                f"{len(table)} rows in {self.table_filename} but "
                f"{len(self.output_points)} output points"
            )
        ds = xr.Dataset(
            {
                name: ((row_dim, self.point_dim), table[[name]].T.to_numpy())
                for name in table
            },
            coords={row_dim: [row]},
        )
        if self.point_coords:
            ds = ds.assign_coords(
                {
                    name: (self.point_dim, self.point_metadata[col].to_numpy())
                    for name, col in self.point_coords.items()
                }
            )
        for name in self.store_parameters:
            if name in case_context:
                ds[f"{name}_forcing"] = ((row_dim,), [float(case_context[name])])

        return ds


class SwanMetaModelWrapper(SwanBoundaryModelWrapper):
    """
    Stationary SWAN cases for a metamodel (e.g. LHS over Hs, Tp, Dir, Spr, WL).

    The case ``hs`` is applied at ``active_node`` (1-based) and zero at the
    other boundary nodes, with the case ``tp``, ``dir``, ``spr`` everywhere,
    so along the boundary Hs decays linearly from the active node to zero at
    its two neighbouring nodes (as in SnapWave): each goal's metamodel learns
    its own contribution, to be summed with the others. Templates read ``wl``
    for ``SET LEVEL``. Each case is postprocessed to a single ``case_num`` row.
    """

    default_parameters = {
        "hs": {
            "type": _SWAN_NUMBER,
            "value": None,
            "description": "Significant wave height at the active node (m).",
        },
        "tp": {"type": _SWAN_NUMBER, "value": None, "description": "Peak period (s)."},
        "dir": {
            "type": _SWAN_NUMBER,
            "value": None,
            "description": "Mean wave direction (nautical, coming from, deg).",
        },
        "spr": {
            "type": _SWAN_NUMBER,
            "value": None,
            "description": "Directional spread (deg).",
        },
        "wl": {"type": _SWAN_NUMBER, "value": None, "description": "Water level (m)."},
        "active_node": {
            "type": (int, np.integer),
            "value": None,
            "description": "1-based boundary node where the case Hs is applied.",
        },
    }

    def __init__(
        self,
        *args,
        split_by: str | None = None,
        split_filename: str = "output_sites_{}.nc",
        **kwargs,
    ) -> None:
        """
        Initialise the metamodel wrapper.

        Parameters
        ----------
        *args, **kwargs
            See :class:`SwanBoundaryModelWrapper`.
        split_by : str, optional
            Case parameter by which the joined output is split into one NetCDF
            per value in ``output_dir``. Default is None (no split).
        split_filename : str, optional
            File name pattern for the split files. Default is
            "output_sites_{}.nc".
        """

        self.split_by = split_by
        self.split_filename = split_filename
        super().__init__(*args, **kwargs)

    def boundary_values(self, case_context: dict) -> dict[str, list]:
        """Case ``hs`` at the active node and 0 at the others; same tp, dir, spr."""

        active = int(case_context["active_node"]) - 1
        n = self.n_nodes

        return {
            "hs": [case_context["hs"] if i == active else 0.0 for i in range(n)],
            "tp": [case_context["tp"]] * n,
            "dir": [case_context["dir"]] * n,
            "spr": [case_context["spr"]] * n,
        }

    def postprocess_case(
        self, case_num: int, case_dir: str, case_context: dict
    ) -> xr.Dataset:
        """
        Read one stationary case on ``(case_num, point_dim)``.

        Parameters
        ----------
        case_num : int
            Case index.
        case_dir : str
            Case folder.
        case_context : dict
            Case parameters.

        Returns
        -------
        xr.Dataset
            Case output.
        """

        ds = self.read_case_table(case_dir, case_context, "case_num", case_num)
        if self.split_by is not None:
            ds = ds.assign_coords(
                {self.split_by: ("case_num", [case_context[self.split_by]])}
            )

        return ds

    def join_postprocessed_files(
        self, postprocessed_files: list[xr.Dataset]
    ) -> xr.Dataset | dict:
        """
        Concatenate postprocessed cases along ``case_num`` (optionally split).

        Parameters
        ----------
        postprocessed_files : list of xr.Dataset
            One dataset per case.

        Returns
        -------
        xr.Dataset or dict
            All cases; with ``split_by``, ``{value: written NetCDF path}``.
        """

        joined = xr.concat(postprocessed_files, dim="case_num")
        if self.split_by is None:
            return joined

        written = {}
        for value, part in joined.groupby(self.split_by):
            path = os.path.join(self.output_dir, self.split_filename.format(value))
            part.drop_vars(self.split_by).to_netcdf(path)
            written[value] = path
        self.logger.info(f"Wrote {len(written)} files split by {self.split_by}.")

        return written


class SwanDynamicModelWrapper(SwanBoundaryModelWrapper):
    """
    One stationary SWAN case per time step of a boundary forcing time series.

    Every boundary node gets its own forcing, interpolated linearly between
    the nodes along the open boundary.

    Build ``metamodel_parameters`` with
    :func:`~bluemath_tk.wrappers.snapwave.snapwave_utils.boundary_time_series`
    (``tref`` and per-node ``hs_nodes``, ``tp_nodes``, ``dir_nodes``,
    ``spr_nodes``, optional ``wl_nodes``). SWAN takes one water level per
    case: templates read ``wl``, the mean of ``wl_nodes`` (0 without it).

    Wind, for templates that switch it on:

    - a wind field (``wind``): each case gets the field at its ``tref`` in
      ``wind.dat``, and the context ``wind_file`` and ``wind_grid`` (for
      ``INPGRID WIND REGULAR {{ wind_grid }}`` / ``READINP WIND 1
      '{{ wind_file }}' 3 0 FREE``);
    - a uniform wind: scalar ``u10`` (m/s) and ``u10dir`` (deg, nautical,
      coming from) case parameters (``WIND {{ u10 }} {{ u10dir }}``).
    """

    default_parameters = {
        "tref": {
            "type": str,
            "value": None,
            "description": "Case time, 'YYYYmmdd HHMMSS'.",
        },
        **{
            f"{var}_nodes": {
                "type": list,
                "value": None,
                "description": f"{var} at each boundary node.",
            }
            for var in ("hs", "tp", "dir", "spr", "wl")
        },
        "u10": {
            "type": _SWAN_NUMBER,
            "value": None,
            "description": "Uniform 10 m wind speed (m/s); 0 = no wind.",
        },
        "u10dir": {
            "type": _SWAN_NUMBER,
            "value": None,
            "description": "Uniform wind direction (deg, nautical, coming from).",
        },
    }

    def __init__(self, *args, wind: xr.Dataset | None = None, **kwargs) -> None:
        """
        Initialise the dynamic wrapper.

        Parameters
        ----------
        *args, **kwargs
            See :class:`SwanBoundaryModelWrapper`.
        wind : xr.Dataset, optional
            ``u10`` and ``v10`` (m/s) on ``(time, latitude, longitude)``,
            evenly spaced, covering the grid and the cases' times (see
            :func:`~bluemath_tk.waves.wind.wind_field_at`). Default is None.
        """

        self.wind = wind
        super().__init__(*args, **kwargs)

    def boundary_values(self, case_context: dict) -> dict[str, list]:
        """Per-node forcing of the time step."""

        return {v: list(case_context[f"{v}_nodes"]) for v in ("hs", "tp", "dir", "spr")}

    def build_case(self, case_context: dict, case_dir: str) -> None:
        """As :meth:`SwanBoundaryModelWrapper.build_case`, plus the case ``wl`` and wind."""

        wl_nodes = case_context.get("wl_nodes")
        case_context["wl"] = float(np.nanmean(wl_nodes)) if wl_nodes else 0.0
        super().build_case(case_context, case_dir)
        if self.wind is None:
            return

        from ...waves.wind import wind_field_at
        from .swan_utils import write_swan_wind

        time = pd.to_datetime(case_context["tref"], format="%Y%m%d %H%M%S")
        case_context["wind_file"] = "wind.dat"
        case_context["wind_grid"] = write_swan_wind(
            os.path.join(case_dir, "wind.dat"), wind_field_at(self.wind, time)
        )

    def postprocess_case(
        self, case_num: int, case_dir: str, case_context: dict
    ) -> xr.Dataset:
        """
        Read one time step on ``(time, point_dim)``.

        Parameters
        ----------
        case_num : int
            Case index.
        case_dir : str
            Case folder.
        case_context : dict
            Case parameters.

        Returns
        -------
        xr.Dataset
            Case output.
        """

        time = pd.to_datetime(case_context["tref"], format="%Y%m%d %H%M%S")

        return self.read_case_table(case_dir, case_context, "time", time)

    def join_postprocessed_files(
        self, postprocessed_files: list[xr.Dataset]
    ) -> xr.Dataset:
        """Concatenate postprocessed time steps along ``time``."""

        return xr.concat(postprocessed_files, dim="time")
