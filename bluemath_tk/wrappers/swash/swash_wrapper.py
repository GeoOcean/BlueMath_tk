import os
import os.path as op
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import find_peaks

from ...waves.spectra import spectral_analysis
from ...waves.statistics import upcrossing
from .._base_wrappers import BaseModelWrapper

np.random.seed(42)  # TODO: check global behavior.


class SwashModelWrapper(BaseModelWrapper):
    """
    Wrapper for the SWASH model.
    https://swash.sourceforge.io/online_doc/swashuse/swashuse.html#input-and-output-files

    Attributes
    ----------
    default_parameters : dict
        The default parameters type for the wrapper.
    available_launchers : dict
        The available launchers for the wrapper.
    postprocess_functions : dict
        The postprocess functions for the wrapper.
    depth_array : np.ndarray
        The depth array (working for 1D arrays for the moment).
    xlenc : int
       The length in meters of the computational domain.
    mxc : int
        The number of computational cells.
    mxinp : int
        The number of input points.
    dxinp : float
        The input points spacing (to calculate the meters).
    n_nodes_per_wavelength : int
        The number of nodes per wavelength.
    deltc : float
        The time step (DeltaT).
    tendc : float
        The total computation time.

    Methods
    -------
    build_cases -> None
        Create the cases folders and render the input files.
    list_available_postprocess_vars -> List[str]
        List available postprocess variables.
    _read_tabfile -> pd.DataFrame
        Read a tab file and return a pandas DataFrame.
    _convert_case_output_files_to_nc -> xr.Dataset
        Convert tab files to netCDF file.
    get_case_percentage_from_file -> float
        Get the case percentage from the output log file.
    monitor_cases -> pd.DataFrame
        Monitor the cases and log relevant information.
    postprocess_case -> xr.Dataset
        Convert tab ouput files to netCDF file.
    join_postprocessed_files -> xr.Dataset
        Join postprocessed files in a single Dataset.
    find_maximas -> Tuple[np.ndarray, np.ndarray]
        Find the individual maxima of an array.
    get_waterlevel -> xr.Dataset
        Get water level from the output netCDF file.
    calculate_runup2 -> xr.Dataset
        Calculates runup 2% (Ru2) from the output netCDF file.
    calculate_runup -> xr.Dataset
        Stores runup from the output netCDF file.
    calculate_setup -> xr.Dataset
        Calculates mean setup (Msetup) from the output netCDF file.
    calculate_statistical_analysis -> xr.Dataset
        Calculates zero-upcrossing analysis to obtain individual wave heights (Hi) and wave periods (Ti).
    calculate_spectral_analysis -> xr.Dataset
        Makes a water level spectral analysis (scipy.signal.welch)
        then separates incident waves, infragravity waves, very low frequency waves.
    """

    default_parameters = {
        "vegetation_height": float,
    }

    available_launchers = {
        "serial": "swash_serial.exe",
        "mpi": "mpirun -np 2 swash_mpi.exe",
        "docker_serial": "docker run --rm -v .:/case_dir -w /case_dir geoocean/rocky8 swash_serial.exe",
        "docker_mpi": "docker run --rm -v .:/case_dir -w /case_dir geoocean/rocky8 mpirun -np 2 swash_mpi.exe",
        "geoocean-cluster": "launchSwash.sh",
    }

    postprocess_functions = {
        "Ru2": "calculate_runup2",
        "RuDist": "calculate_runup",
        "Msetup": "calculate_setup",
        "Hrms": "calculate_statistical_analysis",
        "Hfreqs": "calculate_spectral_analysis",
        "Watlev": "get_waterlevel",
    }

    def __init__(
        self,
        templates_dir: str,
        model_parameters: dict,
        output_dir: str,
        depth_array: np.ndarray,
        dxinp: float,
        n_nodes_per_wavelength: int,
        tendc: int,
        warmup: int = 0,
        templates_name: dict = "all",
        debug: bool = True,
    ) -> None:
        """
        Initialize the SWASH model wrapper.
        """

        super().__init__(
            templates_dir=templates_dir,
            model_parameters=model_parameters,
            output_dir=output_dir,
            templates_name=templates_name,
            default_parameters=self.default_parameters,
        )
        self.set_logger_name(
            name=self.__class__.__name__, level="DEBUG" if debug else "INFO"
        )
        self.depth_array = depth_array
        self.mxinp = len(depth_array) - 1
        self.dxinp = dxinp
        self.xlenc = int(self.mxinp * self.dxinp)
        self.mxc: int = None
        self.n_nodes_per_wavelength = n_nodes_per_wavelength
        self.tendc = tendc
        self.warmup = warmup

    def build_cases(
        self,
        mode: str = "one_by_one",
    ) -> None:
        """
        Create the cases folders and render the input files.

        Parameters
        ----------
        mode : str, optional
            The mode to create the cases. Can be "all_combinations" or "one_by_one".
            Default is "one_by_one".
        """

        if mode == "all_combinations":
            self.cases_context = self.create_cases_context_all_combinations()
        elif mode == "one_by_one":
            self.cases_context = self.create_cases_context_one_by_one()
        else:
            raise ValueError(f"Invalid mode to create cases: {mode}")
        for case_num, case_context in enumerate(self.cases_context):
            case_context["case_num"] = f"{case_num:04}"
            case_dir = op.join(self.output_dir, f"{case_num:04}")
            self.cases_dirs.append(case_dir)
            os.makedirs(case_dir, exist_ok=True)
            self.build_case(
                case_context=case_context,
                case_dir=case_dir,
            )
            for template_name in self.templates_name:
                self.render_file_from_template(
                    template_name=template_name,
                    context=case_context,
                    output_filename=op.join(case_dir, template_name),
                )
        self.logger.info(
            f"{len(self.cases_dirs)} cases created in {mode} mode and saved in {self.output_dir}"
        )

    def list_available_postprocess_vars(self) -> List[str]:
        """
        List available postprocess variables.

        Returns
        -------
        List[str]
            The available postprocess variables.
        """

        return list(self.postprocess_functions.keys())

    @staticmethod
    def _read_tabfile(file_path: str) -> pd.DataFrame:
        """
        Read a tab file and return a pandas DataFrame.
        This function is used to read the output files of SWASH.

        Parameters
        ----------
        file_path : str
            The file path.

        Returns
        -------
        pd.DataFrame
            The pandas DataFrame.
        """

        f = open(file_path, "r")
        lines = f.readlines()
        # read head colums (variables names)
        names = lines[4].split()
        names = names[1:]  # Eliminate '%'
        # read data rows
        values = pd.Series(lines[7:]).str.split(expand=True).values.astype(float)
        df = pd.DataFrame(values, columns=names)
        f.close()

        return df

    def _convert_case_output_files_to_nc(
        self, case_num: int, output_path: str, run_path: str
    ) -> xr.Dataset:
        """
        Convert tab files to netCDF file.

        Parameters
        ----------
        case_num : int
            The case number.
        output_path : str
            The output path.
        run_path : str
            The run path.

        Returns
        -------
        xr.Dataset
            The xarray Dataset.
        """

        df_output = self._read_tabfile(file_path=output_path)
        df_output.set_index(
            ["Xp", "Yp", "Tsec"], inplace=True
        )  # set index to Xp, Yp and Tsec
        ds_ouput = df_output.to_xarray()

        df_run = self._read_tabfile(file_path=run_path)
        df_run.set_index(["Tsec"], inplace=True)
        ds_run = df_run.to_xarray()

        # merge output files to one xarray.Dataset
        ds = xr.merge([ds_ouput, ds_run], compat="no_conflicts")

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

        progress_pattern = r"\[\s*(\d+)%\]"
        with open(output_log_file, "r") as f:
            for line in reversed(f.readlines()):
                match = re.search(progress_pattern, line)
                if match:
                    return f"{match.group(1)} %"

        return "0 %"  # if no progress is found

    def monitor_cases(self) -> pd.DataFrame:
        """
        Monitor the cases and log relevant information.

        Returns
        -------
        pd.DataFrame
            The cases percentage.
        """

        cases_percentage = {}

        for case_dir in self.cases_dirs:
            output_log_file = os.path.join(case_dir, "wrapper_out.log")
            progress = self.get_case_percentage_from_file(
                output_log_file=output_log_file
            )
            cases_percentage[os.path.basename(case_dir)] = progress

        return pd.DataFrame(cases_percentage.items(), columns=["Case", "Percentage"])

    def postprocess_case(
        self, case_num: int, case_dir: str, output_vars: List[str] = None
    ) -> xr.Dataset:
        """
        Convert tab output files to netCDF file.

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

        import warnings

        warnings.filterwarnings("ignore")

        self.logger.info(f"Postprocessing case {case_num} in {case_dir}.")

        if output_vars is None:
            self.logger.info("Postprocessing all available variables.")
            output_vars = list(self.postprocess_functions.keys())

        output_nc_path = os.path.join(case_dir, "output.nc")
        if not os.path.exists(output_nc_path):
            # Convert tab files to netCDF file
            output_path = os.path.join(case_dir, "output.tab")
            run_path = os.path.join(case_dir, "run.tab")
            output_nc = self._convert_case_output_files_to_nc(
                case_num=case_num, output_path=output_path, run_path=run_path
            )
            output_nc.to_netcdf(output_nc_path)
        else:
            self.logger.info("Reading existing output.nc file.")
            output_nc = xr.open_dataset(output_nc_path)

        # Postprocess variables from output.nc
        var_ds_list = []
        for var in output_vars:
            if var in self.postprocess_functions:
                var_ds = getattr(self, self.postprocess_functions[var])(
                    case_num=case_num, case_dir=case_dir, output_nc=output_nc
                )
                var_ds_list.append(var_ds)
            else:
                self.logger.warning(
                    f"Variable {var} is not available for postprocessing."
                )

        # Merge all variables in one Dataset
        ds = xr.merge(var_ds_list, compat="no_conflicts")

        # Save Dataset to netCDF file
        ds.to_netcdf(os.path.join(case_dir, "output_postprocessed.nc"))

        return ds

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

    def find_maximas(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the individual maxima of an array.

        Parameters
        ----------
        x : np.ndarray
            The array (should be the water level time series).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The peaks and the values of the peaks.
        """

        peaks, _ = find_peaks(x=x)

        return peaks, x[peaks]

    def get_waterlevel(
        self, case_num: int, case_dir: str, output_nc: xr.Dataset
    ) -> xr.Dataset:
        """
        Get water level from the output netCDF file.

        Parameters
        ----------
        case_num : int
            The case number.
        case_dir : str
            The case directory.
        output_nc : xr.Dataset
            The output netCDF file.

        Returns
        -------
        xr.Dataset
            The water level.
        """

        # get water level
        return output_nc[["Watlev"]].squeeze()

    def calculate_runup2(
        self, case_num: int, case_dir: str, output_nc: xr.Dataset
    ) -> xr.Dataset:
        """
        Calculates runup 2% (Ru2) from the output netCDF file.

        Parameters
        ----------
        case_num : int
            The case number.
        case_dir : str
            The case directory.
        output_nc : xr.Dataset
            The output netCDF file.

        Returns
        -------
        xr.Dataset
            The runup 2% (Ru2).
        """

        # get runup
        runup = output_nc["Runlev"].values

        # find individual wave uprushes
        _, val_peaks = self.find_maximas(runup)

        # calculate ru2
        ru2 = np.percentile(val_peaks, 98)

        # create xarray Dataset with ru2 value depending on case_num
        ds = xr.Dataset({"Ru2": ("case_num", [ru2])}, {"case_num": [case_num]})

        return ds

    def calculate_runup(
        self, case_num: int, case_dir: str, output_nc: xr.Dataset
    ) -> xr.Dataset:
        """
        Stores runup from the output netCDF file.

        Parameters
        ----------
        case_num : int
            The case number.
        case_dir : str
            The case directory.
        output_nc : xr.Dataset
            The output netCDF file.

        Returns
        -------
        xr.Dataset
            The runup.
        """

        # get runup
        ds = output_nc["Runlev"]

        return ds

    def calculate_setup(
        self, case_num: int, case_dir: str, output_nc: xr.Dataset
    ) -> xr.Dataset:
        """
        Calculates mean setup (Msetup) from the output netCDF file.

        Parameters
        ----------
        case_num : int
            The case number.
        case_dir : str
            The case directory.
        output_nc : xr.Dataset
            The output netCDF file.

        Returns
        -------
        xr.Dataset
            The mean setup (Msetup).
        """

        # create xarray Dataset with mean setup
        ds = output_nc["Watlev"].mean(dim="Tsec")
        ds = ds.to_dataset()

        # eliminate Yp dimension
        ds = ds.squeeze()

        # rename variable
        ds = ds.rename({"Watlev": "Msetup"})

        return ds

    def calculate_statistical_analysis(
        self, case_num: int, case_dir: str, output_nc: xr.Dataset
    ) -> xr.Dataset:
        """
        Calculates zero-upcrossing analysis to obtain individual wave heights (Hi) and wave periods (Ti).

        Parameters
        ----------
        case_num : int
            The case number.
        case_dir : str
            The case directory.
        output_nc : xr.Dataset
            The output netCDF file.

        Returns
        -------
        xr.Dataset
            The statistical analysis.
        """

        # for every X coordinate in domain
        df_Hrms = pd.DataFrame()

        for x in output_nc["Xp"].values:
            dsw = output_nc.sel(Xp=x)

            # obtain series of water level
            series_water = dsw["Watlev"].values
            time_series = dsw["Tsec"].values

            # perform statistical analysis
            # _, Hi = upcrossing(time_series, series_water)
            _, Hi = upcrossing(np.vstack([time_series, series_water]).T)

            # calculate Hrms
            Hrms_x = np.sqrt(np.mean(Hi**2))
            df_Hrms.loc[x, "Hrms"] = Hrms_x

        # convert pd DataFrame to xr Dataset
        df_Hrms.index.name = "Xp"
        ds = df_Hrms.to_xarray()

        # assign coordinate case_num
        ds = ds.assign_coords({"case_num": [output_nc["case_num"].values]})

        return ds

    def calculate_spectral_analysis(
        self, case_num: int, case_dir: str, output_nc: xr.Dataset
    ) -> xr.Dataset:
        """
        Makes a water level spectral analysis (scipy.signal.welch)
        then separates incident waves, infragravity waves, very low frequency waves.

        Parameters
        ----------
        case_num : int
            The case number.
        case_dir : str
            The case directory.
        output_nc : xr.Dataset
            The output netCDF file.

        Returns
        -------
        xr.Dataset
            The spectral analysis.
        """

        delttbl = np.diff(output_nc["Tsec"].values)[1]

        df_H_spectral = pd.DataFrame()

        for x in output_nc["Xp"].values:
            dsw = output_nc.sel(Xp=x)
            series_water = dsw["Watlev"].values

            # calculate significant, SS, IG and VLF wave heighs
            Hs, Hss, Hig, Hvlf = spectral_analysis(series_water, delttbl)

            df_H_spectral.loc[x, "Hs"] = Hs
            df_H_spectral.loc[x, "Hss"] = Hss
            df_H_spectral.loc[x, "ig"] = Hig
            df_H_spectral.loc[x, "Hvlf"] = Hvlf

        # convert pd DataFrame to xr Dataset
        df_H_spectral.index.name = "Xp"
        ds = df_H_spectral.to_xarray()

        # assign coordinate case_num
        ds = ds.assign_coords({"case_num": [output_nc["case_num"].values]})

        return ds
