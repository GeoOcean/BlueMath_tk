import sys
import os
from typing import Tuple, List
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import find_peaks
from .._base_wrappers import BaseModelWrapper
from ...waves.spectra import spectral_analysis
from ...waves.statistics import upcrossing


class SwashModelWrapper(BaseModelWrapper):
    """
    Wrapper for the SWASH model.
    https://swash.sourceforge.io/online_doc/swashuse/swashuse.html#input-and-output-files

    Attributes
    ----------
    swash_exec : str
        The SWASH executable path.
    default_parameters : dict
        The default parameters type for the model.

    Methods
    -------
    set_swash_exec(swash_exec: str) -> None
        Set the SWASH executable path.
    _read_tabfile(file_path: str) -> pd.DataFrame
        Read a tab file and return a pandas DataFrame.
    _convert_case_output_files_to_nc(case_id: int, output_path: str, run_path: str) -> xr.Dataset
        Convert output tabs files to a netCDF file.
    run_model(case_dir: str, log_file: str = "swash_exec.log") -> None
        Run the SWASH model for the specified case.
    """

    default_parameters = {
        "vegetation_height": float,
    }

    postprocess_functions = {
        "Ru2": "calculate_runup2",
        "RuDist": "calculate_runup",
        "Msetup": "calculate_setup",
        "Hrms": "calculate_statistical_analysis",
        "Hfreqs": "calculate_spectral_analysis",
    }

    def __init__(
        self,
        templates_dir: str,
        templates_name: dict,
        model_parameters: dict,
        output_dir: str,
    ) -> None:
        """
        Initialize the SWASH model wrapper.

        Parameters
        ----------
        templates_dir : str
            The directory where the templates are stored.
        templates_name : list
            The names of the templates.
        model_parameters : dict
            The parameters to be used in the templates.
        output_dir : str
            The directory where the output files will be saved.
        """

        super().__init__(
            templates_dir=templates_dir,
            templates_name=templates_name,
            model_parameters=model_parameters,
            output_dir=output_dir,
            default_parameters=self.default_parameters,
        )
        self.set_logger_name(self.__class__.__name__)
        self._swash_exec: str = None

    @property
    def swash_exec(self) -> str:
        return self._swash_exec

    def set_swash_exec(self, swash_exec: str) -> None:
        self._swash_exec = swash_exec

    def list_available_postprocess_vars(self) -> List[str]:
        """
        List available postprocess variables.

        Returns
        -------
        List[str]
            The available postprocess variables.
        """

        return list(self.postprocess_functions.keys())

    def run_model(self, case_dir: str, log_file: str = "swash_exec.log") -> None:
        """
        Run the SWASH model for the specified case.

        Parameters
        ----------
        case_dir : str
            The case directory.
        log_file : str, optional
            The log file name. Default is "swash_exec.log".

        Raises
        ------
        ValueError
            If the SWASH executable was not set.
        """

        if not self.swash_exec:
            raise ValueError("The SWASH executable was not set.")

        # check if windows OS
        is_win = sys.platform.startswith("win")
        if is_win:
            cmd = "cd {0} && {1} input".format(case_dir, self.swash_exec)
        else:
            cmd = "cd {0} && {1} -input input.sws".format(case_dir, self.swash_exec)
        # redirect output
        cmd += f" 2>&1 > {log_file}"
        # execute command
        self._exec_bash_commands(str_cmd=cmd)

    def run_model_with_apptainer(
        self,
        case_dir: str,
        apptainer_image: str,
        apptainer_out_logs: str = "apptainer_out.log",
        apptainer_err_logs: str = "apptainer_err.log",
    ) -> None:
        """
        Run the SWASH model for the specified case using Apptainer.

        Parameters
        ----------
        case_dir : str
            The case directory.
        apptainer_image : str
            The Apptainer image.
        apptainer_out_logs : str, optional
            The Apptainer output log file. Default is "apptainer_out.log".
        apptainer_err_logs : str, optional
            The Apptainer error log file. Default is "apptainer_err.log".
        """

        # Construct the Apptainer command
        apptainer_cmd = f"apptainer exec --bind {case_dir}:/tmp/swash --pwd /tmp/swash {apptainer_image}  swashrun -input input.sws"
        # Execute the Apptainer command
        self._exec_bash_commands(
            str_cmd=apptainer_cmd,
            out_file=os.path.join(case_dir, apptainer_out_logs),
            err_file=os.path.join(case_dir, apptainer_err_logs),
        )

    def run_model_with_docker(
        self,
        case_dir: str,
        docker_image: str = "tausiaj/swash-geoocean:11.01",
        docker_out_logs: str = "docker_out.log",
        docker_err_logs: str = "docker_err.log",
    ) -> None:
        """
        Run the SWASH model for the specified case using Docker.

        Parameters
        ----------
        case_dir : str
            The case directory.
        docker_image : str, optional
            The Docker image. Default is "tausiaj/swash-geoocean:11.01".
        docker_out_logs : str, optional
            The Docker output log file. Default is "docker_out.log".
        docker_err_logs : str, optional
            The Docker error log file. Default is "docker_err.log".
        """

        # Construct the Docker command
        # TODO: Check why --rm flag is not removing the container after execution
        docker_cmd = f"docker run --rm -v {case_dir}:/case_dir -w /case_dir {docker_image} swashrun -input input.sws"
        # Execute the Docker command
        self._exec_bash_commands(
            str_cmd=docker_cmd,
            out_file=os.path.join(case_dir, docker_out_logs),
            err_file=os.path.join(case_dir, docker_err_logs),
        )

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

        # assign correct coordinate case_id
        ds.coords["case_num"] = case_num

        return ds

    def postprocess_case(
        self, case_num: int, case_dir: str, output_vars: List[str] = None
    ) -> xr.Dataset:
        """
        Convert tab ouput files to netCDF file.

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
            output_vars = list(self.postprocess_functions.keys())

        output_nc_path = os.path.join(case_dir, "output.nc")
        if not os.path.exists(output_nc_path):
            # Convert tab files to netCDF file
            output_path = os.path.join(case_dir, "output.tab")
            run_path = os.path.join(case_dir, "run.tab")
            output_nc = self._convert_case_output_files_to_nc(
                case_num=case_num, output_path=output_path, run_path=run_path
            )
            output_nc.to_netcdf(os.path.join(case_dir, "output.nc"))
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
        Find the individual uprushes along the beach profile.

        Parameters
        ----------
        x : np.ndarray
            The water level time series.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The peaks and the values of the peaks.
        """

        peaks, _ = find_peaks(x=x)

        return peaks, x[peaks]

    def calculate_runup2(
        self, case_num: int, case_dir: str, output_nc: xr.Dataset
    ) -> xr.Dataset:
        """
        Calculates runup 2% (Ru2) from the output netCDF file.

        Parameters
        ----------
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
