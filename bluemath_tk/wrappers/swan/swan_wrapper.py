import os
import re
from typing import List

import pandas as pd
import scipy.io as sio
import xarray as xr

from .._base_wrappers import BaseModelWrapper


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
    """

    default_parameters = {
        "hs": float,
        "tp": float,
        "dir": float,
        "spr": float,
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
        model_parameters: dict,
        output_dir: str,
        templates_name: dict = "all",
        debug: bool = True,
    ) -> None:
        """
        Initialize the SWAN model wrapper.
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
