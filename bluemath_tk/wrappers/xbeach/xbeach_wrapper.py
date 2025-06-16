import math
import os
from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr

from .._base_wrappers import BaseModelWrapper


class XBeachModelWrapper(BaseModelWrapper):
    """
    Wrapper for the XBeach model.
    https://xbeach.readthedocs.io/en/latest/

    Attributes
    ----------
    default_parameters : dict
        The default parameters type for the wrapper.
    available_launchers : dict
        The available launchers for the wrapper.
    """

    default_parameters = {
        "comptime": {
            "type": int,
            "value": 3600,
            "description": "The computational time.",
        },
        "wbctype": {
            "type": str,
            "value": "off",
            "description": "The time step for the simulation.",
        },
    }

    available_launchers = {
        "geoocean-cluster": "launchXbeach.sh",
    }

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
        Initialize the XBeach model wrapper.
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

        if case_context["wbctype"] == "jonstable":
            with open(f"{case_dir}/jonswap.txt", "w") as f:
                for _i in range(math.ceil(case_context["comptime"] / 3600)):
                    f.write(
                        f"{case_context['Hs']} {case_context['Tp']} {case_context['Dir']} 3.300000 30.000000 3600.000000 1.000000 \n"
                    )

    def _get_average_var(self, case_nc: xr.Dataset, var: str) -> np.ndarray:
        """
        Get the average value of a variable except for the first hour of the simulation

        Parameters
        ----------
        case_nc : xr.Dataset
            Simulation .nc file.
        var : str
            Variable of interest.

        Returns
        -------
        np.ndarray
            The average value of the variable.
        """

        if var in case_nc:
            return np.mean(
                case_nc[var]
                .isel(meantime=slice(1, int(case_nc.meantime.values[-1])))
                .values,
                axis=0,
            )

    def _get_max_var(self, case_nc: xr.Dataset, var: str) -> np.ndarray:
        """
        Get the Max value of a variable except for the first hour of the simulation

        Parameters
        ----------
        case_nc : xr.Dataset
            Simulation .nc file.
        var : str
            Variable of interest.

        Returns
        -------
        np.ndarray
            The max value of the variable.
        """

        if var in case_nc:
            return np.max(
                case_nc[var]
                .isel(meantime=slice(1, int(case_nc.meantime.values[-1])))
                .values,
                axis=0,
            )

    def monitor_cases(self, value_counts: str = None) -> Union[pd.DataFrame, dict]:
        """
        Monitor the cases based on different model log files.
        """

        cases_status = {}

        for case_dir in self.cases_dirs:
            case_dir_name = os.path.basename(case_dir)
            if os.path.exists(os.path.join(case_dir, "XBlog.txt")):
                if (
                    os.path.exists(os.path.join(case_dir, "XBerror.txt"))
                    & os.path.getsize(os.path.join(case_dir, "XBerror.txt"))
                    != 0
                ):
                    cases_status[case_dir_name] = "XBerror.txt"
                    continue
                else:
                    with open(os.path.join(case_dir, "XBlog.txt"), "r") as f:
                        lines = f.readlines()[-2:]

                    if any("End of program xbeach" in line.lower() for line in lines):
                        cases_status[case_dir_name] = "End of run"
                        continue
                    else:
                        cases_status[case_dir_name] = "Running"
                        continue
            else:
                cases_status[case_dir_name] = "No run"
                continue

        return super().monitor_cases(
            cases_status=cases_status, value_counts=value_counts
        )

    def postprocess_case(
        self,
        case_num: int,
        case_dir: str,
        output_vars: List[str] = None,
        overwrite_output: bool = True,
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
        overwrite_output : bool, optional
            Overwrite the output.nc file. Default is True.

        Returns
        -------
        xr.Dataset
            The postprocessed Dataset.
        """

        import warnings

        warnings.filterwarnings("ignore")

        self.logger.info(f"[{case_num}]: Postprocessing case {case_num} in {case_dir}.")

        output_nc_path = os.path.join(case_dir, "xboutput_postprocessed.nc")
        if not os.path.exists(output_nc_path) or overwrite_output:
            output_raw = xr.open_dataset(os.path.join(case_dir, "xboutput.nc"))

            globalx = output_raw.globalx.values
            globaly = output_raw.globaly.values
            zb = output_raw.zb.values[0]
            y = np.arange(globalx.shape[0])
            x = np.arange(globalx.shape[1])

            ds = xr.Dataset(
                {
                    "globalx": (("y", "x"), globalx),
                    "globaly": (("y", "x"), globaly),
                    "zb": (("y", "x"), zb),
                },
                coords={"y": y, "x": x},
            )

            for var in output_vars:
                if var == "zs_max":
                    maxed = self._get_max_var(case_nc=output_raw, var=var)
                    masked = xr.where(ds["zb"] > 0, np.nan, maxed)
                    ds[var] = (("y", "x"), masked.data)
                else:
                    averaged = self._get_average_var(case_nc=output_raw, var=var)
                    masked = xr.where(ds["zb"] > 0, np.nan, averaged)
                    ds[var] = (("y", "x"), masked.data)

            ds = ds.drop_vars("zb")
            ds.to_netcdf(output_nc_path)

            return ds
        else:
            self.logger.info(
                f"[{case_num}]: Reading existing xboutput_postprocessed.nc file."
            )
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
            The joined xarray.Dataset.
        """

        return xr.concat(postprocessed_files, dim="case_num")
