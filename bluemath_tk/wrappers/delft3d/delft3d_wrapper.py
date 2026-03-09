import os
import os.path as op
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from ...additive.additive import (
    get_regular_grid,
)

from ...tcs.vortex import vortex2delft_3D_FM_nc
from .._base_wrappers import BaseModelWrapper

from .delft3d_utils import (
    generate_grid_forcing_file_D3DFM,
    generate_grid_forcing_file_netCDF_D3DFM,
    sbatch_file_greensurge,
)

class Delft3dModelWrapper(BaseModelWrapper):
    """
    Wrapper for the Delft3d model.

    Attributes
    ----------
    default_parameters : dict
        The default parameters type for the wrapper.
    available_launchers : dict
        The available launchers for the wrapper.
    """

    default_parameters = {}

    available_launchers = {
        "geoocean-cluster": "launchDelft3d.sh",
        "docker_serial": "docker run --rm -v .:/case_dir -w /case_dir geoocean/rocky8 dimr dimr_config.xml",
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
        Initialize the Delft3d model wrapper.
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

        self.sbatch_file_example = sbatch_file_greensurge

        forcing_type = self.fixed_parameters.get("forcing_type", None)
        if forcing_type == "ASCII":
            self.fixed_parameters.setdefault(
                "ExtForceFile", "GreenSurge_GFDcase_wind_ASCII.ext"
            )
        elif forcing_type == "netCDF":
            self.fixed_parameters.setdefault(
                "ExtForceFile", "GreenSurge_GFDcase_wind_netCDF.ext"
            )

    def run_case(
        self,
        case_dir: str,
        launcher: str,
        output_log_file: str = "wrapper_out.log",
        error_log_file: str = "wrapper_error.log",
        postprocess: bool = False,
    ) -> None:
        """
        Run the case based on the launcher specified.

        Parameters
        ----------
        case_dir : str
            The case directory.
        launcher : str
            The launcher to run the case.
        output_log_file : str, optional
            The name of the output log file. Default is "wrapper_out.log".
        error_log_file : str, optional
            The name of the error log file. Default is "wrapper_error.log".
        """

        # Get launcher command from the available launchers
        launcher = self.list_available_launchers().get(launcher, launcher)

        # Run the case in the case directory
        self.logger.info(f"Running case in {case_dir} with launcher={launcher}.")
        output_log_file = op.join(case_dir, output_log_file)
        error_log_file = op.join(case_dir, error_log_file)
        self._exec_bash_commands(
            str_cmd=launcher,
            out_file=output_log_file,
            err_file=error_log_file,
            cwd=case_dir,
        )
        if postprocess:
            self.postprocess_case(case_dir=case_dir)

    def monitor_cases(
        self, dia_file_name: str, value_counts: str = None
    ) -> Union[pd.DataFrame, dict]:
        """
        Monitor the cases based on the status of the .dia files.

        Parameters
        ----------
        dia_file_name : str
            The name of the .dia file to monitor.
        """

        cases_status = {}

        for case_dir in self.cases_dirs:
            case_dir_name = op.basename(case_dir)
            case_dia_file = op.join(case_dir, dia_file_name)
            if op.exists(case_dia_file):
                with open(case_dia_file, "r") as f:
                    lines = f.readlines()
                    if any("finished" in line for line in lines[-15:]):
                        cases_status[case_dir_name] = "FINISHED"
                    else:
                        cases_status[case_dir_name] = "RUNNING"
            else:
                cases_status[case_dir_name] = "NOT STARTED"

        return super().monitor_cases(
            cases_status=cases_status, value_counts=value_counts
        )


class GreenSurgeModelWrapper(Delft3dModelWrapper):
    """
    Wrapper for the Delft3d model for Greensurge.
    """

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
        if case_context.get("SetupType") == "GreenSurge":
            if case_context.get("forcing_type") == "netCDF":
                generate_grid_forcing_file_netCDF_D3DFM(
                    case_context=case_context,
                    case_dir=case_dir,
                    ds_GFD_info=case_context.get("ds_GFD_info"),
                )
            elif case_context.get("forcing_type") == "ASCII":
                if case_context.get("case_num") == 0:
                    ds_GFD_info = case_context.get("ds_GFD_info")
                    lon_grid, lat_grid = get_regular_grid(
                        node_computation_longitude=ds_GFD_info.node_computation_longitude.values,
                        node_computation_latitude=ds_GFD_info.node_computation_latitude.values,
                        node_computation_elements=ds_GFD_info.triangle_computation_connectivity.values,
                    )
                    self.ds_GFD_info = deepcopy(case_context.get("ds_GFD_info"))
                    self.ds_GFD_info["lon_grid"] = np.flip(lon_grid)
                    self.ds_GFD_info["lat_grid"] = lat_grid

                generate_grid_forcing_file_D3DFM(
                    case_context=case_context,
                    case_dir=case_dir,
                    ds_GFD_info=self.ds_GFD_info,
                )
        elif case_context.get("SetupType") == "Dynamic":
            mesh = xr.open_dataset(case_context.get("mesh_path"))
            vortex = case_context.get("vortex")
            forcing = vortex2delft_3D_FM_nc(mesh, vortex)
            forcing.to_netcdf(op.join(case_dir, "forcing.nc"))