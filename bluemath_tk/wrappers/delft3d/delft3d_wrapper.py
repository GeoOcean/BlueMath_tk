import os
import os.path as op
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from ...additive.greensurge import (
    create_triangle_mask,
    create_triangle_mask_from_points,
    get_regular_grid,
)
from ...core.operations import nautical_to_mathematical
from .._base_wrappers import BaseModelWrapper

sbatch_file_example = """#!/bin/bash
#SBATCH --ntasks=1              # Number of tasks (MPI processes)
#SBATCH --partition=geocean     # Standard output and error log
#SBATCH --nodes=1               # Number of nodes to use
#SBATCH --mem=4gb               # Memory per node in GB (see also --mem-per-cpu)
#SBATCH --time=24:00:00

source /home/grupos/geocean/faugeree/miniforge3/etc/profile.d/conda.sh
conda activate GreenSurge

case_dir=$(ls | awk "NR == $SLURM_ARRAY_TASK_ID")
launchDelft3d.sh --case-dir $case_dir

output_file="${case_dir}/dflowfmoutput/GreenSurge_GFDcase_map.nc"
output_file_compressed="${case_dir}/dflowfmoutput/GreenSurge_GFDcase_map_compressed.nc"
output_file_compressed_tmp="${case_dir}/dflowfmoutput/GreenSurge_GFDcase_map_compressed_tmp.nc"

ncap2 -s 'mesh2d_s1=float(mesh2d_s1)' -v -O "$output_file" "$output_file_compressed_tmp" && {
  ncks -4 -L 4 "$output_file_compressed_tmp" "$output_file_compressed"
  rm "$output_file_compressed_tmp"
  [[ "$SLURM_ARRAY_TASK_ID" -ne 1 ]] && rm "$output_file"
}
"""


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

        self.sbatch_file_example = sbatch_file_example

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


import matplotlib.pyplot as plt


def format_matrix(mat):
    return "\n".join(
        " ".join(f"{x:.1f}" if abs(x) > 0.01 else "0" for x in line) for line in mat
    )


def format_zeros(mat_shape):
    return "\n".join("0 " * mat_shape[1] for _ in range(mat_shape[0]))


def actualize_grid_info(
    path_ds_origin: str,
    ds_GFD_calc_info: xr.Dataset,
) -> None:
    """
    Actualizes the grid information in the GFD calculation info dataset
    by adding the node coordinates and triangle connectivity from the original dataset.
    Parameters
    ----------
    path_ds_origin : str
        Path to the original dataset containing the mesh2d node coordinates.
    ds_GFD_calc_info : xr.Dataset
        The dataset containing the GFD calculation information to be updated.
    Returns
    -------
    ds_GFD_calc_info : xr.Dataset
        The updated dataset with the node coordinates and triangle connectivity.
    """

    ds_ori = xr.open_dataset(path_ds_origin)

    ds_GFD_calc_info["node_computation_longitude"] = (
        ("node_cumputation_index",),
        ds_ori.mesh2d_node_x.values,
    )
    ds_GFD_calc_info["node_computation_latitude"] = (
        ("node_cumputation_index",),
        ds_ori.mesh2d_node_y.values,
    )
    ds_GFD_calc_info["triangle_computation_connectivity"] = (
        ("element_computation_index", "triangle_forcing_nodes"),
        (ds_ori.mesh2d_face_nodes.values - 1).astype("int32"),
    )

    return ds_GFD_calc_info


class GreenSurgeModelWrapper(Delft3dModelWrapper):
    """
    Wrapper for the Delft3d model for Greensurge.
    """

    def generate_grid_forcing_file_D3DFM(
        self,
        case_context: dict,
        case_dir: str,
        ds_GFD_info: xr.Dataset,
    ):
        """
        Generate the wind files for a case.

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        ds_GFD_info : xr.Dataset
            The dataset with the GFD information.
        """
        dir_steps = case_context.get("dir_steps")
        real_dirs = np.linspace(0, 360, dir_steps + 1)[:-1]
        i_tes = case_context.get("tesela")
        i_dir = case_context.get("direction")
        real_dir = real_dirs[i_dir]
        dt_forz = case_context.get("dt_forz")
        wind_magnitude = case_context.get("wind_magnitude")
        simul_time = case_context.get("simul_time")

        node_triangle = ds_GFD_info.triangle_forcing_connectivity.isel(
            element_forcing_index=i_tes
        )
        lon_teselas = ds_GFD_info.node_forcing_longitude.isel(
            node_forcing_index=node_triangle
        ).values
        lat_teselas = ds_GFD_info.node_forcing_latitude.isel(
            node_forcing_index=node_triangle
        ).values

        lon_grid = ds_GFD_info.lon_grid.values
        lat_grid = ds_GFD_info.lat_grid.values

        x_llcenter = lon_grid[0]
        y_llcenter = lat_grid[0]

        n_cols = len(lon_grid)
        n_rows = len(lat_grid)

        dx = (lon_grid[-1] - lon_grid[0]) / n_cols
        dy = (lat_grid[-1] - lat_grid[0]) / n_rows
        X0, X1, X2 = lon_teselas
        Y0, Y1, Y2 = lat_teselas

        triangle = [(X0, Y0), (X1, Y1), (X2, Y2)]
        mask = create_triangle_mask(lon_grid, lat_grid, triangle).astype(int)
        mask_int = np.flip(mask, axis=0)  # Ojo

        u = -np.cos(nautical_to_mathematical(real_dir) * np.pi / 180) * wind_magnitude
        v = -np.sin(nautical_to_mathematical(real_dir) * np.pi / 180) * wind_magnitude
        u_mat = mask_int * u
        v_mat = mask_int * v

        self.logger.info(
            f"Creating Tecelda {i_tes} direction {int(real_dir)} with u = {u} and v = {v}"
        )

        file_name_u = op.join(case_dir, "GFD_wind_file.amu")
        file_name_v = op.join(case_dir, "GFD_wind_file.amv")

        with open(file_name_u, "w+") as fu, open(file_name_v, "w+") as fv:
            fu.write(
                "### START OF HEADER\n"
                + "### This file is created by Deltares\n"
                + "### Additional commments\n"
                + "FileVersion = 1.03\n"
                + "filetype = meteo_on_equidistant_grid\n"
                + "NODATA_value = -9999.0\n"
                + f"n_cols = {n_cols}\n"
                + f"n_rows = {n_rows}\n"
                + "grid_unit = degree\n"
                + f"x_llcenter = {x_llcenter}\n"
                + f"y_llcenter = {y_llcenter}\n"
                + f"dx = {dx}\n"
                + f"dy = {dy}\n"
                + "n_quantity = 1\n"
                + "quantity1 = x_wind\n"
                + "unit1 = m s-1\n"
                + "### END OF HEADER\n"
            )
            fv.write(
                "### START OF HEADER\n"
                + "### This file is created by Deltares\n"
                + "### Additional commments\n"
                + "FileVersion = 1.03\n"
                + "filetype = meteo_on_equidistant_grid\n"
                + "NODATA_value = -9999.0\n"
                + f"n_cols = {n_cols}\n"
                + f"n_rows = {n_rows}\n"
                + "grid_unit = degree\n"
                + f"x_llcenter = {x_llcenter}\n"
                + f"y_llcenter = {y_llcenter}\n"
                + f"dx = {dx}\n"
                + f"dy = {dy}\n"
                + "n_quantity = 1\n"
                + "quantity1 = y_wind\n"
                + "unit1 = m s-1\n"
                + "### END OF HEADER\n"
            )
            for time in range(4):
                if time == 0:
                    time_real = time
                elif time == 1:
                    time_real = dt_forz
                elif time == 2:
                    time_real = dt_forz + 0.01
                elif time == 3:
                    time_real = simul_time
                fu.write(f"TIME = {time_real} hours since 2022-01-01 00:00:00 +00:00\n")
                fv.write(f"TIME = {time_real} hours since 2022-01-01 00:00:00 +00:00\n")
                if time in [0, 1]:
                    fu.write(format_matrix(u_mat) + "\n")
                    fv.write(format_matrix(v_mat) + "\n")
                else:
                    fu.write(format_zeros(u_mat.shape) + "\n")
                    fv.write(format_zeros(v_mat.shape) + "\n")

    def generate_grid_forcing_file_netCDF_D3DFM(
        self,
        case_context: dict,
        case_dir: str,
        ds_GFD_info: xr.Dataset,
    ):
        """
        Generate the wind files for a case.

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        ds_GFD_info : xr.Dataset
            The dataset with the GFD information.
        """

        triangle_index = case_context.get("tesela")
        direction_index = case_context.get("direction")
        wind_direction = ds_GFD_info.wind_directions.values[direction_index]
        wind_speed = case_context.get("wind_magnitude")

        connectivity = ds_GFD_info.triangle_forcing_connectivity
        triangle_longitude = ds_GFD_info.node_forcing_longitude.isel(
            node_forcing_index=connectivity
        ).values
        triangle_latitude = ds_GFD_info.node_forcing_latitude.isel(
            node_forcing_index=connectivity
        ).values

        longitude_points_computation = ds_GFD_info.node_computation_longitude.values
        latitude_points_computation = ds_GFD_info.node_computation_latitude.values

        x0, x1, x2 = triangle_longitude[triangle_index, :]
        y0, y1, y2 = triangle_latitude[triangle_index, :]

        triangle_vertices = [(x0, y0), (x1, y1), (x2, y2)]
        triangle_mask = create_triangle_mask_from_points(
            longitude_points_computation, latitude_points_computation, triangle_vertices
        )

        angle_rad = nautical_to_mathematical(wind_direction) * np.pi / 180
        wind_u = -np.cos(angle_rad) * wind_speed
        wind_v = -np.sin(angle_rad) * wind_speed

        windx = np.zeros((4, len(longitude_points_computation)))
        windy = np.zeros((4, len(longitude_points_computation)))

        windx[0:2, triangle_mask] = wind_u
        windy[0:2, triangle_mask] = wind_v

        ds_forcing = ds_GFD_info[
            [
                "time_forcing_index",
                "node_cumputation_index",
                "node_computation_longitude",
                "node_computation_latitude",
            ]
        ]
        ds_forcing = ds_forcing.rename(
            {
                "time_forcing_index": "time",
                "node_cumputation_index": "node",
                "node_computation_longitude": "longitude",
                "node_computation_latitude": "latitude",
            }
        )
        ds_forcing.attrs = {}
        ds_forcing["windx"] = (("time", "node"), windx)
        ds_forcing["windy"] = (("time", "node"), windy)
        ds_forcing["windx"].attrs = {
            "coordinates": "time node",
            "long_name": "Wind speed in x direction",
            "standard_name": "windx",
            "units": "m s-1",
        }
        ds_forcing["windy"].attrs = {
            "coordinates": "time node",
            "long_name": "Wind speed in y direction",
            "standard_name": "windy",
            "units": "m s-1",
        }
        ds_forcing.to_netcdf(op.join(case_dir, "forcing.nc"))

        self.logger.info(
            f"Creating triangle {triangle_index} direction {int(wind_direction)} with u = {wind_u} and v = {wind_v}"
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

        if case_context.get("forcing_type") == "netCDF":
            self.generate_grid_forcing_file_netCDF_D3DFM(
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

            self.generate_grid_forcing_file_D3DFM(
                case_context=case_context,
                case_dir=case_dir,
                ds_GFD_info=self.ds_GFD_info,
            )
        else:
            raise ("Unknown forcing type")

    def postprocess_case(self, case_dir: str) -> None:
        """
        Postprocess the case output file.

        Parameters
        ----------
        case_dir : str
            The case directory.
        """

        output_file = op.join(case_dir, "dflowfmoutput/GreenSurge_GFDcase_map.nc")
        output_file_compressed = op.join(
            case_dir, "dflowfmoutput/GreenSurge_GFDcase_map_compressed.nc"
        )
        output_file_compressed_tmp = op.join(
            case_dir, "dflowfmoutput/GreenSurge_GFDcase_map_compressed_tmp.nc"
        )
        if case_dir == self.output_dir[0]:
            # If the case_dir is the output_dir, we do not remove the original file
            postprocess_command = f"""
                ncap2 -s 'mesh2d_s1=float(mesh2d_s1)' -v -O "{output_file}" "{output_file_compressed_tmp}"
                ncks -4 -L 4 "{output_file_compressed_tmp}" "{output_file_compressed}"
                rm "{output_file_compressed_tmp}"
            """
        else:
            postprocess_command = f"""
                ncap2 -s 'mesh2d_s1=float(mesh2d_s1)' -v -O "{output_file}" "{output_file_compressed_tmp}"
                ncks -4 -L 4 "{output_file_compressed_tmp}" "{output_file_compressed}"
                rm "{output_file_compressed_tmp}"
                rm "{output_file}"
            """

        self._exec_bash_commands(
            str_cmd=postprocess_command,
            cwd=case_dir,
        )

    def postprocess_cases(self, ds_GFD_info: xr.Dataset, parallel: bool = False):
        """
        Postprocess the cases output files.

        Parameters
        ----------
        ds_GFD_info : xr.Dataset
            The dataset with the GFD information.
        parallel : bool, optional
            Whether to run the postprocessing in parallel. Default is False.
        """

        if (
            self.monitor_cases(
                dia_file_name="dflowfmoutput/GreenSurge_GFDcase.dia",
                value_counts="percentage",
            )
            .loc["FINISHED"]
            .values
            != 100.0
        ):
            raise ValueError(
                "Not all cases are finished. Please check the status of the cases."
            )

        path_computation = op.join(
            self.cases_dirs[0], "dflowfmoutput/GreenSurge_GFDcase_map.nc"
        )
        ds_GFD_info = actualize_grid_info(path_computation, ds_GFD_info)
        dirname, basename = os.path.split(ds_GFD_info.attrs["source"])
        name, ext = os.path.splitext(basename)
        new_filepath = os.path.join(dirname, f"{name}_updated{ext}")
        ds_GFD_info.to_netcdf(new_filepath)

        # case_ext = "dflowfmoutput/GreenSurge_GFDcase_map_compressed.nc"
        case_ext = "dflowfmoutput/GreenSurge_GFDcase_map.nc"

        def preprocess(dataset):
            file_name = dataset.encoding.get("source", None)
            dir_i = int(file_name.split("_D_")[-1].split("/")[0])
            tes_i = int(file_name.split("_T_")[-1].split("_D_")[0])
            dataset = (
                dataset[["mesh2d_s1"]]
                .expand_dims(["forcing_cell"])
                .assign_coords(forcing_cell=[tes_i])
            )
            self.logger.info(
                f"Loaded {file_name} with forcing_cell={tes_i} and dir={dir_i}"
            )
            return dataset

        folder_postprocess = op.join(self.output_dir, "GreenSurge_Postprocess")
        os.makedirs(folder_postprocess, exist_ok=True)

        dir_steps = self.fixed_parameters["dir_steps"]
        for idx in range(dir_steps):
            paths = self.cases_dirs[idx::dir_steps]
            file_paths = [op.join(case_dir, case_ext) for case_dir in paths]
            DS = xr.open_mfdataset(
                file_paths,
                parallel=parallel,
                combine="by_coords",
                preprocess=preprocess,
            )
            DS.load().to_netcdf(op.join(folder_postprocess, f"GreenSurge_DB_{idx}.nc"))
