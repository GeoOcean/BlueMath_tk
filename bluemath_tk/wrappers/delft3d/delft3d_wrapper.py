import os
import os.path as op
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from ...additive.greensurge import create_triangle_mask_from_points
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

    def run_case(
        self,
        case_dir: str,
        launcher: str,
        output_log_file: str = "wrapper_out.log",
        error_log_file: str = "wrapper_error.log",
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


# def format_matrix(mat):
#     return "\n".join(
#         " ".join(f"{x:.1f}" if abs(x) > 0.01 else "0" for x in line) for line in mat
#     )


# def format_zeros(mat_shape):
#     return "\n".join("0 " * mat_shape[1] for _ in range(mat_shape[0]))


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

        # Generate wind file
        self.generate_grid_forcing_file_D3DFM(
            case_context=case_context,
            case_dir=case_dir,
            ds_GFD_info=case_context.get("ds_GFD_info"),
        )

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
        dirname, basename = os.path.split(ds_GFD_info.encoding["source"])
        name, ext = os.path.splitext(basename)
        new_filepath = os.path.join(dirname, f"{name}_updated{ext}")
        ds_GFD_info.to_netcdf(new_filepath)

        case_ext = "dflowfmoutput/GreenSurge_GFDcase_map_compressed.nc"

        def preprocess(dataset):
            file_name = dataset.encoding.get("source", "Unknown")
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
