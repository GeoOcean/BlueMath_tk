import os.path as op

import numpy as np
import xarray as xr
from matplotlib.path import Path

from ...core.operations import nautical_to_mathematical
from .._base_wrappers import BaseModelWrapper

sbatch_file_example = """
#!/bin/bash
#SBATCH --ntasks=1              # Number of tasks (MPI processes)
#SBATCH --partition=geocean     # Standard output and error log
#SBATCH --nodes=1               # Number of nodes to use
#SBATCH --mem=4gb               # Memory per node in GB (see also --mem-per-cpu)
#SBATCH --time=24:00:00

case_dir=$(ls | awk "NR == $SLURM_ARRAY_TASK_ID")
launchDelft3d.sh --case-dir $case_dir
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

    available_launchers = {"geoocean-cluster": "launchDelft3d.sh"}

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


def create_triangle_mask(
    lon_grid: np.ndarray, lat_grid: np.ndarray, triangle: np.ndarray
) -> np.ndarray:
    """
    Create a mask for a triangle defined by its vertices.

    Parameters
    ----------
    lon_grid : np.ndarray
        The longitude grid.
    lat_grid : np.ndarray
        The latitude grid.
    triangle : np.ndarray
        The triangle vertices.

    Returns
    -------
    np.ndarray
        The mask for the triangle.
    """

    triangle_path = Path(triangle)
    # if lon_grid.ndim == 1:
    #     lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    points = np.vstack([lon_grid.ravel(), lat_grid.ravel()]).T
    inside_mask = triangle_path.contains_points(points)
    mask = inside_mask.reshape(lon_grid.shape)

    return mask


def format_matrix(mat):
    return "\n".join(
        " ".join(f"{x:.1f}" if abs(x) > 0.01 else "0" for x in line) for line in mat
    )


def format_zeros(mat_shape):
    return "\n".join("0 " * mat_shape[1] for _ in range(mat_shape[0]))


class GreenSurgeModelWrapper(Delft3dModelWrapper):
    """
    Wrapper for the Delft3d model for Greensurge.
    """

    def generate_wnd_files_D3DFM_Tri(
        self,
        case_context: dict,
        case_dir: str,
        ds_GFD_info: xr.Dataset,
        wind_magnitude: float,
        simul_time: int,
        dir_steps: int,
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
        wind_magnitude : float
            The wind magnitude.
        simul_time : int
            The simulation time.
        dir_steps : int
            The number of direction steps.
        """

        ################## NEW PARAMETERS ##################
        real_dirs = np.linspace(0, 360, dir_steps + 1)[:-1]
        i_tes = case_context.get("tesela")
        i_dir = case_context.get("direction")
        real_dir = real_dirs[i_dir]
        dt_forz = case_context.get("dt_forz")
        ####################################################

        node_triangle = ds_GFD_info.node_triangle
        lon_teselas = ds_GFD_info.lon_node.isel(Node=node_triangle).values
        lat_teselas = ds_GFD_info.lat_node.isel(Node=node_triangle).values

        lon_grid = ds_GFD_info.lon_grid.values
        lat_grid = ds_GFD_info.lat_grid.values

        x_llcenter = lon_grid[0]
        y_llcenter = lat_grid[0]

        n_cols = len(lon_grid)
        n_rows = len(lat_grid)

        dx = (lon_grid[-1] - lon_grid[0]) / n_cols
        dy = (lat_grid[-1] - lat_grid[0]) / n_rows
        X0, X1, X2 = lon_teselas[i_tes, :]
        Y0, Y1, Y2 = lat_teselas[i_tes, :]

        triangle = [(X0, Y0), (X1, Y1), (X2, Y2)]
        mask = create_triangle_mask(lon_grid, lat_grid, triangle)
        mask_int = np.flip(mask.astype(int), axis=0)  # Ojo

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
        self.generate_wnd_files_D3DFM_Tri(
            case_context=case_context,
            case_dir=case_dir,
            ds_GFD_info=case_context.get("ds_GFD_info"),
            wind_magnitude=case_context.get("wind_magnitude"),
            simul_time=case_context.get("simul_time"),
            dir_steps=case_context.get("dir_steps"),
        )

        # Copy .nc into each dir
        self.copy_files(
            src=case_context.get("grid_nc_file"),
            dst=op.join(case_dir, op.basename(case_context.get("grid_nc_file"))),
        )
