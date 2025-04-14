import os
import os.path as op

import numpy as np
import xarray as xr
from matplotlib.path import Path

from bluemath_tk.core.operations import nautical_to_mathematical
from bluemath_tk.wrappers.delft3d.delft3d_wrapper import Delft3dModelWrapper


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
    Wrapper for the DelftFM model for Greensurge.
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
        TODO: Document!
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
                # if time == 0 or time == 1:
                #     lines_to_write = []
                #     for line_u, line_v in zip(u_mat, v_mat):
                #         fu.write(" ".join(f"{xu:.1f}" if abs(xu) > 0.01 else "0" for xu in line_u) + "\n")
                #         fv.write(" ".join(f"{xv:.1f}" if abs(xv) > 0.01 else "0" for xv in line_v) + "\n")
                # else:
                #     for line in np.zeros_like(u_mat):
                #         fu.write(" ".join("0" for _ in line) + "\n")
                #         fv.write(" ".join("0" for _ in line) + "\n")
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
            dst=os.path.join(
                case_dir, os.path.basename(case_context.get("grid_nc_file"))
            ),
        )

    def build_cases(
        self,
        mode: str = "one_by_one",
    ) -> None:
        """
        Build the input files for all cases.

        Parameters
        ----------
        mode : str, optional
            The mode to build the cases. Default is "one_by_one".

        Raises
        ------
        ValueError
            If the mode is not valid.
        """

        if mode == "all_combinations":
            self.cases_context = self.create_cases_context_all_combinations()
        elif mode == "one_by_one":
            self.cases_context = self.create_cases_context_one_by_one()
        else:
            raise ValueError(f"Invalid mode to create cases: {mode}")

        for case_num, case_context in enumerate(self.cases_context):
            case_context["case_num"] = case_num
            T = case_context["tesela"]
            D = case_context["direction"]
            case_dir = os.path.join(self.output_dir, f"GF_T{T}_D{D}")
            self.cases_dirs.append(case_dir)
            os.makedirs(case_dir, exist_ok=True)
            case_context.update(self.fixed_parameters)
            self.build_case(
                case_context=case_context,
                case_dir=case_dir,
            )
            for template_name in self.templates_name:
                self.render_file_from_template(
                    template_name=template_name,
                    context=case_context,
                    output_filename=os.path.join(case_dir, template_name),
                )
        self.logger.info(
            f"{len(self.cases_dirs)} cases created in {mode} mode and saved in {self.output_dir}"
        )

        # Save an example sbatch file in the output directory
        with open(f"{self.output_dir}/sbatch_example.sh", "w") as file:
            file.write(self.sbatch_file_example)
        self.logger.info(f"SBATCH example file generated in {self.output_dir}")


class Delft3DfmModelWrapper(Delft3dModelWrapper):
    """
    Wrapper for the Delft3D model with flow mode.
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

        self.copy_files(
            src=case_context.get("grid_nc_file"),
            dst=os.path.join(
                case_dir, os.path.basename(case_context.get("grid_nc_file"))
            ),
        )


# Usage example
if __name__ == "__main__":
    # Define the input parameters
    templates_dir = "/home/grupos/geocean/faugeree/BlueMath_tk/bluemath_tk/wrappers/delft3d/templates/"
    # Get 5 cases using LHS and MDA
    model_parameters = {"friction": [1, 2, 3]}
    output_dir = "/home/grupos/geocean/faugeree/BlueMath_tk/test_cases/delft/"
    # Create an instance of the SWASH model wrapper
    delft3d_wrapper = Delft3dModelWrapper(
        templates_dir=templates_dir,
        model_parameters=model_parameters,
        output_dir=output_dir,
    )
    # Build the input files
    delft3d_wrapper.build_cases(
        mode="one_by_one",
        nc_path="/home/grupos/geocean/faugeree/BlueMath_tk/test_data/Santander_mesh_net.nc",
    )
    # List available launchers
    print(delft3d_wrapper.list_available_launchers())
    # Run the model
    delft3d_wrapper.run_cases_bulk(
        launcher="sbatch --array=0-2%3/home/grupos/geocean/faugeree/BlueMath_tk/test_data"
    )
