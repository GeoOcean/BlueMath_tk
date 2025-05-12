import os
import os.path as op
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from ...additive.greensurge import create_triangle_mask
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

    def postprocess_case(self, case_dir: str) -> None:
        """
        Postprocess the case output file.

        Parameters
        ----------
        case_dir : str
            The case directory.
        """

        output_file = op.join(case_dir, "dflowfmoutput/GreenSurge_GFDcase_map.nc.nc")
        output_file_compressed = op.join(
            case_dir, "dflowfmoutput/GreenSurge_GFDcase_map_compressed.nc"
        )
        postprocess_command = f"""
            ncap2 -s 'mesh2d_s1=float(mesh2d_s1)' -v -O "{output_file}" "{output_file_compressed}"
            ncks -4 -L 4 "{output_file_compressed}" "{output_file_compressed}"
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

        case_ext = "/dflowfmoutput/GreenSurge_GFDcase_map.nc"

        NumT = len(ds_GFD_info.teselas)
        ND = len(ds_GFD_info.Wdir)
        NT = np.arange(NumT)
        NDD = np.arange(ND)
        NNT, DDir_BD = np.meshgrid(NT, NDD)

        NT_str = NNT.flatten().astype(str)
        Dir_BD_str = DDir_BD.flatten().astype(str)

        file_paths = np.char.add(self.output_dir, "/GF_T_")
        file_paths = np.char.add(file_paths, NT_str)
        file_paths = np.char.add(file_paths, "_D_")
        file_paths = np.char.add(file_paths, Dir_BD_str)
        file_paths = np.char.add(file_paths, case_ext)

        self.logger.info(f"Read {len(file_paths)} netcdf files")

        DS_tri = xr.open_dataset(file_paths[0])
        el_calc = DS_tri.mesh2d_face_nodes.values.astype(int) - 1
        mesh2d_node_x = DS_tri.mesh2d_node_x.values
        mesh2d_node_y = DS_tri.mesh2d_node_y.values
        mesh2d_nNodes = len(mesh2d_node_x)
        mesh2d_nNodes = np.arange(1, mesh2d_nNodes + 1, 1)
        celdas = len(el_calc)
        celdas = np.arange(1, celdas + 1, 1)
        NN = [1, 2, 3]
        GFD_calculo_info = xr.Dataset(
            coords={
                "celdas": (("celdas"), celdas),
                "mesh2d_nNodes": (("mesh2d_nNodes"), mesh2d_nNodes),
                "NN": (("NN"), NN),
            },
            data_vars={
                "node_triangle": (("celdas", "NN"), el_calc),
                "mesh2d_node_x": (("mesh2d_nNodes"), mesh2d_node_x),
                "mesh2d_node_y": (("mesh2d_nNodes"), mesh2d_node_y),
            },
        )
        GFD_calculo_info.to_netcdf(
            op.join(self.output_dir, "Data_4_GFD_calculo_info.nc"),
            "w",
            "NETCDF3_CLASSIC",
        )

        def preprocess(dataset):
            file_name = dataset.encoding.get("source", "Unknown")
            dir_i = int(file_name.split("_D_")[-1].split("/")[0])
            tes_i = int(file_name.split("_T_")[-1].split("_D_")[0])
            dataset = (
                dataset[["mesh2d_s1"]]
                .expand_dims(["tes", "dir"])
                .assign_coords(tes=[tes_i], dir=[dir_i])
            )
            self.logger.info(f"Loaded {file_name} with tes={tes_i} and dir={dir_i}")
            return dataset

        folder_postprocess = op.join(self.output_dir, "GreenSurge_DB")
        os.makedirs(folder_postprocess, exist_ok=True)

        D1 = xr.open_mfdataset(
            file_paths,
            parallel=parallel,
            combine="by_coords",
            preprocess=preprocess,
        )

        def save_direction(idx):
            D1.load().isel(dir=idx).to_netcdf(
                op.join(folder_postprocess, f"GreenSurge_DB_{idx}.nc")
            )
            self.logger.info(f"Saved GreenSurge_DB_{idx}.nc")

        list(map(save_direction, NDD))
        T_values = False
        D_values = False
        post_eje = False

        return D1, T_values, D_values, post_eje
