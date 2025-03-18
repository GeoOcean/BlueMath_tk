import os
import os.path as op
import numpy as np
import xarray as xr
from bluemath_tk.wrappers.delft3d.delft3d_wrapper import Delft3dModelWrapper
from matplotlib.path import Path


def create_triangle_mask(lon_grid, lat_grid, triangle):
    triangle_path = Path(triangle)
    # if lon_grid.ndim == 1:
    #     lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    points = np.vstack([lon_grid.ravel(), lat_grid.ravel()]).T
    inside_mask = triangle_path.contains_points(points)
    mask = inside_mask.reshape(lon_grid.shape)

    return mask


def trans_geosdeg2mathdeg(geosdir):
    """
    This function transform 0ºN geospatial data to mathematically
    understandable angles
    Args:
        geosdir (numpy-array): Directions in degrees
    Returns:
        [numpy-array]: Mathematically understandable angles in degrees
    """
    
    geosdir = np.where(geosdir<=90,90-geosdir,geosdir)
    geosdir = np.where((geosdir>90)&(geosdir<=180),-(geosdir-90),geosdir)
    geosdir = np.where((geosdir>180)&(geosdir<=270),-90-(geosdir-180),geosdir)
    geosdir = np.where(geosdir>270,(360-geosdir)+90,geosdir)
    
    return geosdir


class GreenSurgeModelWrapper(Delft3dModelWrapper):
    """
    Wrapper for the DelftFM model for Greensurge.
    """

    def generate_wnd_files_D3DFM_Tri(
            self,
            case_context: dict,
            case_dir: str,
            ds_GFD_info: xr.Dataset,
            wind_magnitude: float = 40.0,
            simul_time: int = 26,
            dir_steps: int = 24,
        ):
        """
        TODO: Document!
        """         

        ################# NEW PARAMETERS ################
        real_dirs = np.linspace(0,360,dir_steps+1)[:-1]
        i_tes = case_context.get("tesela")
        i_dir = case_context.get("direction")

        real_dir = real_dirs[i_dir]

        dt_forz = case_context.get("dt_forz")
        #################################################

        node_triangle = ds_GFD_info.node_triangle
        lon_teselas = ds_GFD_info.lon_node.isel(Node=node_triangle).values
        lat_teselas = ds_GFD_info.lat_node.isel(Node=node_triangle).values

        lon_grid = ds_GFD_info.lon_grid.values
        lat_grid = ds_GFD_info.lat_grid.values

        if np.mean(lon_grid)>0:
            lon_grid = lon_grid-360

        x_llcenter = lon_grid[0]
        y_llcenter = lat_grid[0]

        n_cols = len(lon_grid)
        n_rows = len(lat_grid)

        dx = (lon_grid[-1]-lon_grid[0])/n_cols
        dy = (lat_grid[-1]-lat_grid[0])/n_rows

        quantity = ['x_wind','y_wind']
        unit = ['m s-1','m s-1']
        f_name = ['.amu','.amv']

        # for i_tes in range(len(lon_teselas)):
                
        X0, X1, X2 = lon_teselas[i_tes,:]
        Y0, Y1, Y2 = lat_teselas[i_tes,:]

        triangle = [(X0, Y0), (X1, Y1), (X2, Y2)]
        mask = create_triangle_mask(lon_grid, lat_grid, triangle)
        mask_int = np.flip(mask.astype(int), axis=0)                                            #Ojo

            # for i_dir,dir in enumerate(np.linspace(0,360,dir_steps+1)[:-1]):

        u = np.round(-np.cos(trans_geosdeg2mathdeg(real_dir)*np.pi/180)*wind_magnitude,3)
        v = np.round(-np.sin(trans_geosdeg2mathdeg(real_dir)*np.pi/180)*wind_magnitude,3)
        
        u_mat = mask_int * u
        v_mat = mask_int * v

        #print(f"u = {u} and v = {v}")
        self.logger.info(f"Creating Tecelda {i_tes} direction {int(real_dir)} with u = {u} and v = {v}")
        for i_file in range(2):
            # fill each file with values
            # file name to save u,v and pres values
            file_name = op.join(case_dir, f'GFD_wind_file{f_name[i_file]}')

            with open(file_name,'w+') as f:
                f.write('### START OF HEADER\n'+
                        '### This file is created by Deltares\n'+
                        '### Additional commments\n'+
                        'FileVersion = 1.03\n'+
                        'filetype = meteo_on_equidistant_grid\n'+
                        'NODATA_value = -9999.0\n'+
                        f'n_cols = {n_cols}\n'+
                        f'n_rows = {n_rows}\n'+
                        'grid_unit = degree\n'+
                        f'x_llcenter = {x_llcenter}\n'+
                        f'y_llcenter = {y_llcenter}\n'+
                        f'dx = {dx}\n'+
                        f'dy = {dy}\n'+
                        'n_quantity = 1\n'+
                        f'quantity1 = {quantity[i_file]}\n'+
                        f'unit1 = {unit[i_file]}\n'+
                        '### END OF HEADER\n')
                if i_file==0:
                    Mmat = u_mat
                elif i_file==1:
                    Mmat = v_mat
                for time in range(4):
                    if time==0:
                        time_real=time
                    elif time==1:
                        time_real=dt_forz
                    elif time==2:
                        time_real=dt_forz + 0.01
                    elif time==3:
                        time_real=simul_time
                    f.write(f'TIME = {time_real} hours since 2022-01-01 00:00:00 +00:00\n')
                    if time == 0:
                        for line in Mmat:
                            f.write(" ".join(map(str, line)) + "\n")
                    elif time == 1:

                        for line in Mmat:
                            f.write(" ".join(map(str, line)) + "\n")
                    else:
                        Mmat = Mmat*0

                        for line in Mmat:
                            f.write(" ".join(map(str, line)) + "\n")

    def build_case(
        self,
        case_context: dict,
        case_dir: str,
        ds_GFD_info: xr.Dataset,
        grid_nc_file: str,
        wind_magnitude: float = 40.0,
        simul_time: int = 26,
        dir_steps: int = 24,
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
            ds_GFD_info=ds_GFD_info,
            wind_magnitude=wind_magnitude,
            simul_time=simul_time,
            dir_steps=dir_steps,
        )
        # Copy .nc into each dir
        self.copy_files(
            src=grid_nc_file,
            dst=os.path.join(case_dir, os.path.basename(grid_nc_file))
        )

    def build_cases(
        self,
        ds_GFD_info: xr.Dataset,
        grid_nc_file: str,
        wind_magnitude: float = 40.0,
        simul_time: int = 26,
        dir_steps: int = 24,
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
            If the cases were not properly built
        """

        if mode == "all_combinations":
            self.cases_context = self.create_cases_context_all_combinations()
        elif mode == "one_by_one":
            self.cases_context = self.create_cases_context_one_by_one()
        else:
            raise ValueError(f"Invalid mode to create cases: {mode}")
        for case_num, case_context in enumerate(self.cases_context):
            case_context["case_num"] = case_num
            D = case_context["direction"]
            T = case_context["tesela"]
            case_dir = os.path.join(self.output_dir, f"GF_T{T}_D{D}")
            self.cases_dirs.append(case_dir)
            os.makedirs(case_dir, exist_ok=True)
            for template_name in self.templates_name:
                self.render_file_from_template(
                    template_name=template_name,
                    context=case_context,
                    output_filename=os.path.join(case_dir, template_name),
                )
        self.logger.info(
            f"{len(self.cases_dirs)} cases created in {mode} mode and saved in {self.output_dir}"
        )
        if not self.cases_context or not self.cases_dirs:
            raise ValueError("Cases were not properly built.")
        for case_context, case_dir in zip(self.cases_context, self.cases_dirs):
            self.build_case(
                case_context=case_context,
                case_dir=case_dir,
                ds_GFD_info=ds_GFD_info,
                grid_nc_file=grid_nc_file,
                wind_magnitude=wind_magnitude,
                simul_time=simul_time,
                dir_steps = dir_steps,
            )
        # Define the content to write
        sbatch_file_text = """#!/bin/bash
#SBATCH --ntasks=1              # Number of tasks (MPI processes)
#SBATCH --partition=geocean     # Standard output and error log
#SBATCH --nodes=1               # Number of nodes to use
#SBATCH --mem=4gb               # Memory per node in GB (see also --mem-per-cpu)
#SBATCH --time=24:00:00

case_dir=$(ls | awk "NR == $SLURM_ARRAY_TASK_ID")
launchDelft3d.sh --case-dir $case_dir"""
        # Open the file in write mode ('w') and write the content
        with open(f"{self.output_dir}/ls_sbatch.sh", 'w') as file:
            file.write(sbatch_file_text)
        self.logger.info(
            f"SBATCH example file generated in {self.output_dir}"
        )

class Delft3DfmModelWrapper(Delft3dModelWrapper):
    """
    Wrapper for the SWASH model with vegetation.
    """

    def build_case(
        self,
        grid_nc_file: str,
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
            src=grid_nc_file,
            dst=os.path.join(case_dir, os.path.basename(grid_nc_file)))

    def build_cases(
        self,
        grid_nc_file: str,
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
            If the cases were not properly built
        """

        super().build_cases(mode=mode)
        if not self.cases_context or not self.cases_dirs:
            raise ValueError("Cases were not properly built.")
        for case_context, case_dir in zip(self.cases_context, self.cases_dirs):
            self.build_case(
                grid_nc_file=grid_nc_file,
                case_context=case_context,
                case_dir=case_dir,
            )
        sbatch_file_text = """#!/bin/bash
#SBATCH --ntasks=1              # Number of tasks (MPI processes)
#SBATCH --partition=geocean     # Standard output and error log
#SBATCH --nodes=1               # Number of nodes to use
#SBATCH --mem=4gb               # Memory per node in GB (see also --mem-per-cpu)
#SBATCH --time=24:00:00

case_dir=$(ls | awk "NR == $SLURM_ARRAY_TASK_ID")
launchDelft3d.sh --case-dir $case_dir"""
        # Open the file in write mode ('w') and write the content
        with open(f"{self.output_dir}/ls_sbatch.sh", 'w') as file:
            file.write(sbatch_file_text)
        self.logger.info(
            f"SBATCH example file generated in {self.output_dir}"
        )

# Usage example
if __name__ == "__main__":
    # Define the input parameters
    templates_dir = (
        "/home/grupos/geocean/faugeree/BlueMath_tk/bluemath_tk/wrappers/delft3d/templates/"
    )
    # Get 5 cases using LHS and MDA
    model_parameters = {
        "friction": [1, 2, 3]
    }
    output_dir = "/home/grupos/geocean/faugeree/BlueMath_tk/test_cases/delft/"
    # Create an instance of the SWASH model wrapper
    delft3d_wrapper = Delft3dModelWrapper(
        templates_dir=templates_dir,
        model_parameters=model_parameters,
        output_dir=output_dir,
    )
    # Build the input files
    delft3d_wrapper.build_cases(mode="one_by_one", nc_path="/home/grupos/geocean/faugeree/BlueMath_tk/test_data/Santander_mesh_net.nc")
    # List available launchers
    print(delft3d_wrapper.list_available_launchers())
    # Run the model
    delft3d_wrapper.run_cases_bulk(launcher="sbatch --array=0-2%3/home/grupos/geocean/faugeree/BlueMath_tk/test_data")


