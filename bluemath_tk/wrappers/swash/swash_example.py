import inspect
import os
import os.path as op

import numpy as np

from bluemath_tk.datamining.lhs import LHS
from bluemath_tk.datamining.mda import MDA
from bluemath_tk.wrappers.swash.swash_wrapper import SwashModelWrapper


class ChySwashModelWrapper(SwashModelWrapper):
    """
    Wrapper for the SWASH model with friction.
    """

    default_Cf = 0.0002

    def build_case(
        self,
        case_context: dict,
        case_dir: str,
    ) -> None:
        super().build_case(case_context=case_context, case_dir=case_dir)

        # Build the input friction file
        friction = np.ones((len(self.depth_array))) * self.default_Cf
        friction[
            int(self.fixed_parameters["Cf_ini"]) : int(self.fixed_parameters["Cf_fin"])
        ] = case_context["Cf"]
        np.savetxt(os.path.join(case_dir, "friction.txt"), friction, fmt="%.6f")


# Usage example
if __name__ == "__main__":
    # Define the output directory
    output_dir = "/lustre/geocean/DATA/hidronas1/Chy_cases/"  # CHANGE THIS TO YOUR DESIRED OUTPUT DIRECTORY!
    # Templates directory
    swash_file_path = op.dirname(inspect.getfile(SwashModelWrapper))
    templates_dir = op.join(swash_file_path, "templates")
    
    fixed_parameters = {
        "dxinp": 1.5,         # bathymetry grid spacing
        "default_Cf": 0.002,  # Friction manning coefficient (m^-1/3 s)
        "Cf_ini": 700 / 1.5,  # Friction start cell
        "Cf_fin": 1250 / 1.5, # Friction end cell
        "comptime": 7200,     # Simulation duration (s)
        "warmup": 7200*0.15,  # Warmup duration (s)
        "n_nodes_per_wavelength": 60 # number of nodes per wavelength
    }
    
    # LHS
    variables_to_analyse_in_metamodel = ["Hs", "Hs_L0", "WL", "Cf", "Cr"]
    lhs_parameters = {
        "num_samples": 10000,
        "dimensions_names": variables_to_analyse_in_metamodel,
        "lower_bounds": [0.15, 0.0005, -0.6, 0.025, 0.4],
        "upper_bounds": [1.6, 0.009, 0.356, 0.2, 0.8],
    }


    lhs = LHS(num_dimensions=len(variables_to_analyse_in_metamodel))
    df_dataset = lhs.generate(
        dimensions_names=lhs_parameters.get("dimensions_names"),
        lower_bounds=lhs_parameters.get("lower_bounds"),
        upper_bounds=lhs_parameters.get("upper_bounds"),
        num_samples=lhs_parameters.get("num_samples"),
    )
    # MDA
    mda_parameters = {"num_centers": 800}
    mda = MDA(num_centers=mda_parameters.get("num_centers"))
    mda.fit(data=df_dataset)
    metamodel_parameters = mda.centroids.to_dict(orient="list")

    # ChySwashModelWrapper
    swash_wrapper = ChySwashModelWrapper(
        templates_dir=templates_dir,
        metamodel_parameters=metamodel_parameters,
        fixed_parameters=fixed_parameters,
        output_dir=output_dir,
        depth_array=np.loadtxt(op.join(templates_dir, "depth.bot")),
    )
    # Build the input files
    swash_wrapper.build_cases(mode="one_by_one")

    import sys
    sys.exit()
    #swash_wrapper.set_cases_dirs_from_output_dir()
    # Run the model using docker_serial launcher
    #swash_wrapper.run_cases(launcher="nohup launchSwash.sh", num_workers=30)
    # Post-process the output files
    b=swash_wrapper.list_available_postprocess_vars()
    #b = ['Ru2', 'RuDist', 'Msetup', 'Hrms', 'Hfreqs', 'Watlev']
    vars_to_postprocess = ['Msetup', 'Hrms', 'Hfreqs']
    #postprocessed_data = swash_wrapper.postprocess_case(0,case_dir="/home/grupos/geocean/valvanuz/HySwash/BlueMath_Swash_Cases/0000",output_vars=vars_to_postprocess)
   

    # Verification for cases output arriving to a specific Xp in the profile
    for i in range(2, 7):
        postprocessed_data = swash_wrapper.postprocess_case(i,case_dir="/home/grupos/geocean/valvanuz/HySwash/BlueMath_Swash_Cases/{}".format(str(i).zfill(4)),output_vars=vars_to_postprocess)
        print(postprocessed_data)
        if len(postprocessed_data.sel(case_num=i).dropna(dim='Xp').Xp) < 1750:
            print(i)
        else:
            print('OK')


    import xarray as xr    
    a=xr.open_dataset('/home/grupos/geocean/valvanuz/HySwash/BlueMath_Swash_Cases/0000/output_postprocessed.nc')


    ## Datos sobre postproceso
    
    # Un caso ocupa sin comprimir 1.1GB y en lustre 245MB

    # output.tab ocupa 917M sin comprimir y 96MB comprimido con zip o tar.gz
    # (bluemath_dev) (base) [valvanuz@geocean01 0000]$ ls -lh output.tab 
    # -rwxr-xr-x 1 perezdb geocean 917M Jun 20  2024 output.tab
    # (bluemath_dev) (base) [valvanuz@geocean01 0000]$ du -sh output.tab 
    # 180M    output.tab
    # (bluemath_dev) (base) [valvanuz@geocean01 0000]$ du -sh --apparent-size output.tab 
    # 917M    output.tab


