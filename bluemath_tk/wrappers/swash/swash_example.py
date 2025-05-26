import inspect
import os.path as op

import numpy as np

from bluemath_tk.datamining.lhs import LHS
from bluemath_tk.datamining.mda import MDA
from bluemath_tk.wrappers.swash.swash_wrapper import HySwashVeggyModelWrapper


if __name__ == "__main__":
    # Define the output directory
    output_dir = (
        "test_cases/HySwashVeggy/"  # CHANGE THIS TO YOUR DESIRED OUTPUT DIRECTORY!
    )
    # Templates directory
    swash_file_path = op.dirname(inspect.getfile(HySwashVeggyModelWrapper))
    templates_dir = op.join(swash_file_path, "templates")
    # Fixed parameters
    fixed_parameters = {
        "dxinp": 1,  # bathymetry grid spacing
        "Plants_ini": 750,  # Vegetation start cell
        "Plants_fin": 900,  # Vegetation end cell
        "comptime": 7200,  # Simulation duration (s)
        "warmup": 7200 * 0.15,  # Warmup duration (s)
        "n_nodes_per_wavelength": 80,  # number of nodes per wavelength
    }
    # LHS
    variables_to_analyse_in_metamodel = [
        "Hs",
        "Hs_L0",
        "WL",
        "vegetation_height",
        "plants_density",
    ]
    lhs_parameters = {
        "num_dimensions": 5,
        "num_samples": 10000,
        "dimensions_names": variables_to_analyse_in_metamodel,
        "lower_bounds": [0.5, 0.005, 0, 0, 0],
        "upper_bounds": [2, 0.05, 1, 1.5, 1000],
    }
    lhs = LHS(num_dimensions=len(variables_to_analyse_in_metamodel))
    df_dataset = lhs.generate(
        dimensions_names=lhs_parameters.get("dimensions_names"),
        lower_bounds=lhs_parameters.get("lower_bounds"),
        upper_bounds=lhs_parameters.get("upper_bounds"),
        num_samples=lhs_parameters.get("num_samples"),
    )
    # MDA
    mda_parameters = {"num_centers": 5}
    mda = MDA(num_centers=mda_parameters.get("num_centers"))
    mda.fit(data=df_dataset)
    metamodel_parameters = mda.centroids.to_dict(orient="list")
    # HySwashVeggyModelWrapper
    swash_wrapper = HySwashVeggyModelWrapper(
        templates_dir=templates_dir,
        metamodel_parameters=metamodel_parameters,
        fixed_parameters=fixed_parameters,
        output_dir=output_dir,
        depth_array=np.loadtxt(op.join(templates_dir, "depth.bot")),
    )
    # Build the input files
    swash_wrapper.build_cases(mode="one_by_one")
    # Run the simulations
    swash_wrapper.run_cases(launcher="docker_serial", num_workers=5)
    # Post-process the results
    swash_wrapper.postprocess_cases(
        output_vars=["Ru2", "Runlev", "Msetup", "Hrms", "Hfreqs"]
    )
    print("Done!")
