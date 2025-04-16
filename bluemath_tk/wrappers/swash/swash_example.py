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
    output_dir = "/home/tausiaj/GitHub-GeoOcean/BlueMath_tk/test_cases/CHY"  # CHANGE THIS TO YOUR DESIRED OUTPUT DIRECTORY!
    # Templates directory
    swash_file_path = op.dirname(inspect.getfile(SwashModelWrapper))
    templates_dir = op.join(swash_file_path, "templates")
    # Fixed parameters
    fixed_parameters = {
        "dxinp": 1.5,  # bathymetry grid spacing
        "default_Cf": 0.002,  # Friction manning coefficient (m^-1/3 s)
        "Cf_ini": 700 / 1.5,  # Friction start cell
        "Cf_fin": 1250 / 1.5,  # Friction end cell
        "comptime": 7200,  # Simulation duration (s)
        "warmup": 7200 * 0.15,  # Warmup duration (s)
        "n_nodes_per_wavelength": 60,  # number of nodes per wavelength
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
    mda_parameters = {"num_centers": 5}
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
    # Run the simulations
    swash_wrapper.run_cases(launcher="docker_serial", num_workers=5)
    # Post-process the results
    swash_wrapper.postprocess_cases(
        output_vars=["Msetup", "Hrms", "Hfreqs"], force=True
    )
    print("Done!")
