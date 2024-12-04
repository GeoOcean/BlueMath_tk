import os
import numpy as np
from bluemath_tk.wrappers._base_wrappers import BaseModelWrapper
from bluemath_tk.topo_bathy.profiles import linear


class SwashModelWrapper(BaseModelWrapper):
    """
    Wrapper for the SWASH model.
    https://swash.sourceforge.io/online_doc/swashuse/swashuse.html#input-and-output-files
    """

    default_parameters = {
        "param1": str,
        "vegetation_height": float,
    }

    def __init__(
        self,
        templates_dir: str,
        templates_name: dict,
        model_parameters: dict,
        output_dir: str,
    ):
        super().__init__(
            templates_dir=templates_dir,
            templates_name=templates_name,
            model_parameters=model_parameters,
            output_dir=output_dir,
            default_parameters=self.default_parameters,
        )
        self.set_logger_name(self.__class__.__name__)

    def build_case(self, case_context: dict):
        pass

    def build_cases(self, mode: str = "all_combinations"):
        super().build_cases(mode=mode)

    def run_model(self):
        pass


class MySwashModelWrapper(SwashModelWrapper):
    def build_cases(
        self,
        mode: str = "one_by_one",
        depth: np.ndarray = None,
        waves: np.ndarray = None,
        plants: np.ndarray = None,
    ):
        # Call the base class method to retain the original functionality
        super().build_cases(mode=mode)
        # Create the cases folders and render the input files
        if not self.cases_context or not self.cases_dirs:
            raise ValueError("Cases were not properly built.")
        for case_context, case_dir in zip(self.cases_context, self.cases_dirs):
            if depth is not None:
                # Save the depth to a file
                self.write_array_in_file(
                    array=depth, filename=os.path.join(case_dir, "depth.txt")
                )
            if waves is not None:
                # Save the waves to a file
                self.write_array_in_file(
                    array=waves, filename=os.path.join(case_dir, "waves.bnd")
                )
            if plants is not None:
                # Save the plants to a file
                self.write_array_in_file(
                    array=plants, filename=os.path.join(case_dir, "plants.txt")
                )


# Usage example
if __name__ == "__main__":
    # Define the input parameters
    templates_dir = (
        "C:/Users/UsuarioUC/Documents/BlueMath_tk/bluemath_tk/wrappers/swash/templates/"
    )
    templates_name = ["input.sws"]
    model_parameters = {
        "vegetation_height": [1.0, 2.0, 3.0],
        "num_layers": [1, 2, 3],
    }
    output_dir = "C:/Users/UsuarioUC/Documents/BlueMath_tk/tests_data/swash"
    # Create the depth
    """
    dx:      bathymetry mesh resolution at x axes (m)
    h0:      offshore depth (m)
    bCrest:  beach heigh (m)
    m:       profile slope
    Wfore:   flume length before slope toe (m)
    """
    linear_depth = linear(dx=100, h0=10, bCrest=2, m=1, Wfore=10)
    # Create an instance of the SWASH model wrapper
    swan_model = MySwashModelWrapper(
        templates_dir=templates_dir,
        templates_name=templates_name,
        model_parameters=model_parameters,
        output_dir=output_dir,
    )
    # Build the input files
    swan_model.build_cases(mode="all_combinations", depth=linear_depth)
