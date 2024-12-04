import os
import numpy as np
from bluemath_tk.wrappers._base_wrappers import BaseModelWrapper


class SwanModelWrapper(BaseModelWrapper):
    """
    Wrapper for the SWAN model.
    """

    default_parameters = {}

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


class MySwanModelWrapper(SwanModelWrapper):
    def build_cases(
        self,
        mode: str = "all_combinations",
        bathy: np.ndarray = None,
    ):
        # Call the base class method to retain the original functionality
        super().build_cases(mode=mode)
        # Create the cases folders and render the input files
        if not self.cases_context or not self.cases_dirs:
            raise ValueError("Cases were not properly built.")
        for case_context, case_dir in zip(self.cases_context, self.cases_dirs):
            if bathy is not None:
                # Save the bathymetry to a file
                self.write_array_in_file(
                    array=bathy, filename=os.path.join(case_dir, "depth_main.dat")
                )
                # Generate winds boundary conditions
                wind = np.random.rand(10, 10)
                self.write_array_in_file(
                    array=wind, filename=os.path.join(case_dir, "wind_file.dat")
                )


# Usage example
if __name__ == "__main__":
    # Define the input parameters
    templates_dir = (
        "/home/tausiaj/GitHub-GeoOcean/BlueMath/bluemath_tk/wrappers/swan/templates/"
    )
    templates_name = ["wind_input.swn"]
    model_parameters = {"sea_level": [0, 0.5, 1, 1.5], "spec_type": ["JONSWAP", "PM"]}
    output_dir = "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_cases/swan/"
    # Generate bathymetry
    bathymetry = np.random.rand(100, 100)
    # Create the model
    swan_model = MySwanModelWrapper(
        templates_dir=templates_dir,
        templates_name=templates_name,
        model_parameters=model_parameters,
        output_dir=output_dir,
    )
    # Build the input files
    swan_model.build_cases(mode="all_combinations", bathy=bathymetry)
