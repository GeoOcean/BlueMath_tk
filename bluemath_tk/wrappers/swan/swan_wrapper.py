import os
import numpy as np
from bluemath_tk.wrappers._base_wrappers import BaseModelWrapper


class SwanModelWrapper(BaseModelWrapper):
    """
    Wrapper for the SWAN model.
    https://swanmodel.sourceforge.io/online_doc/swanuse/swanuse.html
    """

    default_parameters = {
        "hs": float,
        "tp": float,
        "dir": float,
        "spr": float,
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

    def build_case_hyswan(self, case_context: dict, case_dir: str):
        if case_context.get("wind"):
            wind = np.random.rand(10, 10)
            self.write_array_in_file(
                array=wind, filename=os.path.join(case_dir, "wind_file.dat")
            )

    def build_cases(self, mode: str = "all_combinations", swan_type: str = "HySwan"):
        super().build_cases(mode=mode)
        if not self.cases_context or not self.cases_dirs:
            raise ValueError("Cases were not properly built.")
        for case_context, case_dir in zip(self.cases_context, self.cases_dirs):
            if swan_type == "HySwan":
                self.build_case_hyswan(case_context=case_context, case_dir=case_dir)

    def run_model(self):
        pass


class MySwanModelWrapper(SwanModelWrapper):
    def build_cases(
        self,
        mode: str = "all_combinations",
    ):
        # Call the base class method to retain the original functionality
        super().build_cases(mode=mode)


# Usage example
if __name__ == "__main__":
    # Define the input parameters
    templates_dir = (
        "/home/tausiaj/GitHub-GeoOcean/BlueMath/bluemath_tk/wrappers/swan/templates/"
    )
    templates_name = ["struc_input.swn"]
    hs = np.linspace(0, 6, 5)
    tp = np.linspace(0, 20, 5)
    dir = np.linspace(0, 360, 5)
    spr = np.linspace(0, 50, 5)
    wind = np.random.choice([False, True], size=5)
    model_parameters = {
        "hs": hs,
        "tp": tp,
        "dir": dir,
        "spr": spr,
        "wind": wind,
    }
    output_dir = "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_cases/swan/"
    # Instantiate the model
    swan_model = SwanModelWrapper(
        templates_dir=templates_dir,
        templates_name=templates_name,
        model_parameters=model_parameters,
        output_dir=output_dir,
    )
    # Build the input files
    swan_model.build_cases(mode="one_by_one", swan_type="HySwan")
