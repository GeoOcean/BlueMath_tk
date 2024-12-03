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
        )
        self.set_logger_name(self.__class__.__name__)
        for model_param, param_value in self.model_parameters.items():
            if model_param not in self.default_parameters:
                self.logger.warning(
                    f"Parameter {model_param} is not in the default_parameters"
                )
            else:
                if isinstance(param_value, list) and all(
                    isinstance(item, float) for item in param_value
                ):
                    self.logger.info(
                        f"Parameter {model_param} has the correct type: {type(param_value)}"
                    )
                else:
                    self.logger.error(
                        f"Parameter {model_param} has the wrong type: {type(param_value)}"
                    )

    def build_cases(self, mode: str = "all_combinations"):
        if mode == "all_combinations":
            cases_context = self.create_cases_context_all_combinations()
        elif mode == "one_by_one":
            cases_context = self.create_cases_context_one_by_one()
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return cases_context


class MySwashModelWrapper(SwashModelWrapper):
    def build_cases(
        self,
        mode: str = "all_combinations",
        depth: np.ndarray = None,
        waves: np.ndarray = None,
        plants: np.ndarray = None,
    ):
        # Call the base class method to retain the original functionality
        cases_context = super().build_cases(mode=mode)
        # Create the cases folders and render the input files
        for case_num, case_context in enumerate(cases_context):
            case_folder = os.path.join(self.output_dir, f"{case_num:04}")
            os.makedirs(case_folder, exist_ok=True)
            for template_name in self.templates_name:
                self.render_file_from_template(
                    template_name=template_name,
                    context=case_context,
                    output_filename=os.path.join(case_folder, template_name),
                )
            if depth is not None:
                # Save the depth to a file
                self.write_array_in_file(
                    array=depth, filename=os.path.join(case_folder, "depth.txt")
                )
            if waves is not None:
                # Save the waves to a file
                self.write_array_in_file(
                    array=waves, filename=os.path.join(case_folder, "waves.bnd")
                )
            if plants is not None:
                # Save the plants to a file
                self.write_array_in_file(
                    array=plants, filename=os.path.join(case_folder, "plants.txt")
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
