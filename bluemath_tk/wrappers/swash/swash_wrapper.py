import subprocess
from bluemath_tk.wrappers._base_wrappers import BaseModelWrapper


class SwashModelWrapper(BaseModelWrapper):
    """
    Wrapper for the SWASH model.
    https://swash.sourceforge.io/online_doc/swashuse/swashuse.html#input-and-output-files
    """

    default_parameters = {
        "param1": "default_type1",
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

    def run_model(self, input_file):
        # Implement the logic to run the SWASH model with the given input file
        # For example, using subprocess to call the SWASH executable
        result = subprocess.run(
            ["swashrun", input_file], capture_output=True, text=True
        )
        return result.stdout, result.stderr


# Usage example
if __name__ == "__main__":
    # Define the input parameters
    templates_dir = (
        "/home/tausiaj/GitHub-GeoOcean/BlueMath/bluemath_tk/wrappers/swash/templates/"
    )
    templates_name = ["input.sws"]
    model_parameters = {
        "vegetation_height": [1, 2, 3],
    }
    output_dir = "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_cases/swash/"
    # Create an instance of the SWASH model wrapper
    swan_model = SwashModelWrapper(
        templates_dir=templates_dir,
        templates_name=templates_name,
        model_parameters=model_parameters,
        output_dir=output_dir,
    )
    # Build the input files
    swan_model.build_cases()
    # Run the model
    # result = swan_model.run_model(input_file)
