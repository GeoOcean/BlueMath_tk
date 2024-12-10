import os
import numpy as np
from bluemath_tk.wrappers.xbeach import XBeachModelWrapper


class MyXBeachModelWrapper(XBeachModelWrapper):
    def build_cases(
        self,
        mode: str = "all_combinations",
        grd: str = None,
        bathy: np.ndarray = None,
        friction: np.ndarray = None,
    ):
        # Call the base class method to retain the original functionality
        super().build_cases(mode=mode)
        # Create the cases folders and render the input files
        if not self.cases_context or not self.cases_dirs:
            raise ValueError("Cases were not properly built.")
        for case_context, case_dir in zip(self.cases_context, self.cases_dirs):
            if grd is not None:
                # copy the grd file to the case folder
                self.copy_files(
                    src=grd, dst=os.path.join(case_dir, os.path.basename(grd))
                )
            if bathy is not None:
                # Save the bathymetry to a file
                self.write_array_in_file(
                    array=bathy, filename=os.path.join(case_dir, "bathy_000.dep")
                )
            if friction is not None:
                # Save the friction to a file
                self.write_array_in_file(
                    array=friction, filename=os.path.join(case_dir, "friction.txt")
                )


# Usage example
if __name__ == "__main__":
    # Define the input parameters
    templates_dir = (
        "/home/tausiaj/GitHub-GeoOcean/BlueMath/bluemath_tk/wrappers/xbeach/templates/"
    )
    templates_name = ["params.txt", "loclist.txt"]
    model_parameters = {
        "thetamax": [360],
        "dtheta": [4, 50, 6],
        "spectra": ["JONSWAP", "DIAZIN"],
    }
    output_dir = "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_cases/xbeach/"
    # Create the bathymetry
    bathy = np.random.randn(100, 100)
    # Create the friction
    friction = np.random.randn(100, 100)
    # Create an instance of the XBEACH model wrapper
    swan_model = MyXBeachModelWrapper(
        templates_dir=templates_dir,
        templates_name=templates_name,
        model_parameters=model_parameters,
        output_dir=output_dir,
    )
    # Build the input files
    swan_model.build_cases(
        mode="all_combinations",
        grd="/home/tausiaj/Downloads/caso_XB/laredo_x5.grd",
        bathy=bathy,
        friction=friction,
    )
