# Numerical Model Wrappers

This section provides general documentation for the model wrappers usage. The wrappers are designed to facilitate the interaction with various numerical models by providing a consistent interface for setting parameters, running simulations, and processing outputs.

For more detailed information, refer to the specific class implementations and their docstrings.

## BaseModelWrapper

The [`BaseModelWrapper`](base_wrapper.md) class serves as the base class for all model wrappers. It provides common functionality that can be extended by specific model wrappers.

## SwashModelWrapper

The [`SwashModelWrapper`](swash_wrapper.md) class is a specific implementation of the `BaseModelWrapper` for the SWASH model. It extends the base functionality to handle SWASH-specific requirements.

### Example Usage

```python
import os
import numpy as np
from bluemath_tk.datamining.lhs import LHS
from bluemath_tk.datamining.mda import MDA
from bluemath_tk.topo_bathy.profiles import linear
from bluemath_tk.waves.series import series_TMA
from bluemath_tk.wrappers.swash.swash_wrapper import SwashModelWrapper


class VeggySwashModelWrapper(SwashModelWrapper):
    """
    Wrapper for the SWASH model with vegetation.
    """

    def build_case(
        self,
        case_context: dict,
        case_dir: str,
        depth: np.ndarray = None,
        plants: np.ndarray = None,
    ) -> None:
        """
        Build the input files for a case.

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        depth : np.ndarray, optional
            The depth array. Default is None.
        plants : np.ndarray, optional
            The plants array. Default is None.
        """

        if depth is not None:
            # Save the depth to a file
            self.write_array_in_file(
                array=depth, filename=os.path.join(case_dir, "depth.bot")
            )
        if plants is not None:
            # Save the plants to a file
            self.write_array_in_file(
                array=plants, filename=os.path.join(case_dir, "plants.txt")
            )
        # Build the input waves
        waves_dict = {
            "H": case_context["Hs"],
            "T": np.sqrt(
                (case_context["Hs"] * 2 * np.pi) / (9.806 * case_context["Hs_L0"])
            ),
            "gamma": 2,
            "warmup": 180,
            "deltat": 1,
            "tendc": 1800,
        }
        waves = series_TMA(waves=waves_dict, depth=depth[0])
        # Save the waves to a file
        self.write_array_in_file(
            array=waves, filename=os.path.join(case_dir, "waves.bnd")
        )

    def build_cases(
        self,
        mode: str = "all_combinations",
        depth: np.ndarray = None,
        plants: np.ndarray = None,
    ) -> None:
        """
        Build the input files for all cases.

        Parameters
        ----------
        mode : str, optional
            The mode to build the cases. Default is "all_combinations".
        depth : np.ndarray, optional
            The depth array. Default is None.
        plants : np.ndarray, optional
            The plants array. Default is None.

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
                case_context=case_context,
                case_dir=case_dir,
                depth=depth,
                plants=plants,
            )


# Usage example
if __name__ == "__main__":
    # Define the input parameters
    templates_dir = (
        "/home/tausiaj/GitHub-GeoOcean/BlueMath/bluemath_tk/wrappers/swash/templates/"
    )
    templates_name = ["input.sws"]
    # Get 5 cases using LHS and MDA
    lhs = LHS(num_dimensions=3)
    lhs_data = lhs.generate(
        dimensions_names=["Hs", "Hs_L0", "vegetation_height"],
        lower_bounds=[0.5, 0.0, 0.0],
        upper_bounds=[3.0, 0.05, 1.5],
        num_samples=500,
    )
    mda = MDA(num_centers=5)
    mda.fit(data=lhs_data)
    model_parameters = mda.centroids.to_dict(orient="list")
    output_dir = "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_cases/swash/"
    # Create the depth
    """
    dx:      bathymetry mesh resolution at x axes (m)
    h0:      offshore depth (m)
    bCrest:  beach heigh (m)
    m:       profile slope
    Wfore:   flume length before slope toe (m)
    """
    linear_depth = linear(dx=0.05, h0=10, bCrest=5, m=1, Wfore=10)
    # Create the plants
    plants = np.zeros(linear_depth.size)
    plants[(linear_depth < 1) & (linear_depth > 0)] = 1.0
    # Create an instance of the SWASH model wrapper
    swan_model = VeggySwashModelWrapper(
        templates_dir=templates_dir,
        templates_name=templates_name,
        model_parameters=model_parameters,
        output_dir=output_dir,
    )
    # Build the input files
    swan_model.build_cases(mode="one_by_one", depth=linear_depth, plants=plants)
    # Set the SWASH executable
    swan_model.set_swash_exec(
        "/home/tausiaj/GeoOcean-Execs/SWASH-10.05-Linux/bin/swashrun"
    )
    # Run the model
    swan_model.run_cases()
```
