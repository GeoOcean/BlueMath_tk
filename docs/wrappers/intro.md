# Numerical Model Wrappers

This section provides general documentation for the model wrappers usage. The wrappers are designed to facilitate the interaction with various numerical models by providing a consistent interface for setting parameters, running simulations, and processing outputs.

For more detailed information, refer to the specific class implementations and their docstrings.

| Model | URL                           | Base Class        | Documentation                        | Owner              |
| ----- | ----------------------------- | ----------------- | ------------------------------------ | ------------------ |
| Swash | https://swash.sourceforge.io/ | SwashModelWrapper | [swash_wrapper.md](swash_wrapper.md) | ricondoa@unican.es |

## BaseModelWrapper

The [`BaseModelWrapper`](base_wrapper.md) class serves as the base class for all model wrappers. It provides common functionality that can be extended by specific model wrappers.

## SwashModelWrapper

The [`SwashModelWrapper`](swash_wrapper.md) class is a specific implementation of the `BaseModelWrapper` for the SWASH model. It extends the base functionality to handle SWASH-specific requirements.

### Example Usage (VeggySwashModelWrapper)

To properly use wrappers, several bullet points must be understood:

1. As shown in the example below, your model of interest base class, ``SwashModelWrapper` in this case, can be inherited to overwrite methods and build / run cases as needed.

2. `build_case` method is essential to properly create the needed files to execute the model in each folder. In the example below, we copy a couple of files and then create an *waves* array that is written in another file.

3. To **RUN** the cases, couple of methos are available: `run_cases` and `run_cases_with_scheduler`. In this section, we will focus on the `run_cases` method, for information regarding the `run_cases_with_scheduler` method, go [here](schedulers.md).
Then, the `run_cases` method allows the user to run the model for the different cases, directory by directory, as it is usually done.

The method parameters are `launcher`, `script`, `params`, `parallel` and `cases_to_run`. Depending on the launcher, there might be some methods available, so please check if the launcher you want to use already has an  implemented method. *If this is the case, overwrite this method in your class if you do not want to use the implemented version.* If your launcher has not an implemented method, you can either implement a method to be called, or use the parameters `launcher`, `script` and `params`, as the wrapper will execute `launcher params script` in each case directory.

If no **launcher** is specified, wrapper will try to run the model locally, so the *executable* file must be set. Other way, wrapper should raise an Error.

```python
import os
import numpy as np
from bluemath_tk.datamining.lhs import LHS
from bluemath_tk.datamining.mda import MDA
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

        # Copy test depth and plants files
        self.copy_files(
            src="C:/Users/UsuarioUC/Documents/BlueMath_tk/test_data/swash-depth.bot",
            dst=os.path.join(case_dir, "depth.bot"),
        )
        self.copy_files(
            src="C:/Users/UsuarioUC/Documents/BlueMath_tk/test_data/swash-plants.txt",
            dst=os.path.join(case_dir, "plants.txt"),
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
        waves = series_TMA(waves=waves_dict, depth=10.0)
        # Save the waves to a file
        self.write_array_in_file(
            array=waves, filename=os.path.join(case_dir, "waves.bnd")
        )

    def build_cases(
        self,
        mode: str = "all_combinations",
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
            )


# Usage example
if __name__ == "__main__":
    # Define the input parameters
    templates_dir = (
        "C:/Users/UsuarioUC/Documents/BlueMath_tk/bluemath_tk/wrappers/swash/templates"
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
    output_dir = "C:/Users/UsuarioUC/Documents/BlueMath_tk/test_cases/swash/"
    # Create an instance of the SWASH model wrapper
    swan_model = VeggySwashModelWrapper(
        templates_dir=templates_dir,
        templates_name=templates_name,
        model_parameters=model_parameters,
        output_dir=output_dir,
    )
    # Build the input files
    swan_model.build_cases(mode="one_by_one")
    # Set the SWASH executable (not used if docker is used)
    swan_model.set_swash_exec(
        "/home/tausiaj/GeoOcean-Execs/SWASH-10.05-Linux/bin/swashrun"
    )
    # Run the model
    swan_model.run_cases(launcher="docker")
```
