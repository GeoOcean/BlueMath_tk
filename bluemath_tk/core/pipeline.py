import itertools
from copy import deepcopy
from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd
import xarray as xr


class BlueMathPipeline:
    """
    A flexible, modular pipeline for chaining together BlueMath models and data processing steps.

    This class allows you to define a sequence of steps, where each step must be a BlueMathModel.
    Each step is defined by a dictionary specifying:
        - 'name': str, a unique identifier for the step.
        - 'model': the model instance to use (or will be created via 'model_init' and 'model_init_params').
        - 'model_init': (optional) a callable/class to instantiate the model.
        - 'model_init_params': (optional) dict of parameters for model initialization.
        - 'fit_method': (optional) str, the method name to call for fitting (default is based on model type).
        - 'fit_params': (optional) dict, parameters for the fit method.
        - 'pipeline_attributes_to_store': (optional) list of attribute names to store for later use.

    The pipeline supports advanced parameter passing, including referencing outputs from previous steps
    and using callables for dynamic parameter computation.
    """

    # Map model class names to their default fit method
    _default_fit_methods = {
        "LHS": "generate",
        "MDA": "fit_predict",
        "KMA": "fit_predict",
        "SOM": "fit_predict",
        "PCA": "fit_transform",
        "RBF": "fit_predict",
    }

    def __init__(self, steps: List[Dict[str, Any]]):
        """
        Initialize the BlueMathPipeline with a sequence of steps.

        Parameters
        ----------
        steps : List[Dict[str, Any]]
            A list of dictionaries, each specifying at least 'name' and 'model',
            and optionally 'model_init', 'model_init_params', 'fit_method', 'fit_params',
            and 'pipeline_attributes_to_store'.
        """

        self.steps = steps
        self._pipeline_attributes = {}  # Stores attributes from previous models for later reference

    @property
    def pipeline_attributes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the stored model attributes from each pipeline step.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            A dictionary mapping step names to dictionaries of stored attributes.

        Raises
        ------
        ValueError
            If the pipeline has not been fit yet and no attributes are stored.
        """

        if len(self._pipeline_attributes) == 0:
            raise ValueError(
                "No model attributes found. Please fit the pipeline first."
            )

        return self._pipeline_attributes

    def fit(self, data: Union[np.ndarray, pd.DataFrame, xr.Dataset] = None):
        """
        Fit all models in the pipeline sequentially, passing the output of each step as input to the next.

        For each step, the model is (optionally) initialized, then fit using the specified method and parameters.
        Parameters and model initialization arguments can be dynamically computed using callables or references
        to previous pipeline attributes.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame, xr.Dataset], optional
            The input data to fit the models. If None, the pipeline expects each step to handle its own data.

        Returns
        -------
        The output of the final step in the pipeline (could be transformed data, predictions, etc.).
        """

        if data is not None:
            data = deepcopy(data)  # Avoid modifying the original input data

        # Iterate over each step in the pipeline
        for step in self.steps:
            # If model needs to be initialized (using model_init and model_init_params)
            if "model_init" in step and "model_init_params" in step:
                for init_param_name, init_param_value in step[
                    "model_init_params"
                ].items():
                    # If the parameter is a callable, call it with (pipeline, step, data)
                    if callable(init_param_value):
                        step["model_init_params"][init_param_name] = init_param_value(
                            self, step, data
                        )
                    # If the parameter is a dict with 'data' and 'function', call the function
                    elif (
                        isinstance(init_param_value, dict)
                        and "data" in init_param_value
                        and "function" in init_param_value
                        and callable(init_param_value["function"])
                    ):
                        # Call the function with (pipeline, step, data)
                        step["model_init_params"][init_param_name] = init_param_value[
                            "function"
                        ](self, step, init_param_value["data"])
                    # If the parameter is the string 'data', replace with the current data
                    elif isinstance(init_param_value, str):
                        if init_param_value == "data":
                            step["model_init_params"][init_param_name] = data

                # Actually instantiate the model with the resolved parameters
                step["model"] = step["model_init"](**step["model_init_params"])

            # Retrieve the model instance for this step
            model = step["model"]
            default_method = self._default_fit_methods.get(type(model).__name__)
            method_name = step.get(
                "fit_method", default_method
            )  # Use step's method or default
            if method_name is None:
                raise ValueError(
                    f"No fit method found for model {type(model).__name__}. Please specify a fit_method in the step."
                )

            # Prepare parameters for the fit method, resolving any callables or references
            params = step.get("fit_params", {}).copy()
            for param_name, param_value in params.items():
                if callable(param_value):
                    params[param_name] = param_value(self, step, data)
                elif (
                    isinstance(param_value, dict)
                    and "data" in param_value
                    and "function" in param_value
                    and callable(param_value["function"])
                ):
                    # Call the function with (data, pipeline, step)
                    params[param_name] = param_value["function"](
                        self, step, param_value["data"]
                    )
                elif isinstance(param_value, str):
                    if param_value == "data":
                        params[param_name] = data

            # Call the fit method on the model with the resolved parameters
            method = getattr(model, method_name)
            try:
                # Some methods expect 'data' as a named argument
                data = method(data=data, **params)
            except Exception as _e:
                # If that fails, try calling without 'data' as a named argument
                data = method(**params)

            # Store specified model attributes for later use, if requested
            if "pipeline_attributes_to_store" in step:
                self._pipeline_attributes[step["name"]] = {
                    attr_name: getattr(model, attr_name)
                    for attr_name in step["pipeline_attributes_to_store"]
                }

        return data

    def _generate_param_combinations(
        self, param_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate all possible combinations of parameters from a parameter grid for a single pipeline step.

        Parameters
        ----------
        param_grid : Dict[str, List[Any]]
            Dictionary mapping parameter names to lists of values to try for each parameter.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries, each representing a unique combination of parameters.
        """

        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))

        return [dict(zip(keys, combo)) for combo in combinations]

    def grid_search(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        param_grid: List[Dict[str, Any]],
        metric: Callable = None,
        target_data: Union[np.ndarray, pd.DataFrame] = None,
        plot: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform a grid search over all possible parameter combinations for all steps in the pipeline.

        This method evaluates every possible combination of parameters (from the provided grids) for each step,
        fits the pipeline, and scores the result using the provided metric or the last model's score method.
        The best parameter set (lowest score) is selected and the pipeline is updated accordingly.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            The input data to fit the models.
        param_grid : List[Dict[str, Any]]
            List of parameter grids for each step in the pipeline. Each grid is a dict mapping parameter names
            to lists of values to try. Parameters can be for either model_init_params or fit_params.
        metric : Callable, optional
            Function to evaluate the final output. Should take (y_true, y_pred) as arguments.
            If None, will use the last model's built-in score method if available.
        target_data : Union[np.ndarray, pd.DataFrame], optional
            Target data to evaluate against if using a custom metric. Required if metric is provided.
        plot : bool, optional
            If True, plot the score for each parameter combination after grid search. Default is False.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
                - 'best_params': the best parameter set for each step
                - 'best_score': the best score achieved
                - 'best_output': the output of the pipeline for the best parameters
                - 'all_results': a list of all parameter sets and their scores/outputs

        Raises
        ------
        ValueError
            If the number of parameter grids does not match the number of pipeline steps,
            or if a metric is provided but no target_data is given.
        """

        if len(param_grid) != len(self.steps):
            raise ValueError(
                "Number of parameter grids must match number of pipeline steps"
            )

        if metric is not None and target_data is None:
            raise ValueError("target_data must be provided when using a custom metric")

        # Generate all possible parameter combinations for each step
        all_param_combinations = []
        for step_params in param_grid:
            step_combinations = self._generate_param_combinations(step_params)
            all_param_combinations.append(step_combinations)

        # Cartesian product: all possible combinations across all steps
        param_combinations = list(itertools.product(*all_param_combinations))

        best_score = float("inf")  # Initialize best score as infinity (lower is better)
        best_params = None
        best_output = None
        all_results = []

        # Iterate over every possible parameter combination
        for step_params in param_combinations:
            pipeline_copy = deepcopy(self)  # Work on a copy to avoid side effects

            # Update each step in the pipeline with the current parameter set
            for step_idx, params in enumerate(step_params):
                step = pipeline_copy.steps[step_idx]
                # Assign parameters to model_init_params or fit_params as appropriate
                if "model_init_params" in step:
                    for param_name, param_value in params.items():
                        if param_name in step["model_init_params"]:
                            step["model_init_params"][param_name] = param_value
                        else:
                            step.setdefault("fit_params", {})[param_name] = param_value
                else:
                    step.setdefault("fit_params", {}).update(params)
                # Re-initialize the model if model_init_params were updated
                if "model_init_params" in step and "model_init" in step:
                    step["model"] = step["model_init"](**step["model_init_params"])

            # Fit the pipeline and get the output for this parameter set
            output = pipeline_copy.fit(data)

            # Score the output using the provided metric or the model's score method
            if metric is not None:
                score = metric(target_data, output)
            else:
                try:
                    score = pipeline_copy.steps[-1]["model"].score(target_data, output)
                except (AttributeError, TypeError):
                    raise ValueError(
                        "Either provide a metric function and target_data, "
                        "or ensure the last model has a score method"
                    )

            # Store the result for this parameter set
            result = {"params": step_params, "score": score, "output": output}
            all_results.append(result)

            # Update the best score/params/output if this is the best so far
            if score < best_score:
                best_score = score
                best_params = step_params
                best_output = output

        # After search, update the pipeline with the best parameters found
        for step_idx, params in enumerate(best_params):
            step = self.steps[step_idx]
            if "model_init_params" in step:
                for param_name, param_value in params.items():
                    if param_name in step["model_init_params"]:
                        step["model_init_params"][param_name] = param_value
                    else:
                        step.setdefault("fit_params", {})[param_name] = param_value
            else:
                step.setdefault("fit_params", {}).update(params)
            if "model_init_params" in step and "model_init" in step:
                step["model"] = step["model_init"](**step["model_init_params"])

        # Plotting if requested
        if plot:
            try:
                import matplotlib.pyplot as plt

                scores = [result["score"] for result in all_results]
                plt.figure(figsize=(6, 4))
                plt.plot(range(len(scores)), scores, marker="o", linestyle="-")
                plt.xlabel("Parameter Combination Index")
                plt.ylabel("Score")
                plt.title("Grid Search Scores for Parameter Combinations")
                plt.grid(True)
                plt.show()
            except ImportError:
                print("matplotlib is not installed. Cannot plot grid search results.")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "best_output": best_output,
            "all_results": all_results,
        }
