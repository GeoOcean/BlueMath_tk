import itertools
from copy import deepcopy
from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd
import xarray as xr


class BlueMathPipeline:
    """
    Flexible pipeline for BlueMath models, allowing each step to specify the model, method names, and parameters.

    Each step is a dict with at least:
        - 'name': str, unique step name
        - 'model': the model instance
      Optionally:
        - 'fit_method': str, method name to call for fitting (default: see _default_fit_methods)
        - 'fit_params': dict, parameters for the fit method (default: {})
        - 'pipeline_attributes': dict, mapping of attribute names to be used in later steps
    """

    # Map model class names to their default fit method
    _default_fit_methods = {
        "PCA": "fit_transform",  # from bluemath_tk.datamining.pca.PCA
        "KMA": "fit_predict",  # from bluemath_tk.datamining.kma.KMA
    }

    def __init__(self, steps: List[Dict[str, Any]]):
        """
        Initialize the BlueMathPipeline.

        Parameters
        ----------
        steps : List[Dict[str, Any]]
            A list of dicts, each specifying at least 'name' and 'model',
            and optionally 'fit_method' and 'fit_params'.
        """

        self.steps = steps
        self._pipeline_attributes = {}  # Store attributes from previous models

    @property
    def pipeline_attributes(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns the model attributes.
        """

        if len(self._pipeline_attributes) == 0:
            raise ValueError(
                "No model attributes found. Please fit the pipeline first."
            )

        return self._pipeline_attributes

    def fit(self, data: Union[np.ndarray, pd.DataFrame, xr.Dataset] = None):
        """
        Fit the pipeline models using the specified method and parameters for each step.
        The default method is determined by the _default_fit_methods dict based on model class name.
        If not found, defaults to 'fit'.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame, xr.Dataset], optional
            The input data to fit the models. Default is None.
        """

        if data is not None:
            data = deepcopy(data)  # Make sure we don't modify the original data

        # Iterate over the steps, fitting the models and storing the attributes
        for step in self.steps:
            if "model_init" in step and "model_init_params" in step:
                for init_param_name, init_param_value in step[
                    "model_init_params"
                ].items():
                    if callable(init_param_value):
                        step["model_init_params"][init_param_name] = init_param_value(
                            self, step, data
                        )
                    elif (
                        isinstance(init_param_value, dict)
                        and "data" in init_param_value
                        and "function" in init_param_value
                        and callable(init_param_value["function"])
                    ):
                        # Call the function with (data, pipeline, step)
                        step["model_init_params"][init_param_name] = init_param_value[
                            "function"
                        ](self, step, init_param_value["data"])
                    elif isinstance(init_param_value, str):
                        if init_param_value == "data":
                            step["model_init_params"][init_param_name] = data

                # Initialize the model with the parameters
                step["model"] = step["model_init"](**step["model_init_params"])

            # Fit the model with the parameters
            model = step["model"]
            model_class_name = type(model).__name__
            default_method = self._default_fit_methods.get(model_class_name, "fit")
            method_name = step.get("fit_method", default_method)

            # Get parameters, resolving any references to previous model attributes
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

            # Call the method with the parameters
            method = getattr(model, method_name)
            try:
                data = method(data=data, **params)
            except Exception as _e:
                # print(f"Error in {model_class_name} with method {method_name}: {e}")
                data = method(**params)

            # Store model attributes for later use if specified
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
        Generate all possible combinations of parameters from the parameter grid.

        Parameters
        ----------
        param_grid : Dict[str, List[Any]]
            Dictionary mapping parameter names to lists of values to try.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing all possible parameter combinations.
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
    ) -> Dict[str, Any]:
        """
        Perform grid search to find the best parameters for each step in the pipeline.
        Evaluates all possible combinations of parameters across all steps together.
        Parameters can be optimized in both model_init_params and fit_params.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            The input data to fit the models.
        param_grid : List[Dict[str, Any]]
            List of parameter grids for each step in the pipeline. Each grid should be a dict
            mapping parameter names to lists of values to try. Parameters can be for either
            model_init_params or fit_params.
        metric : Callable, optional
            Function to evaluate the final output. Should take (y_true, y_pred) as arguments.
            If None, will use the last model's built-in score method if available.
        target_data : Union[np.ndarray, pd.DataFrame], optional
            Target data to evaluate against if using a custom metric. Required if metric is provided.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the best parameters and scores for each step.
        """

        if len(param_grid) != len(self.steps):
            raise ValueError(
                "Number of parameter grids must match number of pipeline steps"
            )

        if metric is not None and target_data is None:
            raise ValueError("target_data must be provided when using a custom metric")

        # Generate all possible combinations of parameters across all steps
        all_param_combinations = []
        for step_params in param_grid:
            step_combinations = self._generate_param_combinations(step_params)
            all_param_combinations.append(step_combinations)

        # Generate all possible combinations of step parameters
        param_combinations = list(itertools.product(*all_param_combinations))

        best_score = float("inf")
        best_params = None
        best_output = None
        all_results = []

        # For each combination of parameters across all steps
        for step_params in param_combinations:
            # Create a copy of the pipeline with current parameters
            pipeline_copy = deepcopy(self)

            # Update parameters for each step
            for step_idx, params in enumerate(step_params):
                step = pipeline_copy.steps[step_idx]

                # Handle model initialization parameters
                if "model_init_params" in step:
                    for param_name, param_value in params.items():
                        if param_name in step["model_init_params"]:
                            # Update model_init_params
                            step["model_init_params"][param_name] = param_value
                        else:
                            # If not in model_init_params, put in fit_params
                            step.setdefault("fit_params", {})[param_name] = param_value
                else:
                    # If no model_init_params, just update fit_params
                    step.setdefault("fit_params", {}).update(params)

                # Reinitialize model if model_init_params were updated
                if "model_init_params" in step and "model_init" in step:
                    step["model"] = step["model_init"](**step["model_init_params"])

            # Fit the pipeline and get predictions
            output = pipeline_copy.fit(data)

            # Calculate score
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

            # Store results
            result = {"params": step_params, "score": score, "output": output}
            all_results.append(result)

            # Update best parameters if this combination is better
            if score < best_score:
                best_score = score
                best_params = step_params
                best_output = output

        # Update the pipeline with best parameters
        for step_idx, params in enumerate(best_params):
            step = self.steps[step_idx]

            # Update model initialization parameters
            if "model_init_params" in step:
                for param_name, param_value in params.items():
                    if param_name in step["model_init_params"]:
                        step["model_init_params"][param_name] = param_value
                    else:
                        step.setdefault("fit_params", {})[param_name] = param_value
            else:
                step.setdefault("fit_params", {}).update(params)

            # Reinitialize model if model_init_params were updated
            if "model_init_params" in step and "model_init" in step:
                step["model"] = step["model_init"](**step["model_init_params"])

        return {
            "best_params": best_params,
            "best_score": best_score,
            "best_output": best_output,
            "all_results": all_results,
        }
