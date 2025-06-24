import functools
from typing import Any, Dict, List

import pandas as pd
import xarray as xr


def validate_data_lhs(func):
    """
    Decorator to validate data in LHS class fit method.

    Parameters
    ----------
    func : callable
        The function to be decorated

    Returns
    -------
    callable
        The decorated function
    """

    @functools.wraps(func)
    def wrapper(
        self,
        dimensions_names: List[str],
        lower_bounds: List[float],
        upper_bounds: List[float],
        num_samples: int,
    ):
        if not isinstance(dimensions_names, list):
            raise TypeError("Dimensions names must be a list")
        if not isinstance(lower_bounds, list):
            raise TypeError("Lower bounds must be a list")
        if not isinstance(upper_bounds, list):
            raise TypeError("Upper bounds must be a list")
        if len(dimensions_names) != len(lower_bounds) or len(lower_bounds) != len(
            upper_bounds
        ):
            raise ValueError(
                "Dimensions names, lower bounds and upper bounds must have the same length"
            )
        if not all(
            [lower <= upper for lower, upper in zip(lower_bounds, upper_bounds)]
        ):
            raise ValueError("Lower bounds must be less than or equal to upper bounds")
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("Variable num_samples must be integer and > 0")
        return func(self, dimensions_names, lower_bounds, upper_bounds, num_samples)

    return wrapper


def validate_data_mda(func):
    """
    Decorator to validate data in MDA class fit method.

    Parameters
    ----------
    func : callable
        The function to be decorated

    Returns
    -------
    callable
        The decorated function
    """

    @functools.wraps(func)
    def wrapper(
        self,
        data: pd.DataFrame,
        directional_variables: List[str] = [],
        custom_scale_factor: dict = {},
        first_centroid_seed: int = None,
        normalize_data: bool = False,
    ):
        if data is None:
            raise ValueError("Data cannot be None")
        elif not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        if not isinstance(directional_variables, list):
            raise TypeError("Directional variables must be a list")
        if not isinstance(custom_scale_factor, dict):
            raise TypeError("Custom scale factor must be a dict")
        if first_centroid_seed is not None:
            if (
                not isinstance(first_centroid_seed, int)
                or first_centroid_seed < 0
                or first_centroid_seed > data.shape[0]
            ):
                raise ValueError(
                    "First centroid seed must be an integer >= 0 and < num of data points"
                )
        if not isinstance(normalize_data, bool):
            raise TypeError("Normalize data must be a boolean")
        return func(
            self,
            data,
            directional_variables,
            custom_scale_factor,
            first_centroid_seed,
            normalize_data,
        )

    return wrapper


def validate_data_kma(func):
    """
    Decorator to validate data in KMA class fit method.

    Parameters
    ----------
    func : callable
        The function to be decorated

    Returns
    -------
    callable
        The decorated function
    """

    @functools.wraps(func)
    def wrapper(
        self,
        data: pd.DataFrame,
        directional_variables: List[str] = [],
        custom_scale_factor: dict = {},
        min_number_of_points: int = None,
        max_number_of_iterations: int = 10,
        normalize_data: bool = False,
        regression_guided: Dict[str, List] = {},
    ):
        if data is None:
            raise ValueError("data cannot be None")
        elif not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not isinstance(directional_variables, list):
            raise TypeError("directional_variables must be a list")
        if not isinstance(custom_scale_factor, dict):
            raise TypeError("custom_scale_factor must be a dict")
        if min_number_of_points is not None:
            if not isinstance(min_number_of_points, int) or min_number_of_points <= 0:
                raise ValueError("min_number_of_points must be integer and > 0")
        if (
            not isinstance(max_number_of_iterations, int)
            or max_number_of_iterations <= 0
        ):
            raise ValueError("max_number_of_iterations must be integer and > 0")
        if not isinstance(normalize_data, bool):
            raise TypeError("normalize_data must be a boolean")
        if not isinstance(regression_guided, dict):
            raise TypeError("regression_guided must be a dictionary")
        if not all(
            isinstance(var, str) and var in data.columns
            for var in regression_guided.get("vars", [])
        ):
            raise TypeError(
                "regression_guided vars must be a list of strings and must exist in data"
            )
        if not all(
            isinstance(alpha, float) and alpha >= 0 and alpha <= 1
            for alpha in regression_guided.get("alpha", [])
        ):
            raise TypeError(
                "regression_guided alpha must be a list of floats between 0 and 1"
            )
        return func(
            self,
            data,
            directional_variables,
            custom_scale_factor,
            min_number_of_points,
            max_number_of_iterations,
            normalize_data,
            regression_guided,
        )

    return wrapper


def validate_data_som(func):
    """
    Decorator to validate data in SOM class fit method.

    Parameters
    ----------
    func : callable
        The function to be decorated

    Returns
    -------
    callable
        The decorated function
    """

    @functools.wraps(func)
    def wrapper(
        self,
        data: pd.DataFrame,
        directional_variables: List[str] = [],
        custom_scale_factor: dict = {},
        num_iteration: int = 1000,
        normalize_data: bool = False,
    ):
        if data is None:
            raise ValueError("Data cannot be None")
        elif not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        if not isinstance(directional_variables, list):
            raise TypeError("Directional variables must be a list")
        if not isinstance(custom_scale_factor, dict):
            raise TypeError("Custom scale factor must be a dict")
        if not isinstance(num_iteration, int) or num_iteration <= 0:
            raise ValueError("Number of iterations must be integer and > 0")
        if not isinstance(normalize_data, bool):
            raise TypeError("Normalize data must be a boolean")
        return func(
            self,
            data,
            directional_variables,
            custom_scale_factor,
            num_iteration,
            normalize_data,
        )

    return wrapper


def validate_data_pca(func):
    """
    Decorator to validate data in PCA class fit method.

    Parameters
    ----------
    func : callable
        The function to be decorated

    Returns
    -------
    callable
        The decorated function
    """

    @functools.wraps(func)
    def wrapper(
        self,
        data: xr.Dataset,
        vars_to_stack: List[str],
        coords_to_stack: List[str],
        pca_dim_for_rows: str,
        windows_in_pca_dim_for_rows: dict = {},
        value_to_replace_nans: dict = {},
        nan_threshold_to_drop: dict = {},
        scale_data: bool = True,
    ):
        if not isinstance(data, xr.Dataset):
            raise TypeError("Data must be an xarray Dataset")
        # Check that all vars_to_stack are in the data
        if not isinstance(vars_to_stack, list) or len(vars_to_stack) == 0:
            raise ValueError("Variables to stack must be a non-empty list")
        for var in vars_to_stack:
            if var not in data.data_vars:
                raise ValueError(f"Variable {var} not found in data")
        # Check that all variables in vars_to_stack have the same coordinates and dimensions
        first_var = vars_to_stack[0]
        first_var_dims = list(data[first_var].dims)
        first_var_coords = list(data[first_var].coords)
        for var in vars_to_stack:
            if list(data[var].dims) != first_var_dims:
                raise ValueError(
                    f"All variables must have the same dimensions. Variable {var} does not match."
                )
            if list(data[var].coords) != first_var_coords:
                raise ValueError(
                    f"All variables must have the same coordinates. Variable {var} does not match."
                )
        # Check that all coords_to_stack are in the data
        if not isinstance(coords_to_stack, list) or len(coords_to_stack) == 0:
            raise ValueError("Coordinates to stack must be a non-empty list")
        for coord in coords_to_stack:
            if coord not in data.coords:
                raise ValueError(f"Coordinate {coord} not found in data.")
        # Check that pca_dim_for_rows is in the data, and window > 0 if provided
        if not isinstance(pca_dim_for_rows, str) or pca_dim_for_rows not in data.dims:
            raise ValueError(
                "PCA dimension for rows must be a string and found in the data dimensions"
            )
        for variable, windows in windows_in_pca_dim_for_rows.items():
            if not isinstance(windows, list):
                raise TypeError("Windows must be a list")
            if not all([isinstance(window, int) and window > 0 for window in windows]):
                raise ValueError("Windows must be a list of integers > 0")
        for variable, threshold in nan_threshold_to_drop.items():
            if not isinstance(threshold, float) or threshold < 0 or threshold > 1:
                raise ValueError("Threshold must be a float between 0 and 1")
        if not isinstance(scale_data, bool):
            raise TypeError("Scale data must be a boolean, either True or False")
        return func(
            self,
            data,
            vars_to_stack,
            coords_to_stack,
            pca_dim_for_rows,
            windows_in_pca_dim_for_rows,
            value_to_replace_nans,
            nan_threshold_to_drop,
            scale_data,
        )

    return wrapper


def validate_data_rbf(func):
    """
    Decorator to validate data in RBF class fit method.

    Parameters
    ----------
    func : callable
        The function to be decorated

    Returns
    -------
    callable
        The decorated function
    """

    @functools.wraps(func)
    def wrapper(
        self,
        subset_data: pd.DataFrame,
        target_data: pd.DataFrame,
        subset_directional_variables: List[str] = [],
        target_directional_variables: List[str] = [],
        subset_custom_scale_factor: dict = {},
        normalize_target_data: bool = True,
        target_custom_scale_factor: dict = {},
        num_workers: int = None,
        iteratively_update_sigma: bool = False,
    ):
        if subset_data is None:
            raise ValueError("Subset data cannot be None")
        elif not isinstance(subset_data, pd.DataFrame):
            raise TypeError("Subset data must be a pandas DataFrame")
        if target_data is None:
            raise ValueError("Target data cannot be None")
        elif not isinstance(target_data, pd.DataFrame):
            raise TypeError("Target data must be a pandas DataFrame")
        if not isinstance(subset_directional_variables, list):
            raise TypeError("Subset directional variables must be a list")
        for directional_variable in subset_directional_variables:
            if directional_variable not in subset_data.columns:
                raise ValueError(
                    f"Directional variable {directional_variable} not found in subset data"
                )
        if not isinstance(target_directional_variables, list):
            raise TypeError("Target directional variables must be a list")
        for directional_variable in target_directional_variables:
            if directional_variable not in target_data.columns:
                raise ValueError(
                    f"Directional variable {directional_variable} not found in target data"
                )
        if not isinstance(subset_custom_scale_factor, dict):
            raise TypeError("Subset custom scale factor must be a dict")
        if not isinstance(normalize_target_data, bool):
            raise TypeError("Normalize target data must be a bool")
        if not isinstance(target_custom_scale_factor, dict):
            raise TypeError("Target custom scale factor must be a dict")
        if num_workers is not None:
            if not isinstance(num_workers, int) or num_workers <= 0:
                raise ValueError("Number of workers must be integer and > 0")
        if not isinstance(iteratively_update_sigma, bool):
            raise TypeError("Iteratively update sigma must be a boolean")
        return func(
            self,
            subset_data,
            target_data,
            subset_directional_variables,
            target_directional_variables,
            subset_custom_scale_factor,
            normalize_target_data,
            target_custom_scale_factor,
            num_workers,
            iteratively_update_sigma,
        )

    return wrapper


def validate_data_xwt(func):
    """
    Decorator to validate data in XWT class fit method.

    Parameters
    ----------
    func : callable
        The function to be decorated

    Returns
    -------
    callable
        The decorated function
    """

    @functools.wraps(func)
    def wrapper(
        self,
        data: xr.Dataset,
        fit_params: Dict[str, Dict[str, Any]] = {},
        variable_to_sort_bmus: str = None,
    ):
        if not isinstance(data, xr.Dataset):
            raise TypeError("Data must be an xarray Dataset")
        if "time" not in data.dims:
            raise ValueError(
                'Time dimension with name "time" not found in data, rename and re-fit'
            )  # TODO: check time is actually datetime
        if not isinstance(fit_params, dict):
            raise TypeError("Fit params must be a dict")
        if "pca" not in fit_params:
            raise ValueError("Fit params must contain PCA parameters")
        if variable_to_sort_bmus is not None:
            if (
                not isinstance(variable_to_sort_bmus, str)
                or variable_to_sort_bmus not in data.data_vars
            ):
                raise TypeError(
                    "variable_to_sort_bmus must be a string and must exist in data variables"
                )
        return func(
            self,
            data,
            fit_params,
            variable_to_sort_bmus,
        )

    return wrapper


def validate_data_calval(func):
    """
    Decorator to validate data in CalVal class fit method.

    Parameters
    ----------
    func : callable
        The function to be decorated

    Returns
    -------
    callable
        The decorated function
    """

    @functools.wraps(func)
    def wrapper(
        self,
        data: pd.DataFrame,
        data_longitude: float,
        data_latitude: float,
        data_to_calibrate: pd.DataFrame,
        max_time_diff: int = 2,
    ):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        if not isinstance(data_longitude, float):
            raise TypeError("Longitude must be a float")
        data_longitude = data_longitude % 360
        if not isinstance(data_latitude, float):
            raise TypeError("Latitude must be a float")
        if not isinstance(data_to_calibrate, pd.DataFrame):
            raise TypeError("Data to calibrate must be a pandas DataFrame")
        if "LONGITUDE" not in data_to_calibrate.columns:
            raise ValueError(
                "Data to calibrate must contain a column named 'LONGITUDE'"
            )
        if "LATITUDE" not in data_to_calibrate.columns:
            raise ValueError("Data to calibrate must contain a column named 'LATITUDE'")
        if "Hs_CAL" not in data_to_calibrate.columns:
            raise ValueError("Data to calibrate must contain a column named 'Hs_CAL'")
        if not isinstance(max_time_diff, int) or max_time_diff <= 0:
            raise ValueError("Maximum time difference must be an integer and > 0")

        return func(
            self,
            data,
            data_longitude,
            data_latitude,
            data_to_calibrate,
            max_time_diff,
        )

    return wrapper
